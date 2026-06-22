"""Ragas evaluation runner — faithfulness + context precision/recall.

OFFLINE / batch only.  This module MUST NOT be imported by any hot-path module
(api/, agent/, retrieval/, generation/).  The two-clocks rule (ADR-006,
CONVENTIONS §10) places all LLM-judging in the offline/eval path.

``RagasRunner`` computes Ragas metrics using the **local judge LLM** (Lynx-8B /
Selene-Mini-8B), never the generator.  This enforces the generator ≠ judge
contamination rule (lld-eval-observability §4/§9) for Ragas as well.

``ragas`` is **lazily imported** inside the method body so the module can be
collected by pytest even when ragas is not installed (only the dev group plus
optional ``eval-judges`` group installs it).  A clear error is raised at call
time if ragas is absent.

Ragas is wired to the judge endpoint via ``LangchainLLMWrapper`` +
``LangchainEmbeddingsWrapper``, not the generator endpoint.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from iw_rag.domain.errors import ConfigError


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One (question, answer, contexts, ground_truth) sample for Ragas.

    Attributes:
        question: The query.
        answer: The generated answer to evaluate.
        contexts: Retrieved context strings used by the answerer.
        ground_truth: Optional reference answer (required for some metrics).
    """

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


@dataclass(frozen=True, slots=True)
class GenerationScores:
    """Per-sample Ragas generation metric scores.

    Attributes:
        question: The evaluated question.
        faithfulness: Ragas faithfulness score in [0, 1].
        context_precision: Ragas context precision in [0, 1].
        context_recall: Ragas context recall in [0, 1].
        factual_correctness: Ragas factual correctness in [0, 1]
            (reference-required; ``None`` when no reference answer).
        noise_sensitivity: Ragas noise sensitivity in [0, 1], LOWER is
            better (reference-required; ``None`` when no reference answer).
        errored: True if any metric failed (NaN / parse error).
        error_detail: Error message when errored=True.
    """

    question: str
    faithfulness: float | None
    context_precision: float | None
    context_recall: float | None
    factual_correctness: float | None = None
    noise_sensitivity: float | None = None
    errored: bool = False
    error_detail: str = ""


# Supported Ragas metrics (the default set per lld-eval-observability §2)
_DEFAULT_METRICS = ("faithfulness", "context_precision", "context_recall")


class RagasRunner:
    """Runs Ragas evaluation using the local offline judge LLM + BGE-M3 embeddings.

    The judge LLM (Lynx-8B / Selene-Mini-8B) is used — NEVER the generator.
    The embedder (BGE-M3) is reused from the ingestion/retrieval path.

    ``ragas`` is lazy-imported; an explicit :class:`~iw_rag.domain.errors.ConfigError`
    is raised if it is absent (it lives in the optional ``eval-judges`` dependency
    group).

    Attributes:
        _judge_llm: The judge LLM client (offline judge endpoint).
        _embed: The embedder client (BGE-M3).
        _generator_model: Generator model name (contamination guard check).
        _judge_model: Judge model name.
    """

    def __init__(
        self,
        judge_llm: Any,
        embed: Any,
        generator_model: str,
        judge_model: str,
    ) -> None:
        """Initialise the RagasRunner and enforce generator ≠ judge.

        Args:
            judge_llm: An LLM client pointed at the offline judge endpoint
                (``VLLM_JUDGE_URL``), not the hot-path generator.
            embed: An embeddings client (BGE-M3) for context-precision/recall.
            generator_model: The generator model name (contamination guard).
            judge_model: The judge model name (must differ from generator).

        Raises:
            ConfigError: If ``judge_model == generator_model`` (contamination).
        """
        from iw_rag.eval.golden_set import assert_generator_not_judge

        assert_generator_not_judge(generator_model, [judge_model])
        self._judge_llm = judge_llm
        self._embed = embed
        self._generator_model = generator_model
        self._judge_model = judge_model

    async def score(
        self,
        samples: Sequence[EvalSample],
        metrics: tuple[str, ...] = _DEFAULT_METRICS,
    ) -> list[GenerationScores]:
        """Score a batch of samples with Ragas faithfulness + context P/R.

        Uses the local judge LLM (Lynx-8B / Selene-Mini-8B) and BGE-M3
        embeddings.  Ragas is lazily imported here; a :class:`ConfigError` is
        raised if it is not installed.  Per-sample NaN / parse failures are
        recorded as ``errored=True`` rather than dropped silently
        (lld-eval-observability §9).

        Args:
            samples: Batch of :class:`EvalSample` inputs.
            metrics: Ragas metric names to compute.  Defaults to faithfulness,
                context_precision, context_recall.

        Returns:
            List of :class:`GenerationScores`, one per sample, in input order.

        Raises:
            ConfigError: If ragas is not installed.
            ConfigError: If the samples list is empty.
        """
        try:
            # Lazy import — ragas lives in the optional eval-judges group.
            # ragas 0.4.x API: SingleTurnSample/EvaluationDataset with the
            # renamed fields (user_input/response/retrieved_contexts/reference);
            # evaluate(llm=, embeddings=) takes the wrappers directly.
            import ragas  # noqa: F401
            from ragas import EvaluationDataset, SingleTurnSample, evaluate
            from ragas.dataset_schema import EvaluationResult
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper
            from ragas.metrics import (
                FactualCorrectness,
                NoiseSensitivity,
                context_precision,
                context_recall,
                faithfulness,
            )
            from ragas.run_config import RunConfig
        except ImportError as exc:
            raise ConfigError(
                "ragas is not installed. Add it via the 'eval-judges' dependency "
                "group: uv sync --group eval-judges"
            ) from exc

        if not samples:
            raise ConfigError("RagasRunner.score called with empty samples list.")

        # Wire to the LOCAL judge endpoint (not the generator).
        llm_wrapper = LangchainLLMWrapper(self._judge_llm)
        embed_wrapper = LangchainEmbeddingsWrapper(self._embed)

        # Build the metric objects (FactualCorrectness/NoiseSensitivity are
        # classes in 0.4.x; the others are module singletons).  The output
        # column name equals each metric's ``.name``.
        metric_map = {
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "factual_correctness": FactualCorrectness(),
            "noise_sensitivity": NoiseSensitivity(),
        }
        active_metrics = [metric_map[m] for m in metrics if m in metric_map]

        # ragas 0.4.x EvaluationDataset / SingleTurnSample (renamed fields).
        dataset = EvaluationDataset(
            samples=[
                SingleTurnSample(
                    user_input=s.question,
                    response=s.answer,
                    retrieved_contexts=list(s.contexts),
                    reference=s.ground_truth,
                )
                for s in samples
            ]
        )

        # The local 8B judge is COMPUTE-bound, not concurrency-bound: a single
        # 8B model on one GPU saturates at low concurrency, so piling on workers
        # only inflates per-request latency (measured e2e p99 ~28 s at 16
        # workers) until requests hit the client timeout.  4 workers is the
        # sweet spot (BENCH-000014: ~34 min, all metrics, no timeouts).  The
        # generous per-job timeout absorbs the inherently long claim-level
        # calls; real speed-ups come from running fewer heavy metrics (see the
        # ``heavy`` toggle in the engine), not more workers.
        run_config = RunConfig(timeout=900, max_workers=4, max_retries=2, max_wait=120)

        # ``evaluate`` is SYNCHRONOUS and CPU/IO-heavy (it drives its own
        # internal event loop over minutes of judge calls).  Calling it inline
        # would block the api's event loop for the whole Ragas phase, freezing
        # every other request (the Operator Console included).  Run it in a
        # worker thread so the api stays responsive during a benchmark.
        import asyncio
        import functools

        results = await asyncio.to_thread(
            functools.partial(
                evaluate,
                dataset=dataset,
                metrics=active_metrics,
                llm=llm_wrapper,
                embeddings=embed_wrapper,
                show_progress=False,
                run_config=run_config,
            )
        )
        assert isinstance(results, EvaluationResult)
        results_df = results.to_pandas()

        scores: list[GenerationScores] = []

        def _safe_float(row: Any, col: str) -> float | None:
            """Extract a metric float from a Ragas result row (None on NaN/missing).

            ragas 0.4.x suffixes some metric columns with the metric's mode —
            e.g. ``factual_correctness(mode=f1)`` and
            ``noise_sensitivity(mode=relevant)`` — so an exact-name lookup
            misses them.  Match the exact column first, then any column whose
            name is ``<col>(...)``.
            """
            import math

            if row is None:
                return None
            actual = (
                col
                if col in row.index
                else next(
                    (c for c in row.index if str(c).startswith(col + "(")),
                    None,
                )
            )
            if actual is None:
                return None
            try:
                f = float(row[actual])
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        for i, sample in enumerate(samples):
            row = results_df.iloc[i] if i < len(results_df) else None
            if row is None:
                scores.append(
                    GenerationScores(
                        question=sample.question,
                        faithfulness=None,
                        context_precision=None,
                        context_recall=None,
                        errored=True,
                        error_detail="No result row returned by Ragas.",
                    )
                )
                continue

            scores.append(
                GenerationScores(
                    question=sample.question,
                    faithfulness=_safe_float(row, "faithfulness"),
                    context_precision=_safe_float(row, "context_precision"),
                    context_recall=_safe_float(row, "context_recall"),
                    factual_correctness=_safe_float(row, "factual_correctness"),
                    noise_sensitivity=_safe_float(row, "noise_sensitivity"),
                    errored=False,
                )
            )
        return scores
