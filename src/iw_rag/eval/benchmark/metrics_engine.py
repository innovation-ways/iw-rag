"""Metrics engine — the per-run T1/T2/T3 unifier (F-00093, S02).

The single per-run computation that unifies the three metric tiers into one
typed result which report assembly (S06) persists into ``benchmark_reports``:

* **T1 (retrieval, deterministic)** — recall@k, nDCG@k, MRR, computed by
  reusing :func:`iw_rag.eval.retrieval_metrics.score_retrieval`.  Never
  reimplement the metric math (CONVENTIONS §1 / lld-eval-observability §2).

* **T2 (judged, reference-free)** — faithfulness (Lynx-8B /
  :class:`~iw_rag.eval.judges.FaithfulnessJudge` and/or Ragas), answer
  relevancy, context relevance, rubric (Selene-Mini-8B /
  :class:`~iw_rag.eval.judges.GradingJudge`).

* **T3 (judged, reference-required)** — context recall, factual
  correctness, noise sensitivity — **only** computed when the dataset
  supplies ``reference_answer`` (AC4 / Invariant 5).  When no reference
  answers exist, the engine makes no judge call for them and the
  corresponding fields are ``None`` (never zero, never guessed).

The engine is **injector-first**: judges and the Ragas runner are passed
in, so unit tests can supply fakes with no network.  The engine never
constructs an LLM client or hits an endpoint.  The generator is never
used as a judge — the existing :class:`FaithfulnessJudge` /
:class:`GradingJudge` / :class:`RagasRunner` already enforce
``assert_generator_not_judge`` at construction time (Invariant 6).

Failure / boundary handling (per F-00093 Feature_Design §Boundary Behavior):

* **Empty golden set** — raises :class:`~iw_rag.domain.errors.ConfigError`
  (never produces an all-zero report).
* **Judge endpoint down** — the affected T2/T3 metrics are recorded as
  ``None`` (errored), the per-tier errored count is incremented, T1
  retrieval metrics and the report are still produced.  The engine
  never falls back to the generator as judge.
* **Per-sample NaN/parse failure** — mirrors
  :mod:`iw_rag.eval.ragas_runner` ``errored`` handling; does not crash
  the run.
* **Reference-required gating** — context recall, factual correctness,
  and noise sensitivity are ``None`` and never judged when the dataset
  has no ``reference_answer`` for any query.

This module is **offline-only** (CONVENTIONS §10 / lld-eval-observability
§5 — two-clocks rule) and MUST NOT be imported by any hot-path module
(``api/``, ``agent/``, ``retrieval/``, ``generation/``).  The
``tests/unit/test_offline_only_import_guard.py`` static check enforces
this.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from iw_rag.domain.errors import ConfigError
from iw_rag.domain.models import RetrievalResult
from iw_rag.eval.repository import GoldenQuery
from iw_rag.eval.retrieval_metrics import score_retrieval

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Protocols — what the engine depends on (injection surface, no constructors)
# ---------------------------------------------------------------------------


class _FaithfulnessJudgeLike(Protocol):
    """The minimal surface the engine needs from a faithfulness judge.

    Mirrors :class:`iw_rag.eval.judges.FaithfulnessJudge` so tests can
    supply a ``MagicMock`` with an async ``judge`` method.
    """

    async def judge(self, question: str, context: str, answer: str) -> Any: ...


class _GradingJudgeLike(Protocol):
    """The minimal surface the engine needs from a grading judge.

    Mirrors :class:`iw_rag.eval.judges.GradingJudge`.
    """

    async def grade(self, rubric: Any, sample: Any, *, position_swap: bool = ...) -> Any: ...


class _RagasRunnerLike(Protocol):
    """The minimal surface the engine needs from a Ragas runner.

    Mirrors :class:`iw_rag.eval.ragas_runner.RagasRunner`.
    """

    async def score(self, samples: Sequence[Any], metrics: Any = ...) -> list[Any]: ...


# A rubric fed to the GradingJudge for "answer relevancy" / rubric grading.
# We import lazily because the judge dataclass lives in the offline judge
# module; tests that don't need grading can pass a fake.
RubricFactory = Callable[[], Any]


def _default_rubric_factory() -> Any:
    """Build a default 0-5 answer-relevancy rubric (lazy import)."""
    from iw_rag.eval.judges import Rubric

    return Rubric(
        criteria=(
            "How relevant and correct is the answer with respect to the "
            "question, given only the retrieved context? Score 0 (irrelevant "
            "or refuses) to 5 (directly answers every part of the question)."
        ),
        max_score=5.0,
    )


# ---------------------------------------------------------------------------
# Public typed result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    """The typed per-run result of :meth:`MetricsEngine.compute`.

    This is the **single** object report assembly (S06) consumes and
    persists as a ``benchmark_reports`` row.  Promoted columns match the
    schema in :mod:`iw_rag.storage.schema`; the :attr:`metrics` blob is
    the full per-tier superset, including per-query score vectors paired
    by query index (so the stats module in S07 can run paired tests).

    Attributes:
        recall_at_5: Mean recall@5 across scored queries (``T1``).
        recall_at_10: Mean recall@10 across scored queries (``T1``).
        ndcg_at_5: Mean nDCG@5 across scored queries (``T1``).
        ndcg_at_10: Mean nDCG@10 across scored queries (``T1``).
        mrr: Mean reciprocal rank across scored queries (``T1``).
        faithfulness_mean: Mean faithfulness across judged samples
            (``T2``).  ``None`` when the judge errored for every sample.
        faithfulness_kappa: Optional judge↔human kappa (e.g. from the
            existing kappa harness); ``None`` when not measured for the
            run.
        context_precision: Mean context precision across judged samples
            (``T2``).  ``None`` when the judge errored.
        context_recall: Mean context recall across judged samples
            (``T3``, reference-required).  ``None`` when the dataset has
            no ``reference_answer`` or the judge errored.
        factual_correctness: Mean factual correctness across judged
            samples (``T3``, reference-required).  ``None`` when the
            dataset has no ``reference_answer`` or the judge errored.
        noise_sensitivity: Mean noise sensitivity across judged samples
            (``T3``, reference-required).  ``None`` when the dataset has
            no ``reference_answer`` or the judge errored.
        n_queries: Number of queries in the run (NOT NULL in the schema).
        n_judged: Number of samples that were sent to a judge; ``None``
            when no judge was injected.
        errored: Count of per-sample judge / runner failures (NaN, parse
            error, endpoint down — anything that did not produce a score).
        metrics: Full per-tier blob (superset of the promoted columns).
            Per-query vectors are keyed by metric name (``"ndcg_at_5"``,
            ``"faithfulness"``, ``"context_recall"`` …) and are paired by
            query index so the stats module can run paired A/B tests.
    """

    recall_at_5: float | None
    recall_at_10: float | None
    ndcg_at_5: float | None
    ndcg_at_10: float | None
    mrr: float | None
    faithfulness_mean: float | None
    faithfulness_kappa: float | None
    context_precision: float | None
    context_recall: float | None
    factual_correctness: float | None
    noise_sensitivity: float | None = None
    n_queries: int = 0
    n_judged: int | None = None
    errored: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Default k values — T1 cutoffs (mirrors retrieval_metrics / lld §2)
# ---------------------------------------------------------------------------

_DEFAULT_KS: tuple[int, ...] = (5, 10)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[float | None]) -> float | None:
    """Compute the mean of a sequence, ignoring ``None`` and NaN.

    Returns ``None`` when the sequence is empty or every value is
    ``None``/NaN — never ``0.0``, so the caller can distinguish "judge
    errored for every sample" from "judge scored zero for every sample".

    Args:
        values: Sequence of optional floats.

    Returns:
        The mean, or ``None`` if no finite value is available.
    """
    finite: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        finite.append(f)
    if not finite:
        return None
    return sum(finite) / len(finite)


def _log_judge_coverage(name: str, scores: Sequence[float | None], first_error: str | None) -> None:
    """Log how many queries a judge actually scored (no more silent nulls).

    A benchmark run used to swallow every judge failure and persist a
    ``report`` whose entire ``Answer quality`` tier was empty with no trace
    of *why* (the judge endpoint was unreachable / every verdict failed to
    parse).  This surfaces that: when **zero** of N queries were scored it
    logs a WARNING carrying the first captured error (usually the
    connection failure that names the dead endpoint); a partial result is
    an INFO; full coverage is a DEBUG.

    Args:
        name: The judge's display name (e.g. ``"faithfulness"``).
        scores: The per-query score list (``None`` where the judge failed).
        first_error: The first captured failure message, or ``None`` when
            no call raised (e.g. the judge scored every query).
    """
    total = len(scores)
    scored = sum(1 for s in scores if s is not None)
    if total == 0:
        return
    if scored == 0:
        logger.warning(
            "benchmark %s judge scored 0/%d queries — the Answer-quality tier "
            "will be empty for this run. First failure: %s",
            name,
            total,
            first_error or "unknown (no score produced)",
        )
    elif scored < total:
        logger.info(
            "benchmark %s judge scored %d/%d queries (%d failed). First failure: %s",
            name,
            scored,
            total,
            total - scored,
            first_error or "n/a",
        )
    else:
        logger.debug("benchmark %s judge scored %d/%d queries.", name, scored, total)


def _has_reference_answers(queries: Sequence[GoldenQuery]) -> bool:
    """``True`` iff at least one query carries a non-empty ``reference_answer``.

    The T3 reference-required metrics (``context_recall``,
    ``factual_correctness``, ``noise_sensitivity``) MUST be skipped when
    the dataset has no reference answers — the engine must not call a
    judge for them in that case (AC4, Invariant 5).

    Args:
        queries: The golden queries for the run.

    Returns:
        ``True`` if any query has a non-empty ``reference_answer``.
    """
    for q in queries:
        ref = (q.reference_answer or "").strip()
        if ref:
            return True
    return False


def _coerce_score(raw: Any, field: str) -> float | None:
    """Best-effort coerce a ragas-style score to ``float`` (None on NaN/fail).

    The fake ``RagasRunner`` in the unit tests returns ``MagicMock`` objects
    with attribute access; the real one returns
    :class:`~iw_rag.eval.ragas_runner.GenerationScores` dataclasses.  Both
    expose the metric as an attribute named ``field`` (``faithfulness``,
    ``context_precision``, …).

    Args:
        raw: A score-like value (dataclass, MagicMock, or ``None``).
        field: The attribute name to read.

    Returns:
        The float score, or ``None`` if it is missing / NaN.
    """
    if raw is None:
        return None
    val: Any = getattr(raw, field, None)
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _has_error(ragas_result: Any) -> bool:
    """Whether a ragas-style per-sample result flagged itself as errored.

    Mirrors :class:`iw_rag.eval.ragas_runner.GenerationScores.errored`.
    """
    if ragas_result is None:
        return True
    return bool(getattr(ragas_result, "errored", False))


# ---------------------------------------------------------------------------
# MetricsEngine
# ---------------------------------------------------------------------------


class MetricsEngine:
    """The per-run T1/T2/T3 unifier.

    Reuses the existing :class:`FaithfulnessJudge` /
    :class:`GradingJudge` / :class:`RagasRunner` wiring; never
    instantiates a generator-as-judge.  Judges and the Ragas runner are
    **injected** so tests can supply fakes with no network and the
    engine itself never reads ``config/models.yaml``.

    Attributes:
        _faith: The injected faithfulness judge.
        _grade: The injected grading judge (rubric-anchored).
        _ragas: The injected Ragas runner, or ``None`` if ragas is not
            available in this deployment (the engine then skips T2/T3
            judged metrics and records them as ``errored``).
        _rubric_factory: Callable returning a rubric for the grading
            judge.  Defaults to a 0–5 answer-relevancy rubric.
        _ks: The T1 k cutoffs scored (defaults to ``(5, 10)``).
    """

    def __init__(
        self,
        *,
        faithfulness_judge: _FaithfulnessJudgeLike,
        grading_judge: _GradingJudgeLike,
        ragas_runner: _RagasRunnerLike | None,
        rubric_factory: RubricFactory | None = None,
        ks: tuple[int, ...] = _DEFAULT_KS,
    ) -> None:
        """Initialise the engine with injected judge wiring.

        Args:
            faithfulness_judge: A :class:`FaithfulnessJudge` (Lynx-8B)
                or equivalent.  Must expose ``async judge(question, context,
                answer)``.  May raise :class:`JudgeUnavailable` on
                endpoint failure.
            grading_judge: A :class:`GradingJudge` (Selene-Mini-8B) or
                equivalent.  Must expose ``async grade(rubric, sample,
                position_swap=...)``.  May raise
                :class:`JudgeUnavailable` on endpoint failure.
            ragas_runner: A :class:`RagasRunner` for the T2/T3 judged
                metrics, or ``None`` if ragas is not installed in this
                deployment.  When ``None``, T2/T3 judged metrics are
                ``None`` (errored) and the engine still produces T1 +
                the report.
            rubric_factory: Optional callable returning the rubric to
                pass to the grading judge.  Defaults to a built-in 0-5
                answer-relevancy rubric.
            ks: T1 k cutoffs scored.  Defaults to ``(5, 10)``.
        """
        self._faith = faithfulness_judge
        self._grade = grading_judge
        self._ragas = ragas_runner
        self._rubric_factory: RubricFactory = rubric_factory or _default_rubric_factory
        self._ks = ks

    async def compute(
        self,
        queries: Sequence[GoldenQuery],
        retrievals: Sequence[RetrievalResult],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        *,
        heavy_ragas: bool = False,
    ) -> BenchmarkMetrics:
        """Compute the per-run metrics and return the typed result.

        All four input sequences MUST be the same length and non-empty.

        Args:
            queries: The golden queries for the run.
            retrievals: The per-query :class:`RetrievalResult` from the
                retriever.  ``result.chunks`` is assumed best-first per
                the spine contract.
            answers: The per-query generated answers.
            contexts: The per-query retrieved context strings used by
                the answerer (one list of strings per query).

        Returns:
            A :class:`BenchmarkMetrics` with the promoted columns and
            the full per-tier blob.

        Raises:
            ConfigError: If any input sequence is empty or the four
                sequences differ in length (the engine must never
                produce an all-zero report from an empty golden set).
        """
        if not queries:
            raise ConfigError(
                "MetricsEngine.compute called with an empty queries list — "
                "the engine refuses to produce a meaningless all-zero report."
            )
        n = len(queries)
        if not (len(retrievals) == len(answers) == len(contexts) == n):
            raise ConfigError(
                f"MetricsEngine.compute received mismatched input lengths: "
                f"queries={n}, retrievals={len(retrievals)}, "
                f"answers={len(answers)}, contexts={len(contexts)}."
            )

        has_refs = _has_reference_answers(queries)

        # T1 — deterministic retrieval metrics (per query, then mean)
        t1 = self._compute_t1(queries, retrievals)

        # T2 + T3 — judged metrics via the injected wiring
        # We always try T2 (reference-free) and, when references exist,
        # T3.  The judge-down case degrades to None/errored — never
        # falls back to the generator.
        t2_t3, judged_per_query, errored = await self._compute_t2_t3(
            queries=queries,
            answers=answers,
            contexts=contexts,
            has_references=has_refs,
            heavy=heavy_ragas,
        )
        # ``n_judged`` counts UNIQUE queries that produced at least one
        # judge verdict (per the schema: a sample is judged, not a
        # judge call).  A query is judged if ANY of the three judges
        # produced a score — the engine never falls back to the
        # generator when judges error.
        judged_count = sum(1 for v in judged_per_query if v)

        # Compose the per-tier blob — superset of the promoted columns.
        # Per-query vectors are paired by query index so the stats
        # module (S07) can run paired A/B tests on the same set of
        # queries between two reports.
        metrics_blob: dict[str, Any] = {
            "recall_at_5": t1["recall_at_5"],
            "recall_at_10": t1["recall_at_10"],
            "ndcg_at_5": t1["ndcg_at_5"],
            "ndcg_at_10": t1["ndcg_at_10"],
            "mrr": t1["mrr"],
            "faithfulness": t2_t3["faithfulness_per_q"],
            "context_precision": t2_t3["context_precision_per_q"],
            "context_recall": t2_t3["context_recall_per_q"],
            "factual_correctness": t2_t3["factual_correctness_per_q"],
            "noise_sensitivity": t2_t3["noise_sensitivity_per_q"],
            # Engine sub-phase wall-clock (faithfulness / grading / ragas ms),
            # surfaced so the runner can build the report's step-timing waterfall.
            "_phase_ms": t2_t3.get("_phase_ms", {}),
        }

        return BenchmarkMetrics(
            recall_at_5=t1["recall_at_5_mean"],
            recall_at_10=t1["recall_at_10_mean"],
            ndcg_at_5=t1["ndcg_at_5_mean"],
            ndcg_at_10=t1["ndcg_at_10_mean"],
            mrr=t1["mrr_mean"],
            faithfulness_mean=t2_t3["faithfulness_mean"],
            faithfulness_kappa=None,  # populated by the kappa harness (out of scope here)
            context_precision=t2_t3["context_precision_mean"],
            context_recall=t2_t3["context_recall_mean"],
            factual_correctness=t2_t3["factual_correctness_mean"],
            noise_sensitivity=t2_t3["noise_sensitivity_mean"],
            n_queries=n,
            n_judged=judged_count,
            errored=errored,
            metrics=metrics_blob,
        )

    # ------------------------------------------------------------------ T1
    def _compute_t1(
        self,
        queries: Sequence[GoldenQuery],
        retrievals: Sequence[RetrievalResult],
    ) -> dict[str, Any]:
        """Score T1 retrieval metrics per query and return per-vec + means.

        Reuses :func:`iw_rag.eval.retrieval_metrics.score_retrieval`
        (CONVENTIONS §1 — never reimplement the math).  Zero-gold queries
        score 0.0 and are included in the means; this matches the
        retrieval_metrics primitives which already return ``0.0`` for
        empty gold.

        Args:
            queries: The golden queries (for ``relevant_chunk_ids``).
            retrievals: The per-query retrieval results.

        Returns:
            Dict with ``recall_at_5_mean``, ``recall_at_10_mean``,
            ``ndcg_at_5_mean``, ``ndcg_at_10_mean``, ``mrr_mean``, and
            the corresponding per-query vectors.
        """
        per_q_recall: dict[int, list[float]] = {k: [] for k in self._ks}
        per_q_ndcg: dict[int, list[float]] = {k: [] for k in self._ks}
        per_q_mrr: list[float] = []

        for q, r in zip(queries, retrievals, strict=True):
            scores = score_retrieval(r, q.relevant_chunk_ids, ks=self._ks)
            for k in self._ks:
                per_q_recall[k].append(scores.recall[k])
                per_q_ndcg[k].append(scores.ndcg[k])
            per_q_mrr.append(scores.mrr)

        return {
            "recall_at_5_mean": _safe_mean(per_q_recall[5]) if 5 in self._ks else None,
            "recall_at_10_mean": _safe_mean(per_q_recall[10]) if 10 in self._ks else None,
            "ndcg_at_5_mean": _safe_mean(per_q_ndcg[5]) if 5 in self._ks else None,
            "ndcg_at_10_mean": _safe_mean(per_q_ndcg[10]) if 10 in self._ks else None,
            "mrr_mean": _safe_mean(per_q_mrr),
            "recall_at_5": per_q_recall.get(5, []),
            "recall_at_10": per_q_recall.get(10, []),
            "ndcg_at_5": per_q_ndcg.get(5, []),
            "ndcg_at_10": per_q_ndcg.get(10, []),
            "mrr": per_q_mrr,
        }

    # ------------------------------------------------------------------ T2/T3
    async def _compute_t2_t3(
        self,
        *,
        queries: Sequence[GoldenQuery],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        has_references: bool,
        heavy: bool = False,
    ) -> tuple[dict[str, Any], list[bool], int]:
        """Compute T2 (reference-free) and T3 (reference-required) metrics.

        Strategy:

        1. **Faithfulness** (T2) — call the injected
           :class:`FaithfulnessJudge` once per (query, answer, context).
           Endpoint-down → ``None`` for the per-query value and
           ``errored += 1`` (the run does not crash; T1 + latency/resource
           + the report still produce).
        2. **Rubric / answer relevancy** (T2) — call the injected
           :class:`GradingJudge` with a 0–5 rubric.  Endpoint-down →
           ``None`` + ``errored += 1``.
        3. **Ragas batch** (T2 + T3) — build
           :class:`~iw_rag.eval.ragas_runner.EvalSample`s and call
           :meth:`RagasRunner.score`.  When references are absent, only
           the reference-free metrics are requested
           (``faithfulness`` + ``context_precision``); the runner is
           never asked for ``context_recall`` / ``factual_correctness`` /
           ``noise_sensitivity`` (AC4 / Invariant 5).
        4. **Per-sample NaN/parse failure** — the ragas output's
           ``errored=True`` flag is honoured; the engine does not crash
           the run (mirrors :mod:`iw_rag.eval.ragas_runner`).

        Args:
            queries: The golden queries (for ``question`` text and
                ``reference_answer``).
            answers: The per-query generated answers.
            contexts: The per-query context strings.
            has_references: ``True`` if the dataset has at least one
                ``reference_answer`` (T3 is allowed; otherwise T3 is
                skipped without a judge call).

        Returns:
            ``(payload, judged_per_query, errored)`` where ``payload`` has
            ``faithfulness_mean``/``context_precision_mean``/
            ``context_recall_mean``/``factual_correctness_mean``/
            ``noise_sensitivity_mean`` (all ``float | None``) and the
            corresponding per-query vectors; ``judged_per_query`` is a
            bool list (one per query) marking whether any judge
            produced a score for that query; ``errored`` is the total
            per-sample judge / runner failures.
        """
        from iw_rag.domain.errors import JudgeUnavailable

        n = len(queries)
        faith_per_q: list[float | None] = [None] * n
        grade_per_q: list[float | None] = [None] * n
        context_precision_per_q: list[float | None] = [None] * n
        context_recall_per_q: list[float | None] = [None] * n
        factual_per_q: list[float | None] = [None] * n
        noise_per_q: list[float | None] = [None] * n
        # Per-query "was this query judged?" flag — used to compute
        # ``n_judged`` (counted once per unique query, not per judge
        # call).  A query is judged if any of the three judges produced
        # a score.
        judged_per_q: list[bool] = [False] * n

        errored = 0

        # 1) Faithfulness (T2) — pointwise, per query
        faith_start = time.perf_counter()
        first_faith_err: str | None = None
        for i, (q, ans, ctxs) in enumerate(zip(queries, answers, contexts, strict=True)):
            ctx_text = "\n\n".join(ctxs)
            try:
                verdict = await self._faith.judge(q.question, ctx_text, ans)
                score = float(getattr(verdict, "score", 0.0))
                if math.isnan(score) or math.isinf(score):
                    faith_per_q[i] = None
                    errored += 1
                    if first_faith_err is None:
                        first_faith_err = "non-finite judge score"
                else:
                    faith_per_q[i] = score
                    judged_per_q[i] = True
            except JudgeUnavailable as exc:
                faith_per_q[i] = None
                errored += 1
                if first_faith_err is None:
                    first_faith_err = f"JudgeUnavailable: {exc}"
            except (TypeError, ValueError, AttributeError) as exc:
                faith_per_q[i] = None
                errored += 1
                if first_faith_err is None:
                    first_faith_err = f"{type(exc).__name__}: {exc}"
        _log_judge_coverage("faithfulness", faith_per_q, first_faith_err)
        faithfulness_ms = (time.perf_counter() - faith_start) * 1000.0

        grade_start = time.perf_counter()

        # 2) Rubric / answer relevancy (T2) — pointwise, per query
        # We use a single shared rubric across the run; the value
        # contributes to ``metrics["rubric"]`` per-query and is folded
        # into a ``rubric`` mean for traceability.  (Not a promoted
        # ``benchmark_reports`` column — schema mirrors R-00121 §5.2.)
        rubric = self._rubric_factory()
        # We need the actual GradingJudge to import GradeSample lazily
        # — the engine's public API does not require it (it is typed
        # structurally via ``_GradingJudgeLike``).
        from iw_rag.eval.judges import GradeSample

        first_grade_err: str | None = None
        for i, (q, ans, ctxs) in enumerate(zip(queries, answers, contexts, strict=True)):
            sample = GradeSample(question=q.question, answer=ans, context="\n\n".join(ctxs))
            try:
                verdict = await self._grade.grade(rubric, sample)
                score = float(getattr(verdict, "score", 0.0))
                if math.isnan(score) or math.isinf(score):
                    grade_per_q[i] = None
                    errored += 1
                    if first_grade_err is None:
                        first_grade_err = "non-finite judge score"
                else:
                    grade_per_q[i] = score
                    judged_per_q[i] = True
            except JudgeUnavailable as exc:
                grade_per_q[i] = None
                errored += 1
                if first_grade_err is None:
                    first_grade_err = f"JudgeUnavailable: {exc}"
            except (TypeError, ValueError, AttributeError) as exc:
                grade_per_q[i] = None
                errored += 1
                if first_grade_err is None:
                    first_grade_err = f"{type(exc).__name__}: {exc}"
        _log_judge_coverage("rubric/grading", grade_per_q, first_grade_err)
        grading_ms = (time.perf_counter() - grade_start) * 1000.0

        # 3) Ragas batch (T2 + T3) — single call for efficiency.
        #
        # The WHOLE Ragas suite is OPT-IN via ``heavy`` (params.ragas_full).
        # Ragas is intrinsically slow on the local single-GPU 8B judge — it
        # makes per-context (context precision/recall) and per-claim (factual
        # correctness / noise sensitivity) LLM calls, so even the "cheap" subset
        # runs ~28 min and the full suite ~70 min (concurrency does not help —
        # the judge is compute-bound).  So the DEFAULT run scores only
        # faithfulness (Lynx, above) + the deterministic T1 retrieval metrics,
        # finishing in ~8 min; ``ragas_full`` turns the whole suite on.
        # ``faithfulness`` is never requested from Ragas — it comes from the
        # dedicated FaithfulnessJudge (Lynx); the engine never reads Ragas's.
        if not heavy:
            requested_metrics: tuple[str, ...] = ()
        elif has_references:
            requested_metrics = (
                "context_precision",
                "context_recall",
                "factual_correctness",
                "noise_sensitivity",
            )
        else:
            requested_metrics = ("context_precision",)

        ragas_start = time.perf_counter()
        if self._ragas is not None and requested_metrics:
            from iw_rag.eval.ragas_runner import EvalSample

            samples: list[EvalSample] = [
                EvalSample(
                    question=q.question,
                    answer=ans,
                    contexts=list(ctxs),
                    ground_truth=q.reference_answer or "",
                )
                for q, ans, ctxs in zip(queries, answers, contexts, strict=True)
            ]

            try:
                results = await self._ragas.score(samples, metrics=requested_metrics)
            except JudgeUnavailable as exc:
                results = []
                errored += 1
                logger.warning(
                    "benchmark Ragas scoring unavailable (%s) — context_precision / "
                    "context_recall / factual_correctness will be empty for this run.",
                    exc,
                )

            for i, res in enumerate(results):
                if i >= n:
                    break
                if _has_error(res):
                    errored += 1
                    continue
                # Each metric goes to its own slot.  Reference-free
                # metrics are recorded only when the runner was
                # actually asked for them; otherwise the per-query
                # slot stays None.
                if "context_precision" in requested_metrics:
                    cp = _coerce_score(res, "context_precision")
                    if cp is not None:
                        context_precision_per_q[i] = cp
                if "context_recall" in requested_metrics:
                    cr = _coerce_score(res, "context_recall")
                    if cr is not None:
                        context_recall_per_q[i] = cr
                    fc = _coerce_score(res, "factual_correctness")
                    if fc is not None:
                        factual_per_q[i] = fc
                if "noise_sensitivity" in requested_metrics:
                    ns = _coerce_score(res, "noise_sensitivity")
                    if ns is not None:
                        noise_per_q[i] = ns
                judged_per_q[i] = True
        elif self._ragas is not None and not requested_metrics:
            # Fast mode (ragas_full off): Ragas was deliberately skipped, NOT
            # errored — faithfulness (Lynx) still scored the run above.
            pass
        else:
            # Ragas not available: T2/T3 judged metrics are not
            # recorded.  The run still produces T1 + the report.
            # We do NOT raise — the engine degrades gracefully.
            errored += n  # all judged slots for this run are missing
        ragas_ms = (time.perf_counter() - ragas_start) * 1000.0

        # 4) Compose means.  Reference-required means are None when no
        # references exist — the per-query vectors are also None so the
        # stats module sees consistent lengths.
        return (
            {
                "_phase_ms": {
                    "faithfulness_ms": faithfulness_ms,
                    "grading_ms": grading_ms,
                    "ragas_ms": ragas_ms,
                },
                "faithfulness_mean": _safe_mean(faith_per_q),
                "context_precision_mean": _safe_mean(context_precision_per_q),
                "context_recall_mean": (
                    _safe_mean(context_recall_per_q) if has_references else None
                ),
                "factual_correctness_mean": (_safe_mean(factual_per_q) if has_references else None),
                "noise_sensitivity_mean": (_safe_mean(noise_per_q) if has_references else None),
                "faithfulness_per_q": faith_per_q,
                "context_precision_per_q": context_precision_per_q,
                "context_recall_per_q": context_recall_per_q,
                "factual_correctness_per_q": factual_per_q,
                "noise_sensitivity_per_q": noise_per_q,
                # The rubric mean is exposed via the blob only; the
                # schema does not promote it to a column.
                "rubric_per_q": grade_per_q,
            },
            judged_per_q,
            errored,
        )


__all__ = ["BenchmarkMetrics", "MetricsEngine"]
