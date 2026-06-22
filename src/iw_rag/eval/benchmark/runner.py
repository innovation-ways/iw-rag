"""Benchmark Studio — run orchestration service (F-00095, S01).

The engine that orchestrates **one** full benchmark and streams its
progress as ``{"event": ..., "data": ...}`` dicts — the same shape the
existing :meth:`iw_rag.api.service.AgentQueryService.answer_question_stream`
yields.  The S02 API layer wraps these dicts into SSE bytes; the
runner itself yields dicts so it stays transport-agnostic and
unit-testable.

**Orchestration sequence** (F-00095 design Scope item 1; R-00123 §6;
R-00122 final Recommendation):

1. **Resolve** the requested ``model_config`` → emit ``queued``.
2. **Activate** the generator via the F-00094 swap-controller and
   **wait for health-gated readiness**.  Emit ``swapping`` before the
   wait; record the wall time of activation→ready as
   ``model_load_ms`` in its OWN bucket — it MUST NEVER be folded into
   per-query or e2e latency (Invariant 4; R-00122 §6).  Emit ``ready``
   when the controller reports ready.  On timeout, emit a terminal
   ``error`` ``{reason_code: "swap_timeout", message, trace_ref}`` and
   persist NO report (Boundary row 1).
3. **Snapshot VRAM baseline** after warmup, **start the GPU sampler**,
   and **scrape vLLM ``/metrics``** for the pre-run baseline.
4. **Run the selected golden dataset** through the live
   retrieval+generation pipeline, one query at a time.  Emit
   ``running`` with the query index ``i`` (and total) per query.
5. **Compute the metric set** via the F-00093 metrics engine — T1
   (deterministic retrieval), T2 (judged generation), T3
   (reference-required, **omitted** when the dataset has no reference
   answers), latency (``model_load_ms`` + per-stage + vLLM
   percentiles), VRAM (baseline/peak/incremental), and judge κ.  Emit
   ``scoring``.
6. **Persist exactly ONE** ``benchmark_reports`` row via the F-00093
   repository, whose ``model_config`` equals the submitted selection
   (Invariants 2 & 5).  On a successful run persist exactly once; on
   any failure/abort persist zero rows.
7. Emit a terminal ``done`` carrying the persisted ``report_id``.

Any mid-run failure ends with a terminal ``error`` event
``{message, trace_ref}`` (generic message only — never exception
text, retrieved content, or PII; mirror :func:`iw_rag.api.routes._sse_generator`
``except`` block).  On client disconnect the in-flight run must not
leave a half-written report (Boundary row "SSE client disconnects
mid-run").

**Consumed collaborators are injected as Protocols.**  F-00093 and
F-00094 are authored in parallel, so the surfaces the runner consumes
MAY not exist yet.  The minimal Protocols below let the runner be
tested with fakes today and bind to the real implementations at the
composition root (``app.py``, wired in S02) once the dependencies
land.  If a real name is already importable, the S02 composition root
binds the real one and the Protocol disappears behind the call site.

This module is **offline-only by contract** in the sense that the
runner itself never reaches a hosted/cloud API — all collaborators
(judges, generator, embedder, reranker) are local OSS, and the runner
never imports a cloud SDK (Invariant 1).  The S05 perf collectors are
imported lazily inside the body so the module imports cleanly when
those optional deps are absent.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Protocol

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import GeneratedAnswer, RetrievalResult
from iw_rag.eval.benchmark.assembly import assemble_report
from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics
from iw_rag.eval.benchmark.perf import (
    GpuBaselineSnapshot,
    GpuMetrics,
    ModelLoadTiming,
    VllmServerMetrics,
    combine_vllm_deltas,
)
from iw_rag.eval.repository import GoldenQuery

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SwapTimeout(Exception):
    """Raised when the swap-controller's readiness poll exceeds the
    health-gate timeout.

    Caught by :meth:`BenchmarkRunner.run` and translated to a terminal
    SSE ``error`` event with ``reason_code: "swap_timeout"`` (Boundary
    row 1).  The runner persists ZERO reports on this path.
    """


# ---------------------------------------------------------------------------
# Collaborator Protocols (F-00093 / F-00094 seams; injected at S02)
# ---------------------------------------------------------------------------


class SwapControllerProtocol(Protocol):
    """The minimal surface the runner needs from the F-00094
    swap-controller.

    Mirrors :class:`iw_rag.serving.swap_controller_client.SwapControllerClient`
    plus a ``wait_for_ready`` readiness poll — the real client (or the
    in-process :class:`~iw_rag.serving.swap_controller.SwapController`
    when bound at the S02 composition root) implements both.  Tests
    supply a fake that may raise :class:`SwapTimeout` to exercise
    Boundary row 1.

    Attributes:
        activate(model_config): Trigger a generator swap.
        wait_for_ready(*, timeout_s): Block until the swap-controller
            reports ready; raise :class:`SwapTimeout` on timeout.
    """

    async def activate(self, model_config: dict[str, Any]) -> dict[str, Any]: ...

    async def wait_for_ready(self, *, timeout_s: float) -> dict[str, Any]: ...


class MetricsEngineProtocol(Protocol):
    """The minimal surface the runner needs from the F-00093 metrics
    engine.

    Mirrors :class:`iw_rag.eval.benchmark.metrics_engine.MetricsEngine.compute`.
    The engine itself injects the judges / Ragas runner / grading
    rubric; the runner never constructs any of them.
    """

    async def compute(
        self,
        queries: Sequence[GoldenQuery],
        retrievals: Sequence[RetrievalResult],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        *,
        heavy_ragas: bool = ...,
    ) -> BenchmarkMetrics: ...


class ReportRepositoryProtocol(Protocol):
    """The minimal surface the runner needs from the F-00093
    ``benchmark_reports`` repository.

    Mirrors :meth:`iw_rag.eval.benchmark.repository.BenchmarkReportRepository.create_report`.
    The runner uses only the ``create_report`` surface; the repository
    is append-only (no update / delete — Invariant 2).
    """

    async def create_report(self, **kwargs: Any) -> str: ...


class GpuSamplerProtocol(Protocol):
    """The minimal surface the runner needs from the F-00093
    :class:`GpuSampler`.

    Mirrors :class:`iw_rag.eval.benchmark.perf.GpuSampler`.  Graceful
    degradation (no pynvml / no GPU) is the sampler's contract — the
    runner just records the start/stop calls.
    """

    def snapshot_baseline(self) -> GpuBaselineSnapshot: ...

    def start(self) -> None: ...

    def mark_run_start(self) -> None: ...

    def stop(self) -> GpuMetrics: ...


class VllmScraperProtocol(Protocol):
    """The minimal surface the runner needs from the F-00093
    :class:`VllmMetricsScraper`.

    Mirrors :class:`iw_rag.eval.benchmark.perf.VllmMetricsScraper.scrape`.
    The runner scrapes twice (before + after) and lets the S05 layer
    compute deltas — the runner just records the calls.
    """

    async def scrape(self) -> VllmServerMetrics: ...


class GoldenSetLoaderProtocol(Protocol):
    """The minimal surface the runner needs from the F-00095 golden set
    loader.

    The loader is injected so the runner stays decoupled from the
    S02-layer resolver (which reads from Postgres).  Tests supply a
    fake that returns a canned :class:`GoldenQuery` list; production
    binds the real loader (which resolves the YAML + the live
    ``relevant_chunk_ids`` like :mod:`scripts.run_golden_eval`).
    """

    async def load(self, dataset_id: str) -> list[GoldenQuery]: ...


class QueryRunnerProtocol(Protocol):
    """The minimal surface the runner needs from the per-query
    retrieval+generation pipeline.

    Production binds the live :class:`iw_rag.retrieval.spine.RetrievalSpine`
    + :class:`iw_rag.generation.answer.AnswerAssembler` pair (or the
    full :class:`iw_rag.api.service.AgentQueryService` when S02 wires
    the agent loop); tests supply a fake that replays a canned
    per-query script.
    """

    async def run_query(
        self,
        question: str,
        ctx: AccessContext,
        i: int,
        total: int,
    ) -> tuple[RetrievalResult, GeneratedAnswer, list[str]]: ...


# ---------------------------------------------------------------------------
# BenchmarkRunner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """The F-00095 run-orchestration service.

    Orchestrates one full benchmark (resolve → swap → readiness → GPU
    sample → run golden set → compute metrics → persist) and yields
    progress events as ``{"event": str, "data": dict}`` dicts — the
    shape the S02 API layer wraps into SSE bytes (mirroring
    :func:`iw_rag.api.routes._sse_generator`).

    All collaborators are **injected** as Protocols so the runner is
    unit-testable with fakes (NO GPU, NO live swap, NO live
    retrieval+generation) and binds to the real F-00093 / F-00094
    surfaces at the S02 composition root.

    Attributes:
        _swap: The F-00094 swap-controller client (real or fake).
        _engine: The F-00093 metrics engine (real or fake).
        _repo: The F-00093 ``benchmark_reports`` repository.
        _gpu: The F-00093 :class:`GpuSampler` (no-op when no GPU).
        _vllm: The F-00093 :class:`VllmMetricsScraper` (no-op when the
            Prometheus endpoint is absent).
        _golden: The F-00095 golden set loader.
        _qr: The per-query retrieval+generation pipeline.
        _health_timeout_s: The swap-controller readiness-poll budget.
        _per_query_timeout_s: The per-query wall-clock budget (the
            runner measures, not enforces — ``asyncio.wait_for`` could
            be added later if a hard cap is required).
    """

    def __init__(
        self,
        *,
        swap_controller: SwapControllerProtocol,
        metrics_engine: MetricsEngineProtocol,
        repository: ReportRepositoryProtocol,
        gpu_sampler: GpuSamplerProtocol,
        vllm_scraper: VllmScraperProtocol,
        golden_loader: GoldenSetLoaderProtocol,
        query_runner: QueryRunnerProtocol,
        conclusion_fn: Any | None = None,
        health_timeout_s: float = 300.0,
        per_query_timeout_s: float = 120.0,
    ) -> None:
        """Initialise the runner with the injected collaborators.

        Args:
            swap_controller: The F-00094 swap-controller client.  Real
                production wiring: the
                :class:`~iw_rag.serving.swap_controller_client.SwapControllerClient`
                on ``app.state``.  Tests inject a fake.
            metrics_engine: The F-00093 metrics engine.  Real wiring:
                :class:`~iw_rag.eval.benchmark.metrics_engine.MetricsEngine`.
            repository: The F-00093 ``benchmark_reports`` repository.
                Real wiring:
                :class:`~iw_rag.eval.benchmark.repository.BenchmarkReportRepository`
                (which is itself bound to an async session).
            gpu_sampler: The F-00093 :class:`GpuSampler` (graceful
                degradation to a no-op when no GPU / no pynvml).
            vllm_scraper: The F-00093 :class:`VllmMetricsScraper`
                (graceful degradation to an empty result when the
                Prometheus endpoint is absent).
            golden_loader: The F-00095 golden set loader (resolves
                ``dataset_id`` → list of :class:`GoldenQuery`).
            query_runner: The per-query retrieval+generation pipeline.
            health_timeout_s: The swap-controller readiness-poll budget
                in seconds (default 300 — matches
                ``llama-swap.yaml`` ``healthCheckTimeout``).
            per_query_timeout_s: The per-query wall-clock budget in
                seconds (default 120).  The runner measures the per-
                stage ms for the latency blob; this value is reserved
                for a future hard cap (a future S02/S05 hardening).
        """
        self._swap = swap_controller
        self._engine = metrics_engine
        self._repo = repository
        self._gpu = gpu_sampler
        self._vllm = vllm_scraper
        self._golden = golden_loader
        self._qr = query_runner
        # Optional run-time conclusion writer: ``async (this_row, dataset_id)
        # -> str | None``.  Built by the S02 composition (local judge LLM +
        # same-dataset comparables read).  ``None`` → no conclusion (the run
        # still persists everything else).
        self._conclusion_fn = conclusion_fn
        self._health_timeout_s = float(health_timeout_s)
        self._per_query_timeout_s = float(per_query_timeout_s)

    # ------------------------------------------------------------------ run

    async def run(
        self,
        model_config: dict[str, Any],
        *,
        trace_id: str = "",
        access_context: AccessContext | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Run one full benchmark; yield progress events as dicts.

        The async-generator pattern mirrors
        :meth:`iw_rag.api.service.AgentQueryService.answer_question_stream`
        — the S02 API layer wraps each yielded dict in an SSE
        ``event: <type>\\ndata: <json>\\n\\n`` envelope.

        Args:
            model_config: The submitted ``model_config`` dict
                (``{generator, embedder, reranker, grader} + params +
                dataset_id``).  Stored verbatim on the persisted row
                (Invariant 5).
            trace_id: The Phoenix trace id for the run.  Forwarded to
                the terminal ``error`` event's ``trace_ref`` and to
                the persisted row (so a follow-up diagnose can find
                the timeline).
            access_context: The authenticated :class:`AccessContext`
                (carried through to the per-query pipeline; ``None``
                allowed for the offline batch path).

        Yields:
            Progress event dicts ``{"event": str, "data": dict}`` in
            the canonical order:
            ``queued → swapping → ready → running×N → scoring → done``.
            Any failure (including the swap-timeout boundary) ends
            with a terminal ``error`` event carrying only
            ``{message, trace_ref}`` (and the swap-timeout variant
            adds ``reason_code: "swap_timeout"``).  No exception ever
            propagates out of this async generator — the terminal
            ``error`` is the failure surface (mirror
            :func:`iw_rag.api.routes._sse_generator`).
        """
        # Mirror the SSE error contract from _sse_generator: the
        # terminal error is generic — never the exception text,
        # retrieved content, or PII.  trace_ref is the only forensic
        # handle.
        try:
            async for ev in self._run_unsafe(model_config, trace_id, access_context):
                yield ev
        except SwapTimeout:
            yield {
                "event": "error",
                "data": {
                    "reason_code": "swap_timeout",
                    "message": "swap-controller never reported ready",
                    "trace_ref": trace_id,
                },
            }
        except Exception:
            # Generic message; never the exception text.  trace_ref
            # is the only forensic handle (lld-api §error mapping).
            import logging

            logging.getLogger("iw_rag.benchmark").exception(
                "benchmark run failed (trace_ref=%s)", trace_id
            )
            yield {
                "event": "error",
                "data": {
                    "message": "benchmark run failed; see trace for details",
                    "trace_ref": trace_id,
                },
            }

    # ------------------------------------------------------------------ _run_unsafe

    async def _run_unsafe(
        self,
        model_config: dict[str, Any],
        trace_id: str,
        access_context: AccessContext | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """The unguarded orchestration; raises on failure.

        This is the body the public :meth:`run` wraps in a generic
        ``try/except`` so a mid-run failure becomes a terminal SSE
        ``error`` rather than a 500.

        Args:
            model_config: The submitted ``model_config`` dict.
            trace_id: The Phoenix trace id.
            access_context: The authenticated :class:`AccessContext`
                (or ``None`` for the offline batch path).

        Yields:
            Progress event dicts.  On any uncaught failure the
            exception propagates to the public wrapper.

        Raises:
            SwapTimeout: When the swap-controller's readiness poll
                exceeds the health-gate timeout (caught by the public
                wrapper and translated to a ``reason_code:
                "swap_timeout"`` terminal error).
        """
        # 1. Resolve the model_config + golden set
        yield {"event": "queued", "data": {"model_config": dict(model_config)}}
        dataset_id = str(model_config.get("dataset_id") or "benchmark_golden")
        golden_queries = await self._golden.load(dataset_id)
        total = len(golden_queries)

        # 2. Activate + wait for readiness.
        #
        # ``model_load_ms`` brackets the FULL activation→ready window: on a
        # cold swap the model load happens INSIDE ``activate`` (the proxy
        # spawns/loads the generator), so timing only the trailing readiness
        # poll undercounts it to ~0 ms.  ``load_start`` therefore sits before
        # ``activate``.
        #
        # The GPU sampler is started BEFORE activation so it observes the
        # whole swap: the VRAM trough while the prior generator is unloaded
        # (the idle floor) is captured as ``vram_baseline_mb``, and
        # ``vram_incremental_mb`` becomes the resident model's footprint
        # (run-time peak − floor) — vLLM eagerly pre-allocates its KV-cache
        # pool at load, so a post-load baseline would equal the peak
        # (incremental ≡ 0).  ``mark_run_start`` (after ready, below) bounds
        # the utilisation + run-time-peak window so the load period does not
        # pollute them.  ``snapshot_baseline`` adds a guaranteed pre-swap
        # reading in case the sampler interval misses the trough.
        yield {"event": "swapping", "data": {"model_config": dict(model_config)}}
        self._gpu.snapshot_baseline()
        self._gpu.start()
        load_start = time.perf_counter()
        await self._swap.activate(dict(model_config))
        # ``asyncio.wait_for`` raises :class:`asyncio.TimeoutError` on
        # timeout; the swap-controller's own client may also raise
        # :class:`SwapTimeout` from its wait_for_ready body (Boundary
        # row 1 path).  We re-raise as :class:`SwapTimeout` uniformly
        # so the public wrapper handles a single exception type.
        try:
            await self._swap.wait_for_ready(timeout_s=self._health_timeout_s)
        except SwapTimeout:
            raise
        except Exception as exc:
            # Any other readiness failure (controller-side timeout,
            # asyncio.wait_for exhaustion) is translated to SwapTimeout
            # so the public surface is uniform.
            raise SwapTimeout(
                f"swap-controller never reported ready within {self._health_timeout_s}s"
            ) from exc
        model_load_ms = (time.perf_counter() - load_start) * 1000.0
        yield {"event": "ready", "data": {"model_load_ms": model_load_ms}}

        # 3. Mark the measurement-window start (the sampler has been running
        # since before the swap to catch the VRAM trough; util + run-time
        # peak are computed only from here on) and scrape the pre-run vLLM
        # /metrics (best-effort; the before+after pair is consumed by the
        # S05 layer downstream).
        self._gpu.mark_run_start()
        # Run-start /metrics snapshot (the baseline for the before→after token
        # + prefix-cache deltas, combined after generation below).
        vllm_before = await self._vllm.scrape()

        # 4. Run the golden set through the live pipeline
        generate_start = time.perf_counter()
        retrievals: list[RetrievalResult] = []
        answers: list[str] = []
        contexts: list[list[str]] = []
        per_stage_ms: dict[str, list[float]] = {}
        for i, q in enumerate(golden_queries):
            stage_start = time.perf_counter()
            retrieval, answer, ctxs = await self._qr.run_query(
                q.question,
                access_context if access_context is not None else _offline_ctx(),
                i,
                total,
            )
            elapsed_ms = (time.perf_counter() - stage_start) * 1000.0
            retrievals.append(retrieval)
            answers.append(answer.answer)
            contexts.append(ctxs)
            # Per-stage aggregation: we record one bucket per query
            # for the "retrieve + generate" composite.  A future
            # instrumentation pass can split the timing across the
            # retriever / reranker / generator stages; for S01 the
            # per-query latency vector is the minimum invariant
            # R-00122 §2 requires.
            per_stage_ms.setdefault("retrieve_generate_ms", []).append(elapsed_ms)
            yield {
                "event": "running",
                "data": {"i": i, "total": total, "question": q.question},
            }
        generate_ms = (time.perf_counter() - generate_start) * 1000.0

        # Scrape the vLLM /metrics for the report NOW — BEFORE scoring.  The
        # judge phase below cold-swaps the generator OUT for the judge model
        # (two-phase judging, F-00094 exclusive group); afterwards the
        # generator container is stopped and its /metrics endpoint is gone, so
        # a post-scoring scrape returns empty (every ``vllm_server.*`` field
        # NULL → blank TTFT/TPOT/E2E/queue p99).  Capturing it here — generator
        # still resident, all N generations complete — is the cumulative
        # end-of-generation snapshot the report wants.  Combine it with the
        # run-start baseline so the per-run token + prefix-cache DELTAS
        # populate (these feed the Scoreboard's tok/s board).
        vllm_after = await self._vllm.scrape()
        vllm_metrics = combine_vllm_deltas(vllm_before, vllm_after)

        # 5. Score (Tier-1 + Tier-2 + Tier-3 via the F-00093 engine)
        yield {"event": "scoring", "data": {"n_queries": total}}
        score_start = time.perf_counter()
        # ``params.ragas_full`` (default False) opts into the EXPENSIVE Ragas
        # metrics (factual_correctness + noise_sensitivity) — off by default so
        # the run stays fast (~10-15 min) instead of ~30-60 min on the local
        # 8B judge.  context precision/recall + faithfulness always run.
        params = model_config.get("params") or {}
        heavy_ragas = bool(params.get("ragas_full", False))
        metrics = await self._engine.compute(
            golden_queries, retrievals, answers, contexts, heavy_ragas=heavy_ragas
        )
        score_ms = (time.perf_counter() - score_start) * 1000.0

        # Persist a per-item sample record (question + generated answer +
        # retrieved contexts) into the metrics blob, paired by query index with
        # the per-query metric vectors.  This powers the multi-run Compare
        # view's expandable qualitative diff (each run's answer/context per
        # item).  Bounded in size (contexts capped + truncated) so the JSONB
        # blob stays reasonable.  Reports from before this change carry no
        # ``samples`` key and the Compare view degrades to metrics-only.
        metrics.metrics["samples"] = self._build_item_samples(golden_queries, answers, contexts)

        # 6. Persist exactly ONE benchmark_reports row.  The
        # F-00093 assembly is the single S06 surface that maps the
        # metrics + perf results to a row payload.  model_load_ms
        # is a TOP-LEVEL key in the latency JSONB; it is NEVER
        # folded into per-stage / per-query ms (Invariant 4).
        gpu_metrics = self._gpu.stop()
        model_load = ModelLoadTiming(elapsed_ms=model_load_ms, result="ok", error=None)
        # Flatten per-stage vectors to means (R-00122 §2 JSON shape).
        per_stage_means: dict[str, float] = {
            name: float(sum(values) / len(values)) if values else 0.0
            for name, values in per_stage_ms.items()
        }
        # Per-phase wall-clock — the "how long did each step take" breakdown
        # (model swap/load, the generation loop, the scoring/judging phase).
        # Surfaced as a waterfall in the report view + PDF.  ``score_ms`` is
        # the whole judge phase (faithfulness + grading + Ragas); the engine
        # also records its own sub-phase split in ``metrics["_phase_ms"]``.
        phase_ms: dict[str, float] = {
            "model_load_ms": model_load_ms,
            "generate_ms": generate_ms,
            "score_ms": score_ms,
        }
        engine_phases = (metrics.metrics or {}).get("_phase_ms")
        if isinstance(engine_phases, dict):
            for k, v in engine_phases.items():
                if isinstance(v, int | float):
                    phase_ms[str(k)] = float(v)

        # Run-time conclusion (stored in the immutable report's metrics blob at
        # insert).  Written by a LOCAL judge model already resident after
        # scoring — so no extra swap — and grounded in the measured numbers +
        # same-dataset comparables.  Best-effort: a failure never fails the run.
        if self._conclusion_fn is not None:
            this_row: dict[str, Any] = {
                "report_label": self._build_report_label(model_config),
                "model_config": model_config,
                "faithfulness_mean": metrics.faithfulness_mean,
                "ndcg_at_5": metrics.ndcg_at_5,
                "context_recall": metrics.context_recall,
                "factual_correctness": metrics.factual_correctness,
                "latency": {
                    "e2e_p99_s": vllm_metrics.e2e_p99,
                    "per_stage_ms": dict(per_stage_means),
                },
                "resource": {
                    "gpu": {
                        "vram_peak_mb": gpu_metrics.vram_peak_mb,
                        "vram_incremental_mb": gpu_metrics.vram_incremental_mb,
                    }
                },
            }
            try:
                conclusion_text = await self._conclusion_fn(this_row, dataset_id)
            except Exception:  # noqa: BLE001 — never fail the run on the conclusion
                logger.warning("benchmark conclusion writer raised", exc_info=True)
                conclusion_text = None
            if conclusion_text:
                metrics.metrics["conclusion"] = {
                    "text": conclusion_text,
                    "basis": "run-time",
                }
        # The row id surface is ``create_report`` — we let the
        # assembly emit the row, then we look up the id (the
        # assembly does not return the id; the repo does — but the
        # S06 layer returns the id directly).  We thread a temporary
        # ``dataset_id`` and ``run_id`` here; the real run id will
        # be plumbed in S02 from the auth + session layer.
        report_id = await self._assemble_and_persist(
            dataset_id=dataset_id,
            model_config=model_config,
            metrics=metrics,
            per_stage_means=per_stage_means,
            phase_ms=phase_ms,
            vllm_metrics=vllm_metrics,
            gpu_metrics=gpu_metrics,
            model_load=model_load,
            trace_id=trace_id,
        )

        # 7. Done
        yield {"event": "done", "data": {"report_id": report_id}}

    # ------------------------------------------------------------------ _assemble_and_persist

    async def _assemble_and_persist(
        self,
        *,
        dataset_id: str,
        model_config: dict[str, Any],
        metrics: BenchmarkMetrics,
        per_stage_means: dict[str, float],
        phase_ms: dict[str, float],
        vllm_metrics: VllmServerMetrics,
        gpu_metrics: GpuMetrics,
        model_load: ModelLoadTiming,
        trace_id: str,
    ) -> str:
        """Assemble the row payload via :func:`assemble_report` and
        hand it to the repository.

        The function is the **only** place the runner calls the
        F-00093 assembly; it threads the latency / resource JSONB
        together from the typed perf results.  ``created_by`` is
        ``None`` here — the S02 composition root sets it from the
        authenticated principal (Invariant 3 — never derived from
        the LLM, never from a request body field).  The ``run_id``
        is a synthetic ``str(uuid4())``; the S02 layer may supply
        the real ``eval_runs`` id when it exists.

        Args:
            dataset_id: The dataset the run belongs to.
            model_config: The submitted ``model_config`` (stored
                verbatim, Invariant 5).
            metrics: The :class:`BenchmarkMetrics` from the F-00093
                engine.
            per_stage_means: Per-stage mean ms dict (R-00122 §2
                JSON shape).
            vllm_metrics: The post-run :class:`VllmServerMetrics`.
            gpu_metrics: The :class:`GpuMetrics` from
                :meth:`GpuSampler.stop`.
            model_load: The :class:`ModelLoadTiming` carrying the
                cold-swap wall-clock (Invariant 4 — its OWN bucket).
            trace_id: The Phoenix trace id (carried for forensic
                correlation; not a column).

        Returns:
            The persisted report's id (str).
        """
        import uuid

        # The S02 layer will supply the real eval_runs id; for the
        # offline / un-wired path we use a synthetic uuid so the
        # repository's str↔uuid boundary conversion succeeds.
        run_id = str(uuid.uuid4())
        report_id = await assemble_report(
            repository=self._repo,
            run_id=run_id,
            dataset_id=dataset_id,
            report_label=self._build_report_label(model_config),
            git_sha=None,
            created_by=None,  # Invariant 3 — S02 wires the real principal
            model_config=dict(model_config),
            metrics=metrics,
            per_stage_ms=per_stage_means,
            phase_ms=phase_ms,
            vllm_metrics=vllm_metrics,
            gpu_metrics=gpu_metrics,
            model_load=model_load,
            ci_95=None,
        )
        # Carry the trace id onto the report surface (defensive: the
        # assembly's repo contract doesn't take trace_id today, so
        # the runner's caller reads it from the per-step timeline).
        del trace_id
        return report_id

    @staticmethod
    def _build_item_samples(
        queries: Sequence[GoldenQuery],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        *,
        max_contexts: int = 8,
        max_ctx_chars: int = 1200,
        max_answer_chars: int = 4000,
    ) -> list[dict[str, Any]]:
        """Build the bounded per-item sample records for the metrics blob.

        One record per query (paired by index with the per-query metric
        vectors): the question, the generated answer, and the retrieved
        context strings — each truncated so the persisted JSONB stays
        small (the Compare view only needs enough to eyeball the
        qualitative difference between runs).

        Args:
            queries: The golden queries (question text).
            answers: The per-query generated answers.
            contexts: The per-query retrieved context strings.
            max_contexts: Max context strings kept per item.
            max_ctx_chars: Max characters kept per context string.
            max_answer_chars: Max characters kept per answer.

        Returns:
            A list of ``{question, answer, contexts}`` dicts, one per query.
        """
        samples: list[dict[str, Any]] = []
        for q, ans, ctxs in zip(queries, answers, contexts, strict=True):
            trimmed = [str(c)[:max_ctx_chars] for c in list(ctxs)[:max_contexts]]
            samples.append(
                {
                    "question": q.question,
                    "answer": (ans or "")[:max_answer_chars],
                    "contexts": trimmed,
                }
            )
        return samples

    @staticmethod
    def _build_report_label(model_config: dict[str, Any]) -> str:
        """Compose a human-readable label for the report row.

        The generator id is the primary disambiguator; a NON-default dataset is
        appended by its human ``dataset_name`` (e.g. ``qwen3-14b@my-set``).  The
        opaque ``dataset_id`` uuid is NEVER appended — the run route resolves the
        logical name to an ``eval_datasets`` uuid before the run, so appending
        ``dataset_id`` would leak an unreadable uuid into every label.

        Args:
            model_config: The submitted ``model_config`` dict (carries the
                human ``dataset_name`` for a non-default dataset).

        Returns:
            A short label string.
        """
        gen = str(model_config.get("generator") or "model")
        name = str(model_config.get("dataset_name") or "")
        if name and name not in ("benchmark_golden", "benchmark-golden"):
            return f"{gen}@{name}"
        return gen


# ---------------------------------------------------------------------------
# Offline access context helper
# ---------------------------------------------------------------------------


def _offline_ctx() -> AccessContext:
    """A minimal :class:`AccessContext` for the offline batch path.

    The runner accepts ``None`` and synthesises this context so the
    per-query pipeline can run without an authenticated principal.
    Production callers (S02) always pass a real context from auth.

    Returns:
        A bare-bones :class:`AccessContext` with no groups.
    """
    return AccessContext(principal_id="offline", principal_groups=(), is_operator=False)


__all__ = [
    "BenchmarkRunner",
    "GoldenSetLoaderProtocol",
    "GpuSamplerProtocol",
    "MetricsEngineProtocol",
    "QueryRunnerProtocol",
    "ReportRepositoryProtocol",
    "SwapControllerProtocol",
    "SwapTimeout",
    "VllmScraperProtocol",
]
