"""Report assembly — the mapping from engine + perf results to a row payload.

The single function :func:`assemble_report` is the **only** way a
``benchmark_reports`` row gets created.  It takes the
:class:`~iw_rag.eval.benchmark.metrics_engine.BenchmarkMetrics` result
from the S02 metrics engine, the S05 perf collector results
(:class:`~iw_rag.eval.benchmark.perf.VllmServerMetrics`,
:class:`~iw_rag.eval.benchmark.perf.GpuMetrics`,
:class:`~iw_rag.eval.benchmark.perf.ModelLoadTiming`), a per-stage-ms
dict (R-00122 §2), an optional bootstrap ``ci_95`` dict (S07), the
static ``model_config`` snapshot, and the authenticated principal id
(``created_by``).  It then maps these onto the column payload of a
``benchmark_reports`` row and hands it to a
:class:`~iw_rag.eval.benchmark.repository.BenchmarkReportRepository`-shaped
object (the real DB-backed one in production, a fake in unit tests).

The function is the **only** S06 mutable surface — the repository
itself is insert-only / read-only (Invariant 2 — reports are
append-only history), and the assembly never derives ``created_by``
from the LLM, from a request body, or from anything other than the
explicit parameter (Invariant 3 — principal stored from auth only).

Boundary behavior (per F-00093 Feature_Design §Boundary Behavior):

* **Empty golden set** — the metrics engine raises :class:`ConfigError`
  upstream, but :func:`assemble_report` defensively raises
  :class:`ConfigError` too if it ever receives ``n_queries == 0`` —
  the table's ``n_queries`` column is ``NOT NULL`` and a meaningless
  all-zero row is forbidden.
* **No GPU / pynvml unavailable** — the assembly does not require
  GPU data; ``GpuMetrics(available=False)`` round-trips to
  ``resource.gpu.available == False`` with all numeric fields
  ``None``.  The report persists.
* **Reference-required metrics absent** — when the engine produces
  ``None`` for ``context_recall`` / ``factual_correctness``
  (AC4 / Invariant 5), the assembly writes them as ``None`` to the
  row — never ``0.0``, never silently dropped.

This module is **offline-only** (CONVENTIONS §10 / lld-eval-observability
§5 — two-clocks rule) and MUST NOT be imported by any hot-path module
(``api/``, ``agent/``, ``retrieval/``, ``generation/``).  The
``tests/unit/test_offline_only_import_guard.py`` static check enforces
this.
"""

from __future__ import annotations

from typing import Any, Protocol

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics
from iw_rag.eval.benchmark.perf import GpuMetrics, ModelLoadTiming, VllmServerMetrics


class _BenchmarkReportRepositoryLike(Protocol):
    """The minimum surface :func:`assemble_report` needs from a repository.

    Mirrors :class:`iw_rag.eval.benchmark.repository.BenchmarkReportRepository`
    so the assembly can be unit-tested with an in-memory fake (no DB).
    The real repository implements ``create_report`` with the
    keyword-only explicit signature below (T1 + T2 + T3 promoted
    columns + the JSONB blobs + counts); the fake in
    ``tests/unit/eval/benchmark/test_assembly.py`` mirrors it
    verbatim so the test exercises the production call shape.
    """

    async def create_report(
        self,
        *,
        run_id: str,
        dataset_id: str,
        report_label: str,
        git_sha: str | None,
        created_by: str | None,
        model_config: dict[str, Any],
        recall_at_5: float | None,
        recall_at_10: float | None,
        ndcg_at_5: float | None,
        ndcg_at_10: float | None,
        mrr: float | None,
        faithfulness_mean: float | None,
        faithfulness_kappa: float | None,
        context_precision: float | None,
        context_recall: float | None,
        factual_correctness: float | None,
        metrics_row: dict[str, Any],
        latency: dict[str, Any] | None,
        resource: dict[str, Any] | None,
        ci_95: dict[str, Any] | None,
        n_queries: int,
        n_judged: int | None,
    ) -> str: ...


def _gpu_to_dict(gpu: GpuMetrics) -> dict[str, Any]:
    """Serialize a :class:`GpuMetrics` to a JSON-safe dict for ``resource.gpu``.

    The dataclass's ``available`` flag is preserved verbatim so a
    no-GPU run (``available=False``) is still surfaced and the column
    is unambiguous on read-back.  Numeric fields are ``None`` when
    the GPU was unavailable.
    """
    return {
        "available": gpu.available,
        "util_peak_pct": gpu.util_peak_pct,
        "util_mean_pct": gpu.util_mean_pct,
        "util_p95_pct": gpu.util_p95_pct,
        "vram_peak_mb": gpu.vram_peak_mb,
        "vram_baseline_mb": gpu.vram_baseline_mb,
        "vram_incremental_mb": gpu.vram_incremental_mb,
        "device_name": gpu.device_name,
    }


def _vllm_to_dict(vllm: VllmServerMetrics) -> dict[str, Any]:
    """Serialize a :class:`VllmServerMetrics` to a JSON-safe dict.

    The ``raw`` field (the parsed Prometheus map) is **excluded** —
    it is a test-traceability field, not a metric.  The promoted
    scalar fields are the canonical record.
    """
    return {
        "ttft_p99": vllm.ttft_p99,
        "tpot_p99": vllm.tpot_p99,
        "e2e_p99": vllm.e2e_p99,
        "queue_time_p99": vllm.queue_time_p99,
        "kv_cache_peak_pct": vllm.kv_cache_peak_pct,
        "prefix_cache_hit_rate": vllm.prefix_cache_hit_rate,
        "prompt_tokens": vllm.prompt_tokens,
        "generation_tokens": vllm.generation_tokens,
        "prompt_tokens_delta": vllm.prompt_tokens_delta,
        "generation_tokens_delta": vllm.generation_tokens_delta,
        "prefix_cache_queries_delta": vllm.prefix_cache_queries_delta,
        "prefix_cache_hits_delta": vllm.prefix_cache_hits_delta,
    }


def _model_load_to_dict(model_load: ModelLoadTiming) -> dict[str, Any]:
    """Serialize a :class:`ModelLoadTiming` to a JSON-safe dict."""
    return {
        "elapsed_ms": model_load.elapsed_ms,
        "error": model_load.error,
    }


async def assemble_report(
    *,
    repository: _BenchmarkReportRepositoryLike,
    run_id: str,
    dataset_id: str,
    report_label: str,
    git_sha: str | None,
    created_by: str | None,
    model_config: dict[str, Any],
    metrics: BenchmarkMetrics,
    per_stage_ms: dict[str, float],
    vllm_metrics: VllmServerMetrics,
    gpu_metrics: GpuMetrics,
    model_load: ModelLoadTiming,
    ci_95: dict[str, Any] | None = None,
    phase_ms: dict[str, float] | None = None,
) -> str:
    """Assemble a :class:`BenchmarkMetrics` + perf results into a stored row.

    The single public S06 surface for turning a finished eval run into
    a persisted ``benchmark_reports`` row.  Maps the T1 / T2 / T3
    promoted columns from :class:`BenchmarkMetrics` to the matching
    float columns, packs the S05 perf collector results into
    ``latency`` / ``resource`` JSONB, passes through ``ci_95`` and
    ``metrics`` JSONB verbatim, and hands the row to the repository's
    ``create_report`` method.

    Args:
        repository: A :class:`BenchmarkReportRepository`-shaped object
            (the real DB-backed one in production; a fake in unit
            tests).  The assembly uses only the ``create_report``
            surface.
        run_id: The :class:`eval_runs` id this report rolls up.
            str at the boundary (the real repository converts to uuid).
        dataset_id: The :class:`eval_datasets` id this report belongs
            to.  str at the boundary.
        report_label: Human-readable label, e.g.
            ``"qwen3-14b-2026-06-10"``.
        git_sha: The git SHA of the configuration snapshot, when
            available.
        created_by: The authenticated principal id from the auth
            layer (Invariant 3).  Stored verbatim, never derived from
            the LLM or from a request body field.  ``None`` only when
            the offline batch path has no authenticated principal
            (orchestrator-set).
        model_config: The static model configuration snapshot — what
            was loaded at the time of the run.  Stored as JSONB so
            two reports can be lined up only when this column matches.
        metrics: The :class:`BenchmarkMetrics` from the S02 engine.
        per_stage_ms: A dict of per-stage wall-clock ms (R-00122 §2),
            e.g. ``{"retrieve_ms": 85.0, "rerank_ms": 35.0,
            "generate_ms": 950.0, ...}``.  Aggregated offline from
            the existing ``observability/timeline.py`` ``StepRecord``s
            — the assembly does not run a new query.
        vllm_metrics: The :class:`VllmServerMetrics` from the S05
            scraper.  ``available=False`` is fine; the dict round-trips
            with all numeric fields ``None``.
        gpu_metrics: The :class:`GpuMetrics` from the S05 sampler.
        model_load: The :class:`ModelLoadTiming` from the S05
            ``time_model_load_ms`` helper.
        ci_95: The per-metric 95% bootstrap CIs from the S07 stats
            module, when available.  ``None``/``{}`` is fine — the
            column is nullable in the schema.

    Returns:
        The new report's id (str, a uuid-stringified form for the
        real repository; an arbitrary key in the fake).

    Raises:
        ConfigError: If ``metrics.n_queries == 0`` (Boundary Behavior
            row 1 — the engine raises upstream, but assembly asserts
            defensively so a future bug that bypasses the engine is
            caught at the persistence boundary, not at the table).
    """
    # Boundary: refuse to persist a meaningless all-zero report.
    if metrics.n_queries <= 0:
        raise ConfigError(
            f"assemble_report refused: metrics.n_queries={metrics.n_queries}. "
            "The metrics engine raises upstream on an empty golden set; this "
            "assertion is defensive in case the engine is bypassed."
        )

    # T1 — promote the means from BenchmarkMetrics to the float columns.
    t1 = {
        "recall_at_5": metrics.recall_at_5,
        "recall_at_10": metrics.recall_at_10,
        "ndcg_at_5": metrics.ndcg_at_5,
        "ndcg_at_10": metrics.ndcg_at_10,
        "mrr": metrics.mrr,
    }
    # T2 — reference-free judged columns.  ``faithfulness_kappa``
    # stays None when the kappa harness did not run for this run
    # (it is not a metric of the per-query score, it is a judge↔human
    # agreement statistic from a separate harness).
    t2 = {
        "faithfulness_mean": metrics.faithfulness_mean,
        "faithfulness_kappa": metrics.faithfulness_kappa,
        "context_precision": metrics.context_precision,
    }
    # T3 — reference-required judged columns.  Pass through None
    # verbatim when the dataset had no reference answers (AC4 /
    # Invariant 5) — the engine already guarantees the engine
    # never called a judge for them.
    t3 = {
        "context_recall": metrics.context_recall,
        "factual_correctness": metrics.factual_correctness,
    }

    # latency JSONB — per-stage ms + the vLLM-derived percentiles
    # + model_load_ms.  R-00122 §"Recommendation" JSON shape.
    latency_payload: dict[str, Any] = {
        "per_stage_ms": dict(per_stage_ms),
        "ttft_p99_s": vllm_metrics.ttft_p99,
        "tpot_p99_s": vllm_metrics.tpot_p99,
        "e2e_p99_s": vllm_metrics.e2e_p99,
        "queue_time_p99_s": vllm_metrics.queue_time_p99,
        "model_load_ms": model_load.elapsed_ms,
        # Per-phase wall-clock breakdown ("how long each step took"): the model
        # swap/load, the generation loop, the scoring/judging phase (+ the
        # engine's faithfulness/grading/ragas split).  Rendered as a waterfall.
        "phase_ms": dict(phase_ms) if phase_ms else None,
    }

    # resource JSONB — gpu + vllm_server.  GPU may be unavailable;
    # the dict still carries the ``available=False`` flag so the
    # reader can tell the difference between "ran without a GPU"
    # and "GPU was not sampled".
    resource_payload: dict[str, Any] = {
        "gpu": _gpu_to_dict(gpu_metrics),
        "vllm_server": _vllm_to_dict(vllm_metrics),
    }

    # Assemble the row payload.  str ids are passed verbatim — the
    # real repository converts to uuid at the boundary (matching
    # EvalRepository's str↔uuid boundary convention).  The call
    # uses explicit keyword args (one per column) so the repository
    # signature mirrors the schema and a future schema change
    # surfaces as a test compile error rather than a silent
    # mis-mapping.
    return await repository.create_report(
        run_id=run_id,
        dataset_id=dataset_id,
        report_label=report_label,
        git_sha=git_sha,
        created_by=created_by,  # Invariant 3 — verbatim, never derived
        model_config=dict(model_config),
        # T1 promoted columns (means from BenchmarkMetrics)
        recall_at_5=t1["recall_at_5"],
        recall_at_10=t1["recall_at_10"],
        ndcg_at_5=t1["ndcg_at_5"],
        ndcg_at_10=t1["ndcg_at_10"],
        mrr=t1["mrr"],
        # T2 promoted columns
        faithfulness_mean=t2["faithfulness_mean"],
        faithfulness_kappa=t2["faithfulness_kappa"],
        context_precision=t2["context_precision"],
        # T3 promoted columns (None when no reference answers — AC4)
        context_recall=t3["context_recall"],
        factual_correctness=t3["factual_correctness"],
        # JSONB columns
        metrics_row=dict(metrics.metrics),
        latency=latency_payload,
        resource=resource_payload,
        ci_95=dict(ci_95) if ci_95 else None,
        # Counts
        n_queries=metrics.n_queries,
        n_judged=metrics.n_judged,
    )


__all__ = [
    "assemble_report",
]
