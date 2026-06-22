"""Unit tests for the report-assembly path (eval/benchmark/assembly.py).

TDD RED-first driver for S06.  These tests are written **before** the
assembly module and exercise the *mapping contract* between the
``MetricsEngine`` output (``BenchmarkMetrics``), the S05 perf
collectors (``VllmServerMetrics`` / ``GpuMetrics`` / ``ModelLoadTiming``),
and the row payload persisted via :class:`BenchmarkReportRepository`.

A **fake repository** is used in place of the real DB-backed one so the
tests stay hermetic.  The real ``BenchmarkReportRepository`` is owned by
S09 (integration tests with testcontainers Postgres).

The tests pin:

* All T1 promoted columns (``recall_at_5/10``, ``ndcg_at_5/10``, ``mrr``)
  are populated from the ``BenchmarkMetrics`` input.
* All T2 promoted columns (``faithfulness_mean``, ``context_precision``)
  are populated.
* All T3 promoted columns (``context_recall``, ``factual_correctness``)
  are populated when the metrics engine returned them.
* The ``latency`` / ``resource`` JSONB payloads are populated from the
  S05 typed results.
* The ``ci_95`` JSONB round-trips when provided.
* The ``n_queries`` and ``n_judged`` columns are populated.
* The ``model_config`` JSONB round-trips.
* The ``created_by`` column is **exactly** the principal id passed in
  (Invariant 3 — never derived from the LLM, never from the request body).
* Empty golden set (n_queries == 0) is rejected at the assembly boundary
  (Boundary Behavior row 1) — the engine already raises, but the
  assembly asserts defensively.
* Reference-required metrics stay ``None`` when the engine produced
  ``None`` (AC4 / Invariant 5) and are passed through as ``None`` —
  NOT silently dropped, NOT replaced with ``0.0``.
* The ``metrics`` blob carries the per-query vectors paired by index
  (so the S07 stats module can run paired A/B tests).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeBenchmarkReportRepository:
    """In-memory fake of :class:`BenchmarkReportRepository`.

    Mirrors the **real** repository's ``create_report`` keyword-only
    signature verbatim (T1 + T2 + T3 promoted columns + the JSONB
    blobs + counts) so the assembly module can be unit-tested against
    a realistic surface.  The fake stores the captured kwargs as a
    flat dict and exposes ``get_report`` / ``list_reports`` over the
    same in-memory map.
    """

    def __init__(self) -> None:
        self._rows: dict[str, dict[str, Any]] = {}
        self.last_create_kwargs: dict[str, Any] | None = None

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
    ) -> str:
        """Insert one row; return its generated id."""
        new_id = f"row-{len(self._rows) + 1}"
        row = {
            "run_id": run_id,
            "dataset_id": dataset_id,
            "report_label": report_label,
            "git_sha": git_sha,
            "created_by": created_by,
            "model_config": model_config,
            "recall_at_5": recall_at_5,
            "recall_at_10": recall_at_10,
            "ndcg_at_5": ndcg_at_5,
            "ndcg_at_10": ndcg_at_10,
            "mrr": mrr,
            "faithfulness_mean": faithfulness_mean,
            "faithfulness_kappa": faithfulness_kappa,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "factual_correctness": factual_correctness,
            "metrics_row": metrics_row,
            "latency": latency,
            "resource": resource,
            "ci_95": ci_95,
            "n_queries": n_queries,
            "n_judged": n_judged,
        }
        self._rows[new_id] = row
        self.last_create_kwargs = dict(row)
        return new_id

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Read one row by id (or ``None``)."""
        rec = self._rows.get(report_id)
        return None if rec is None else dict(rec)

    async def list_reports(self, dataset_id: str) -> list[dict[str, Any]]:
        """List rows for a dataset (newest first).  Not exercised here."""
        return [dict(r) for r in self._rows.values() if r.get("dataset_id") == dataset_id]


def _bm_full() -> BenchmarkMetrics:
    """Build a fully-populated ``BenchmarkMetrics`` (T1+T2+T3)."""
    return BenchmarkMetrics(
        recall_at_5=0.8,
        recall_at_10=0.9,
        ndcg_at_5=0.7,
        ndcg_at_10=0.75,
        mrr=0.65,
        faithfulness_mean=0.92,
        faithfulness_kappa=None,  # not measured for this run
        context_precision=0.85,
        context_recall=0.6,
        factual_correctness=0.5,
        noise_sensitivity=0.4,
        n_queries=3,
        n_judged=3,
        errored=0,
        metrics={
            "recall_at_5": [1.0, 0.5, 0.9],
            "recall_at_10": [1.0, 0.8, 0.9],
            "ndcg_at_5": [1.0, 0.4, 0.7],
            "ndcg_at_10": [1.0, 0.5, 0.75],
            "mrr": [1.0, 0.5, 0.5],
            "faithfulness": [1.0, 0.9, 0.86],
            "context_precision": [0.9, 0.8, 0.85],
            "context_recall": [0.7, 0.5, 0.6],
            "factual_correctness": [0.6, 0.4, 0.5],
            "noise_sensitivity": [0.5, 0.3, 0.4],
        },
    )


def _bm_no_references() -> BenchmarkMetrics:
    """Build a ``BenchmarkMetrics`` for a dataset with no reference answers.

    T3 slots are ``None`` and the engine produced no per-query values for
    them (AC4 / Invariant 5).  T1 + T2 still populate.
    """
    return BenchmarkMetrics(
        recall_at_5=0.5,
        recall_at_10=0.7,
        ndcg_at_5=0.4,
        ndcg_at_10=0.6,
        mrr=0.45,
        faithfulness_mean=0.7,
        faithfulness_kappa=None,
        context_precision=0.6,
        context_recall=None,  # no reference answers
        factual_correctness=None,  # no reference answers
        noise_sensitivity=None,  # no reference answers
        n_queries=2,
        n_judged=2,
        errored=0,
        metrics={
            "recall_at_5": [0.4, 0.6],
            "recall_at_10": [0.6, 0.8],
            "ndcg_at_5": [0.3, 0.5],
            "ndcg_at_10": [0.5, 0.7],
            "mrr": [0.4, 0.5],
            "faithfulness": [0.7, 0.7],
            "context_precision": [0.6, 0.6],
            "context_recall": [None, None],
            "factual_correctness": [None, None],
            "noise_sensitivity": [None, None],
        },
    )


def _bm_empty() -> BenchmarkMetrics:
    """Build a ``BenchmarkMetrics`` for the empty-golden-set boundary case.

    The engine raises before this is ever built, so this is a guard
    against a future bug where the engine is bypassed.  The assembly
    must defensively reject ``n_queries == 0``.
    """
    return BenchmarkMetrics(
        recall_at_5=None,
        recall_at_10=None,
        ndcg_at_5=None,
        ndcg_at_10=None,
        mrr=None,
        faithfulness_mean=None,
        faithfulness_kappa=None,
        context_precision=None,
        context_recall=None,
        factual_correctness=None,
        noise_sensitivity=None,
        n_queries=0,
        n_judged=None,
        errored=0,
        metrics={},
    )


def _vllm() -> MagicMock:
    """Build a fake ``VllmServerMetrics`` for assembly tests."""
    m = MagicMock()
    m.ttft_p99 = 0.045
    m.tpot_p99 = 0.020
    m.e2e_p99 = 1.2
    m.queue_time_p99 = 0.005
    m.kv_cache_peak_pct = 0.72
    m.prefix_cache_hit_rate = 0.6
    m.prompt_tokens = 1024
    m.generation_tokens = 256
    m.prompt_tokens_delta = 1024
    m.generation_tokens_delta = 256
    m.prefix_cache_queries_delta = 5
    m.prefix_cache_hits_delta = 3
    m.raw = None
    return m


def _gpu(available: bool = True) -> MagicMock:
    """Build a fake ``GpuMetrics`` for assembly tests."""
    m = MagicMock()
    m.available = available
    m.util_peak_pct = 87.0
    m.util_mean_pct = 60.0
    m.util_p95_pct = 80.0
    m.vram_peak_mb = 1800
    m.vram_baseline_mb = 1000
    m.vram_incremental_mb = 800.0
    m.device_name = "NVIDIA H100"
    return m


def _model_load() -> MagicMock:
    """Build a fake ``ModelLoadTiming`` for assembly tests."""
    m = MagicMock()
    m.elapsed_ms = 1250.5
    m.result = "ok"
    m.error = None
    return m


def _per_stage_ms() -> dict[str, float]:
    """Build a fake per-stage ms dict (R-00122 §2)."""
    return {
        "retrieve_ms": 85.0,
        "encode_ms": 22.0,
        "rerank_ms": 35.0,
        "generate_ms": 950.0,
        "total_ms": 1092.0,
    }


# ---------------------------------------------------------------------------
# RED: public API surface
# ---------------------------------------------------------------------------


def test_assembly_module_exposes_assemble_report() -> None:
    """The assembly module MUST expose an async ``assemble_report`` function."""
    from iw_rag.eval.benchmark import assembly as asm

    assert hasattr(asm, "assemble_report"), "assembly.assemble_report is the public S06 surface"
    assert callable(asm.assemble_report)


# ---------------------------------------------------------------------------
# RED: empty golden set is rejected
# ---------------------------------------------------------------------------


async def test_assembly_rejects_empty_golden_set() -> None:
    """``n_queries == 0`` MUST raise — the engine already raises, but
    the assembly asserts defensively (Boundary Behavior row 1)."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    with pytest.raises((ConfigError, ValueError), match="assemble_report refused"):
        await assemble_report(
            repository=repo,  # type: ignore[arg-type]
            run_id="run-1",
            dataset_id="ds-1",
            report_label="baseline",
            git_sha="abc1234",
            created_by="alice",
            model_config={"embedder": "bge-m3"},
            metrics=_bm_empty(),
            per_stage_ms=_per_stage_ms(),
            vllm_metrics=_vllm(),
            gpu_metrics=_gpu(),
            model_load=_model_load(),
            ci_95={"ndcg_at_5": (0.5, 0.9)},
        )


# ---------------------------------------------------------------------------
# RED: the row payload is correct for a full T1+T2+T3 report
# ---------------------------------------------------------------------------


async def test_assembly_persists_full_t1_t2_t3_payload() -> None:
    """All promoted T1/T2/T3 columns land in the row payload."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="11111111-1111-1111-1111-111111111111",
        dataset_id="22222222-2222-2222-2222-222222222222",
        report_label="qwen3-14b-2026-06-10",
        git_sha="abc1234",
        created_by="alice",
        model_config={"embedder": "bge-m3", "generator": "qwen3-14b"},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={"ndcg_at_5": (0.5, 0.9), "recall_at_10": (0.6, 0.95)},
    )
    assert report_id == "row-1"
    row = await repo.get_report(report_id)
    assert row is not None

    # Identity + provenance columns
    assert row["run_id"] == "11111111-1111-1111-1111-111111111111"
    assert row["dataset_id"] == "22222222-2222-2222-2222-222222222222"
    assert row["report_label"] == "qwen3-14b-2026-06-10"
    assert row["git_sha"] == "abc1234"
    assert row["created_by"] == "alice"

    # T1 promoted columns — pass-through from BenchmarkMetrics.
    assert row["recall_at_5"] == pytest.approx(0.8)
    assert row["recall_at_10"] == pytest.approx(0.9)
    assert row["ndcg_at_5"] == pytest.approx(0.7)
    assert row["ndcg_at_10"] == pytest.approx(0.75)
    assert row["mrr"] == pytest.approx(0.65)

    # T2 promoted columns.
    assert row["faithfulness_mean"] == pytest.approx(0.92)
    assert row["faithfulness_kappa"] is None  # not measured for the run
    assert row["context_precision"] == pytest.approx(0.85)

    # T3 promoted columns.
    assert row["context_recall"] == pytest.approx(0.6)
    assert row["factual_correctness"] == pytest.approx(0.5)

    # Counts.
    assert row["n_queries"] == 3
    assert row["n_judged"] == 3


# ---------------------------------------------------------------------------
# RED: latency / resource JSONB payloads are populated from S05 results
# ---------------------------------------------------------------------------


async def test_assembly_packs_latency_and_resource_jsonb() -> None:
    """``latency`` carries per-stage ms + TTFT/TPOT/E2E/queue + model_load_ms.
    ``resource`` carries GPU + vLLM server metrics.

    (R-00122 §"Recommendation" — the JSONB shape.)
    """
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="bob",
        model_config={},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None

    # latency JSONB
    latency = row["latency"]
    assert isinstance(latency, dict)
    assert latency["per_stage_ms"] == _per_stage_ms()
    assert latency["ttft_p99_s"] == pytest.approx(0.045)
    assert latency["tpot_p99_s"] == pytest.approx(0.020)
    assert latency["e2e_p99_s"] == pytest.approx(1.2)
    assert latency["queue_time_p99_s"] == pytest.approx(0.005)
    assert latency["model_load_ms"] == pytest.approx(1250.5)

    # resource JSONB
    resource = row["resource"]
    assert isinstance(resource, dict)
    assert resource["gpu"]["available"] is True
    assert resource["gpu"]["util_peak_pct"] == pytest.approx(87.0)
    assert resource["gpu"]["vram_incremental_mb"] == pytest.approx(800.0)
    assert resource["gpu"]["device_name"] == "NVIDIA H100"
    assert resource["vllm_server"]["ttft_p99"] == pytest.approx(0.045)
    assert resource["vllm_server"]["prefix_cache_hit_rate"] == pytest.approx(0.6)
    assert resource["vllm_server"]["prompt_tokens_delta"] == 1024


# ---------------------------------------------------------------------------
# RED: ci_95 + metrics blobs round-trip
# ---------------------------------------------------------------------------


async def test_assembly_round_trips_ci_95_and_metrics_blob() -> None:
    """``ci_95`` and ``metrics`` JSONB columns round-trip verbatim."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    bm = _bm_full()
    ci = {"ndcg_at_5": (0.5, 0.9), "recall_at_10": (0.6, 0.95)}
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config={"embedder": "bge-m3"},
        metrics=bm,
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95=ci,
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["ci_95"] == ci
    # The assembly passes the per-tier blob via the repository's
    # ``metrics_row`` kwarg (mirroring the SQL column name ``metrics``).
    assert row["metrics_row"] == bm.metrics  # full per-tier blob, paired by index
    # The per-query vector for ndcg_at_5 has length == n_queries.
    assert len(row["metrics_row"]["ndcg_at_5"]) == row["n_queries"]


# ---------------------------------------------------------------------------
# RED: created_by is the principal id (Invariant 3)
# ---------------------------------------------------------------------------


async def test_assembly_created_by_is_exactly_the_passed_principal() -> None:
    """``created_by`` MUST be the principal id passed in, verbatim.

    Invariant 3 — never derived from the LLM, never from a request body
    field.  The fake's ``created_by`` here is ``"user:alice-1234"``;
    the engine never sees this string, so the only way it could end up
    in the row is via the parameter.  A future bug that derived
    ``created_by`` from ``created_by or "anonymous"`` or from a
    ``request.principal`` would fail this test.
    """
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="user:alice-1234",  # explicit, formatted principal
        model_config={},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["created_by"] == "user:alice-1234"


# ---------------------------------------------------------------------------
# RED: reference-required metrics stay None when dataset has no refs (AC4)
# ---------------------------------------------------------------------------


async def test_assembly_keeps_reference_required_metrics_none() -> None:
    """``context_recall`` / ``factual_correctness`` MUST be ``None``
    when the engine produced ``None`` (AC4 / Invariant 5).  The assembly
    must not silently replace them with ``0.0``."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config={},
        metrics=_bm_no_references(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["context_recall"] is None
    assert row["factual_correctness"] is None
    # T1 + T2 still populate.
    assert row["recall_at_5"] == pytest.approx(0.5)
    assert row["faithfulness_mean"] == pytest.approx(0.7)
    assert row["context_precision"] == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# RED: GPU degradation (Boundary row 3 — no GPU / pynvml unavailable)
# ---------------------------------------------------------------------------


async def test_assembly_handles_gpu_unavailable_gracefully() -> None:
    """When ``GpuMetrics.available is False``, the report still persists.
    ``resource.gpu`` carries the ``available=False`` flag; numeric
    fields are ``None``/absent."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config={},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(available=False),  # no GPU
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None  # report persisted
    assert row["resource"]["gpu"]["available"] is False


# ---------------------------------------------------------------------------
# RED: model_config round-trips
# ---------------------------------------------------------------------------


async def test_assembly_round_trips_model_config() -> None:
    """The ``model_config`` JSONB round-trips the passed dict verbatim."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    cfg = {
        "embedder": "bge-m3",
        "generator": "qwen3-14b",
        "reranker": "bge-reranker-v2-m3",
        "retrieval": {"candidate_k": 40, "top_k": 8},
    }
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config=cfg,
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["model_config"] == cfg


# ---------------------------------------------------------------------------
# Boundary: ``created_by=None`` is the offline batch path (the
# orchestrator sets it when there is no authenticated principal;
# the integration test for the same contract is
# ``test_created_by_accepts_none_for_offline_batch_path``).  Mirrored
# here as a unit-test pin so a future regression to
# ``created_by or "anonymous"`` or ``created_by or ""`` is caught at
# the assembly boundary, not just at the DB.
# ---------------------------------------------------------------------------


async def test_assembly_created_by_none_is_preserved_verbatim() -> None:
    """``created_by=None`` is stored verbatim (never coerced to a
    placeholder string).  The orchestrator is the only place that
    decides what ``created_by`` means; the assembly is a pure mapper."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="offline-batch",
        git_sha=None,
        created_by=None,  # the offline batch path
        model_config={},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["created_by"] is None
    # And the captured kwargs at the call site show the assembly did not
    # mutate the value.
    assert repo.last_create_kwargs is not None
    assert repo.last_create_kwargs["created_by"] is None


# ---------------------------------------------------------------------------
# Boundary: ``ci_95`` is ``None`` (the S07 stats module was not run).
# The assembly must store ``NULL``, not ``{}``, so a downstream renderer
# can distinguish "no CI computed" from "the BCa CI happened to be the
# degenerate (0, 0)".
# ---------------------------------------------------------------------------


async def test_assembly_ci_95_none_stores_null() -> None:
    """``ci_95=None`` (the S07 stats module was not run) is stored as
    ``NULL``, not as an empty dict."""
    from iw_rag.eval.benchmark.assembly import assemble_report

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config={},
        metrics=_bm_full(),
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95=None,  # S07 not run
    )
    row = await repo.get_report(report_id)
    assert row is not None
    assert row["ci_95"] is None


# ---------------------------------------------------------------------------
# Boundary: T1 retrieval means are the **per-query means** of the
# score_retrieval output — the assembly must NOT re-average; the
# contract is "means come from the engine, assembly is a pure mapper".
# ---------------------------------------------------------------------------


async def test_assembly_does_not_recompute_means_from_per_query() -> None:
    """If the means in ``BenchmarkMetrics`` do not match the mean of
    the per-query vectors, the assembly preserves the means
    verbatim (it does NOT recompute from the per-query blob).  This
    pins the "assembly is a pure mapper" contract: the engine is
    the single source of truth for the headline means."""
    from iw_rag.eval.benchmark.assembly import assemble_report
    from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics

    # Per-query vectors say ~0.90, but the promoted mean is 0.42 (the
    # engine is allowed to weight samples differently in the future).
    # The assembly must use 0.42, NOT a fresh mean of the blob.
    bm = BenchmarkMetrics(
        recall_at_5=0.42,
        recall_at_10=0.50,
        ndcg_at_5=0.55,
        ndcg_at_10=0.60,
        mrr=0.30,
        faithfulness_mean=0.70,
        faithfulness_kappa=None,
        context_precision=0.65,
        context_recall=0.50,
        factual_correctness=0.40,
        n_queries=3,
        n_judged=3,
        errored=0,
        metrics={
            "recall_at_5": [0.9, 0.9, 0.9],  # mean = 0.9, but row says 0.42
            "recall_at_10": [0.9, 0.9, 0.9],
            "ndcg_at_5": [0.9, 0.9, 0.9],
            "ndcg_at_10": [0.9, 0.9, 0.9],
            "mrr": [0.9, 0.9, 0.9],
            "faithfulness": [0.9, 0.9, 0.9],
            "context_precision": [0.9, 0.9, 0.9],
            "context_recall": [0.9, 0.9, 0.9],
            "factual_correctness": [0.9, 0.9, 0.9],
            "noise_sensitivity": [0.9, 0.9, 0.9],
        },
    )

    repo = FakeBenchmarkReportRepository()
    report_id = await assemble_report(
        repository=repo,  # type: ignore[arg-type]
        run_id="run-1",
        dataset_id="ds-1",
        report_label="baseline",
        git_sha=None,
        created_by="alice",
        model_config={},
        metrics=bm,
        per_stage_ms=_per_stage_ms(),
        vllm_metrics=_vllm(),
        gpu_metrics=_gpu(),
        model_load=_model_load(),
        ci_95={},
    )
    row = await repo.get_report(report_id)
    assert row is not None
    # Promoted means are the engine's, not the blob's.
    assert row["recall_at_5"] == pytest.approx(0.42)
    assert row["ndcg_at_5"] == pytest.approx(0.55)
    # The blob round-trips verbatim.
    assert row["metrics_row"]["recall_at_5"] == [0.9, 0.9, 0.9]
