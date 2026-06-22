"""Unit tests for the Benchmark Studio run-orchestration runner (F-00095, S01).

TDD RED-first driver for S01.  These tests pin the runner's behaviour
before the implementation.  The runner is the engine that orchestrates
one full benchmark and streams its progress as ``{"event": ...,
"data": ...}`` dicts (the API layer in S02 wraps them into SSE bytes).

The tests use **fakes** (no live GPU, no live swap-controller, no live
retrieval+generation pipeline):

* A fake **swap-controller client** that records ``activate`` /
  ``wait_for_ready`` calls and can simulate a swap-timeout.
* A fake **metrics engine** that returns a canned
  :class:`BenchmarkMetrics` and records its call.
* A fake **report repository** that records the ``create_report`` call.
* A fake **GPU sampler** that records ``start`` / ``stop`` / baseline
  calls and returns a canned :class:`GpuMetrics`.
* A fake **VllmMetricsScraper** that returns a canned
  :class:`VllmServerMetrics`.
* A fake **query runner** that replays canned per-query
  ``(RetrievalResult, answer, contexts)`` tuples.
* A fake **golden set loader** that returns canned
  :class:`GoldenQuery` lists (and can omit reference answers to exercise
  AC4's Tier-3 omission path).

The tests pin:

* **AC1** — happy path emits ``queued → swapping → ready → running×N
  → scoring → done`` in order, ``done`` carries a ``report_id``, and the
  persisted report's ``model_config`` matches the submission.
* **Boundary row 1** — swap-timeout path emits a terminal ``error``
  with ``reason_code: "swap_timeout"`` and persists ZERO reports.
* **Boundary row 2** — dataset without reference answers omits Tier-3
  (``context_recall`` / ``factual_correctness`` are ``None``) and
  still persists one report.
* **Invariant 4** — ``model_load_ms`` is recorded separately from
  per-query latency (the per_stage_ms dict fed to assembly does NOT
  contain ``model_load_ms``; ``model_load.elapsed_ms`` IS fed to
  assembly; the row's ``latency`` JSONB contains both — but
  ``model_load_ms`` is its own key, never summed into per-stage /
  per-query ms).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

from iw_rag.domain.models import GeneratedAnswer, RetrievalResult, RetrievedChunk
from iw_rag.eval.benchmark.metrics_engine import BenchmarkMetrics
from iw_rag.eval.benchmark.perf import GpuMetrics, ModelLoadTiming, VllmServerMetrics
from iw_rag.eval.repository import GoldenQuery

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeSwapControllerClient:
    """In-memory fake of the swap-controller client (F-00094).

    Mirrors the minimal surface :class:`BenchmarkRunner` consumes:
    :meth:`activate` and :meth:`wait_for_ready`.  By default both
    succeed (no-op activation, immediate ready).  Tests can set
    ``swap_timeout=True`` to make :meth:`wait_for_ready` raise
    :class:`SwapTimeout` instead of returning.
    """

    def __init__(self, *, swap_timeout: bool = False) -> None:
        """Configure the fake.

        Args:
            swap_timeout: When ``True``, :meth:`wait_for_ready` raises
                :class:`SwapTimeout` to exercise the AC2 failure path.
        """
        self._swap_timeout = swap_timeout
        self.activate_calls: list[dict[str, Any]] = []
        self.wait_calls: list[dict[str, Any]] = []
        self.last_status: dict[str, Any] = {"state": "ready", "active_model_id": "gen-x"}

    async def activate(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Record the activation call and return a ready status dict.

        Args:
            model_config: The model config payload (the same shape
                passed to :meth:`BenchmarkRunner.run`).

        Returns:
            A status dict mirroring the swap-controller's response.
        """
        self.activate_calls.append({"model_config": dict(model_config)})
        return {"state": "swapping", "active_model_id": model_config.get("generator")}

    async def wait_for_ready(self, *, timeout_s: float) -> dict[str, Any]:
        """Block (in real life) until ready; here just record and decide.

        Args:
            timeout_s: The health-gate timeout in seconds.

        Returns:
            A ready status dict.

        Raises:
            SwapTimeout: When the fake was configured with
                ``swap_timeout=True``.
        """
        self.wait_calls.append({"timeout_s": timeout_s})
        if self._swap_timeout:
            from iw_rag.eval.benchmark.runner import SwapTimeout

            raise SwapTimeout(f"swap never became ready within {timeout_s}s")
        return self.last_status


class FakeMetricsEngine:
    """In-memory fake of the F-00093 :class:`MetricsEngine`.

    Returns a canned :class:`BenchmarkMetrics` from a queue (one
    instance per call), so a test can assert the per-call shape.  When
    the queue is exhausted the engine records an error (caller
    assertions verify the right number of calls).
    """

    def __init__(self, scripted: list[BenchmarkMetrics]) -> None:
        """Configure the fake.

        Args:
            scripted: The :class:`BenchmarkMetrics` results to replay,
                one per ``compute`` call.  When the list is exhausted
                subsequent calls return the LAST entry (avoids an
                off-by-one flake).
        """
        self._scripted = list(scripted) if scripted else []
        self.calls: list[dict[str, Any]] = []

    async def compute(
        self,
        queries: Sequence[GoldenQuery],
        retrievals: Sequence[RetrievalResult],
        answers: Sequence[str],
        contexts: Sequence[Sequence[str]],
        *,
        heavy_ragas: bool = False,
    ) -> BenchmarkMetrics:
        """Replay the scripted result and record the call inputs.

        Args:
            queries: The golden queries (recorded).
            retrievals: The per-query retrievals (recorded).
            answers: The per-query answers (recorded).
            contexts: The per-query contexts (recorded).

        Returns:
            The next :class:`BenchmarkMetrics` from the script.
        """
        self.calls.append(
            {
                "queries": list(queries),
                "retrievals": list(retrievals),
                "answers": list(answers),
                "contexts": [list(c) for c in contexts],
            }
        )
        if not self._scripted:
            return BenchmarkMetrics(
                recall_at_5=0.0,
                recall_at_10=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                mrr=0.0,
                faithfulness_mean=None,
                faithfulness_kappa=None,
                context_precision=None,
                context_recall=None,
                factual_correctness=None,
                noise_sensitivity=None,
                n_queries=len(queries),
                n_judged=None,
                errored=0,
                metrics={},
            )
        idx = min(len(self.calls) - 1, len(self._scripted) - 1)
        return self._scripted[idx]


class FakeReportRepository:
    """In-memory fake of the F-00093 :class:`BenchmarkReportRepository`.

    Mirrors the real repository's ``create_report`` keyword-only
    signature verbatim (subset sufficient for the runner's call) and
    records every call.  ``get_report`` / ``list_reports`` are no-ops.
    """

    def __init__(self) -> None:
        """Initialise the fake."""
        self._rows: dict[str, dict[str, Any]] = {}
        self.create_calls: list[dict[str, Any]] = []

    async def create_report(self, **kwargs: Any) -> str:
        """Insert one row; return its generated id.

        Args:
            **kwargs: The row payload.

        Returns:
            A unique row id.
        """
        new_id = f"row-{len(self._rows) + 1}"
        self._rows[new_id] = dict(kwargs)
        self.create_calls.append(dict(kwargs))
        return new_id

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Read one row by id (or ``None``)."""
        rec = self._rows.get(report_id)
        return None if rec is None else dict(rec)

    async def list_reports(self, dataset_id: str) -> list[dict[str, Any]]:
        """List rows for a dataset (newest first).  Not exercised here."""
        return [dict(r) for r in self._rows.values() if r.get("dataset_id") == dataset_id]


class FakeGpuSampler:
    """In-memory fake of the F-00093 :class:`GpuSampler`."""

    def __init__(self, baseline_mb: int = 1000) -> None:
        """Configure the fake.

        Args:
            baseline_mb: The baseline VRAM to return from
                :meth:`snapshot_baseline`.
        """
        self._baseline_mb = baseline_mb
        self.snapshot_calls: list[None] = []
        self.start_calls: list[None] = []
        self.mark_calls: list[None] = []
        self.stop_calls: list[None] = []

    def snapshot_baseline(self) -> Any:
        """Record a snapshot and return a canned baseline."""
        self.snapshot_calls.append(None)
        snap = MagicMock()
        snap.available = True
        snap.vram_used_mb = self._baseline_mb
        snap.device_name = "NVIDIA H100"
        return snap

    def start(self) -> None:
        """Record a start call."""
        self.start_calls.append(None)

    def mark_run_start(self) -> None:
        """Record a measurement-window mark call."""
        self.mark_calls.append(None)

    def stop(self) -> GpuMetrics:
        """Record a stop call and return a canned aggregate."""
        self.stop_calls.append(None)
        return GpuMetrics(
            available=True,
            util_peak_pct=80.0,
            util_mean_pct=55.0,
            util_p95_pct=72.0,
            vram_peak_mb=1500,
            vram_baseline_mb=self._baseline_mb,
            vram_incremental_mb=500.0,
            device_name="NVIDIA H100",
        )


class FakeVllmScraper:
    """In-memory fake of the F-00093 :class:`VllmMetricsScraper`."""

    def __init__(self, before: VllmServerMetrics, after: VllmServerMetrics) -> None:
        """Configure the fake.

        Args:
            before: Returned from the first :meth:`scrape` call.
            after: Returned from the second :meth:`scrape` call.
        """
        self._before = before
        self._after = after
        self.scrape_calls: list[None] = []

    async def scrape(self) -> VllmServerMetrics:
        """Replay the before/after sequence and return a metric."""
        self.scrape_calls.append(None)
        return self._before if len(self.scrape_calls) == 1 else self._after


class FakeGoldenSetLoader:
    """In-memory fake of the F-00095 golden set loader.

    Returns a canned list of :class:`GoldenQuery` records.  The list
    may omit ``reference_answer`` to exercise the Tier-3 omission path.
    """

    def __init__(self, queries: list[GoldenQuery]) -> None:
        """Configure the fake.

        Args:
            queries: The queries to return from :meth:`load`.
        """
        self._queries = list(queries)
        self.load_calls: list[str] = []

    async def load(self, dataset_id: str) -> list[GoldenQuery]:
        """Record the call and return the canned queries.

        Args:
            dataset_id: The dataset id being requested (recorded).

        Returns:
            The list of :class:`GoldenQuery` records.
        """
        self.load_calls.append(dataset_id)
        return list(self._queries)


class FakeQueryRunner:
    """In-memory fake of the per-query retrieval+generation runner.

    Replays a script of ``(RetrievalResult, GeneratedAnswer, contexts)``
    tuples — one per call.  Each call records its ``i``/``total`` and
    the ``question`` text (so tests can assert the per-query
    ``running`` event ordering).
    """

    def __init__(
        self,
        script: list[tuple[RetrievalResult, GeneratedAnswer, list[str]]],
    ) -> None:
        """Configure the fake.

        Args:
            script: The per-query ``(retrieval, answer, contexts)``
                tuples to replay.
        """
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def run_query(
        self,
        question: str,
        ctx: Any,
        i: int,
        total: int,
    ) -> tuple[RetrievalResult, GeneratedAnswer, list[str]]:
        """Replay the next scripted tuple and record the call.

        Args:
            question: The query text (recorded).
            ctx: The access context (recorded, structurally).
            i: The zero-based query index (recorded).
            total: The total query count (recorded).

        Returns:
            The next ``(RetrievalResult, GeneratedAnswer, contexts)``
            tuple from the script.
        """
        self.calls.append({"question": question, "i": i, "total": total, "ctx": ctx})
        if not self._script:
            r = RetrievalResult(chunks=[], retriever="spine", acl_applied=True)
            a = GeneratedAnswer(answer="", grounded=False, citations=[], used_chunk_ids=[])
            return r, a, []
        idx = min(len(self.calls) - 1, len(self._script) - 1)
        return self._script[idx]


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _golden_with_refs(n: int = 3) -> list[GoldenQuery]:
    """Build ``n`` golden queries WITH reference answers."""
    return [
        GoldenQuery(
            id=f"q{i}",
            question=f"question-{i}",
            reference_answer=f"ref-{i}",
            relevant_chunk_ids=[f"c{i}"],
            query_type="factual",
        )
        for i in range(n)
    ]


def _golden_no_refs(n: int = 3) -> list[GoldenQuery]:
    """Build ``n`` golden queries WITHOUT reference answers (AC4 path)."""
    return [
        GoldenQuery(
            id=f"q{i}",
            question=f"question-{i}",
            reference_answer=None,
            relevant_chunk_ids=[f"c{i}"],
            query_type="factual",
        )
        for i in range(n)
    ]


def _retrieval(ids: list[str]) -> RetrievalResult:
    """Build a best-first :class:`RetrievalResult`."""
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                document_id="doc-chunk",
                chunk_id=i,
                content=f"content-{i}",
                score=1.0 - n * 0.1,
                source_uri="u",
            )
            for n, i in enumerate(ids)
        ],
        retriever="spine",
        acl_applied=True,
    )


def _answer(text: str) -> GeneratedAnswer:
    """Build a :class:`GeneratedAnswer`."""
    return GeneratedAnswer(answer=text, grounded=True, citations=[], used_chunk_ids=[])


def _bm_with_refs() -> BenchmarkMetrics:
    """A full T1+T2+T3 :class:`BenchmarkMetrics`."""
    return BenchmarkMetrics(
        recall_at_5=0.8,
        recall_at_10=0.9,
        ndcg_at_5=0.7,
        ndcg_at_10=0.75,
        mrr=0.65,
        faithfulness_mean=0.92,
        faithfulness_kappa=None,
        context_precision=0.85,
        context_recall=0.6,
        factual_correctness=0.5,
        noise_sensitivity=0.4,
        n_queries=3,
        n_judged=3,
        errored=0,
        metrics={
            "ndcg_at_5": [0.7] * 3,
            "faithfulness": [0.92] * 3,
            "context_recall": [0.6] * 3,
        },
    )


def _bm_no_refs() -> BenchmarkMetrics:
    """A no-references :class:`BenchmarkMetrics` (T3 omitted, AC4)."""
    return BenchmarkMetrics(
        recall_at_5=0.5,
        recall_at_10=0.7,
        ndcg_at_5=0.4,
        ndcg_at_10=0.6,
        mrr=0.45,
        faithfulness_mean=0.7,
        faithfulness_kappa=None,
        context_precision=0.6,
        context_recall=None,
        factual_correctness=None,
        noise_sensitivity=None,
        n_queries=3,
        n_judged=3,
        errored=0,
        metrics={
            "ndcg_at_5": [0.4] * 3,
            "faithfulness": [0.7] * 3,
            "context_recall": [None] * 3,
        },
    )


def _model_config() -> dict[str, Any]:
    """A canned ``model_config`` payload."""
    return {
        "generator": "qwen3-14b",
        "embedder": "bge-m3",
        "reranker": "bge-reranker-v2-m3",
        "grader": "judge_faithfulness",
        "dataset_id": "benchmark_golden",
        "params": {"temperature": 0.1, "top_k": 8},
    }


def _vllm_metrics() -> VllmServerMetrics:
    """A canned :class:`VllmServerMetrics`."""
    return VllmServerMetrics(
        ttft_p99=0.045,
        tpot_p99=0.020,
        e2e_p99=1.2,
        queue_time_p99=0.005,
        kv_cache_peak_pct=0.72,
        prefix_cache_hit_rate=0.6,
        prompt_tokens=1024,
        generation_tokens=256,
        prompt_tokens_delta=1024,
        generation_tokens_delta=256,
        prefix_cache_queries_delta=5,
        prefix_cache_hits_delta=3,
        raw={},
    )


def _model_load(ms: float = 1500.0) -> ModelLoadTiming:
    """A canned :class:`ModelLoadTiming`."""
    return ModelLoadTiming(elapsed_ms=ms, result="ok", error=None)


def _make_runner(
    *,
    swap_timeout: bool = False,
    queries: list[GoldenQuery] | None = None,
    bm: BenchmarkMetrics | None = None,
    script: list[tuple[RetrievalResult, GeneratedAnswer, list[str]]] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a :class:`BenchmarkRunner` with the canned collaborators.

    Args:
        swap_timeout: When ``True``, configure the swap controller to
            raise :class:`SwapTimeout` on ``wait_for_ready``.
        queries: The golden queries (defaults to with-references).
        bm: The metrics engine result (defaults to with-references).
        script: The per-query script (defaults to a 3-query replay).

    Returns:
        ``(runner, fakes)`` — the runner plus a dict of the fake
        collaborators (for assertion helpers).
    """
    from iw_rag.eval.benchmark.runner import BenchmarkRunner

    qs = queries if queries is not None else _golden_with_refs(3)
    metrics = bm if bm is not None else _bm_with_refs()
    per_query_script = (
        script
        if script is not None
        else [
            (
                _retrieval([f"c{i}"]),
                _answer(f"answer-{i}"),
                [f"content-{i}"],
            )
            for i in range(3)
        ]
    )

    swap = FakeSwapControllerClient(swap_timeout=swap_timeout)
    engine = FakeMetricsEngine([metrics])
    repo = FakeReportRepository()
    gpu = FakeGpuSampler()
    vllm = FakeVllmScraper(
        before=_vllm_metrics(),
        after=_vllm_metrics(),
    )
    golden = FakeGoldenSetLoader(qs)
    qr = FakeQueryRunner(per_query_script)

    runner = BenchmarkRunner(
        swap_controller=cast_any(swap),
        metrics_engine=cast_any(engine),
        repository=cast_any(repo),
        gpu_sampler=cast_any(gpu),
        vllm_scraper=cast_any(vllm),
        golden_loader=cast_any(golden),
        query_runner=cast_any(qr),
        health_timeout_s=300.0,
        per_query_timeout_s=120.0,
    )
    fakes = {
        "swap": swap,
        "engine": engine,
        "repo": repo,
        "gpu": gpu,
        "vllm": vllm,
        "golden": golden,
        "query_runner": qr,
    }
    return runner, fakes


def cast_any(obj: object) -> Any:
    """Cast an object to ``Any`` for the structural-Protocol injection."""
    return obj  # type: ignore[return-value]


async def _drain(runner: Any, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run the runner to completion and collect all events."""
    return [ev async for ev in runner.run(model_config, trace_id="trace-1")]


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_runner_module_exposes_benchmark_runner() -> None:
    """The runner module MUST expose a :class:`BenchmarkRunner` class."""
    from iw_rag.eval.benchmark import runner as rmod

    assert hasattr(rmod, "BenchmarkRunner"), "BenchmarkRunner is the public S01 surface"
    assert callable(rmod.BenchmarkRunner)


def test_build_report_label_uses_human_dataset_name_never_uuid() -> None:
    """The report label appends the human ``dataset_name`` for a non-default
    dataset and NEVER the opaque resolved ``dataset_id`` uuid."""
    from iw_rag.eval.benchmark.runner import BenchmarkRunner

    uuid = "f6c10fa1-c229-4a18-a91d-5966b61bfebc"
    # Default golden dataset → bare generator (no suffix).
    assert (
        BenchmarkRunner._build_report_label(
            {"generator": "qwen3-8b-awq", "dataset_id": uuid, "dataset_name": "benchmark_golden"}
        )
        == "qwen3-8b-awq"
    )
    # Non-default named dataset → appended by its HUMAN name.
    assert (
        BenchmarkRunner._build_report_label(
            {"generator": "qwen3-14b", "dataset_id": uuid, "dataset_name": "my-set"}
        )
        == "qwen3-14b@my-set"
    )
    # No human name available (uuid-only submission) → never leak the uuid.
    assert (
        BenchmarkRunner._build_report_label({"generator": "qwen3-14b", "dataset_id": uuid})
        == "qwen3-14b"
    )


def test_runner_module_exposes_protocols() -> None:
    """The runner module MUST expose the four collaborator Protocols.

    F-00093/F-00094 are parallel work — the runner is testable with
    fakes today and binds to the real implementations at the
    composition root in S02.
    """
    from iw_rag.eval.benchmark import runner as rmod

    for name in (
        "SwapControllerProtocol",
        "MetricsEngineProtocol",
        "ReportRepositoryProtocol",
        "GpuSamplerProtocol",
        "VllmScraperProtocol",
        "GoldenSetLoaderProtocol",
        "QueryRunnerProtocol",
    ):
        assert hasattr(rmod, name), f"runner.{name} is part of the public Protocol surface"


# ---------------------------------------------------------------------------
# AC1: happy-path event order + persisted model_config
# ---------------------------------------------------------------------------


async def test_happy_path_event_order_and_done_report_id() -> None:
    """AC1: ``queued → swapping → ready → running×N → scoring → done``
    in order, ``done`` carries a ``report_id``, and the persisted
    report's ``model_config`` matches the submission."""
    runner, fakes = _make_runner()

    events = await _drain(runner, _model_config())

    # Event-type sequence (strip the data payload)
    types = [e["event"] for e in events]
    expected_prefix = ["queued", "swapping", "ready"]
    assert types[: len(expected_prefix)] == expected_prefix
    # 3 running events, one per query
    running_events = [e for e in events if e["event"] == "running"]
    assert len(running_events) == 3
    # running events carry i + total
    for i, ev in enumerate(running_events):
        assert ev["data"]["i"] == i
        assert ev["data"]["total"] == 3
    # scoring then done
    assert types[-2] == "scoring"
    assert types[-1] == "done"
    # done carries report_id — FakeReportRepository assigns "row-1" for the first row
    done = events[-1]
    assert done["data"].get("report_id") == "row-1"
    # exactly one report persisted
    assert len(fakes["repo"].create_calls) == 1
    # the persisted model_config equals the submission (Invariant 5)
    persisted = fakes["repo"].create_calls[0]
    assert persisted["model_config"] == _model_config()
    # created_by is None (the runner has no principal id injection in this path)
    assert persisted["created_by"] is None


# ---------------------------------------------------------------------------
# Boundary row 1: swap-timeout → error + zero reports
# ---------------------------------------------------------------------------


async def test_swap_timeout_emits_terminal_error_and_persists_zero_reports() -> None:
    """Boundary row 1: readiness never arrives → terminal ``error``
    ``{reason_code:\"swap_timeout\", message, trace_ref}`` + ZERO rows."""
    runner, fakes = _make_runner(swap_timeout=True)

    events = await _drain(runner, _model_config())

    types = [e["event"] for e in events]
    # The sequence must be: queued → swapping → ready is NEVER emitted → error
    assert types[0] == "queued"
    assert types[1] == "swapping"
    # The terminal event is an error
    assert types[-1] == "error"
    err = events[-1]["data"]
    assert err.get("reason_code") == "swap_timeout"
    assert "message" in err
    assert err.get("trace_ref") == "trace-1"
    # No "ready", no "running", no "scoring", no "done"
    assert "ready" not in types
    assert "running" not in types
    assert "scoring" not in types
    assert "done" not in types
    # No report persisted
    assert fakes["repo"].create_calls == []
    # Per-query runner was never called
    assert fakes["query_runner"].calls == []
    # Metrics engine was never called
    assert fakes["engine"].calls == []


# ---------------------------------------------------------------------------
# Boundary row 2: dataset without reference answers → Tier-3 omitted, still persists
# ---------------------------------------------------------------------------


async def test_dataset_without_references_omits_tier3_and_persists() -> None:
    """Boundary row 2: when the golden set has no reference answers,
    the metrics engine returns ``None`` for context_recall /
    factual_correctness and the runner still persists ONE report."""
    qs = _golden_no_refs(3)
    bm = _bm_no_refs()
    runner, fakes = _make_runner(queries=qs, bm=bm)

    events = await _drain(runner, _model_config())

    types = [e["event"] for e in events]
    # The full sequence still happens — Tier-3 omission does not abort
    assert types[0] == "queued"
    assert types[-1] == "done"
    # exactly one report persisted
    assert len(fakes["repo"].create_calls) == 1
    persisted = fakes["repo"].create_calls[0]
    # Tier-3 (context_recall, factual_correctness) are None in the
    # row payload (Boundary row 2 + Invariant 5 — never 0.0, never
    # silently dropped).
    assert persisted["context_recall"] is None
    assert persisted["factual_correctness"] is None
    # T1 + T2 still populated
    assert persisted["recall_at_5"] == 0.5
    assert persisted["faithfulness_mean"] == 0.7


# ---------------------------------------------------------------------------
# Invariant 4: model_load_ms is its own bucket, never folded in
# ---------------------------------------------------------------------------


async def test_model_load_ms_is_separate_from_per_query_latency() -> None:
    """Invariant 4: ``model_load_ms`` is its own metric and is NEVER
    folded into per-query or e2e latency.  The row's ``latency``
    JSONB MUST contain ``model_load_ms`` AS A SEPARATE KEY, and the
    per-stage dict MUST NOT include it."""
    runner, fakes = _make_runner()

    await _drain(runner, _model_config())

    # Persisted latency blob has model_load_ms
    persisted = fakes["repo"].create_calls[0]
    latency = persisted["latency"]
    assert "model_load_ms" in latency
    # Per-stage ms dict does NOT carry model_load_ms — the
    # per_query timing helper writes only per-stage / per-query ms.
    per_stage = latency["per_stage_ms"]
    assert "model_load_ms" not in per_stage
    # The model_load_ms key is positive and lives at the top level
    assert latency["model_load_ms"] >= 0.0
    # The GPU sampler is started once (before the swap, to catch the VRAM
    # trough), the measurement window is marked once (after ready), and the
    # sampler is stopped once.
    assert len(fakes["gpu"].start_calls) == 1
    assert len(fakes["gpu"].mark_calls) == 1
    assert len(fakes["gpu"].stop_calls) == 1


async def test_model_load_ms_includes_activation_time() -> None:
    """``model_load_ms`` MUST bracket the full activation→ready window.

    Regression: on a cold swap the model load happens INSIDE
    ``swap.activate`` (the proxy spawns/loads the generator); timing only
    the trailing ``wait_for_ready`` poll undercounts ``model_load_ms`` to
    ~0.  With a slow ``activate`` the recorded ``model_load_ms`` must be
    at least the activation duration.
    """
    runner, fakes = _make_runner()
    swap = fakes["swap"]
    original_activate = swap.activate

    async def slow_activate(model_config: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.05)  # 50 ms — the bulk of a cold-swap load
        return await original_activate(model_config)

    swap.activate = slow_activate  # type: ignore[method-assign]

    await _drain(runner, _model_config())

    latency = fakes["repo"].create_calls[0]["latency"]
    # Pre-fix this was ~0 (activate sat outside the timed window); now it
    # includes the activation time.
    assert latency["model_load_ms"] >= 50.0


async def test_gpu_sampler_spans_the_swap_then_marks_run_start() -> None:
    """The GPU sampler MUST start (and snapshot the baseline) BEFORE the swap,
    and mark the measurement window AFTER activation.

    Regression: with the sampler started only after ``ready`` the cold-swap
    VRAM trough (the idle floor) was never sampled, so ``vram_baseline_mb``
    equalled the peak and ``vram_incremental_mb`` collapsed to 0 (vLLM
    pre-allocates its KV-cache pool at load).  Sampling across the swap and
    marking the run start afterwards makes the incremental the resident
    model's footprint while keeping util/run-peak undiluted by the load.
    """
    runner, fakes = _make_runner()
    swap, gpu = fakes["swap"], fakes["gpu"]
    order: list[str] = []

    original_activate = swap.activate
    original_snapshot = gpu.snapshot_baseline
    original_start = gpu.start
    original_mark = gpu.mark_run_start

    async def recording_activate(model_config: dict[str, Any]) -> dict[str, Any]:
        order.append("activate")
        return await original_activate(model_config)

    def recording_snapshot() -> Any:
        order.append("snapshot_baseline")
        return original_snapshot()

    def recording_start() -> None:
        order.append("start")
        original_start()

    def recording_mark() -> None:
        order.append("mark_run_start")
        original_mark()

    swap.activate = recording_activate  # type: ignore[method-assign]
    gpu.snapshot_baseline = recording_snapshot  # type: ignore[method-assign]
    gpu.start = recording_start  # type: ignore[method-assign]
    gpu.mark_run_start = recording_mark  # type: ignore[method-assign]

    await _drain(runner, _model_config())

    # Baseline snapshot + sampler start both precede activation; the
    # measurement-window mark follows it (post-ready).
    assert order.index("snapshot_baseline") < order.index("activate")
    assert order.index("start") < order.index("activate")
    assert order.index("activate") < order.index("mark_run_start"), (
        f"sampler must span the swap and mark the run after activation; order was {order}"
    )


# ---------------------------------------------------------------------------
# Generic mid-run failure → terminal error, no orphan row
# ---------------------------------------------------------------------------


async def test_mid_run_query_failure_emits_error_and_persists_zero() -> None:
    """Any mid-run failure (e.g. the query runner raising) must end
    with a terminal ``error`` ``{message, trace_ref}`` and persist
    ZERO reports (Invariant 2 — exactly one row per successful run)."""

    class _ExplodingQueryRunner:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def run_query(
            self,
            question: str,
            ctx: Any,
            i: int,
            total: int,
        ) -> tuple[RetrievalResult, GeneratedAnswer, list[str]]:
            self.calls.append({"i": i})
            raise RuntimeError("simulated mid-run explosion")

    from iw_rag.eval.benchmark.runner import BenchmarkRunner

    qs = _golden_with_refs(3)
    swap = FakeSwapControllerClient()
    engine = FakeMetricsEngine([_bm_with_refs()])
    repo = FakeReportRepository()
    gpu = FakeGpuSampler()
    vllm = FakeVllmScraper(before=_vllm_metrics(), after=_vllm_metrics())
    golden = FakeGoldenSetLoader(qs)
    qr = _ExplodingQueryRunner()

    runner = BenchmarkRunner(
        swap_controller=cast_any(swap),
        metrics_engine=cast_any(engine),
        repository=cast_any(repo),
        gpu_sampler=cast_any(gpu),
        vllm_scraper=cast_any(vllm),
        golden_loader=cast_any(golden),
        query_runner=cast_any(qr),
        health_timeout_s=300.0,
        per_query_timeout_s=120.0,
    )

    events = await _drain(runner, _model_config())
    types = [e["event"] for e in events]
    assert types[-1] == "error"
    err = events[-1]["data"]
    # Generic message only — never the exception text
    assert "simulated mid-run explosion" not in err.get("message", "")
    assert err.get("trace_ref") == "trace-1"
    # No report persisted (Invariant 2)
    assert repo.create_calls == []


# ---------------------------------------------------------------------------
# Local-only: no cloud SDK imports
# ---------------------------------------------------------------------------


def test_runner_does_not_import_cloud_sdks() -> None:
    """Invariant 1: the runner module MUST NOT import a cloud SDK.

    The hard rule (F-00095 design Invariant 1) is ``git grep``-able;
    this test pins it at the import level so a future careless
    addition is caught at unit-test time.
    """
    import iw_rag.eval.benchmark.runner as rmod

    src = (rmod.__file__) or ""
    # Read the source and assert no obvious cloud SDK names appear
    # in the imports / references.
    forbidden = ("openai", "anthropic", "cohere", "boto3", "google.cloud", "azure")
    assert src.endswith(".py")
    with open(src) as fh:
        body = fh.read()
    for token in forbidden:
        assert token not in body, (
            f"runner.py references a cloud SDK '{token}' — Invariant 1 (no cloud) is broken"
        )


# ---------------------------------------------------------------------------
# Persistence contract: repository is called exactly once on success
# ---------------------------------------------------------------------------


async def test_runner_persists_exactly_one_row_on_success() -> None:
    """Invariant 2: on a successful run, the repository's
    ``create_report`` is called EXACTLY ONCE."""
    runner, fakes = _make_runner()

    await _drain(runner, _model_config())

    assert len(fakes["repo"].create_calls) == 1


async def test_runner_event_data_is_sse_shape() -> None:
    """Every emitted event MUST be a dict with ``event`` (str) and
    ``data`` (dict) — the same shape the API layer in S02 wraps into
    SSE bytes."""
    runner, _ = _make_runner()

    events = await _drain(runner, _model_config())

    assert events, "the runner must emit at least one event"
    for ev in events:
        assert isinstance(ev, dict)
        assert isinstance(ev.get("event"), str)
        assert isinstance(ev.get("data"), dict)
    # Pin the exact event names present in a successful run.
    event_names = {ev["event"] for ev in events}
    known_events = {"queued", "swapping", "ready", "running", "scoring", "done"}
    assert event_names == known_events, (
        f"Expected exactly these event names in a successful run: {known_events!r}; "
        f"got: {event_names!r}"
    )


async def test_runner_records_swap_activate_and_wait_in_order() -> None:
    """The runner MUST call ``swap.activate`` THEN
    ``swap.wait_for_ready`` — never the other way round."""
    runner, fakes = _make_runner()

    await _drain(runner, _model_config())

    assert len(fakes["swap"].activate_calls) == 1
    assert len(fakes["swap"].wait_calls) == 1
    # The activate call payload mirrors the submitted model_config
    assert fakes["swap"].activate_calls[0]["model_config"] == _model_config()
    # The wait call uses the health-gate timeout
    assert fakes["swap"].wait_calls[0]["timeout_s"] == 300.0


async def test_runner_persists_per_item_samples_in_metrics_blob() -> None:
    """The persisted ``metrics`` blob carries a ``samples`` record per query.

    The multi-run Compare view's expandable qualitative diff reads
    ``metrics.samples[i] = {question, answer, contexts}`` paired by query
    index.  The runner must persist it from the golden questions + the
    per-query generated answers + retrieved contexts."""
    script = [
        (_retrieval([f"c{i}"]), _answer(f"answer-{i}"), [f"ctx-{i}-a", f"ctx-{i}-b"])
        for i in range(3)
    ]
    runner, fakes = _make_runner(script=script)

    await _drain(runner, _model_config())

    calls = fakes["repo"].create_calls
    assert len(calls) == 1
    samples = calls[0]["metrics_row"].get("samples")
    assert isinstance(samples, list)
    assert len(samples) == 3
    # Paired by index: question text from the golden set, answer + contexts
    # from the per-query pipeline.
    assert samples[0]["question"] == "question-0"
    assert samples[0]["answer"] == "answer-0"
    assert samples[0]["contexts"] == ["ctx-0-a", "ctx-0-b"]
    assert samples[2]["answer"] == "answer-2"


def test_build_item_samples_truncates_contexts_and_answer() -> None:
    """``_build_item_samples`` caps context count + per-string length + answer length."""
    from iw_rag.eval.benchmark.runner import BenchmarkRunner

    queries = _golden_with_refs(1)
    answers = ["A" * 9000]
    contexts = [[f"ctx-{i}-" + ("x" * 5000) for i in range(20)]]
    samples = BenchmarkRunner._build_item_samples(queries, answers, contexts)
    assert len(samples) == 1
    rec = samples[0]
    # Answer truncated to the 4000-char cap.
    assert len(rec["answer"]) == 4000
    # At most 8 contexts kept, each truncated to the 1200-char cap.
    assert len(rec["contexts"]) == 8
    assert all(len(c) <= 1200 for c in rec["contexts"])
    assert rec["question"] == "question-0"


async def test_runner_calls_metrics_engine_with_per_query_records() -> None:
    """The metrics engine MUST be called with the per-query records
    collected during ``running`` (n queries in, n queries out)."""
    runner, fakes = _make_runner()

    await _drain(runner, _model_config())

    assert len(fakes["engine"].calls) == 1
    call = fakes["engine"].calls[0]
    assert len(call["queries"]) == 3
    assert len(call["retrievals"]) == 3
    assert len(call["answers"]) == 3
    assert len(call["contexts"]) == 3
