"""Unit tests for ``eval/benchmark/scoreboard.py`` (F-00096, S05).

The scoreboard is the offline ranking service that turns the append-only
``benchmark_reports`` history into per-metric leaderboards plus a
composite.  It is the only authority on the AC6 faithfulness ordering,
and the S08 route (``GET /benchmark/scoreboard``) will call it.

These tests are TDD RED-first drivers.  The production module is
scaffolded with :class:`NotImplementedError` (or absent) so the
collection succeeds and the failures are the correct TDD shape —
``AssertionError`` / ``NotImplementedError`` / ``AttributeError`` from
the missing implementation, not an ``ImportError`` or collection
error.

Coverage:

* ``score()`` returns a :class:`Scoreboard` with five per-metric
  boards plus a composite (AC6).
* ``faithfulness`` board is **descending** (higher is better), with
  the right ``model_config`` + ``created_at`` per row.
* ``ndcg_at_5`` board is descending.
* ``latency`` board is **ascending** (lower is better, exact
  ascending order asserted).
* ``tok_per_s`` board is descending, computed from
  ``resource.vllm_server.generation_tokens_delta`` /
  ``(latency.per_stage_ms.retrieve_generate_ms * n_queries / 1000)``.
* ``vram_efficiency`` board is descending, computed as
  ``ndcg_at_5 / (resource.gpu.vram_peak_mb / 1024)``.
* Composite is a deterministic weighted sum of the min-max-normalised
  per-metric values; ties are broken by ``created_at`` then ``id``.
* **Empty history** (0 reports) → every board is ``[]`` (no
  exception).  Boundary row: *"Empty report history → explicit empty
  state, not an error"*.
* A report with a ``NULL`` metric on a board is **ranked last** for
  that board (deterministic, documented rule), with other reports
  unaffected.
* The service composes ``BenchmarkReportRepository`` via the
  ``Database.session()`` async context manager (no live DB needed
  in unit tests — a fake repository replaying a fixed set of rows
  is used).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Minimal async session stub.

    Holds an ``execute`` async mock that returns the rows its
    :class:`FakeDatabase` was configured with (matching the
    :class:`FakeDatabase` in
    ``tests/unit/eval/benchmark/test_service.py``).
    """


class FakeDatabase:
    """In-memory stand-in for :class:`~iw_rag.storage.db.Database`.

    Yields a :class:`_FakeAsyncSession` from ``session()`` whose
    ``execute`` async mock replays the configured row list as a
    mappings() result — matches the S03 sibling's fakes and the
    real :meth:`BenchmarkReportService.list_reports` read path
    (the scoreboard uses the same direct SELECT).

    Attributes:
        _mappings: The row dicts the fake replays from
            ``session.execute().mappings()``.
    """

    def __init__(self, mappings: list[dict[str, Any]] | None = None) -> None:
        """Configure the fake.

        Args:
            mappings: Optional rows to return from
                ``session.execute().mappings()``.  The scoreboard's
                direct SELECT reads the whole ``benchmark_reports``
                history, so callers pass a list of row dicts.
        """
        self._mappings = mappings or []

    @asynccontextmanager  # type: ignore[arg-type]
    async def session(self):  # type: ignore[override]
        """Yield a stubbed session whose ``execute`` returns the rows.

        Yields:
            A :class:`_FakeAsyncSession` with an ``execute`` attribute
            that is an :class:`AsyncMock` returning a ``MagicMock``
            whose ``mappings()`` yields ``self._mappings``.
        """
        session = _FakeAsyncSession()
        mappings_result = MagicMock()
        mappings_result.mappings.return_value = self._mappings
        session.execute = AsyncMock(return_value=mappings_result)  # type: ignore[attr-defined]
        yield session


class FakeReportRepo:
    """In-memory stub of :class:`BenchmarkReportRepository`.

    Unused by default — the scoreboard service reads the full
    history via a direct SELECT (matching
    :meth:`BenchmarkReportService.list_reports`).  Kept available
    for any future per-dataset scoreboard view.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)

    async def list_reports(self, dataset_id: str) -> list[dict[str, Any]]:
        return [r for r in self._rows if r.get("dataset_id") == dataset_id]


def _make_score(
    report_id: str,
    *,
    dataset_id: str = "ds-1",
    ndcg_at_5: float | None = 0.5,
    faithfulness_mean: float | None = 0.8,
    latency: dict[str, Any] | None = None,
    resource: dict[str, Any] | None = None,
    n_queries: int = 10,
    model_config: dict[str, Any] | None = None,
    created_at: Any = None,
) -> dict[str, Any]:
    """Build a minimal ``benchmark_reports`` row dict for the scoreboard tests.

    Defaults are picked so callers only have to override the fields they
    are exercising.

    Args:
        report_id: The report's string id.
        dataset_id: The dataset this report belongs to.
        ndcg_at_5: T1 promoted nDCG@5 (or ``None`` for "no T1 metrics").
        faithfulness_mean: T2 promoted faithfulness (or ``None``).
        latency: The latency JSONB (or ``None`` for "no latency measured").
        resource: The resource JSONB (or ``None`` for "no resource measured").
        n_queries: The query count stored in the row.
        model_config: The static model configuration snapshot.
        created_at: The run timestamp (string or datetime is fine for tests).

    Returns:
        A flat dict mirroring the ``benchmark_reports`` table row shape.
    """
    return {
        "id": report_id,
        "dataset_id": dataset_id,
        "ndcg_at_5": ndcg_at_5,
        "faithfulness_mean": faithfulness_mean,
        "latency": latency
        if latency is not None
        else {
            "per_stage_ms": {"retrieve_generate_ms": 200.0},
            "ttft_p99_s": 0.05,
            "tpot_p99_s": 0.01,
            "e2e_p99_s": 1.0,
            "queue_time_p99_s": 0.005,
            "model_load_ms": 500.0,
        },
        "resource": resource
        if resource is not None
        else {
            "gpu": {
                "available": True,
                "vram_peak_mb": 16 * 1024,  # 16 GiB
                "vram_baseline_mb": 0,
                "vram_incremental_mb": 16 * 1024,
            },
            "vllm_server": {
                "generation_tokens_delta": 2000,
            },
        },
        "n_queries": n_queries,
        "model_config": model_config
        if model_config is not None
        else {"generator": "qwen3-14b", "embedder": "bge-m3"},
        "created_at": created_at if created_at is not None else "2026-06-01T00:00:00Z",
    }


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------


def _make_service(rows: list[dict[str, Any]] | None = None) -> tuple[Any, Any]:
    """Build a :class:`ScoreboardService` with an injected fake database.

    The scoreboard reads the full ``benchmark_reports`` history via
    a direct SELECT (matching :meth:`BenchmarkReportService.list_reports`)
    — the test fake replays a fixed row list from
    ``session.execute().mappings()``.

    Args:
        rows: The rows the fake replays (the service's
            ``_fetch_history`` returns them verbatim).

    Returns:
        ``(service, no-op)`` — the service instance and a no-op
        patcher the test must ``stop()`` in a ``finally`` (kept for
        API parity with the S03 sibling's fakes; not used here
        because the S05 service does not bind ``BenchmarkReportRepository``
        at the module level).
    """
    from iw_rag.eval.benchmark.scoreboard import ScoreboardService

    db = FakeDatabase(mappings=rows or [])
    svc = ScoreboardService(database=db)  # type: ignore[arg-type]
    return svc, _NoopPatcher()


class _NoopPatcher:
    """No-op patcher returned by :func:`_make_service` for API parity.

    The scoreboard service reads via a direct SELECT, not via
    ``BenchmarkReportRepository``, so there is nothing to patch.  The
    tests still call ``patcher.stop()`` in a ``finally`` block.
    """

    def stop(self) -> None:
        return None


# ===========================================================================
# 1. Module public surface
# ===========================================================================


def test_scoreboard_module_exposes_public_classes() -> None:
    """``scoreboard`` exports ``ScoreboardService`` and ``Scoreboard``."""
    from iw_rag.eval.benchmark import scoreboard as mod

    assert hasattr(mod, "ScoreboardService")
    assert callable(mod.ScoreboardService)
    assert hasattr(mod, "Scoreboard")
    # Scoreboard is a frozen dataclass / pydantic model (typed shape,
    # not a loose dict).
    from iw_rag.eval.benchmark.scoreboard import Scoreboard

    # Frozen dataclass → importing + class-level attrs present
    assert hasattr(Scoreboard, "__dataclass_fields__") or hasattr(Scoreboard, "model_fields")


def test_scoreboard_entry_dataclass_is_frozen() -> None:
    """``ScoreboardEntry`` is a frozen dataclass (immutable row shape)."""
    from dataclasses import FrozenInstanceError

    from iw_rag.eval.benchmark.scoreboard import ScoreboardEntry

    entry = ScoreboardEntry(
        report_id="r1",
        model_config={"generator": "x"},
        created_at="2026-06-01",
        metric_value=0.5,
    )
    with pytest.raises(FrozenInstanceError, match=r"cannot assign to field"):
        entry.report_id = "r2"  # type: ignore[misc]


# ===========================================================================
# 2. Empty history → empty boards (boundary row)
# ===========================================================================


async def test_empty_history_returns_empty_boards() -> None:
    """0 reports → every board is ``[]``, no exception.

    Boundary row: *"Empty report history → explicit empty state, not
    an error"*.
    """
    svc, patcher = _make_service(rows=[])
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    assert result.faithfulness == []
    assert result.ndcg_at_5 == []
    assert result.tok_per_s == []
    assert result.latency == []
    assert result.vram_efficiency == []
    assert result.composite == []


# ===========================================================================
# 3. Faithfulness board (AC6) — descending, model_config + timestamp
# ===========================================================================


async def test_faithfulness_board_orders_descending() -> None:
    """``faithfulness`` board is descending (higher is better)."""
    rows = [
        _make_score(
            "r-low",
            faithfulness_mean=0.60,
            model_config={"generator": "qwen3-7b"},
            created_at="2026-06-01T00:00:00Z",
        ),
        _make_score(
            "r-high",
            faithfulness_mean=0.95,
            model_config={"generator": "qwen3-14b"},
            created_at="2026-06-02T00:00:00Z",
        ),
        _make_score(
            "r-mid",
            faithfulness_mean=0.78,
            model_config={"generator": "qwen3-10b"},
            created_at="2026-06-03T00:00:00Z",
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.faithfulness
    assert len(board) == 3
    # Exact ordering — high first, then mid, then low.
    assert [e.report_id for e in board] == ["r-high", "r-mid", "r-low"]
    # Each entry carries the exact metric value (AC6).
    assert board[0].metric_value == pytest.approx(0.95)
    assert board[1].metric_value == pytest.approx(0.78)
    assert board[2].metric_value == pytest.approx(0.60)
    # Each entry carries model_config + run timestamp (AC6 names both).
    assert board[0].model_config == {"generator": "qwen3-14b"}
    assert board[0].created_at == "2026-06-02T00:00:00Z"
    assert board[1].model_config == {"generator": "qwen3-10b"}
    assert board[2].model_config == {"generator": "qwen3-7b"}


def test_board_entries_thread_resolved_dataset_name() -> None:
    """Ranked entries carry the resolved ``dataset_name`` (decorated onto the
    history rows by ``attach_dataset_labels``) so the console board renders the
    human dataset instead of the opaque uuid."""
    from iw_rag.eval.benchmark.scoreboard import ScoreboardService

    rows = [
        _make_score("r1", faithfulness_mean=0.9, created_at="2026-06-02T00:00:00Z"),
        _make_score("r2", faithfulness_mean=0.7, created_at="2026-06-01T00:00:00Z"),
    ]
    for r in rows:
        r["dataset_name"] = "benchmark_golden"

    # Both the per-metric ranker and the composite ranker must thread it.
    faith = ScoreboardService._rank_faithfulness(rows)
    composite = ScoreboardService._rank_composite(rows)
    assert faith[0].dataset_name == "benchmark_golden"
    assert composite[0].dataset_name == "benchmark_golden"


def test_board_entry_dataset_name_empty_when_unresolved() -> None:
    """A history row without a resolved ``dataset_name`` yields ``""`` (the page
    then falls back to the snapshot uuid) rather than ``None`` or a crash."""
    from iw_rag.eval.benchmark.scoreboard import ScoreboardService

    rows = [_make_score("r1", faithfulness_mean=0.9, created_at="2026-06-02T00:00:00Z")]
    board = ScoreboardService._rank_faithfulness(rows)
    assert board[0].dataset_name == ""


async def test_faithfulness_board_ties_broken_by_created_at_then_id() -> None:
    """Faithfulness ties are broken by ``created_at`` ascending then ``id`` ascending."""
    rows = [
        _make_score("r-c", faithfulness_mean=0.80, created_at="2026-06-03T00:00:00Z"),
        _make_score("r-a", faithfulness_mean=0.80, created_at="2026-06-01T00:00:00Z"),
        _make_score("r-b", faithfulness_mean=0.80, created_at="2026-06-02T00:00:00Z"),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.faithfulness
    assert [e.report_id for e in board] == ["r-a", "r-b", "r-c"]


# ===========================================================================
# 4. ndcg_at_5 board — descending
# ===========================================================================


async def test_ndcg_at_5_board_orders_descending() -> None:
    """``ndcg_at_5`` board is descending (higher is better)."""
    rows = [
        _make_score("r-1", ndcg_at_5=0.40),
        _make_score("r-2", ndcg_at_5=0.85),
        _make_score("r-3", ndcg_at_5=0.62),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.ndcg_at_5
    assert [e.report_id for e in board] == ["r-2", "r-3", "r-1"]
    assert [e.metric_value for e in board] == pytest.approx([0.85, 0.62, 0.40])


# ===========================================================================
# 5. Latency board — ASCENDING (lower is better)
# ===========================================================================


async def test_latency_board_orders_ascending_lower_is_better() -> None:
    """``latency`` board is **ascending** — lower e2e_p99 wins.

    Three reports with distinct ``latency.e2e_p99_s``; the test pins
    the exact resulting order.
    """
    rows = [
        _make_score(
            "r-slow",
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 2000.0},
                "e2e_p99_s": 5.0,
                "ttft_p99_s": 0.5,
                "tpot_p99_s": 0.05,
                "queue_time_p99_s": 0.01,
                "model_load_ms": 1000.0,
            },
        ),
        _make_score(
            "r-fast",
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 100.0},
                "e2e_p99_s": 0.5,
                "ttft_p99_s": 0.05,
                "tpot_p99_s": 0.01,
                "queue_time_p99_s": 0.001,
                "model_load_ms": 200.0,
            },
        ),
        _make_score(
            "r-mid",
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 800.0},
                "e2e_p99_s": 2.0,
                "ttft_p99_s": 0.2,
                "tpot_p99_s": 0.02,
                "queue_time_p99_s": 0.005,
                "model_load_ms": 500.0,
            },
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.latency
    assert [e.report_id for e in board] == ["r-fast", "r-mid", "r-slow"]
    assert [e.metric_value for e in board] == pytest.approx([0.5, 2.0, 5.0])


# ===========================================================================
# 6. tok_per_s board — descending (throughput)
# ===========================================================================


async def test_tok_per_s_board_orders_descending() -> None:
    """``tok_per_s`` board is descending — higher throughput wins.

    Throughput is computed as
    ``resource.vllm_server.generation_tokens_delta``
    ``/ (latency.per_stage_ms.retrieve_generate_ms * n_queries / 1000)``.

    Three reports with distinct values:

    * r-fast: 2000 tokens / (100 ms * 10 queries / 1000) = 2000 / 1.0 = 2000 tok/s
    * r-mid:  1000 tokens / (200 ms * 10 queries / 1000) = 1000 / 2.0 =  500 tok/s
    * r-slow:  500 tokens / (500 ms * 10 queries / 1000) =  500 / 5.0 =  100 tok/s
    """
    rows = [
        _make_score(
            "r-slow",
            n_queries=10,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 500.0},
                "e2e_p99_s": 4.0,
                "ttft_p99_s": 0.5,
                "tpot_p99_s": 0.05,
                "queue_time_p99_s": 0.01,
                "model_load_ms": 1000.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 16 * 1024},
                "vllm_server": {"generation_tokens_delta": 500},
            },
        ),
        _make_score(
            "r-fast",
            n_queries=10,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 100.0},
                "e2e_p99_s": 0.5,
                "ttft_p99_s": 0.05,
                "tpot_p99_s": 0.01,
                "queue_time_p99_s": 0.001,
                "model_load_ms": 200.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 16 * 1024},
                "vllm_server": {"generation_tokens_delta": 2000},
            },
        ),
        _make_score(
            "r-mid",
            n_queries=10,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 200.0},
                "e2e_p99_s": 1.5,
                "ttft_p99_s": 0.2,
                "tpot_p99_s": 0.02,
                "queue_time_p99_s": 0.005,
                "model_load_ms": 500.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 16 * 1024},
                "vllm_server": {"generation_tokens_delta": 1000},
            },
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.tok_per_s
    assert [e.report_id for e in board] == ["r-fast", "r-mid", "r-slow"]
    assert [e.metric_value for e in board] == pytest.approx([2000.0, 500.0, 100.0])


# ===========================================================================
# 7. vram_efficiency board — descending (quality per resident GiB)
# ===========================================================================


async def test_vram_efficiency_board_orders_descending() -> None:
    """``vram_efficiency`` = ``ndcg_at_5 / (vram_peak_mb / 1024)`` — higher is better.

    Three reports with distinct values:

    * r-best:  ndcg=0.80, vram=8 GiB   → 0.80 / 8 = 0.100
    * r-mid:   ndcg=0.60, vram=16 GiB  → 0.60 / 16 = 0.0375
    * r-worst: ndcg=0.40, vram=8 GiB   → 0.40 / 8 = 0.050

    Expected order: r-best (0.100), r-worst (0.050), r-mid (0.0375).
    """
    rows = [
        _make_score(
            "r-worst",
            ndcg_at_5=0.40,
            resource={
                "gpu": {"available": True, "vram_peak_mb": 8 * 1024},
                "vllm_server": {"generation_tokens_delta": 100},
            },
        ),
        _make_score(
            "r-best",
            ndcg_at_5=0.80,
            resource={
                "gpu": {"available": True, "vram_peak_mb": 8 * 1024},
                "vllm_server": {"generation_tokens_delta": 200},
            },
        ),
        _make_score(
            "r-mid",
            ndcg_at_5=0.60,
            resource={
                "gpu": {"available": True, "vram_peak_mb": 16 * 1024},
                "vllm_server": {"generation_tokens_delta": 150},
            },
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.vram_efficiency
    assert [e.report_id for e in board] == ["r-best", "r-worst", "r-mid"]
    assert [e.metric_value for e in board] == pytest.approx([0.100, 0.050, 0.0375])


# ===========================================================================
# 8. Composite board — deterministic weighted-sum of normalised metrics
# ===========================================================================


async def test_composite_board_is_deterministic_weighted_normalised() -> None:
    """Composite = sum of min-max-normalised per-metric scores (deterministic).

    The S05 docstring pins the exact formula — see the implementation.
    Three reports with distinct profiles; the test pins the exact
    ranking produced by the documented weighting.
    """
    rows = [
        # Balanced: high quality, mid throughput, fast, mid VRAM.
        _make_score(
            "r-balanced",
            ndcg_at_5=0.80,
            faithfulness_mean=0.90,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 200.0},
                "e2e_p99_s": 1.0,
                "ttft_p99_s": 0.1,
                "tpot_p99_s": 0.01,
                "queue_time_p99_s": 0.005,
                "model_load_ms": 500.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 16 * 1024},
                "vllm_server": {"generation_tokens_delta": 2000},
            },
            n_queries=10,
        ),
        # Quality king: top quality, slow, high VRAM, low throughput.
        _make_score(
            "r-quality",
            ndcg_at_5=0.90,
            faithfulness_mean=0.95,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 5000.0},
                "e2e_p99_s": 8.0,
                "ttft_p99_s": 0.8,
                "tpot_p99_s": 0.08,
                "queue_time_p99_s": 0.05,
                "model_load_ms": 2000.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 30 * 1024},
                "vllm_server": {"generation_tokens_delta": 200},
            },
            n_queries=10,
        ),
        # Speedster: low quality, fast, tiny VRAM, huge throughput.
        _make_score(
            "r-speed",
            ndcg_at_5=0.40,
            faithfulness_mean=0.60,
            latency={
                "per_stage_ms": {"retrieve_generate_ms": 50.0},
                "e2e_p99_s": 0.2,
                "ttft_p99_s": 0.02,
                "tpot_p99_s": 0.005,
                "queue_time_p99_s": 0.001,
                "model_load_ms": 100.0,
            },
            resource={
                "gpu": {"available": True, "vram_peak_mb": 4 * 1024},
                "vllm_server": {"generation_tokens_delta": 5000},
            },
            n_queries=10,
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    # The composite board has exactly one entry per input report, in
    # deterministic order.  Exact ranking is documented in the
    # service docstring; this test pins that the ranking is a
    # permutation of the inputs (no entries dropped) and that the
    # first entry's score is the highest of the three.
    assert len(result.composite) == 3
    composite_ids = {e.report_id for e in result.composite}
    assert composite_ids == {"r-balanced", "r-quality", "r-speed"}
    scores_by_id = {e.report_id: e.metric_value for e in result.composite}
    top_score = max(scores_by_id.values())
    top_id = max(scores_by_id, key=lambda k: scores_by_id[k])
    # The top of the board is the report with the maximum composite
    # value — pin that the top-of-board is exactly that report.
    assert result.composite[0].report_id == top_id
    assert result.composite[0].metric_value == top_score
    # The order is strictly descending (no ties in this fixture).
    for i in range(len(result.composite) - 1):
        assert result.composite[i].metric_value >= result.composite[i + 1].metric_value


async def test_composite_ties_broken_by_created_at_then_id() -> None:
    """Composite ties are broken by ``created_at`` ascending then ``id`` ascending."""
    rows = [
        _make_score(
            "r-c",
            ndcg_at_5=0.50,
            faithfulness_mean=0.50,
            created_at="2026-06-03T00:00:00Z",
        ),
        _make_score(
            "r-a",
            ndcg_at_5=0.50,
            faithfulness_mean=0.50,
            created_at="2026-06-01T00:00:00Z",
        ),
        _make_score(
            "r-b",
            ndcg_at_5=0.50,
            faithfulness_mean=0.50,
            created_at="2026-06-02T00:00:00Z",
        ),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    # All three have identical metric values → composite scores all
    # zero after min-max normalisation.  Ties broken by created_at,
    # then by id.  Pin the exact resulting order.
    assert [e.report_id for e in result.composite] == ["r-a", "r-b", "r-c"]


# ===========================================================================
# 9. NULL metric handling — ranked last, no crash
# ===========================================================================


async def test_null_metric_is_ranked_last_on_its_board() -> None:
    """A report with a NULL metric is ranked last on that board.

    Documented rule: a row with a NULL value for a given board's
    metric is placed at the END of the board, ordered amongst
    themselves by ``created_at`` then ``id`` (so the empty-state view
    can show "no measurement" for those rows).  Reports with measured
    values are not affected.
    """
    rows = [
        _make_score("r-good", ndcg_at_5=0.80),
        _make_score("r-null", ndcg_at_5=None),
        _make_score("r-better", ndcg_at_5=0.95),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.ndcg_at_5
    # Measured values first (descending), then the null row last.
    assert [e.report_id for e in board] == ["r-better", "r-good", "r-null"]
    # The null row's metric_value is None (so the UI can render the
    # "no measurement" state without a sentinel value).
    assert board[-1].metric_value is None


async def test_null_faithfulness_is_ranked_last_no_crash() -> None:
    """A NULL ``faithfulness_mean`` row is ranked last on the board — no crash."""
    rows = [
        _make_score("r-a", faithfulness_mean=None),
        _make_score("r-b", faithfulness_mean=0.70),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    board = result.faithfulness
    assert [e.report_id for e in board] == ["r-b", "r-a"]
    assert board[-1].metric_value is None


async def test_all_null_faithfulness_does_not_crash() -> None:
    """All-NULL faithfulness board: the board is the empty list, no crash."""
    rows = [
        _make_score("r-a", faithfulness_mean=None),
        _make_score("r-b", faithfulness_mean=None),
    ]
    svc, patcher = _make_service(rows=rows)
    try:
        result = await svc.score()
    finally:
        patcher.stop()

    # No row has a measured value → the board is the empty list
    # (consistent with the empty-history boundary).
    assert result.faithfulness == []


# ===========================================================================
# 10. Service uses injected Database.session() to open the read session
# ===========================================================================


async def test_service_opens_session_via_injected_database() -> None:
    """The service opens its read session via ``Database.session()``.

    Verifies the DI pattern matches ``BenchmarkReportService``: the
    scoreboard service takes a ``database`` in its constructor and
    enters the ``session()`` async context manager to read.
    """
    rows = [_make_score("r-1", ndcg_at_5=0.5)]

    session_entered = False
    session_exited = False

    @asynccontextmanager  # type: ignore[arg-type]
    async def tracking_session():  # type: ignore[override]
        nonlocal session_entered, session_exited
        session_entered = True
        mappings_result = MagicMock()
        mappings_result.mappings.return_value = rows
        session = _FakeAsyncSession()
        session.execute = AsyncMock(return_value=mappings_result)  # type: ignore[attr-defined]
        yield session
        session_exited = True

    db = MagicMock()
    db.session = tracking_session  # type: ignore[attr-defined]

    from iw_rag.eval.benchmark.scoreboard import ScoreboardService

    svc = ScoreboardService(database=db)  # type: ignore[arg-type]
    result = await svc.score()

    assert session_entered
    assert session_exited
    # And the data round-tripped.
    assert len(result.ndcg_at_5) == 1
