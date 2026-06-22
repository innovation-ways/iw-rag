"""Unit tests for eval/benchmark/service.py — BenchmarkReportService.

The service is read-only and composes the append-only
``BenchmarkReportRepository`` (F-00093 S06) with the ``stats.compare``
module.  These tests exercise the three public methods without a real
Postgres connection: a ``FakeDatabase`` is injected whose ``session()``
context manager yields a fake session, and ``BenchmarkReportRepository``
is patched out so its ``get_report`` / ``list_reports`` methods replay
scripted data.

Tests cover:
* ``compare()`` same-dataset ``ValidationError`` guard (service.py:115-120)
  with ``match=``.
* ``compare()`` vector extraction from ``metrics`` blobs — exact extracted
  vectors and the ``_COMPARE_METRICS`` filter.
* ``compare()`` returns ``None`` when either report id is unknown.
* ``list_reports()`` shaping — ordering and dict contents.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from iw_rag.domain.errors import ValidationError

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Minimal async session stub — enough for the service's usage pattern.

    The service passes the session to ``BenchmarkReportRepository``; we patch
    that class away, so the session only needs to exist (not be functional).
    """


class FakeDatabase:
    """In-memory stand-in for :class:`~iw_rag.storage.db.Database`.

    Yields a :class:`_FakeAsyncSession` from ``session()``.  The service
    never calls any SQL directly on the session object (only
    ``list_reports()`` does, but we patch ``session.execute`` there).
    """

    def __init__(
        self,
        mappings: list[dict[str, Any]] | None = None,
        datasets: list[Any] | None = None,
    ) -> None:
        """Configure the fake.

        Args:
            mappings: Optional rows to return from ``session.execute().mappings()``.
                Used by ``list_reports()``, which issues raw SELECT.
            datasets: Optional ``eval_datasets`` row-likes (``.id`` / ``.name`` /
                ``.version``) returned from ``session.execute().all()`` — used by
                ``_attach_dataset_labels`` to resolve the human dataset name.
        """
        self._mappings = mappings or []
        self._datasets = datasets or []

    @asynccontextmanager  # type: ignore[arg-type]
    async def session(self):  # type: ignore[override]
        """Yield a minimal async session with a stubbed ``execute`` method.

        Yields:
            A :class:`_FakeAsyncSession` with an ``execute`` attribute that
            replays ``self._mappings`` via ``.mappings()`` and ``self._datasets``
            via ``.all()`` (the two read shapes the service uses).
        """
        session = _FakeAsyncSession()
        # One result mock serves both read shapes: report SELECT → .mappings();
        # eval_datasets SELECT (dataset-label resolution) → .all().
        result = MagicMock()
        result.mappings.return_value = self._mappings
        result.all.return_value = self._datasets
        session.execute = AsyncMock(return_value=result)  # type: ignore[attr-defined]
        yield session


def _fake_report(
    report_id: str,
    dataset_id: str,
    *,
    metrics: dict[str, Any] | None = None,
    n_queries: int = 5,
) -> dict[str, Any]:
    """Build a minimal report row dict.

    Args:
        report_id: The report's string id.
        dataset_id: The dataset this report belongs to.
        metrics: Optional per-query metric vectors (JSONB blob).
        n_queries: The query count stored in the row.

    Returns:
        A flat dict mirroring the ``benchmark_reports`` table row shape.
    """
    return {
        "id": report_id,
        "dataset_id": dataset_id,
        "n_queries": n_queries,
        "metrics": metrics or {},
        "recall_at_5": 0.6,
        "ndcg_at_5": 0.5,
        "faithfulness_mean": 0.8,
    }


class FakeReportRepo:
    """In-memory stub of :class:`~iw_rag.eval.benchmark.repository.BenchmarkReportRepository`.

    Replays a pre-built ``{id: row_dict}`` store from ``get_report``.

    Attributes:
        _store: Mapping of report id string to row dict.
    """

    def __init__(self, store: dict[str, dict[str, Any]]) -> None:
        """Configure the in-memory store.

        Args:
            store: The id → row dict map to replay.
        """
        self._store = store

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Return the row for ``report_id``, or ``None``.

        Args:
            report_id: The report's string id.

        Returns:
            The stored row dict, or ``None`` when the id is unknown.
        """
        return self._store.get(report_id)

    async def list_reports(self, dataset_id: str) -> list[dict[str, Any]]:
        """Return all rows for ``dataset_id`` (unused in service unit tests).

        Args:
            dataset_id: The dataset id to filter by.

        Returns:
            A list of matching row dicts.
        """
        return [r for r in self._store.values() if r.get("dataset_id") == dataset_id]


class _FakeDatasetRow:
    """``eval_datasets`` row-like exposing the three columns the service reads.

    Args:
        id: The dataset uuid (matched against a report row's ``dataset_id``).
        name: The human dataset name (e.g. ``benchmark_golden``).
        version: The dataset version (e.g. ``v1``).
    """

    def __init__(self, id: str, name: str, version: str) -> None:
        self.id = id
        self.name = name
        self.version = version


def _make_service(
    repo_store: dict[str, dict[str, Any]] | None = None,
    list_rows: list[dict[str, Any]] | None = None,
    datasets: list[Any] | None = None,
) -> Any:
    """Build a :class:`BenchmarkReportService` with injected fakes.

    Patches ``BenchmarkReportRepository`` at the service module level so
    ``compare()`` / ``get_report()`` receive the scripted data without a
    real DB session.

    Args:
        repo_store: The id → row dict map used by ``get_report`` calls.
        list_rows: Rows returned by ``list_reports()`` (raw SELECT path).

    Returns:
        ``(service, patch_ctx)`` — the service instance and the active patch.
        Call ``patch_ctx.stop()`` after the test.
    """
    from iw_rag.eval.benchmark.service import BenchmarkReportService

    store = repo_store or {}
    fake_repo = FakeReportRepo(store)

    # Patch BenchmarkReportRepository inside the service module so any
    # ``BenchmarkReportRepository(session)`` call returns our fake.
    patcher = patch(
        "iw_rag.eval.benchmark.service.BenchmarkReportRepository",
        return_value=fake_repo,
    )
    patcher.start()

    db = FakeDatabase(mappings=list_rows or [], datasets=datasets or [])
    svc = BenchmarkReportService(database=db)  # type: ignore[arg-type]
    return svc, patcher


# ===========================================================================
# 1. compare() — same-dataset ValidationError guard
# ===========================================================================


async def test_compare_different_datasets_raises_validation_error() -> None:
    """``compare()`` raises ``ValidationError`` when the two reports are on different datasets."""
    store = {
        "report-a": _fake_report("report-a", dataset_id="ds-1"),
        "report-b": _fake_report("report-b", dataset_id="ds-2"),  # different!
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(
            ValidationError,
            match=r"compare across different datasets is not allowed",
        ):
            await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()


async def test_compare_different_datasets_error_message_includes_dataset_ids() -> None:
    """The ``ValidationError`` message names both mismatched dataset ids."""
    store = {
        "report-a": _fake_report("report-a", dataset_id="dataset-alpha"),
        "report-b": _fake_report("report-b", dataset_id="dataset-beta"),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        with pytest.raises(ValidationError, match=r"dataset-alpha"):
            await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()


# ===========================================================================
# 2. compare() — returns None when either report is missing
# ===========================================================================


async def test_compare_returns_none_when_report_a_missing() -> None:
    """``compare()`` returns ``None`` when report A does not exist."""
    store = {
        "report-b": _fake_report("report-b", dataset_id="ds-1"),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        result = await svc.compare("no-such-report", "report-b")
        assert result is None
    finally:
        patcher.stop()


async def test_compare_returns_none_when_report_b_missing() -> None:
    """``compare()`` returns ``None`` when report B does not exist."""
    store = {
        "report-a": _fake_report("report-a", dataset_id="ds-1"),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        result = await svc.compare("report-a", "no-such-report")
        assert result is None
    finally:
        patcher.stop()


async def test_compare_returns_none_when_both_reports_missing() -> None:
    """``compare()`` returns ``None`` when neither report id exists."""
    svc, patcher = _make_service(repo_store={})
    try:
        result = await svc.compare("ghost-a", "ghost-b")
        assert result is None
    finally:
        patcher.stop()


# ===========================================================================
# 3. compare() — vector extraction from metrics blobs
# ===========================================================================


async def test_compare_extracts_only_metrics_present_in_both_reports() -> None:
    """``compare()`` filters ``_COMPARE_METRICS`` to those present as lists in BOTH blobs.

    ``ndcg_at_5`` is in both, ``faithfulness`` only in A, ``context_recall``
    only in B, ``factual_correctness`` in neither.  Only ``ndcg_at_5`` passes
    the filter (service.py:124-128).
    """
    pytest.importorskip("scipy", reason="scipy required for stats.compare")

    store = {
        "report-a": _fake_report(
            "report-a",
            dataset_id="ds-shared",
            n_queries=3,
            metrics={
                "ndcg_at_5": [0.5, 0.6, 0.7],
                "faithfulness": [0.8, 0.9, 0.85],
                # context_recall absent
            },
        ),
        "report-b": _fake_report(
            "report-b",
            dataset_id="ds-shared",
            n_queries=3,
            metrics={
                "ndcg_at_5": [0.4, 0.65, 0.75],
                # faithfulness absent
                "context_recall": [0.5, 0.6, 0.7],
            },
        ),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        result = await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()

    assert result is not None
    # Only the metric present in BOTH blobs as a list is compared.
    metric_names = [r.metric for r in result.rows]
    assert "ndcg_at_5" in metric_names
    assert "faithfulness" not in metric_names
    assert "context_recall" not in metric_names


async def test_compare_passes_exact_vectors_to_stats() -> None:
    """``compare()`` passes the exact per-query vectors to ``stats.compare``.

    We intercept ``stats.compare`` to capture its arguments and verify
    the extracted vectors match the report blobs exactly.
    """
    ndcg_a = [0.5, 0.6, 0.7]
    ndcg_b = [0.55, 0.65, 0.75]
    store = {
        "report-a": _fake_report(
            "report-a",
            dataset_id="ds-vec",
            n_queries=3,
            metrics={"ndcg_at_5": ndcg_a},
        ),
        "report-b": _fake_report(
            "report-b",
            dataset_id="ds-vec",
            n_queries=3,
            metrics={"ndcg_at_5": ndcg_b},
        ),
    }
    svc, repo_patcher = _make_service(repo_store=store)

    captured: list[dict[str, Any]] = []

    def _fake_compare(
        a_vectors: dict[str, Any],
        b_vectors: dict[str, Any],
        *,
        metrics: Any,
        n_queries: int,
        **_kwargs: Any,
    ) -> MagicMock:
        captured.append(
            {
                "a_vectors": dict(a_vectors),
                "b_vectors": dict(b_vectors),
                "metrics": list(metrics),
                "n_queries": n_queries,
            }
        )
        return MagicMock()  # ComparisonResult-like stub

    try:
        with patch("iw_rag.eval.benchmark.service.stats.compare", side_effect=_fake_compare):
            await svc.compare("report-a", "report-b")
    finally:
        repo_patcher.stop()

    assert len(captured) == 1
    call = captured[0]
    assert call["a_vectors"]["ndcg_at_5"] == ndcg_a
    assert call["b_vectors"]["ndcg_at_5"] == ndcg_b
    assert call["n_queries"] == 3
    assert "ndcg_at_5" in call["metrics"]


async def test_compare_skips_non_list_metric_values() -> None:
    """``compare()`` skips a metric whose blob value is not a list (e.g. a scalar mean)."""
    pytest.importorskip("scipy", reason="scipy required for stats.compare")

    store = {
        "report-a": _fake_report(
            "report-a",
            dataset_id="ds-skip",
            n_queries=2,
            metrics={
                "ndcg_at_5": [0.6, 0.7],
                "faithfulness": 0.75,  # scalar, not a list — should be skipped
            },
        ),
        "report-b": _fake_report(
            "report-b",
            dataset_id="ds-skip",
            n_queries=2,
            metrics={
                "ndcg_at_5": [0.55, 0.65],
                "faithfulness": 0.80,  # also scalar — skipped
            },
        ),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        result = await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()

    assert result is not None
    metric_names = [r.metric for r in result.rows]
    assert "faithfulness" not in metric_names  # was scalar, not a list
    assert "ndcg_at_5" in metric_names


async def test_compare_empty_metrics_blob_returns_result_with_no_rows() -> None:
    """``compare()`` with empty ``metrics`` blobs produces an empty rows result."""
    store = {
        "report-a": _fake_report("report-a", dataset_id="ds-empty", n_queries=2, metrics={}),
        "report-b": _fake_report("report-b", dataset_id="ds-empty", n_queries=2, metrics={}),
    }
    svc, patcher = _make_service(repo_store=store)

    captured_metrics: list[list[str]] = []

    def _fake_compare(
        a_vectors: dict[str, Any],
        b_vectors: dict[str, Any],
        *,
        metrics: Any,
        n_queries: int,
        **_kwargs: Any,
    ) -> MagicMock:
        captured_metrics.append(list(metrics))
        return MagicMock()

    try:
        with patch("iw_rag.eval.benchmark.service.stats.compare", side_effect=_fake_compare):
            await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()

    # No eligible metrics — compare is called with an empty metrics list.
    assert len(captured_metrics) == 1
    assert captured_metrics[0] == []


# ===========================================================================
# 4. compare() — n_queries from report row
# ===========================================================================


async def test_compare_passes_n_queries_from_report_a() -> None:
    """``compare()`` reads ``n_queries`` from report A's row (service.py:129)."""
    store = {
        "report-a": _fake_report("report-a", dataset_id="ds-nq", n_queries=42, metrics={}),
        "report-b": _fake_report("report-b", dataset_id="ds-nq", n_queries=42, metrics={}),
    }
    svc, patcher = _make_service(repo_store=store)

    captured_n: list[int] = []

    def _fake_compare(
        a_vectors: Any,
        b_vectors: Any,
        *,
        metrics: Any,
        n_queries: int,
        **_kwargs: Any,
    ) -> MagicMock:
        captured_n.append(n_queries)
        return MagicMock()

    try:
        with patch("iw_rag.eval.benchmark.service.stats.compare", side_effect=_fake_compare):
            await svc.compare("report-a", "report-b")
    finally:
        patcher.stop()

    assert captured_n == [42]


# ===========================================================================
# 5. list_reports() — shaping
# ===========================================================================


async def test_list_reports_returns_all_rows_as_dicts() -> None:
    """``list_reports()`` returns every row as a dict, in DB order."""
    rows = [
        {"id": "r1", "dataset_id": "ds-1", "created_at": "2026-01-02", "ndcg_at_5": 0.7},
        {"id": "r2", "dataset_id": "ds-1", "created_at": "2026-01-01", "ndcg_at_5": 0.6},
    ]
    svc, patcher = _make_service(list_rows=rows)
    try:
        result = await svc.list_reports()
    finally:
        patcher.stop()

    assert len(result) == 2
    assert result[0]["id"] == "r1"
    assert result[1]["id"] == "r2"


async def test_list_reports_attaches_human_dataset_name_and_version() -> None:
    """``list_reports()`` resolves the opaque ``dataset_id`` uuid to the
    ``eval_datasets`` name + version so callers render ``benchmark_golden (v1)``
    instead of the uuid."""
    rows = [
        {"id": "r1", "dataset_id": "ds-uuid-1", "created_at": "2026-01-02"},
        {"id": "r2", "dataset_id": "ds-uuid-1", "created_at": "2026-01-01"},
    ]
    datasets = [_FakeDatasetRow("ds-uuid-1", "benchmark_golden", "v1")]
    svc, patcher = _make_service(list_rows=rows, datasets=datasets)
    try:
        result = await svc.list_reports()
    finally:
        patcher.stop()

    assert result[0]["dataset_name"] == "benchmark_golden"
    assert result[0]["dataset_version"] == "v1"
    # Every row sharing the dataset gets the same resolved label.
    assert result[1]["dataset_name"] == "benchmark_golden"


async def test_list_reports_dataset_label_is_none_when_fk_unresolved() -> None:
    """A dangling ``dataset_id`` (no matching ``eval_datasets`` row) yields
    ``None`` for both label fields so the render sites fall back to the uuid."""
    rows = [{"id": "r1", "dataset_id": "ghost-uuid", "created_at": "2026-01-02"}]
    svc, patcher = _make_service(list_rows=rows, datasets=[])
    try:
        result = await svc.list_reports()
    finally:
        patcher.stop()

    assert result[0]["dataset_name"] is None
    assert result[0]["dataset_version"] is None


async def test_list_reports_empty_table_returns_empty_list() -> None:
    """``list_reports()`` returns ``[]`` when the table has no rows."""
    svc, patcher = _make_service(list_rows=[])
    try:
        result = await svc.list_reports()
    finally:
        patcher.stop()

    assert result == []


async def test_list_reports_row_fields_round_trip() -> None:
    """``list_reports()`` row dicts carry all fields from the DB mapping."""
    rows = [
        {
            "id": "abc",
            "dataset_id": "ds-x",
            "recall_at_5": 0.8,
            "ndcg_at_5": 0.7,
            "faithfulness_mean": 0.9,
            "created_at": "2026-06-01",
        }
    ]
    svc, patcher = _make_service(list_rows=rows)
    try:
        result = await svc.list_reports()
    finally:
        patcher.stop()

    assert len(result) == 1
    row = result[0]
    assert row["id"] == "abc"
    assert row["recall_at_5"] == pytest.approx(0.8)
    assert row["ndcg_at_5"] == pytest.approx(0.7)
    assert row["faithfulness_mean"] == pytest.approx(0.9)


# ===========================================================================
# 6. get_report() — delegates to repository
# ===========================================================================


async def test_get_report_returns_row_when_found() -> None:
    """``get_report()`` delegates to the repository and returns the row dict."""
    store = {
        "r99": _fake_report("r99", dataset_id="ds-1"),
    }
    svc, patcher = _make_service(repo_store=store)
    try:
        result = await svc.get_report("r99")
    finally:
        patcher.stop()

    assert result is not None
    assert result["id"] == "r99"


async def test_get_report_returns_none_when_not_found() -> None:
    """``get_report()`` returns ``None`` when the report id does not exist."""
    svc, patcher = _make_service(repo_store={})
    try:
        result = await svc.get_report("no-such-id")
    finally:
        patcher.stop()

    assert result is None


# ===========================================================================
# 7. Service module public surface
# ===========================================================================


def test_service_module_exposes_benchmark_report_service() -> None:
    """The service module exposes ``BenchmarkReportService`` as its public class."""
    from iw_rag.eval.benchmark import service as svc_mod

    assert hasattr(svc_mod, "BenchmarkReportService")
    assert callable(svc_mod.BenchmarkReportService)


def test_compare_metrics_constant_includes_required_winner_metrics() -> None:
    """``_COMPARE_METRICS`` contains both ``ndcg_at_5`` and ``faithfulness``.

    The winner rule (R-00121 §5.4) requires BOTH metrics to be present in
    the comparison set; they must appear in ``_COMPARE_METRICS``.
    """
    from iw_rag.eval.benchmark.service import _COMPARE_METRICS

    assert set(_COMPARE_METRICS) == {
        "ndcg_at_5",
        "faithfulness",
        "context_recall",
        "factual_correctness",
    }
