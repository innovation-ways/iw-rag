"""Unit tests for production composition helpers in benchmark_composition.py.

Covers the two main units that lack test coverage on this branch:

* :func:`_build_ragas_runner` — the graceful-degrade path when ``langchain_openai``
  is absent (always the case in this environment).
* :func:`build_conclusion_fn` — both the import-failure degrade path (returns
  ``None``) and the success path (returns a callable whose async body invokes the
  writer and returns the generated conclusion text).
* The inner ``_conclusion_fn`` closure — exercises ``_read_same_dataset_reports``
  via a fake database session that mimics the exception-degrade path (empty-list
  fallback) and the happy-path (list of rows forwarded to the writer).
* :func:`_read_same_dataset_reports` — the success path (returns list of dicts)
  and the degrade path (returns []).
* :func:`resolve_dataset_uuid` — the uuid pass-through, the named-match path, the
  populated-fallback path, and the NotFoundError path.

All tests are hermetic: no DB, no network, no real LLM.
"""

from __future__ import annotations

import uuid as _uuid_mod
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from iw_rag.api.benchmark_composition import (
    _build_ragas_runner,
    _read_same_dataset_reports,
    build_conclusion_fn,
    resolve_dataset_uuid,
)

# ---------------------------------------------------------------------------
# Fakes / stubs
# ---------------------------------------------------------------------------


class _FakeModelEntry:
    """Minimal model-entry stub exposing the two fields the composition root reads.

    Args:
        endpoint: Served endpoint URL.
        served_name: Model ID on the serving engine.
    """

    def __init__(
        self, endpoint: str = "http://localhost:8001/v1", served_name: str = "selene-2b"
    ) -> None:
        self.endpoint = endpoint
        self.served_name = served_name


class _FakeModelRegistry:
    """Fake ``ModelRegistry`` whose ``get`` always returns the same entry stub.

    Args:
        entry: The ``_FakeModelEntry`` to return from every ``get`` call.
    """

    def __init__(self, entry: _FakeModelEntry | None = None) -> None:
        self._entry = entry or _FakeModelEntry()

    def get(self, logical_name: str) -> _FakeModelEntry:
        """Return the configured entry regardless of logical name."""
        return self._entry


class _FakeEvalJudges:
    """Stub for ``settings.eval.judges`` (only the field the composition reads)."""

    general_model: str = "judge_general"


class _FakeEval:
    """Stub for ``settings.eval`` subsection."""

    judges: _FakeEvalJudges = _FakeEvalJudges()


class _FakeSettings:
    """Minimal settings stub satisfying all paths exercised in this test module.

    Args:
        entry: Model entry returned by the registry.
    """

    def __init__(self, entry: _FakeModelEntry | None = None) -> None:
        self.models = _FakeModelRegistry(entry or _FakeModelEntry())
        self.eval = _FakeEval()


class _FakeRaisingModelRegistry:
    """Registry whose ``get`` raises ``KeyError`` to trigger the except path."""

    def get(self, logical_name: str) -> _FakeModelEntry:
        raise KeyError(f"unknown model: {logical_name!r}")


class _FakeRaisingSettings:
    """Settings whose model registry always raises (forces degrade-to-None paths)."""

    def __init__(self) -> None:
        self.models = _FakeRaisingModelRegistry()
        self.eval = _FakeEval()


class _FakeMappings:
    """Iterable of row-like objects returned by sqlalchemy result.mappings().

    Args:
        rows: List of dicts to expose as mapping rows.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def __iter__(self) -> Any:
        return iter(self._rows)


class _FakeExecuteResult:
    """Fake SQLAlchemy execute result supporting .mappings() and .all().

    Args:
        rows: The rows returned by .mappings() (as dicts) or .all() (as objects).
        all_rows: The list returned by .all() (overrides rows when set).
    """

    def __init__(
        self, rows: list[dict[str, Any]] | None = None, all_rows: list[Any] | None = None
    ) -> None:
        self._rows = rows or []
        self._all_rows = all_rows if all_rows is not None else []

    def mappings(self) -> _FakeMappings:
        """Return a fake mappings view over the configured rows."""
        return _FakeMappings(self._rows)

    def all(self) -> list[Any]:
        """Return the configured all-rows list."""
        return self._all_rows


class _FakeAsyncSession:
    """Async context manager that delegates execute to a configurable coroutine.

    Args:
        execute_result: What session.execute should return (or raise if an exception).
    """

    def __init__(self, execute_result: Any = None) -> None:
        self._result = execute_result

    async def __aenter__(self) -> _FakeAsyncSession:
        return self

    async def __aexit__(self, *args: Any) -> bool:
        return False

    async def execute(self, stmt: Any) -> Any:
        """Return the configured result or raise if it is an exception."""
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class _FakeDatabase:
    """Database stub whose ``session()`` returns the configured async session.

    Args:
        session_result: What the session's execute will return.
    """

    def __init__(self, session_result: Any = None) -> None:
        self._session = _FakeAsyncSession(session_result)

    def session(self) -> _FakeAsyncSession:
        """Return the pre-built fake session."""
        return self._session


class _FakeRaisingDatabase:
    """Database stub whose ``session()`` raises so the except path is exercised."""

    def session(self) -> _FakeAsyncSession:
        raise RuntimeError("db unavailable")


# ---------------------------------------------------------------------------
# _build_ragas_runner — graceful degrade when langchain_openai is absent
# ---------------------------------------------------------------------------


def test_build_ragas_runner_returns_none_when_langchain_absent() -> None:
    """Returns None when langchain_openai is not installed (always in this env)."""
    settings = _FakeSettings()
    generator = _FakeModelEntry(served_name="qwen3-14b")
    judge_entry = _FakeModelEntry(served_name="selene-2b")
    result = _build_ragas_runner(settings, generator, judge_entry)
    assert result is None


def test_build_ragas_runner_returns_none_regardless_of_settings() -> None:
    """Absent langchain_openai causes None regardless of what settings expose."""
    result = _build_ragas_runner(_FakeRaisingSettings(), _FakeModelEntry(), _FakeModelEntry())
    assert result is None


# ---------------------------------------------------------------------------
# build_conclusion_fn — degrade path (settings raise → None returned)
# ---------------------------------------------------------------------------


def test_build_conclusion_fn_returns_none_when_settings_raise() -> None:
    """Degrade path: when settings.models.get raises, build_conclusion_fn returns None."""
    result = build_conclusion_fn(_FakeRaisingSettings(), _FakeDatabase())
    assert result is None


def test_build_conclusion_fn_returns_none_when_writer_construction_fails() -> None:
    """Degrade path: any error constructing the writer yields None (the run then
    persists without a conclusion) rather than propagating."""
    with patch(
        "iw_rag.eval.benchmark.conclusion.ConclusionWriter.__init__",
        side_effect=RuntimeError("stubbed init failure"),
    ):
        result = build_conclusion_fn(_FakeSettings(), _FakeDatabase())
    assert result is None


# ---------------------------------------------------------------------------
# build_conclusion_fn — success path (returns a callable)
# ---------------------------------------------------------------------------


def test_build_conclusion_fn_returns_callable_on_success() -> None:
    """When settings are valid, build_conclusion_fn returns a non-None callable."""
    result = build_conclusion_fn(_FakeSettings(), _FakeDatabase())
    # The degrade path returns None; success returns the async _conclusion_fn closure.
    assert callable(result)


async def test_conclusion_fn_returns_conclusion_string_from_writer() -> None:
    """The _conclusion_fn closure invokes the writer and returns its string."""
    # Build the real closure with a valid settings stub.
    conclusion_fn = build_conclusion_fn(_FakeSettings(), _FakeDatabase())
    assert conclusion_fn is not None

    expected_text = "This run trades VRAM for faithfulness — prefer it when grounding matters."

    # Patch writer.write + the DB read inside the closure (no real DB access).
    with (
        patch(
            "iw_rag.eval.benchmark.conclusion.ConclusionWriter.write",
            new_callable=AsyncMock,
            return_value=expected_text,
        ),
        patch(
            "iw_rag.api.benchmark_composition._read_same_dataset_reports",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        this_row: dict[str, Any] = {
            "report_label": "qwen3-14b",
            "model_config": {"generator": "qwen3-14b"},
            "faithfulness_mean": 0.82,
            "ndcg_at_5": 0.61,
            "latency": {"e2e_p99_s": 2.1, "per_stage_ms": {"retrieve_generate_ms": 900.0}},
            "resource": {"gpu": {"vram_peak_mb": 24000, "vram_incremental_mb": 23000}},
        }
        result = await conclusion_fn(this_row, "abc123")
    assert result == expected_text


async def test_conclusion_fn_returns_none_when_writer_returns_none() -> None:
    """The _conclusion_fn closure propagates None when the writer degrades to None.

    writer.write() degrades to None on client failure (its own except catches all).
    The closure's return value must reflect that None directly.
    """
    conclusion_fn = build_conclusion_fn(_FakeSettings(), _FakeDatabase())
    assert conclusion_fn is not None

    with (
        patch(
            "iw_rag.eval.benchmark.conclusion.ConclusionWriter.write",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch(
            "iw_rag.api.benchmark_composition._read_same_dataset_reports",
            new_callable=AsyncMock,
            return_value=[],
        ),
    ):
        this_row: dict[str, Any] = {
            "report_label": "qwen3-14b",
            "model_config": {"generator": "qwen3-14b"},
            "faithfulness_mean": 0.80,
            "ndcg_at_5": 0.60,
            "latency": {},
            "resource": {},
        }
        result = await conclusion_fn(this_row, "abc123")
    assert result is None


# ---------------------------------------------------------------------------
# _read_same_dataset_reports — degrade to empty list on exception
# ---------------------------------------------------------------------------


async def test_read_same_dataset_reports_returns_empty_list_on_db_error() -> None:
    """When the database raises, the function returns [] (best-effort comparables)."""
    from iw_rag.api.benchmark_composition import _read_same_dataset_reports

    broken_db = _FakeRaisingDatabase()
    result = await _read_same_dataset_reports(broken_db, "benchmark-golden")
    assert result == []


async def test_read_same_dataset_reports_returns_empty_list_on_invalid_uuid() -> None:
    """A non-uuid, non-name dataset_id that causes uuid.UUID to raise → empty list."""
    # Pass a value that will cause uuid.UUID() to raise ValueError, which is caught.
    broken_db = _FakeRaisingDatabase()
    result = await _read_same_dataset_reports(broken_db, "not-a-uuid-and-db-is-broken")
    assert result == []


async def test_read_same_dataset_reports_returns_rows_from_session() -> None:
    """Success path: rows returned by session.execute are converted to dicts."""
    good_uuid = str(_uuid_mod.uuid4())
    row_dict = {
        "id": good_uuid,
        "faithfulness_mean": 0.75,
        "report_label": "qwen3-8b",
        "model_config": {"generator": "qwen3-8b"},
        "latency": {},
        "resource": {},
    }
    db = _FakeDatabase(session_result=_FakeExecuteResult(rows=[row_dict]))
    result = await _read_same_dataset_reports(db, good_uuid)
    assert len(result) == 1
    assert result[0]["report_label"] == "qwen3-8b"
    assert result[0]["faithfulness_mean"] == 0.75


# ---------------------------------------------------------------------------
# resolve_dataset_uuid — uuid passthrough, named match, fallback, not-found
# ---------------------------------------------------------------------------


async def test_resolve_dataset_uuid_returns_uuid_string_verbatim() -> None:
    """A valid UUID string is returned as-is without touching the database."""
    valid_uuid = str(_uuid_mod.uuid4())
    # Database is broken — it must never be called on the uuid path.
    result = await resolve_dataset_uuid(_FakeRaisingDatabase(), valid_uuid)
    assert result == valid_uuid


async def test_resolve_dataset_uuid_returns_named_match_when_found() -> None:
    """A dataset name that matches a row with n > 0 resolves to its id string."""
    ds_id = _uuid_mod.uuid4()

    class _NamedRow:
        id = ds_id
        name = "benchmark_golden"
        n = 5

    db = _FakeDatabase(session_result=_FakeExecuteResult(all_rows=[_NamedRow()]))
    result = await resolve_dataset_uuid(db, "benchmark_golden")
    assert result == str(ds_id)


async def test_resolve_dataset_uuid_returns_named_match_for_dash_variant() -> None:
    """The name ``benchmark-golden`` matches a row stored as ``benchmark_golden``."""
    ds_id = _uuid_mod.uuid4()

    class _NamedRow:
        id = ds_id
        name = "benchmark_golden"  # stored with underscore
        n = 3

    db = _FakeDatabase(session_result=_FakeExecuteResult(all_rows=[_NamedRow()]))
    # Query uses dash variant; the function normalises it to both.
    result = await resolve_dataset_uuid(db, "benchmark-golden")
    assert result == str(ds_id)


async def test_resolve_dataset_uuid_falls_back_to_most_populated_dataset() -> None:
    """When no row name matches, the most populated dataset is returned as fallback."""
    ds_id = _uuid_mod.uuid4()

    class _PopulatedRow:
        id = ds_id
        name = "other_dataset"
        n = 8

    class _EmptyRow:
        id = _uuid_mod.uuid4()
        name = "another_dataset"
        n = 0

    db = _FakeDatabase(session_result=_FakeExecuteResult(all_rows=[_PopulatedRow(), _EmptyRow()]))
    result = await resolve_dataset_uuid(db, "does-not-exist")
    assert result == str(ds_id)


async def test_resolve_dataset_uuid_raises_not_found_when_no_dataset_populated() -> None:
    """NotFoundError is raised when the database contains no populated dataset."""
    from iw_rag.domain.errors import NotFoundError

    class _EmptyRow:
        id = _uuid_mod.uuid4()
        name = "empty_set"
        n = 0

    db = _FakeDatabase(session_result=_FakeExecuteResult(all_rows=[_EmptyRow()]))
    with pytest.raises(NotFoundError, match="no populated eval dataset found"):
        await resolve_dataset_uuid(db, "does-not-exist")


async def test_resolve_dataset_uuid_raises_not_found_when_table_empty() -> None:
    """NotFoundError is raised when there are no datasets at all in the table."""
    from iw_rag.domain.errors import NotFoundError

    db = _FakeDatabase(session_result=_FakeExecuteResult(all_rows=[]))
    with pytest.raises(NotFoundError, match="no populated eval dataset found"):
        await resolve_dataset_uuid(db, "benchmark_golden")
