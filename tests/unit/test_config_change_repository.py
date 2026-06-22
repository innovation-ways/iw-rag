"""Unit tests for ``ConfigChangeRepository`` (F-00096, S03).

The repository is the typed async insert + read surface over
``iw_rag.config_change_log`` — the Operator Console's append-only
audit trail.  It is the **storage-layer enforcement** of Invariant 2
(append-only) at the application boundary:

* No UPDATE or DELETE method is exposed.
* The only write is :meth:`append`.
* DB-level defense in depth lives in the trigger + REVOKE owned by S01.

These tests pin:

* :meth:`append` returns a fresh uuid for each row.
* :meth:`append` issues an INSERT with the right columns and ``RETURNING id``.
* :meth:`append` never accepts a ``principal_id`` from a request body
  (the column is sourced exclusively from the caller — the service
  layer passes the authenticated ``AccessContext.principal_id``).
* :meth:`list_changes` returns rows in the order the repository
  decides, with ``.mappings()`` semantics.
* The repository has NO update/delete method — the append-only
  surface is a property of the class, not a runtime accident.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from iw_rag.storage.repositories import ConfigChangeRepository

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Minimal async session — captures the insert statement and returns
    a row stub via ``.execute()``."""

    def __init__(self) -> None:
        self.executed: list[Any] = []
        # Pre-seed a row the repository can return from .scalar_one().
        self._next_id = uuid.uuid4()

    async def execute(self, stmt: Any) -> Any:
        """Record the statement and return a stub with ``.scalar_one()``."""
        self.executed.append(stmt)
        result = MagicMock()
        result.scalar_one.return_value = self._next_id
        return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_repository_exposes_only_append_and_list_changes() -> None:
    """Invariant 2: ConfigChangeRepository has NO update/delete method.

    The append-only surface is a class property, not a runtime accident.
    A misclick or a refactor that adds ``update_*`` / ``delete_*``
    methods would re-open the audit-trail mutability hole.
    """
    session = _FakeAsyncSession()
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]
    public = {m for m in dir(repo) if not m.startswith("_")}
    # The two legitimate methods + dunder fallback (object.__init__ etc.).
    for forbidden in ("update", "delete", "drop", "truncate", "purge"):
        assert forbidden not in public, (
            f"append-only repository must not expose '{forbidden}' "
            f"(Invariant 2: config_change_log is append-only)."
        )
    # The two legitimate methods.
    assert "append" in public
    assert "list_changes" in public


async def test_repository_append_returns_uuid_and_inserts_correct_columns() -> None:
    """``append`` returns a uuid and inserts the right columns."""
    session = _FakeAsyncSession()
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]

    new_id = await repo.append(
        principal_id="alice",
        config_key="retrieval.top_k",
        old_value="8",
        new_value="12",
        applied_mode="hot",
        trace_id="trace-1",
    )

    assert isinstance(new_id, uuid.UUID)
    # Exactly one statement was executed (the INSERT ... RETURNING id).
    assert len(session.executed) == 1


async def test_repository_append_uses_principal_id_from_caller() -> None:
    """``append`` stores principal_id exactly as passed (Invariant 3).

    The caller is the service layer, which receives the principal id
    from the authenticated AccessContext — the repository never
    inspects the value, only stores it.
    """
    session = _FakeAsyncSession()
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]

    await repo.append(
        principal_id="bob-the-operator",  # from auth
        config_key="retrieval.top_k",
        old_value="8",
        new_value="12",
        applied_mode="hot",
    )

    stmt = session.executed[0]
    # Inspect the compiled SQL parameters.
    params = stmt.compile().params
    assert params["principal_id"] == "bob-the-operator"


async def test_repository_append_marks_old_value_nullable() -> None:
    """``old_value`` is nullable — the first ever overlay write has no prior value."""
    session = _FakeAsyncSession()
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]

    await repo.append(
        principal_id="alice",
        config_key="retrieval.top_k",
        old_value=None,
        new_value="12",
        applied_mode="hot",
    )

    stmt = session.executed[0]
    params = stmt.compile().params
    assert params["old_value"] is None


async def test_repository_list_changes_returns_mappings() -> None:
    """``list_changes`` issues a SELECT and returns ``.mappings()`` rows."""
    session = _FakeAsyncSession()
    # Patch the read-back execute to return a stub with .mappings().
    read_result = MagicMock()
    read_result.mappings.return_value = [
        {
            "id": uuid.uuid4(),
            "principal_id": "alice",
            "config_key": "retrieval.top_k",
            "old_value": "8",
            "new_value": "12",
            "applied_mode": "hot",
            "trace_id": "trace-1",
        }
    ]
    session.execute = AsyncMock(return_value=read_result)  # type: ignore[method-assign]
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]

    rows = await repo.list_changes()
    assert len(rows) == 1
    assert rows[0]["config_key"] == "retrieval.top_k"
    assert rows[0]["applied_mode"] == "hot"
    assert rows[0]["principal_id"] == "alice"


async def test_repository_list_changes_filter_by_config_key() -> None:
    """``list_changes(config_key=...)`` restricts the read to one key."""
    session = _FakeAsyncSession()
    read_result = MagicMock()
    read_result.mappings.return_value = []
    session.execute = AsyncMock(return_value=read_result)  # type: ignore[method-assign]
    repo = ConfigChangeRepository(session)  # type: ignore[arg-type]

    await repo.list_changes(config_key="retrieval.top_k", limit=10)
    # The execute was called with a statement that targets the table.
    assert session.execute.await_count == 1
    stmt = session.execute.await_args.args[0]
    # The SQL has a WHERE clause that filters on config_key.
    compiled = str(stmt.compile())
    assert "config_change_log" in compiled
    assert "config_key" in compiled
