"""Integration tests for the F-00096 ``config_change_log`` table.

F-00096 S17 — defensive coverage for the Operator Console's
append-only audit trail.  Mirrors the
``tests/integration/storage/test_audit_append_only.py`` pattern (the
F-00093 ``audit_log`` precedent the S01 migration was modelled on).

Uses **own-fixture** migration application: the standard
``tests/integration/conftest.py`` ``pg_dsn`` fixture runs
``alembic upgrade head`` against the rebased chain, but the rebased
chain strips the F-00093 0008 migration (PENDING sentinel) AND the
0009 migration's ``down_revision`` points to 0008 — so 0009 won't
auto-apply.  This file owns its own ``db_container`` /
``migrated_engine`` fixtures so the migration path is explicit:

  1. Copy the alembic versions dir to a temp dir WITHOUT 0008.
  2. ``command.upgrade(cfg, "head")`` against the temp dir — applies
     0001-0007.
  3. Import the 0009 module from the original dir and call its
     ``upgrade()`` directly so the table + the trigger + the REVOKE
     land (matches the contract in ``0009_config_change_log.py``).

Test coverage (per F-00096 Feature Design §Acceptance Criteria and
§Boundary Behavior + the F-00096 §Invariants 2 + 9):

* **AC1 / Invariant 9** — migration round-trip: apply the chain
  through 0009 against the testcontainer and assert the
  ``config_change_log`` table + its immutability trigger exist
  (head = ``0009_config_change_log``).
* **Invariant 2 (append-only)** — a raw UPDATE on
  ``iw_rag.config_change_log`` raises the trigger error
  (``config_change_log is append-only``); a raw DELETE raises the
  same error.  The row is unchanged after the failed mutation.
* **ConfigChangeRepository.append round-trip** — the storage surface
  ``append`` writes the documented columns; ``list_changes`` reads
  them back; every column round-trips exactly (id, ts present,
  ``principal_id``, ``config_key``, ``old_value``, ``new_value``,
  ``applied_mode`` in {hot, pending_restart}, ``trace_id``).

All tests are deterministic — no live model endpoints, no GPU, no
network, no mocked DB.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from alembic.config import Config
from sqlalchemy import text

from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = (
    "paradedb/paradedb@sha256:dbd64f0eb2716b8466afe54b91fec782f4107a8d0e6bd3660b89c88a1ded2ead"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_VERSIONS = _REPO_ROOT / "alembic" / "versions"
# The migration applied in isolation (bare CREATE TABLE — see fixture docstring).
_MIGRATION_0009 = _ALEMBIC_VERSIONS / "0009_config_change_log.py"


# ---------------------------------------------------------------------------
# Fixtures — owned here (not from tests/integration/conftest.py) because
# the rebased chain in the shared fixture would skip 0009 (it depends
# on the stripped 0008).  Mirror test_benchmark_reports_repo.py.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _isolated_alembic_versions() -> Generator[Path, None, None]:
    """Copy ``alembic/versions`` into a temp dir holding only the 0001-0007 base.

    This fixture applies 0009 *in isolation* (its ``upgrade()`` issues a
    bare ``CREATE TABLE`` — not ``IF NOT EXISTS`` — so it must NOT also be
    applied by ``alembic upgrade head``).  We therefore strip 0009 itself
    plus everything layered around it: 0008 (a real predecessor 0009 does
    not depend on) and **every migration on top of 0009** (0010+).

    Stripping the whole ``>= 0008`` tail — rather than a hard-coded
    ``(0008, 0009)`` pair — keeps the fixture correct as new migrations
    are added: a later revision left in the temp dir whose
    ``down_revision`` (e.g. 0009) was stripped dangles, and alembic's
    static chain resolver raises ``KeyError`` while building the revision
    map.  With the tail removed, ``upgrade head`` lands cleanly at 0007
    and 0009 is applied manually below.
    """
    with tempfile.TemporaryDirectory(prefix="iw-rag-cc-log-alembic-") as tmp:
        tmp_path = Path(tmp)
        alembic_dir = tmp_path / "alembic"
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        for src in _ALEMBIC_VERSIONS.glob("*.py"):
            shutil.copy(src, versions_dir / src.name)
        # Strip 0008 and the entire 0009+ tail so the temp chain is exactly
        # 0001-0007 (head=0007); 0009 is applied manually in migrated_engine.
        for candidate in versions_dir.glob("*.py"):
            prefix = candidate.name[:4]
            if prefix.isdigit() and int(prefix) >= 8:
                candidate.unlink()
        # Need env.py + script.py.mako for alembic to run.
        shutil.copy(_ALEMBIC_VERSIONS.parent / "env.py", alembic_dir / "env.py")
        shutil.copy(
            _ALEMBIC_VERSIONS.parent / "script.py.mako",
            alembic_dir / "script.py.mako",
        )
        yield alembic_dir


@pytest.fixture(scope="module")
def db_container() -> Generator[str, None, None]:
    """A fresh paradedb container; yields the psycopg-driver DSN."""
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image=PARADEDB_IMAGE,
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as pg:
        raw_url = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        yield dsn


@pytest.fixture(scope="module")
def migrated_engine(
    db_container: str, _isolated_alembic_versions: Path
) -> Generator[sqlalchemy.Engine, None, None]:
    """Run migrations against the container + apply 0009 manually.

    Steps:
      1. Point alembic at the isolated alembic dir (no 0008 / 0009) +
         the container DSN, then ``command.upgrade(cfg, "head")`` —
         applies 0001-0007.  The ``alembic_version`` table ends up at
         0007.
      2. Import the 0009 migration module from the ORIGINAL versions
         dir (not the temp one) and call its ``upgrade()`` against a
         raw ``sqlalchemy`` connection.  This creates the
         ``iw_rag.config_change_log`` table + the
         ``config_change_log_key_time`` index + the
         ``config_change_log_immutable`` BEFORE UPDATE OR DELETE
         trigger + the REVOKE on the application role.

    Settings env is set on entry and cleaned on exit so the next test
    that loads ``get_settings()`` is not affected.
    """
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    import iw_rag.settings as settings_mod
    from alembic import command

    # Point Settings at the container (module-scoped, so we set env
    # directly; monkeypatch wouldn't survive the module scope).
    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    # 1. Apply 0001-0007.
    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_isolated_alembic_versions))
    cfg.set_main_option("sqlalchemy.url", db_container)
    command.upgrade(cfg, "head")

    # 2. Apply 0009 manually.
    spec = importlib.util.spec_from_file_location("m_0009", str(_MIGRATION_0009))
    m_0009 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m_0009)

    engine = sqlalchemy.create_engine(db_container)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            m_0009.upgrade()

    yield engine

    engine.dispose()
    for key in (
        "IW_RAG__DATABASE__DSN",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
        "IW_RAG__MODELS__RERANKER__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
    ):
        os.environ.pop(key, None)
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_cc_log(migrated_engine: sqlalchemy.Engine) -> None:
    """Truncate ``config_change_log`` before each test for isolation.

    Uses ``DELETE`` to exercise the trigger's defense-in-depth path
    would fail; uses ``TRUNCATE`` instead (TRUNCATE bypasses
    BEFORE-row triggers and is allowed even on tables that REVOKE
    UPDATE/DELETE — so the test setup is unaffected by the immutability
    guarantee the tests are pinning).
    """
    with migrated_engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.config_change_log"))


# ---------------------------------------------------------------------------
# Migration round-trip — table + trigger + index
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_migration_0009_creates_config_change_log_table(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The 0009 migration creates the ``iw_rag.config_change_log`` table.

    Every documented column (per F-00096 §"Database Changes") is
    present: ``id`` (uuid pk), ``ts``, ``principal_id``, ``config_key``,
    ``old_value``, ``new_value``, ``applied_mode``, ``trace_id``.
    """
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'iw_rag' AND table_name = 'config_change_log' "
                "ORDER BY ordinal_position"
            )
        ).fetchall()
    cols = {r[0] for r in rows}
    expected = {
        "id",
        "ts",
        "principal_id",
        "config_key",
        "old_value",
        "new_value",
        "applied_mode",
        "trace_id",
    }
    missing = expected - cols
    assert not missing, f"config_change_log missing columns: {missing}"


@pytest.mark.integration
def test_migration_0009_creates_immutability_trigger(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The 0009 migration creates the ``config_change_log_immutable`` trigger.

    The trigger function is ``iw_rag.config_change_no_mutate()``
    (PL/pgSQL); the trigger fires BEFORE UPDATE OR DELETE for each
    affected row (defense-in-depth layer 1, complementing the REVOKE
    on the application role).  Pinned here so a future migration
    rename / drop breaks loudly.
    """
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT trigger_name, event_manipulation, action_timing "
                "FROM information_schema.triggers "
                "WHERE event_object_schema = 'iw_rag' "
                "AND event_object_table = 'config_change_log'"
            )
        ).fetchall()
    triggers = {(r[0], r[1], r[2]) for r in rows}
    # The trigger must exist for both UPDATE and DELETE, BEFORE timing.
    assert ("config_change_log_immutable", "UPDATE", "BEFORE") in triggers, (
        f"UPDATE BEFORE trigger missing; got {triggers}"
    )
    assert ("config_change_log_immutable", "DELETE", "BEFORE") in triggers, (
        f"DELETE BEFORE trigger missing; got {triggers}"
    )


@pytest.mark.integration
def test_migration_0009_creates_key_time_index(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The 0009 migration creates the ``config_change_log_key_time`` index.

    Backs the operator-review query "all changes for a key, newest
    first" (mirror of ``audit_log_principal_time``).
    """
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'iw_rag' AND tablename = 'config_change_log'"
            )
        ).fetchall()
    index_names = {r[0] for r in rows}
    # Strong form: the index MUST be present (set membership, exact).
    assert "config_change_log_key_time" in index_names, (
        f"expected config_change_log_key_time; got {index_names}"
    )
    # Strong form: the index covers the documented
    # ``(config_key, ts)`` columns (regression guard against a
    # future migration silently re-shaping the index).
    with migrated_engine.connect() as conn:
        index_cols = conn.execute(
            text(
                "SELECT a.attname "
                "FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid "
                "  AND a.attnum = ANY(i.indkey) "
                "WHERE i.indexrelid = 'iw_rag.config_change_log_key_time'::regclass"
            )
        ).fetchall()
    cols = {r[0] for r in index_cols}
    assert cols == {"config_key", "ts"}, (
        f"expected config_change_log_key_time to cover (config_key, ts); got {cols}"
    )


# ---------------------------------------------------------------------------
# Append-only — UPDATE blocked by trigger
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_update_is_blocked_by_trigger(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """A raw UPDATE on ``iw_rag.config_change_log`` raises (trigger-enforced append-only).

    Invariant 2 — the ``config_change_log_immutable`` BEFORE UPDATE
    OR DELETE trigger raises ``config_change_log is append-only``
    for ANY session (the testcontainer is superuser; the trigger
    is the observable enforcement layer for that case).  The
    REVOKE on the application role is a defense-in-depth layer for
    the non-superuser case (effective in production; testcontainers
    connects as superuser, so the trigger is what fires here).
    """
    _truncate_cc_log(migrated_engine)

    # First insert a row (the seed for the UPDATE attempt).
    with migrated_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO iw_rag.config_change_log "
                "(principal_id, config_key, old_value, new_value, applied_mode) "
                "VALUES ('alice', 'guardrails.input.threshold', '0.5', '0.7', 'hot')"
            )
        )

    # The UPDATE must raise with the trigger's exact error message.
    with (
        pytest.raises(sqlalchemy.exc.DBAPIError, match="config_change_log is append-only"),
        migrated_engine.begin() as conn,
    ):
        conn.execute(
            text(
                "UPDATE iw_rag.config_change_log "
                "SET new_value = '0.99' "
                "WHERE config_key = 'guardrails.input.threshold'"
            )
        )

    # The row is unchanged.
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT new_value FROM iw_rag.config_change_log "
                "WHERE config_key = 'guardrails.input.threshold'"
            )
        ).fetchall()
    assert rows == [("0.7",)]


# ---------------------------------------------------------------------------
# Append-only — DELETE blocked by trigger
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_is_blocked_by_trigger(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """A raw DELETE on ``iw_rag.config_change_log`` raises (trigger-enforced append-only)."""
    _truncate_cc_log(migrated_engine)

    with migrated_engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO iw_rag.config_change_log "
                "(principal_id, config_key, old_value, new_value, applied_mode) "
                "VALUES ('bob', 'retrieval.top_k', '8', '12', 'pending_restart')"
            )
        )

    with (
        pytest.raises(sqlalchemy.exc.DBAPIError, match="config_change_log is append-only"),
        migrated_engine.begin() as conn,
    ):
        conn.execute(
            text("DELETE FROM iw_rag.config_change_log WHERE config_key = 'retrieval.top_k'")
        )

    # The row survived the failed DELETE.
    with migrated_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM iw_rag.config_change_log")).scalar_one()
    assert count == 1


# ---------------------------------------------------------------------------
# applied_mode CHECK constraint — only the two legal values
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_applied_mode_check_constraint_rejects_unknown_value(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """An INSERT with a non-canonical ``applied_mode`` is rejected by the CHECK constraint.

    The migration DDL pins the column to ``hot | pending_restart``
    (CHECK in the DDL).  A future refactor that adds a third value
    (e.g. ``"scheduled"``) would be a migration-breaking change;
    this test pins the surface area.
    """
    _truncate_cc_log(migrated_engine)

    with (
        pytest.raises(sqlalchemy.exc.DBAPIError, match="applied_mode|check"),
        migrated_engine.begin() as conn,
    ):
        conn.execute(
            text(
                "INSERT INTO iw_rag.config_change_log "
                "(principal_id, config_key, old_value, new_value, applied_mode) "
                "VALUES ('alice', 'guardrails.input.threshold', '0.5', '0.7', 'scheduled')"
            )
        )

    # No row was written.
    with migrated_engine.connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM iw_rag.config_change_log")).scalar_one()
    assert count == 0


# ---------------------------------------------------------------------------
# ConfigChangeRepository.append round-trip — every column round-trips
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_repository_append_round_trips_every_column(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """``ConfigChangeRepository.append`` round-trips every documented column.

    F-00096 §"Database Changes" pins the column set: ``id`` (uuid pk),
    ``ts`` (server default now()), ``principal_id`` (NOT NULL,
    Invariant 3 — sourced from auth), ``config_key`` (NOT NULL),
    ``old_value`` (nullable — first ever write has no prior),
    ``new_value`` (NOT NULL), ``applied_mode`` (NOT NULL, CHECK
    hot|pending_restart), ``trace_id`` (nullable).  Pinned here so
    the S03 service's storage contract is observed end-to-end.
    """
    import iw_rag.settings as settings_mod
    from iw_rag.storage.repositories import ConfigChangeRepository

    _truncate_cc_log(migrated_engine)
    settings_mod.get_settings.cache_clear()

    async def _round_trip() -> dict[str, Any]:
        dsn = migrated_engine.url.render_as_string(hide_password=False)
        # The migrated_engine is the sync testcontainer URL; the
        # Database factory expects DatabaseSettings (not a raw dsn).
        db = Database(
            DatabaseSettings(
                dsn=dsn,
                pool_min=1,
                pool_max=2,
            )
        )
        try:
            async with db.session() as session:
                repo = ConfigChangeRepository(session)
                new_id = await repo.append(
                    principal_id="operator-charlie",
                    config_key="guardrails.input.threshold",
                    old_value="0.5",
                    new_value="0.7",
                    applied_mode="hot",
                    trace_id="trace-cc-log-1",
                )
                await session.commit()
                return {
                    "id": new_id,
                    "dsn": dsn,
                }
        finally:
            await db.dispose()

    inserted = asyncio.run(_round_trip())
    new_id = inserted["id"]
    dsn = inserted["dsn"]

    # Read it back via a fresh sync connection (the testcontainer
    # URL works for both sync and async — the trigger is session-
    # transparent).
    engine = sqlalchemy.create_engine(dsn)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, principal_id, config_key, old_value, "
                "new_value, applied_mode, trace_id, ts "
                "FROM iw_rag.config_change_log WHERE id = :rid"
            ),
            {"rid": new_id},
        ).first()
    engine.dispose()

    assert row is not None, "row vanished after append — round-trip failed"
    # The exact round-trip:
    assert row[0] == new_id
    assert row[1] == "operator-charlie"
    assert row[2] == "guardrails.input.threshold"
    assert row[3] == "0.5"
    assert row[4] == "0.7"
    assert row[5] == "hot"
    assert row[6] == "trace-cc-log-1"
    # ``ts`` was set by the server default — must be non-NULL.
    assert row[7] is not None


# ---------------------------------------------------------------------------
# list_changes round-trip — read-back surface
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_repository_list_changes_returns_most_recent_first(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """``ConfigChangeRepository.list_changes`` returns rows newest-first (ts DESC)."""
    import iw_rag.settings as settings_mod
    from iw_rag.storage.repositories import ConfigChangeRepository

    _truncate_cc_log(migrated_engine)
    settings_mod.get_settings.cache_clear()
    dsn = migrated_engine.url.render_as_string(hide_password=False)

    async def _seed_and_list() -> tuple[list[dict[str, Any]], str]:
        db = Database(DatabaseSettings(dsn=dsn, pool_min=1, pool_max=2))
        try:
            async with db.session() as session:
                repo = ConfigChangeRepository(session)
                # Two rows; the second one (newer ts) must come first
                # on read-back.
                await repo.append(
                    principal_id="alice",
                    config_key="retrieval.top_k",
                    old_value="8",
                    new_value="10",
                    applied_mode="pending_restart",
                    trace_id=None,
                )
                await session.commit()
                # Sleep a beat so the second row's ts is later (the
                # server default is now() per-row; the testcontainer
                # clock is microsecond-resolution).
                await asyncio.sleep(0.01)
                async with db.session() as session2:
                    repo2 = ConfigChangeRepository(session2)
                    await repo2.append(
                        principal_id="bob",
                        config_key="retrieval.top_k",
                        old_value="10",
                        new_value="12",
                        applied_mode="pending_restart",
                        trace_id="trace-list-2",
                    )
                    await session2.commit()
                # Read back newest-first, filtered to the one key.
                async with db.session() as session3:
                    repo3 = ConfigChangeRepository(session3)
                    rows = await repo3.list_changes(config_key="retrieval.top_k", limit=10)
                return rows, dsn
        finally:
            await db.dispose()

    rows, _ = asyncio.run(_seed_and_list())
    # Newest first: the bob row (later ts) is at the top.
    assert rows[0]["principal_id"] == "bob"
    assert rows[0]["new_value"] == "12"
    assert rows[0]["applied_mode"] == "pending_restart"
    assert rows[0]["trace_id"] == "trace-list-2"
    # The older alice row is second.
    assert rows[1]["principal_id"] == "alice"
    assert rows[1]["new_value"] == "10"
    assert rows[1]["trace_id"] is None
