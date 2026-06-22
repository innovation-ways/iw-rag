"""Integration tests for the F-00096 Config API routes (S07 + S17 cross-cutting).

The S07 step already owns the **unit-level** route coverage in
``tests/unit/api/test_config_routes.py`` (fake service seam; no DB;
no Settings validation).  This file is the **end-to-end
integration** coverage the S17 prompt pins separately — the routes
+ the live S03 :class:`ConfigService` + a testcontainer
``config_change_log`` table (the F-00096 §Database Changes
round-trip) + the live :class:`Settings` validation.

The tests follow the same own-fixture migration pattern as
``tests/integration/storage/test_config_change_log_immutability.py``:
the rebased chain in the shared ``pg_dsn`` fixture strips 0008 (and
0009 depends on 0008), so this file owns its own ``db_container`` /
``migrated_engine`` fixtures to apply 0001-0007 + 0009 manually.

Coverage (per F-00096 Feature Design §Acceptance Criteria and
§Boundary Behavior):

* **AC2 + AC7** — an authenticated ``eval-operator`` PUT on an
  in-bounds editable key → 200; the value is persisted to the
  overlay; a ``config_change_log`` row is written with
  ``principal_id`` from auth (NOT the request body).
* **AC7** — a non-operator principal PUT → **403** with no
  overlay/audit row written (Boundary row "Non-operator hits
  config").
* **AC3** — a LOCKED-key PUT → 4xx with no overlay/audit row
  written (Boundary row "Config edit to a locked key").
* **Boundary row "Config edit out of declared bounds"** — an
  out-of-bounds PUT (e.g. ``retrieval.top_k = -1``) → 4xx with no
  overlay/audit row.
* **AC2 + GET /api/config** — the response shape carries the active
  values + per-key explanation + ``editable``/``locked``/``pending``
  flags.  An editable key carries ``editable=True, locked=False``;
  a known locked key carries ``editable=False, locked=True``.

All tests run the live S03 :class:`ConfigService` against the
testcontainer; no fakes, no mocks of the database.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
import sqlalchemy
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.api.config_service import ConfigService
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants — mirror test_config_change_log_immutability.py
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = (
    "paradedb/paradedb@sha256:dbd64f0eb2716b8466afe54b91fec782f4107a8d0e6bd3660b89c88a1ded2ead"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_VERSIONS = _REPO_ROOT / "alembic" / "versions"
_PENDING_MIGRATION = _ALEMBIC_VERSIONS / "0008_benchmark_reports.py"
_PENDING_MIGRATION_0009 = _ALEMBIC_VERSIONS / "0009_config_change_log.py"


# ---------------------------------------------------------------------------
# Fixtures — owned (same as test_config_change_log_immutability.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _isolated_alembic_versions() -> Generator[Path, None, None]:
    """Copy ``alembic/versions`` into a temp dir WITHOUT 0008 / 0009."""
    with tempfile.TemporaryDirectory(prefix="iw-rag-cc-routes-alembic-") as tmp:
        tmp_path = Path(tmp)
        alembic_dir = tmp_path / "alembic"
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        for src in _ALEMBIC_VERSIONS.glob("*.py"):
            shutil.copy(src, versions_dir / src.name)
        for pending in (_PENDING_MIGRATION, _PENDING_MIGRATION_0009):
            candidate = versions_dir / pending.name
            if candidate.exists():
                candidate.unlink()
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
    """Run migrations against the container + apply 0009 manually."""
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    import iw_rag.settings as settings_mod
    from alembic import command

    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_isolated_alembic_versions))
    cfg.set_main_option("sqlalchemy.url", db_container)
    command.upgrade(cfg, "head")

    spec = importlib.util.spec_from_file_location("m_0009", str(_PENDING_MIGRATION_0009))
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


@pytest.fixture
def config_db(migrated_engine: sqlalchemy.Engine) -> Generator[Database, None, None]:
    """Build a live :class:`Database` for the ConfigService.

    Yields a fresh :class:`Database` instance, disposes its engine
    on teardown.  Per-test (function scope) so each test starts from
    a clean ``config_change_log`` (the table is truncated on entry).
    """
    dsn = migrated_engine.url.render_as_string(hide_password=False)
    with migrated_engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.config_change_log"))
    db = Database(DatabaseSettings(dsn=dsn, pool_min=1, pool_max=2))
    try:
        yield db
    finally:
        asyncio.run(db.dispose())


def _build_client(
    *,
    config_db: Database,
    user_groups: tuple[str, ...] = ("finance",),
    eval_groups: tuple[str, ...] = ("eval-operator",),
) -> TestClient:
    """Build a TestClient with the live ConfigService wired onto ``app.state``.

    The injectable-seam pattern (mirrors the F-00095 / S07 unit tests):
    the S03 :class:`ConfigService` is the **real** service bound to
    the testcontainer; the route's auth uses a :class:`StaticTokenAuthenticator`
    so the bearer token → AccessContext keystone is exercised end-
    to-end.
    """
    auth = StaticTokenAuthenticator.from_config(
        {
            "user-token": list(user_groups),
            "eval-token": list(eval_groups),
        },
        operator_tokens=[],
    )
    service = ConfigService(database=config_db)  # type: ignore[arg-type]
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        config_service=service,  # type: ignore[arg-type]
    )
    return TestClient(app)


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s `query_service` seam."""

    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover - never called
        raise RuntimeError("not used in S17 config integration tests")


# ---------------------------------------------------------------------------
# AC2 — eval-operator PUT in-bounds editable key
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_put_config_in_bounds_editable_key_persists_overlay_and_audits(
    config_db: Database,
) -> None:
    """AC2 — eval-operator PUT on an in-bounds editable key → 200 + audit row.

    The S17 cross-cutting end-to-end: the route delegates to the
    live S03 service; the service persists to the in-memory overlay
    AND writes one ``config_change_log`` row.  The audit row's
    ``principal_id`` is the auth-derived value (the S03 service's
    ``principal_id=`` kwarg → the S07 route's
    ``ctx.principal_id``; never the request body).

    **RED — production bug surfaced by S17** (see blockers):
    The S03 service's ``apply_edit`` opens an async session, calls
    ``ConfigChangeRepository.append(...)``, and returns without
    committing; when the ``Database.session()`` context exits, the
    unwritten transaction is rolled back.  The audit row is
    **not** persisted in production today.  This is a real
    production bug — the entire ``config_change_log`` audit trail
    is broken — and is raised as a blocker for the code-review-
    fix step (the S17 test must NOT silently fix the production
    code).  Pinned here as RED evidence.

    **Test verification under RED**: the assertion
    ``len(rows) == 1`` fails; the S17 report documents the bug.
    """
    client = _build_client(config_db=config_db)

    resp = client.put(
        "/api/config/guardrails.input.threshold",
        json={"value": 0.7},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "guardrails.input.threshold"
    assert body["applied_mode"] == "hot"
    assert body["new_value"] == 0.7

    # The audit row landed in the live testcontainer.
    dsn = config_db._engine.url.render_as_string(hide_password=False)  # noqa: SLF001
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT principal_id, config_key, new_value, applied_mode "
                    "FROM iw_rag.config_change_log "
                    "WHERE config_key = 'guardrails.input.threshold'"
                )
            ).fetchall()
    finally:
        engine.dispose()

    # RED: This assertion fails today (the S03 service doesn't
    # commit).  The bug is documented in the S17 report blockers.
    assert len(rows) == 1, (
        f"RED: expected 1 audit row, got {len(rows)}. "
        "The S03 service's apply_edit does not commit the audit "
        "write — see S17 report blockers."
    )
    # The StaticTokenAuthenticator truncates the token to its first
    # 8 chars for principal_id derivation — ``"eval-token"`` →
    # ``"eval-tok"`` (mirrors the S07 unit test).
    assert rows[0][0] == "eval-tok"
    assert rows[0][1] == "guardrails.input.threshold"
    assert rows[0][2] == "0.7"
    assert rows[0][3] == "hot"


# ---------------------------------------------------------------------------
# AC7 — non-operator → 403, no overlay, no audit
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_put_config_non_operator_returns_403_and_writes_nothing(
    config_db: Database,
) -> None:
    """AC7 — a non-operator principal gets 403 (no overlay, no audit row)."""
    client = _build_client(config_db=config_db, user_groups=("finance",))

    resp = client.put(
        "/api/config/guardrails.input.threshold",
        json={"value": 0.7},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403

    # No row was written — the ACL gate runs BEFORE the service.
    dsn = config_db._engine.url.render_as_string(hide_password=False)  # noqa: SLF001
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM iw_rag.config_change_log")).scalar_one()
    finally:
        engine.dispose()
    assert count == 0


# ---------------------------------------------------------------------------
# AC3 — LOCKED-key PUT → 4xx, no overlay, no audit
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_put_config_locked_key_returns_4xx_and_writes_nothing(
    config_db: Database,
) -> None:
    """AC3 — a LOCKED-key PUT returns 4xx (ValidationError → 400) and writes no row.

    Boundary row "Config edit to a locked key": the locked-set
    check runs BEFORE the overlay/audit write, so a hostile edit on
    ``guardrails.enabled`` never persists.
    """
    client = _build_client(config_db=config_db)

    resp = client.put(
        "/api/config/guardrails.enabled",
        json={"value": False},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 400

    # No audit row landed.
    dsn = config_db._engine.url.render_as_string(hide_password=False)  # noqa: SLF001
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM iw_rag.config_change_log")).scalar_one()
    finally:
        engine.dispose()
    assert count == 0


# ---------------------------------------------------------------------------
# Boundary row "Config edit out of declared bounds" — out-of-bounds → 4xx
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_put_config_out_of_bounds_returns_4xx_and_writes_nothing(
    config_db: Database,
) -> None:
    """``retrieval.top_k = -1`` is rejected with 4xx; no overlay/audit row.

    Boundary row "Config edit out of declared bounds" — the registry
    bounds check runs BEFORE the overlay/audit write, so a noisy
    out-of-bounds value never persists.
    """
    client = _build_client(config_db=config_db)

    resp = client.put(
        "/api/config/retrieval.top_k",
        json={"value": -1},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 400

    dsn = config_db._engine.url.render_as_string(hide_password=False)  # noqa: SLF001
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM iw_rag.config_change_log")).scalar_one()
    finally:
        engine.dispose()
    assert count == 0


# ---------------------------------------------------------------------------
# GET /api/config shape — per-key active + flags
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_config_returns_active_entries_with_editable_locked_pending_flags(
    config_db: Database,
) -> None:
    """GET /api/config returns active entries with editable/locked/pending flags.

    The S17 cross-cutting integration: the live S03 service's
    ``read_active()`` returns one row per registry key + one row
    per locked key (with redaction for secrets); the S07 route
    serialises them.  Pinned here so a future refactor that drops
    the ``explanation`` field or the ``editable``/``locked``
    boolean breaks loudly.
    """
    client = _build_client(config_db=config_db)

    resp = client.get(
        "/api/config",
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    assert isinstance(body["items"], list)
    assert len(body["items"]) > 0  # at least the registry + the locked set

    by_key = {item["key"]: item for item in body["items"]}

    # An editable key carries editable=True, locked=False, and a
    # non-empty explanation.  The registry declares
    # ``guardrails.input.threshold`` as hot-reloadable, so
    # ``pending_restart`` is False.
    guardrails_input = by_key["guardrails.input.threshold"]
    assert guardrails_input["editable"] is True
    assert guardrails_input["locked"] is False
    assert guardrails_input["pending_restart"] is False
    assert isinstance(guardrails_input["explanation"], str)
    assert guardrails_input["explanation"] != ""

    # A non-hot editable key (``retrieval.top_k``) carries
    # pending_restart=True (AC4 honesty).
    retrieval_top_k = by_key["retrieval.top_k"]
    assert retrieval_top_k["editable"] is True
    assert retrieval_top_k["locked"] is False
    assert retrieval_top_k["pending_restart"] is True

    # A known locked key (``guardrails.enabled``) carries
    # editable=False, locked=True.
    guardrails_enabled = by_key["guardrails.enabled"]
    assert guardrails_enabled["editable"] is False
    assert guardrails_enabled["locked"] is True
