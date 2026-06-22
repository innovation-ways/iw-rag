"""Unit tests for ``iw_rag.api.config_service`` (F-00096, S03).

The overlay service is the **security keystone** of the Operator Console's
live-edit Config view.  It enforces Invariants 1, 2, 3, 5 of the design:

* Invariant 1 — locked keys can NEVER be edited.
* Invariant 2 — the audit row is append-only (never UPDATE/DELETE).
* Invariant 3 — ``principal_id`` on the audit row comes from auth, never
  from a request body.
* Invariant 5 — applying an overlay never disables a fail-closed/ACL/
  local-only invariant (locked set covers guardrails.enabled, the DSN,
  the model endpoints, the local-only switches).

These tests pin the apply path's behavioural contract end-to-end:

* **Hot key in-bounds** → ``applied_mode == "hot"``,
  ``get_settings.cache_clear()`` is invoked, one audit row is appended
  with the exact ``principal_id`` from the auth context.
* **Out-of-bounds value** (``top_k = -1``) → ``ValidationError`` raised,
  no audit row written.
* **Locked key** (``guardrails.enabled``) → ``ValidationError`` raised,
  no overlay/audit row written (AC3).
* **Non-hot editable key** (baked at startup) → ``applied_mode ==
  "pending_restart"``, cache NOT cleared (AC4).
* **Audit row shape** — no raw secret values; ``principal_id`` exactly
  the auth-supplied value (Invariant 3).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.errors import ValidationError

# ---------------------------------------------------------------------------
# Overlay-state hygiene — every test in this module runs against a
# pristine overlay so a successful ``apply_edit`` from one test cannot
# leak into the next test's ``Settings()`` build.  The autouse fixture
# clears both the in-process overlay dict and the registered overlay
# source (the source is re-registered after the test by re-importing
# the module, mirroring the production startup path).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_overlay_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear the overlay dict and ``get_settings`` cache around every test.

    The ConfigService writes to a module-level dict (``_OVERLAY``) and
    calls ``get_settings.cache_clear()`` on hot edits; both pieces of
    state are process-global, so a missing clean-up would let one
    test's overlay value leak into the next test's ``Settings()`` build
    and break test isolation.
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    cs._OVERLAY.clear()  # noqa: SLF001  (test seam, intentional)
    settings_mod.get_settings.cache_clear()
    yield
    cs._OVERLAY.clear()  # noqa: SLF001
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Minimal async session — no SQL is run against the in-memory fake."""


class FakeDatabase:
    """In-memory stand-in for :class:`iw_rag.storage.db.Database`.

    The service stores audit rows in a list and replays them through
    ``session.execute()`` for the read-back path.
    """

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self.executed: list[Any] = []

    @property
    def rows(self) -> list[dict[str, Any]]:
        """The current in-memory row set (copy)."""
        return list(self._rows)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[_FakeAsyncSession]:
        """Yield a session whose ``execute()`` returns a row stub.

        The repository's ``append`` issues ``insert(...).returning(id)``;
        we return a uuid the repo can read back.  We also persist the
        captured parameter dict into ``self._rows`` so read-back assertions
        can see what the repo tried to insert.
        """
        session = _FakeAsyncSession()
        session.execute = AsyncMock(  # type: ignore[attr-defined]
            side_effect=self._execute
        )
        session.commit = AsyncMock()  # type: ignore[attr-defined]  # no-op: in-memory fake
        yield session

    async def _execute(self, stmt: Any, *args: Any, **kwargs: Any) -> Any:
        """Capture the insert + return a stub result with a new uuid."""
        self.executed.append(stmt)
        new_id = uuid.uuid4()
        row = {
            "id": new_id,
            "ts": None,
            "principal_id": None,
            "config_key": None,
            "old_value": None,
            "new_value": None,
            "applied_mode": None,
            "trace_id": None,
        }
        # Try to extract the parameter dict from the SQLAlchemy statement.
        try:
            compiled = stmt.compile()
            params = compiled.params
            for k, v in params.items():
                if k in row:
                    row[k] = v
        except Exception:
            pass
        self._rows.append(row)
        result = MagicMock()
        result.scalar_one.return_value = new_id
        return result


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------


def _stub_get_settings(settings: Any) -> Any:
    """A ``get_settings`` stub that returns ``settings`` and has ``cache_clear``.

    The service does ``settings_module.get_settings.cache_clear()`` to
    apply a hot edit, so the test stub needs both ``__call__`` (returns
    the settings) and ``cache_clear`` (a no-op mock by default).
    """
    from unittest.mock import MagicMock

    fake = MagicMock(return_value=settings)
    fake.cache_clear = MagicMock()
    return fake


def _build_settings(**env_overrides: str) -> Any:
    """Build a fresh Settings object with the given env vars injected."""
    import iw_rag.settings as settings_mod

    base = {
        "IW_RAG__DATABASE__DSN": "postgresql+psycopg://u:p@h/db",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT": "http://vllm:8001/v1",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT": "http://infinity:7997",
        "IW_RAG__MODELS__RERANKER__ENDPOINT": "http://infinity:7997",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT": "http://vllm-judge:8002/v1",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT": "http://vllm-judge:8002/v1",
    }
    base.update(env_overrides)
    settings_mod.get_settings.cache_clear()
    with pytest.MonkeyPatch.context() as mp:
        for k, v in base.items():
            mp.setenv(k, v)
        return settings_mod.Settings()


def _make_service(
    database: FakeDatabase | None = None,
) -> tuple[Any, FakeDatabase]:
    """Build a ConfigService with injected fakes."""
    from iw_rag.api.config_service import ConfigService

    database = database if database is not None else FakeDatabase()
    service = ConfigService(database=database)
    return service, database


# ---------------------------------------------------------------------------
# Tests — apply_edit: hot key
# ---------------------------------------------------------------------------


async def test_apply_edit_in_bounds_hot_key_applies_hot_and_audits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-bounds edit on a hot-reloadable key applies HOT and writes one audit row.

    The principal_id on the audit row is exactly the value from the auth
    context (Invariant 3 — never from a request body, which here carries
    the bogus 'request-body-principal' string).
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    result = await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.7,
        principal_id="alice",
        trace_id="trace-xyz",
    )

    assert result.applied_mode == "hot"
    assert result.config_key == "guardrails.input.threshold"
    assert result.principal_id == "alice"
    assert result.old_value == settings.guardrails.input.threshold
    assert result.new_value == 0.7
    # The audit row was written.
    assert len(database.rows) == 1
    audit = database.rows[0]
    assert audit["principal_id"] == "alice"
    assert audit["config_key"] == "guardrails.input.threshold"
    assert audit["applied_mode"] == "hot"
    assert audit["trace_id"] == "trace-xyz"


async def test_apply_edit_in_bounds_hot_key_clears_get_settings_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot key: get_settings.cache_clear() is invoked exactly once."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    fake_get_settings = _stub_get_settings(settings)
    monkeypatch.setattr(settings_mod, "get_settings", fake_get_settings)

    service, _ = _make_service()

    result = await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.6,
        principal_id="alice",
    )

    assert result.applied_mode == "hot"
    # cache_clear() is the contract — verified by the test name.
    fake_get_settings.cache_clear.assert_called_once()


# ---------------------------------------------------------------------------
# Tests — apply_edit: out-of-bounds value
# ---------------------------------------------------------------------------


async def test_apply_edit_out_of_bounds_value_raises_no_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Out-of-bounds value (top_k = -1) raises ValidationError, no audit row.

    AC — "Config edit out of declared bounds" boundary row.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    with pytest.raises(ValidationError, match="bounds|range|out of range|top_k"):
        await service.apply_edit(
            key="retrieval.top_k",
            new_value=-1,
            principal_id="alice",
        )

    # No audit row was written — boundary contract.
    assert database.rows == []


# ---------------------------------------------------------------------------
# Tests — apply_edit: locked key
# ---------------------------------------------------------------------------


async def test_apply_edit_locked_key_raises_no_overlay_no_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Editing a locked key (guardrails.enabled) raises, no overlay, no audit.

    AC3 / boundary "Config edit to a locked key".
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    with pytest.raises(ValidationError, match="locked|guardrails\\.enabled"):
        await service.apply_edit(
            key="guardrails.enabled",
            new_value=False,
            principal_id="alice",
        )

    # No overlay row, no audit row.
    assert database.rows == []


async def test_apply_edit_database_dsn_locked_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The DSN is locked — the secret cannot be edited via the overlay."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    with pytest.raises(ValidationError, match="locked|database\\.dsn"):
        await service.apply_edit(
            key="database.dsn",
            new_value="postgresql+psycopg://attacker:pwn@h/db",
            principal_id="alice",
        )
    assert database.rows == []


async def test_apply_edit_unknown_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown key (not in registry, not in locked set) is rejected."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    with pytest.raises(ValidationError, match="unknown|not editable"):
        await service.apply_edit(
            key="this.key.does.not.exist",
            new_value=42,
            principal_id="alice",
        )
    assert database.rows == []


# ---------------------------------------------------------------------------
# Tests — apply_edit: pending-restart (non-hot) editable key
# ---------------------------------------------------------------------------


async def test_apply_edit_non_hot_editable_key_marks_pending_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-hot editable key (retrieval.top_k) returns pending_restart, no cache clear.

    The cache is NOT cleared because a startup-baked client (the spine's
    RetrievalConfig) consumed the value — the running request path is
    unchanged until restart (AC4).
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    fake_get_settings = _stub_get_settings(settings)
    monkeypatch.setattr(settings_mod, "get_settings", fake_get_settings)

    service, database = _make_service()

    result = await service.apply_edit(
        key="retrieval.top_k",
        new_value=12,
        principal_id="alice",
    )

    assert result.applied_mode == "pending_restart"
    assert result.config_key == "retrieval.top_k"
    assert result.principal_id == "alice"
    assert result.old_value == settings.retrieval.top_k
    assert result.new_value == 12
    # Cache was NOT cleared (AC4).
    fake_get_settings.cache_clear.assert_not_called()
    # Audit row was written.
    assert len(database.rows) == 1
    assert database.rows[0]["applied_mode"] == "pending_restart"
    assert database.rows[0]["config_key"] == "retrieval.top_k"


# ---------------------------------------------------------------------------
# Tests — read_active
# ---------------------------------------------------------------------------


async def test_read_active_returns_entries_for_each_registry_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_active() returns the active value + per-key description for each registered key.

    The shape is what S07 will serialise: an iterable of entries with
    active value, description, editable/locked flag, and a pending_restart
    indicator.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, _ = _make_service()

    entries = service.read_active()
    assert isinstance(entries, list)
    assert len(entries) > 0
    # Each entry has the expected fields.
    by_key = {e.key: e for e in entries}
    top_k = by_key["retrieval.top_k"]
    assert top_k.description
    assert top_k.editable is True
    assert top_k.locked is False
    assert top_k.active_value == settings.retrieval.top_k
    # non-hot editable key → pending_restart=True
    assert top_k.pending_restart is True


async def test_read_active_includes_locked_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """read_active() includes the locked keys as locked, value redacted (not editable)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, _ = _make_service()

    entries = service.read_active()
    by_key = {e.key: e for e in entries}

    guardrails_enabled = by_key.get("guardrails.enabled")
    assert guardrails_enabled is not None
    assert guardrails_enabled.locked is True
    assert guardrails_enabled.editable is False


# ---------------------------------------------------------------------------
# Tests — audit row shape / no secret leakage / principal_id provenance
# ---------------------------------------------------------------------------


async def test_audit_row_uses_exact_principal_id_from_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's principal_id is EXACTLY the value from the auth context (Invariant 3)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.6,
        principal_id="bob-the-operator",  # from auth, not the body
    )

    assert len(database.rows) == 1
    assert database.rows[0]["principal_id"] == "bob-the-operator"


async def test_audit_row_does_not_store_dsn_or_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row never contains the DSN or any secret (Invariant 5, design Notes)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()

    # The DSN edit is rejected (locked key), so no audit row.
    with pytest.raises(ValidationError, match="locked|database\\.dsn"):
        await service.apply_edit(
            key="database.dsn",
            new_value="postgresql+psycopg://leaked:pwn@h/db",
            principal_id="alice",
        )
    assert database.rows == []


# ---------------------------------------------------------------------------
# Tests — AC2 runtime hot-apply (S04 regression: F-1)
# ---------------------------------------------------------------------------
# The S04 review (F-1) found that the hot-apply path was a no-op at
# runtime: the service cleared the ``get_settings`` lru_cache but no
# code consulted the overlay, so the next ``get_settings()`` rebuilt
# from the unchanged env/yaml and returned the OLD value.  These tests
# pin the fix end-to-end — a successful ``apply_edit`` on a hot
# key MUST be observed by the next ``get_settings()`` call (the running
# request path the operator expects to be live-edited).


async def test_apply_edit_hot_key_value_observed_by_next_get_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2 regression: a hot ``apply_edit`` is observed by the next ``get_settings()`` call.

    Builds a real ``Settings`` from env, applies a hot edit, then calls
    the cached ``get_settings()`` directly (the same accessor the
    per-request guard layer uses) and asserts the new value is in
    place.  This is the runtime contract that F-1 broke and that
    ``_OverlaySettingsSource`` restores.
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    _build_settings()
    settings_mod.get_settings.cache_clear()

    # Sanity: the initial value is the env-baked default (Settings
    # default for input.threshold).
    initial = settings_mod.get_settings()
    initial_threshold = initial.guardrails.input.threshold
    # Apply a hot edit (the autouse fixture cleared the overlay so the
    # post-edit value is unambiguously the one we just wrote).
    service, _ = _make_service()
    result = await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.42,
        principal_id="alice",
    )
    assert result.applied_mode == "hot"

    # The next per-request ``get_settings()`` rebuild sees the overlay.
    next_settings = settings_mod.get_settings()
    assert next_settings.guardrails.input.threshold == 0.42
    assert next_settings.guardrails.input.threshold != initial_threshold

    # And the overlay is queryable via the public accessor.
    assert cs._OVERLAY["guardrails.input.threshold"] == 0.42  # noqa: SLF001


async def test_apply_edit_pending_restart_keeps_cached_value_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC4 regression: a pending-restart ``apply_edit`` does NOT change the live ``get_settings()``.

    Startup-baked keys (e.g. ``retrieval.top_k``) are intentionally NOT
    hot-reloadable: the spine's ``RetrievalConfig`` was built once at
    startup and the running request path is unchanged until restart
    (AC4).  The overlay IS written (so the value persists across a
    restart and the next post-restart ``get_settings()`` would pick it
    up via the overlay source), but the cached ``get_settings()``
    snapshot is NOT cleared and the live value stays at the pre-edit
    default.
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    _build_settings()
    settings_mod.get_settings.cache_clear()

    initial = settings_mod.get_settings()
    initial_top_k = initial.retrieval.top_k

    service, _ = _make_service()
    result = await service.apply_edit(
        key="retrieval.top_k",
        new_value=17,
        principal_id="alice",
    )
    assert result.applied_mode == "pending_restart"

    # The cached ``get_settings()`` snapshot is unchanged (AC4).
    next_settings = settings_mod.get_settings()
    assert next_settings.retrieval.top_k == initial_top_k
    # The overlay DOES hold the new value (so a post-restart load
    # picks it up via the overlay source).
    assert cs._OVERLAY["retrieval.top_k"] == 17  # noqa: SLF001
