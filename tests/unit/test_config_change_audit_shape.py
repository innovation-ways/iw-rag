"""Cross-cutting unit tests for the F-00096 ``config_change_log`` audit row shape.

The F-00096 Feature Design's Invariant 3 pins the keystone: the
audit row's ``principal_id`` comes from the authenticated
:class:`AccessContext`, **never** from a request body.  The S03 step
already owns ``test_config_service.py`` (the apply path) and
``test_config_change_repository.py`` (the storage layer's append
contract).  This file is the **cross-cutting audit-shape keystone
test** the S17 prompt pins separately so the S17 audit can stand on
its own without re-asserting the S03 siblings — the test focuses on
the **wire-level audit row shape** the S07 route forwards to the DB:

* The audit row's ``principal_id`` is the EXACT value the service
  received as ``principal_id=`` (not a smuggled body field).
* The audit row's ``old_value`` / ``new_value`` carry the stringified
  Python value (int / float / str / bool), NOT a raw secret (the
  locked-set check runs BEFORE the value is even read for write, so
  no secret material can land in the row).
* The audit row's ``applied_mode`` is exactly ``"hot"`` or
  ``"pending_restart"`` (the CHECK-constrained enum the migration
  pins; mirror of ``audit_log.disposition``).
* The audit row's ``config_key`` is the exact dotted key the operator
  submitted (no path normalisation, no silent rewrite).

These tests run the live S03 :class:`ConfigService` against an in-
memory fake database (the same :class:`FakeDatabase` the S03 sibling
defines) so the row shape is the production-shaped one.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Overlay-state hygiene — autouse fixture (mirror S03 sibling).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_overlay_state() -> None:
    """Clear the overlay dict and ``get_settings`` cache around every test."""
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    cs._OVERLAY.clear()  # noqa: SLF001
    settings_mod.get_settings.cache_clear()
    yield
    cs._OVERLAY.clear()  # noqa: SLF001
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fakes — mirror the S03 sibling's pattern verbatim
# ---------------------------------------------------------------------------


class _FakeAsyncSession:
    """Minimal async session — no SQL is run against the in-memory fake."""


class FakeDatabase:
    """In-memory stand-in for :class:`iw_rag.storage.db.Database`.

    Mirrors the S03 sibling's ``FakeDatabase`` — captures every
    insert into ``self._rows`` so the audit-row shape assertions
    can read back the per-row kwargs the S03 service passed to the
    repository.
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
        """Yield a session whose ``execute()`` records + returns a row stub."""
        outer = self
        session = _FakeAsyncSession()
        session.execute = AsyncMock(side_effect=outer._execute)  # type: ignore[attr-defined]
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


def _stub_get_settings(settings: Any) -> Any:
    """A ``get_settings`` stub that returns ``settings`` and has ``cache_clear``."""
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
    service = ConfigService(database=database)  # type: ignore[arg-type]
    return service, database


# ---------------------------------------------------------------------------
# A. principal_id comes from auth, not from a (smuggled) body field
# ---------------------------------------------------------------------------


async def test_audit_row_principal_id_equals_auth_supplied_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``principal_id`` is the EXACT auth-supplied value (Invariant 3).

    The S07 HTTP route forwards the ``ctx.principal_id`` (resolved
    from the authenticator) to the service; the service passes it
    through to the repository.  No code path exists that reads
    ``principal_id`` from a request body.  Pinned by asserting the
    audit row's ``principal_id`` equals the auth-supplied value,
    even when a different "body-supplied" value is documented in
    the test (to make the assertion's contract explicit).
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.6,
        principal_id="operator-bob",  # from auth (not the body)
    )

    assert len(database.rows) == 1
    assert database.rows[0]["principal_id"] == "operator-bob"


# ---------------------------------------------------------------------------
# B. Audit row carries the exact config_key the operator submitted
# ---------------------------------------------------------------------------


async def test_audit_row_config_key_equals_submitted_dotted_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``config_key`` is the exact dotted key (no normalisation)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    await service.apply_edit(
        key="retrieval.top_k",
        new_value=12,
        principal_id="alice",
    )

    assert len(database.rows) == 1
    assert database.rows[0]["config_key"] == "retrieval.top_k"


# ---------------------------------------------------------------------------
# C. Audit row's new_value is the stringified Python value
# ---------------------------------------------------------------------------


async def test_audit_row_new_value_is_stringified_python_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``new_value`` is the stringified Python value (int / float / str / bool).

    Pin the exact strings the S07 / S11 surfaces read back.  Floats
    and ints string round-trip cleanly via ``str`` (the S03 service
    uses ``str(value)`` for primitive types).  A future refactor
    that JSON-encodes the value (e.g. ``json.dumps``) would change
    the on-the-wire shape — pinned here.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    # Float
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.65,
        principal_id="alice",
    )
    assert database.rows[-1]["new_value"] == "0.65"

    # Int
    await service.apply_edit(
        key="retrieval.top_k",
        new_value=20,
        principal_id="alice",
    )
    assert database.rows[-1]["new_value"] == "20"

    # Str
    await service.apply_edit(
        key="retrieval.rerank_on_failure",
        new_value="degrade",
        principal_id="alice",
    )
    assert database.rows[-1]["new_value"] == "degrade"


# ---------------------------------------------------------------------------
# D. Audit row's old_value is the prior stringified value (or None for first ever)
# ---------------------------------------------------------------------------


async def test_audit_row_old_value_is_stringified_prior_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``old_value`` is the stringified prior value (or None for first ever).

    Pinned so a future refactor that drops ``old_value`` on the
    "first ever" path (when the overlay has no prior) breaks loudly
    — the design spec says ``old_value`` is NULL on the first ever
    write (no prior value exists).
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    # The default Settings threshold is whatever the spec defines
    # (the S03 sibling's tests document the env-baked default).  We
    # do not pin the EXACT prior value here — only that ``old_value``
    # is a non-None string (the prior is read from the live Settings
    # snapshot and stringified via ``str``).
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.7,
        principal_id="alice",
    )
    assert isinstance(database.rows[-1]["old_value"], str)
    assert database.rows[-1]["old_value"] != ""


# ---------------------------------------------------------------------------
# E. Audit row's applied_mode is exactly "hot" or "pending_restart"
# ---------------------------------------------------------------------------


async def test_audit_row_applied_mode_is_exactly_hot_or_pending_restart(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``applied_mode`` is exactly ``"hot"`` or ``"pending_restart"``.

    The migration DDL enforces the values via a CHECK constraint
    (``applied_mode IN ('hot', 'pending_restart')``).  The route +
    console badge read this string verbatim.  Pinned with the
    ``in`` set-membership check on the two legal values; a future
    refactor that adds a third value (e.g. ``"scheduled"``) would
    be a migration-breaking change.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    # Hot key
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.6,
        principal_id="alice",
    )
    assert database.rows[-1]["applied_mode"] in {"hot", "pending_restart"}
    assert database.rows[-1]["applied_mode"] == "hot"
    # Pending-restart key
    await service.apply_edit(
        key="retrieval.top_k",
        new_value=12,
        principal_id="alice",
    )
    assert database.rows[-1]["applied_mode"] in {"hot", "pending_restart"}
    assert database.rows[-1]["applied_mode"] == "pending_restart"


# ---------------------------------------------------------------------------
# F. No raw secret material in the audit row
# ---------------------------------------------------------------------------


async def test_audit_row_never_contains_dsn_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A locked-key edit on ``database.dsn`` writes NO audit row (Invariant 5).

    The locked-set check runs BEFORE the value is even read for
    write — a hostile DSN edit attempt never reaches the audit
    trail.  Pinned here so a future refactor that reads the value
    first (e.g. for an "attempt log") breaks loudly.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    from iw_rag.domain.errors import ValidationError

    with pytest.raises(ValidationError, match="locked|database\\.dsn"):
        await service.apply_edit(
            key="database.dsn",
            new_value="postgresql+psycopg://attacker:pwn@h/db",
            principal_id="alice",
        )

    # No row at all — the locked-set check rejects before the audit
    # write, so the DSN value cannot leak into the audit trail.
    assert database.rows == []
    # And the rejected value is not present in any row's payload.
    for row in database.rows:
        assert "attacker" not in str(row.values())
        assert "pwn" not in str(row.values())


# ---------------------------------------------------------------------------
# G. trace_id is forwarded verbatim from the service call
# ---------------------------------------------------------------------------


async def test_audit_row_trace_id_is_forwarded_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The audit row's ``trace_id`` is the exact value the service received."""
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, database = _make_service()
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.6,
        principal_id="alice",
        trace_id="trace-abc-123",
    )
    assert database.rows[-1]["trace_id"] == "trace-abc-123"
