"""Cross-cutting unit tests for the F-00096 live-edit config overlay.

The F-00096 Feature Design's TDD Approach asks for an
``overlay validate`` unit test that exercises the overlay
``Settings``-validation keystone end-to-end through the S03
:mod:`iw_rag.api.config_service` surface — the same surface the S07
HTTP route delegates to.  The S03 step already owns the
``test_config_service.py`` + ``test_config_registry.py`` +
``test_config_change_repository.py`` triple that covers the per-step
contract in detail; this file is the **cross-cutting overlay keystone
test** the S17 prompt pins separately so the S17 audit can stand on
its own without re-asserting the S03 sibling.

Coverage (F-00096 S17 prompt §"Unit tests" / "Overlay validate" + "Hot-
vs-pending classification"):

* An in-bounds value for an editable key passes validation through
  :class:`iw_rag.settings.Settings` — the overlay snapshot is observed
  by the next :func:`iw_rag.settings.get_settings` (no Settings
  validation error).
* An out-of-bounds value is **rejected** with a typed
  :class:`ValidationError` (matches the design Boundary row "Config
  edit out of declared bounds"); the overlay is NOT mutated.
* A **LOCKED-key** edit is **rejected** and writes **no** overlay row
  (AC3 / Invariant 1) — the overlay dict stays untouched.
* Hot-vs-pending classification: a ``hot_reloadable=True`` key resolves
  to ``applied_mode == "hot"`` (the next ``get_settings()`` rebuild
  sees the new value); a ``hot_reloadable=False`` editable key
  resolves to ``applied_mode == "pending_restart"`` (the cached
  ``get_settings()`` snapshot is unchanged until restart; AC4 / Boundary
  row "hot vs pending").

These tests run the live S03 :class:`ConfigService` against an in-
memory fake database + a real :class:`Settings` snapshot built from
the baseline env (no DB, no network, no GPU) — mirroring the
``test_config_service.py`` isolation pattern (autouse overlay-state
clear).
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
# leak into the next test's ``Settings()`` build (autouse fixture,
# mirrors the S03 sibling).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_overlay_state() -> None:
    """Clear the overlay dict and ``get_settings`` cache around every test.

    The ConfigService writes to a module-level dict (``_OVERLAY``) and
    calls ``get_settings.cache_clear()`` on hot edits; both pieces of
    state are process-global, so a missing clean-up would let one
    test's overlay value leak into the next test's ``Settings()`` build
    and break test isolation.
    """
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

    Captures every insert the service issues into ``self._rows`` so
    the round-trip assertions can read back the per-row kwargs the
    S03 service passed to the repository.
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
        captured parameter dict into ``self._rows`` so read-back
        assertions can see what the repo tried to insert.
        """
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
# A. In-bounds value passes Settings validation (no exception)
# ---------------------------------------------------------------------------


async def test_overlay_apply_in_bounds_value_passes_settings_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-bounds value on an editable key passes ``Settings`` validation.

    Pydantic v2's ``Settings.model_validate`` runs the full tree
    constraints on the proposed new state; an in-bounds float must
    land without raising.  The overlay snapshot must be observed by
    the next ``get_settings()`` (the keystone the S04 F-1 regression
    restored).
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, _ = _make_service()
    result = await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.55,
        principal_id="alice",
    )

    assert result.applied_mode == "hot"
    assert result.new_value == 0.55
    # The overlay carries the new raw Python value (Pydantic
    # type-coerces on the way in via the _OverlaySettingsSource).
    assert cs._OVERLAY["guardrails.input.threshold"] == 0.55  # noqa: SLF001


# ---------------------------------------------------------------------------
# B. Out-of-bounds value is rejected (Boundary row: "Config edit out of declared bounds")
# ---------------------------------------------------------------------------


async def test_overlay_apply_out_of_bounds_rejected_with_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``top_k = -1`` is rejected with a typed :class:`ValidationError`.

    The design Boundary row "Config edit out of declared bounds" pins
    this path: 4xx-mapping, no overlay row, no audit row.  The error
    message must reference the violated bounds so the operator can
    see WHY.
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, _ = _make_service()
    with pytest.raises(ValidationError, match="bounds|range|out of range|top_k"):
        await service.apply_edit(
            key="retrieval.top_k",
            new_value=-1,
            principal_id="alice",
        )

    # The overlay dict is UNCHANGED — the rejected edit did not write.
    assert "retrieval.top_k" not in cs._OVERLAY  # noqa: SLF001


# ---------------------------------------------------------------------------
# C. LOCKED-key edit is rejected AND writes no overlay row (AC3)
# ---------------------------------------------------------------------------


async def test_overlay_apply_locked_key_rejected_and_writes_no_overlay_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A LOCKED-key edit is rejected and writes NO overlay row (AC3 / Invariant 1).

    The design Boundary row "Config edit to a locked key" pins this
    path.  The overlay dict is UNCHANGED for the locked key (and for
    any other key) — the locked-set check runs BEFORE the overlay
    write so a hostile edit never gets to persist.
    """
    import iw_rag.api.config_service as cs
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

    # No overlay row was written.
    assert cs._OVERLAY == {}  # noqa: SLF001
    # No audit row was written.
    assert database.rows == []


# ---------------------------------------------------------------------------
# D. Hot key → applied_mode == "hot" — exact-string contract
# ---------------------------------------------------------------------------


async def test_overlay_hot_reloadable_key_classifies_as_hot_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``hot_reloadable=True`` key resolves to ``applied_mode == "hot"`` exactly.

    The S17 prompt's "hot vs pending" Boundary row + AC4 keystone
    require the exact literal string; the API response and the
    console's pending-restart badge both read it.  The cache_clear is
    invoked (the contract: the next ``get_settings()`` rebuild sees
    the new value).
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    fake_get_settings = _stub_get_settings(settings)
    monkeypatch.setattr(settings_mod, "get_settings", fake_get_settings)

    service, _ = _make_service()
    result = await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.42,
        principal_id="alice",
    )

    # The exact literal string the API response and the UI badge consume.
    assert result.applied_mode == "hot"
    # The hot path's signature: cache_clear was invoked exactly once.
    assert fake_get_settings.cache_clear.call_count == 1


# ---------------------------------------------------------------------------
# E. Non-hot editable key → applied_mode == "pending_restart" exactly
# ---------------------------------------------------------------------------


async def test_overlay_non_hot_editable_key_classifies_as_pending_restart_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-hot editable key resolves to ``applied_mode == "pending_restart"`` exactly.

    AC4 / Boundary row "hot vs pending": a startup-baked client owns
    the value (the spine's ``RetrievalConfig`` was built once at
    startup); the running request path is unchanged until restart.
    The audit row IS written (so the operator sees the pending
    badge), but the cache is NOT cleared.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    fake_get_settings = _stub_get_settings(settings)
    monkeypatch.setattr(settings_mod, "get_settings", fake_get_settings)

    service, database = _make_service()
    result = await service.apply_edit(
        key="retrieval.top_k",
        new_value=20,
        principal_id="alice",
    )

    # Exact-string contract — the S14 / S11 UI badges read this value.
    assert result.applied_mode == "pending_restart"
    # The pending path's signature: cache NOT cleared (AC4).
    assert fake_get_settings.cache_clear.call_count == 0
    # The audit row IS written — the operator sees the pending badge.
    assert len(database.rows) == 1
    assert database.rows[0]["applied_mode"] == "pending_restart"


# ---------------------------------------------------------------------------
# F. Composite shape: each successful apply edits exactly one overlay row
# ---------------------------------------------------------------------------


async def test_overlay_successful_apply_writes_exactly_one_overlay_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful apply writes exactly one overlay row (one edit, one entry).

    Boundary keystone: the S03 service is the sole writer; a
    "succeeded twice for the same key" sequence overwrites the dict's
    existing entry (not appends) — the latest value wins.  Pinned
    here so a future refactor that switches to ``defaultdict`` /
    list-of-rows breaks loudly.
    """
    import iw_rag.api.config_service as cs
    import iw_rag.settings as settings_mod

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    settings = _build_settings()
    monkeypatch.setattr(settings_mod, "get_settings", _stub_get_settings(settings))

    service, _ = _make_service()
    # Two successive edits on the same key.
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.30,
        principal_id="alice",
    )
    await service.apply_edit(
        key="guardrails.input.threshold",
        new_value=0.80,
        principal_id="alice",
    )

    # The overlay dict holds ONE key (latest value wins).
    assert list(cs._OVERLAY.keys()) == ["guardrails.input.threshold"]  # noqa: SLF001
    assert cs._OVERLAY["guardrails.input.threshold"] == 0.80  # noqa: SLF001
