"""ConfigService — the live-edit overlay service for the Operator Console (F-00096, S03).

The service is the **security keystone** of the Console's Config view: it
implements the F-00096 §Invariants 1, 2, 3, 5 contract end-to-end:

* Invariant 1 — locked keys can NEVER be edited (rejected server-side,
  before any overlay/audit row is written).
* Invariant 2 — the audit row is append-only (the repository exposes
  only ``append``; no UPDATE/DELETE).
* Invariant 3 — ``principal_id`` on the audit row is sourced from the
  authenticated :class:`AccessContext` (``principal_id=`` kwarg), never
  from a request body field.
* Invariant 5 — applying an overlay never disables a fail-closed/ACL/
  local-only invariant (the registry's locked set covers
  ``guardrails.enabled``, guard/judge/model endpoints, the DSN, and the
  local-only switches).

The service has two public methods:

* :meth:`read_active` — returns the active values + per-key description
  + ``editable|locked`` + ``pending_restart`` flag for each registry
  key (the GET payload shape S07 serialises).
* :meth:`apply_edit` — validates + persists + applies (hot via
  ``get_settings.cache_clear()`` or flagged ``pending_restart`` for
  startup-baked keys) + writes the audit row.

This module imports the registry (no I/O) and the repository (DB),
but the service itself is pure orchestration: it does not own a
session — every session is opened per call from the injected
``database`` (``session()`` async context manager) so the service is
safe to hold as a long-lived singleton on ``app.state``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import ValidationError as PydanticValidationError

from iw_rag import settings as settings_module
from iw_rag.config_registry import (
    classify_key,
    get_registry_entry,
    iter_editable_entries,
    iter_locked_keys,
)
from iw_rag.domain.errors import ValidationError
from iw_rag.settings import Settings, set_overlay_source
from iw_rag.storage.repositories import ConfigChangeRepository


def _get_settings() -> Settings:
    """Return the current :class:`Settings` (looked up via the module).

    Lookup is done via :mod:`iw_rag.settings` at call time so tests can
    patch the module attribute and the service picks up the patch
    (the service module imports :mod:`iw_rag.settings`, not the bare
    ``get_settings`` symbol).
    """
    return settings_module.get_settings()


# ---------------------------------------------------------------------------
# Read-active shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ActiveConfigEntry:
    """One row in the GET /api/config payload.

    Attributes:
        key: Dotted config key.
        description: One-line operator-facing description.
        active_value: The current value (stringified for display).
        editable: True iff the key is in the allowlist AND not locked.
        locked: True iff the key is in the locked set.
        pending_restart: True iff the entry is editable but a
            startup-baked client owns it (the running request path is
            unchanged until restart; the edit will apply after restart).
    """

    key: str
    description: str
    active_value: Any
    editable: bool
    locked: bool
    pending_restart: bool


# ---------------------------------------------------------------------------
# Apply-edit result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApplyEditResult:
    """The result of a successful ``apply_edit`` call.

    Attributes:
        config_key: The dotted key that was edited.
        old_value: The prior value (raw type, e.g. ``int`` /
            ``float`` / ``str`` — captured from the active
            :class:`Settings` before the apply).
        new_value: The new value that was applied (raw type, as
            supplied by the operator).
        applied_mode: ``"hot"`` (applied to the next request) or
            ``"pending_restart"`` (startup-baked client owns the key).
        principal_id: The authenticated principal id that requested the
            edit (echoed back for the response body; never sourced from
            the request body — Invariant 3).
    """

    config_key: str
    old_value: Any
    new_value: Any
    applied_mode: Literal["hot", "pending_restart"]
    principal_id: str


# ---------------------------------------------------------------------------
# In-memory overlay (S03) — the durable store the next ``get_settings()``
# load reads from.  Holds the per-key RAW Python values (int / float /
# str / bool) written by successful apply_edit calls (locked-key /
# out-of-bounds / unknown rejects do not write here).
#
# The overlay is consulted by :class:`iw_rag.settings.Settings` via the
# ``_OverlaySource`` registered in :mod:`iw_rag.settings` — that source
# calls :func:`get_overlay_init_kwargs` at ``Settings()`` construction
# time, which converts the dotted keys into a nested init-kwargs dict
# the model validates (Pydantic type-coerces the values into the field
# types on the way in, so a stored ``0.7`` lands as ``float`` on
# ``Settings.guardrails.input.threshold``).
#
# S17 / a follow-up may replace this with a Postgres overlay table or an
# env-file the settings loader consults (the design §3 step 4 mentions
# both as options).  The contract that other layers see is the
# ``_OVERLAY`` dict; the backend can swap without changing callers.
# ---------------------------------------------------------------------------

_OVERLAY: dict[str, Any] = {}


def get_overlay_init_kwargs() -> dict[str, Any]:
    """Return the overlay as a nested ``{top_field: {nested: value}}`` init-kwargs dict.

    The :class:`iw_rag.settings.Settings` overlay source calls this at
    construction time.  Each dotted key in :data:`_OVERLAY` is split
    on ``"."`` and assembled into a nested dict whose top-level key
    matches a :class:`Settings` field (e.g. ``"guardrails.input.threshold"``
    becomes ``{"guardrails": {"input": {"threshold": 0.7}}}``).  Pydantic
    then type-coerces the values and validates the tree, so a stored
    raw ``int`` / ``float`` / ``str`` / ``bool`` lands on the correct
    field type.

    Returns:
        A nested init-kwargs dict.  Empty when the overlay has no
        successful edits.
    """
    nested: dict[str, Any] = {}
    for dotted_key, value in _OVERLAY.items():
        parts = dotted_key.split(".")
        leaf: dict[str, Any] = {parts[-1]: value}
        for part in reversed(parts[:-1]):
            leaf = {part: leaf}
        # Deep-merge with anything already assembled for a sibling key
        # (two dotted keys sharing a top-level prefix — e.g. both
        # ``guardrails.input.threshold`` and ``guardrails.output.x``
        # — are merged into a single ``{"guardrails": {...}}`` entry).
        _deep_merge_into(nested, leaf)
    return nested


def _deep_merge_into(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """In-place deep-merge of ``overlay`` into ``base``.

    Like :func:`_deep_merge` but mutates ``base``.  List values are
    replaced wholesale (no element-wise merge).
    """
    for k, v in overlay.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge_into(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stringify(value: Any) -> str:
    """Stringify a value for the audit row + active-value display.

    Uses ``repr`` for non-primitives so a str-equality test on the
    audit row's ``old_value``/``new_value`` is stable.  Floats and ints
    string round-trip cleanly via ``str``.

    Args:
        value: A primitive value (int, float, str, bool, None).

    Returns:
        A stable string representation.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _read_dotted(settings: Settings, key: str) -> Any:
    """Read the value at ``key`` from ``settings`` (e.g. ``"retrieval.top_k"``).

    Args:
        settings: The current :class:`Settings` instance.
        key: Dotted config key.

    Returns:
        The current value at that path.
    """
    obj: Any = settings
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj


def _build_nested_update(dotted_key: str, value: Any) -> dict[str, Any]:
    """Build a nested update dict for ``Settings.model_validate``.

    Translates ``"retrieval.top_k"`` → ``{"retrieval": {"top_k": value}}``
    so the caller can merge it into a flat ``model_dump()`` before
    re-validating.

    Args:
        dotted_key: Dotted config key.
        value: The new value to set at that path.

    Returns:
        A nested dict suitable for shallow-merge with ``model_dump()``.
    """
    parts = dotted_key.split(".")
    leaf: dict[str, Any] = {parts[-1]: value}
    for part in reversed(parts[:-1]):
        leaf = {part: leaf}
    return leaf


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base`` and return a new dict.

    Pure function — does not mutate either input.  List values are
    replaced wholesale (no element-wise merge).

    Args:
        base: The base dict (typically a ``model_dump()`` output).
        overlay: The overlay dict (typically ``_build_nested_update(...)``).

    Returns:
        A new dict with overlay keys replacing base keys.
    """
    result = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# ConfigService
# ---------------------------------------------------------------------------


# Default accessor — tests inject a stub.
SettingsGetter = Callable[[], Settings]


class ConfigService:
    """The overlay service: validate, persist, apply, audit.

    Constructor-injected ``database`` (``session()`` async context
    manager) and an optional ``settings_getter`` callable (default: the
    module-level :func:`iw_rag.settings.get_settings`).  Tests inject
    fakes for both.
    """

    def __init__(
        self,
        database: Any,
        *,
        settings_getter: SettingsGetter | None = None,
    ) -> None:
        """Bind the service to a database + a settings accessor.

        Args:
            database: An object exposing a ``session()`` async context
                manager that yields an ``AsyncSession`` bound to the
                ``config_change_log`` table.
            settings_getter: Optional callable returning the current
                :class:`Settings`.  Defaults to :func:`_get_settings`,
                which resolves :func:`iw_rag.settings.get_settings` at
                call time so tests can patch the module attribute.
        """
        self._database = database
        self._settings_getter: SettingsGetter = settings_getter or _get_settings

    # -------------------------------------------------------------------
    # read_active
    # -------------------------------------------------------------------

    def read_active(self) -> list[ActiveConfigEntry]:
        """Return the active values + per-key metadata for every registry key.

        Walks the editable allowlist (registry) and the locked set
        (frozenset).  Each entry is one :class:`ActiveConfigEntry`; the
        S07 route serialises this list verbatim as the GET
        ``/api/config`` payload.

        Locked keys are included with their ``description`` and
        ``active_value`` from the current ``Settings`` (e.g. the
        guardrails boolean), but ``active_value`` is redacted for
        secret-shaped keys (DSN, download token secret) — secrets
        are surface-only in the locked set, never copied into the
        audit trail.

        Synchronous: no DB or network I/O is performed, only a read
        of the in-process :class:`Settings` snapshot.

        Returns:
            A list of :class:`ActiveConfigEntry`, one per registered
            key.
        """
        settings = self._settings_getter()
        out: list[ActiveConfigEntry] = []
        # Editable allowlist first (gives the operator the edit form).
        for key, entry in iter_editable_entries():
            if classify_key(key) != "editable":
                # Defensive: should not happen (locked-wins inside the
                # registry), but if a future maintainer adds a key to
                # both sets, classify wins.
                continue
            active = self._safe_read(settings, key)
            out.append(
                ActiveConfigEntry(
                    key=key,
                    description=entry.description,
                    active_value=active,
                    editable=True,
                    locked=False,
                    pending_restart=not entry.hot_reloadable,
                )
            )
        # Locked set (read-only entries).
        for key in iter_locked_keys():
            if key == "models":
                # Parent prefix covering models.* — skip as a
                # standalone row; sub-rows would enumerate every
                # registry entry.  The S07 view can render a single
                # "model registry (locked)" group instead.
                continue
            out.append(
                ActiveConfigEntry(
                    key=key,
                    description="(locked) — security-critical / infra; not editable.",
                    active_value=self._safe_read(settings, key),
                    editable=False,
                    locked=True,
                    pending_restart=False,
                )
            )
        return out

    def _safe_read(self, settings: Settings, key: str) -> Any:
        """Read the value at ``key`` from ``settings``; return ``"***"`` on failure.

        Some locked keys (the DSN, the download-token secret) are
        secrets — the read path returns a redaction marker so the
        Console never echoes the secret value into the response.

        Args:
            settings: The current :class:`Settings` instance.
            key: Dotted config key.

        Returns:
            The value, or the redaction marker ``"***"`` when the
            key resolves to a secret.
        """
        if key in _REDACTED_KEYS:
            return "***"
        try:
            return _read_dotted(settings, key)
        except AttributeError:
            return None

    # -------------------------------------------------------------------
    # apply_edit
    # -------------------------------------------------------------------

    async def apply_edit(
        self,
        *,
        key: str,
        new_value: Any,
        principal_id: str,
        trace_id: str | None = None,
    ) -> ApplyEditResult:
        """Validate + persist + apply one config edit.

        Flow (F-00096 design §3, AC2/AC3/AC4):

        1. Reject if ``key`` is in the LOCKED set or ``unknown`` →
           :class:`ValidationError`.  No overlay row, no audit row.
        2. Reject if ``new_value`` is outside the registry ``bounds`` →
           :class:`ValidationError`.  No rows.
        3. Validate the proposed value through ``Settings`` (construct
           a cloned ``Settings`` with the overlay applied, let pydantic
           validate).  On failure → :class:`ValidationError`, no rows.
        4. Persist the edit to the in-memory overlay.
        5. If the registry entry is ``hot_reloadable``: clear
           ``get_settings.cache_clear()`` so the next per-request
           ``get_settings()`` returns the new value → ``applied_mode
           = "hot"`` (AC2).  Else: ``applied_mode = "pending_restart"``
           (AC4); do not clear the cache.
        6. Append a ``config_change_log`` row via
           :class:`ConfigChangeRepository.append`.

        Args:
            key: Dotted config key (e.g. ``"retrieval.top_k"``).
            new_value: The new value (must satisfy the registry bounds
                AND ``Settings`` validation).
            principal_id: The authenticated principal id (from
                :class:`AccessContext`; never from a request body).
            trace_id: Optional request correlation id.

        Returns:
            The :class:`ApplyEditResult` for the response body.

        Raises:
            ValidationError: When the key is locked/unknown, the value
                is out of bounds, or ``Settings`` validation fails.
        """
        settings = self._settings_getter()

        # --- Step 1: classify the key --------------------------------
        classification = classify_key(key)
        if classification == "locked":
            raise ValidationError(
                f"Config key '{key}' is locked — it cannot be edited. "
                "Locked keys cover security/infra invariants "
                "(F-00096 Invariant 5)."
            )
        if classification == "unknown":
            raise ValidationError(
                f"Config key '{key}' is not editable (not in the registry). "
                "Only allowlisted RAG-behaviour keys can be edited via the overlay."
            )

        entry = get_registry_entry(key)
        assert entry is not None  # classify_key said editable

        # --- Step 2: registry bounds check ----------------------------
        if not entry.bounds.check(new_value):
            raise ValidationError(
                f"Value for '{key}' is out of declared bounds "
                f"(kind={entry.bounds.kind}, "
                f"min={entry.bounds.min}, max={entry.bounds.max}, "
                f"choices={list(entry.bounds.choices) or None})."
            )

        # --- Step 3: validate through Settings -----------------------
        old_value = _read_dotted(settings, key)
        new_dump = _deep_merge(
            settings.model_dump(),
            _build_nested_update(key, new_value),
        )
        try:
            Settings.model_validate(new_dump)
        except PydanticValidationError as exc:
            raise ValidationError(f"Value for '{key}' failed Settings validation: {exc}") from exc

        # --- Step 4: persist to overlay -------------------------------
        # Store the raw Python value (int / float / str / bool).  The
        # ``_OverlaySource`` registered in ``iw_rag.settings`` reads
        # :data:`_OVERLAY` at ``Settings()`` construction time and
        # Pydantic type-coerces on the way in; the audit row stringifies
        # explicitly at step 6 (we don't want to lose a round-trip via
        # string storage for a value that started as int / float / bool).
        _OVERLAY[key] = new_value

        # --- Step 5: hot vs pending_restart ---------------------------
        if entry.hot_reloadable:
            # AC2 — clear the lru_cache so the next get_settings() call
            # rebuilds from env+yaml+overlay and picks up the new value.
            # Looked up on the module at call time so a test that
            # patches ``iw_rag.settings.get_settings`` with a mock
            # also gets its ``cache_clear`` invoked.
            settings_module.get_settings.cache_clear()
            applied_mode: Literal["hot", "pending_restart"] = "hot"
        else:
            # AC4 — do NOT clear the cache; the running request path is
            # unchanged until restart.  The audit row is still written
            # so the operator sees the pending-restart badge.
            applied_mode = "pending_restart"

        # --- Step 6: audit row ----------------------------------------
        new_value_str = _stringify(new_value)
        old_value_str = _stringify(old_value) if old_value is not None else None
        async with self._database.session() as session:
            repo = ConfigChangeRepository(session)
            await repo.append(
                principal_id=principal_id,
                config_key=key,
                old_value=old_value_str,
                new_value=new_value_str,
                applied_mode=applied_mode,
                trace_id=trace_id,
            )
            await session.commit()

        return ApplyEditResult(
            config_key=key,
            old_value=old_value,
            new_value=new_value,
            applied_mode=applied_mode,
            principal_id=principal_id,
        )


# ---------------------------------------------------------------------------
# Keys whose active value must never be surfaced (redaction markers).
# ---------------------------------------------------------------------------
# The locked set is the security gate; the read path additionally
# redacts the value of secret-shaped keys so a stolen browser
# session or a careless log dump cannot leak the secret into the
# Config view.  The DSN and the download-token secret are the two
# well-known env-only secrets carried by :class:`Settings`.
_REDACTED_KEYS: frozenset[str] = frozenset(
    {
        "database.dsn",
        "api.download_token_secret",
    }
)


# ---------------------------------------------------------------------------
# Register the overlay provider with iw_rag.settings (F-00096 AC2).
# ---------------------------------------------------------------------------
# Done at import time so any later ``Settings()`` construction (incl.
# the one triggered by the service's ``cache_clear()`` after a hot
# edit) picks up the overlay values.  Tests can ``set_overlay_source(None)``
# to temporarily bypass the overlay, or call ``set_overlay_source(...)``
# again with a custom provider.
set_overlay_source(get_overlay_init_kwargs)
