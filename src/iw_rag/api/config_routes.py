"""HTTP routes for the F-00096 Operator Console Config API (S07).

The Console is a **thin API client**; all validation, the editable-key
allowlist, the locked set, and the audit-write live server-side in the
S03 ``ConfigService`` (``iw_rag.api.config_service``).  This module is
a thin, operator-gated HTTP facade over that service — the routes
*never* open a DB session, *never* check the registry, and *never*
read the principal from the request body (Invariant 3).

**Auth-only ACL keystone.**  The ``eval-operator`` group required to
apply a config edit is read from the resolved
:class:`~iw_rag.domain.acl.AccessContext` (auth-only), NEVER from the
request body.  The :class:`ConfigUpdateRequest` body carries only
``value`` — structurally identical to :class:`QueryRequest` /
:class:`ActivateRequest` / :class:`BenchmarkRunRequest`.

**Domain-error → HTTP mapping** is delegated to the existing
``_ERROR_STATUS`` table in :mod:`iw_rag.api.app`: a locked-key edit
and an out-of-bounds value both surface as
:class:`~iw_rag.domain.errors.ValidationError` → **4xx** (the route
MUST NOT catch-and-swallow).  A non-operator principal is rejected by
the :data:`EvalOperatorCtx` dependency → **403**.  No new exception
handlers are added here.

Routes:

* ``GET /api/config`` — list active config entries + per-key
  explanation + editable/locked/pending flags.  Open to any
  authenticated principal (mirrors ``list_benchmark_reports``).
* ``PUT /api/config/{key}`` — operator-gated, validated, applied,
  audited.  Delegates to the S03 service's
  :meth:`~iw_rag.api.config_service.ConfigService.apply_edit`; the
  service is the sole source of active-value / editable / locked /
  pending semantics (AC2; design Scope "Config view + live-edit").
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from iw_rag.api.benchmark_routes import EvalOperatorCtx
from iw_rag.api.routes import AccessCtx
from iw_rag.api.schemas import (
    ConfigItemOut,
    ConfigResponse,
    ConfigUpdateRequest,
    ConfigUpdateResponse,
)
from iw_rag.config_registry import get_registry_entry
from iw_rag.domain.errors import ValidationError

logger = logging.getLogger(__name__)

config_router = APIRouter()


# ---------------------------------------------------------------------------
# Collaborator resolution (mirrors benchmark_routes._benchmark_service)
# ---------------------------------------------------------------------------


def _config_service(request: Request) -> Any:
    """Resolve the S03 :class:`ConfigService` from ``app.state``.

    Raises:
        ValidationError: When the service is not wired onto
            ``app.state.config_service`` (e.g. an unwired test
            deployment) so the existing ``_ERROR_STATUS`` table maps
            the error to 400 instead of a 5xx.  Mirrors
            :func:`iw_rag.api.benchmark_routes._benchmark_service`.
    """
    service: Any = getattr(request.app.state, "config_service", None)
    if service is None:
        raise ValidationError(
            "config service is not wired on app.state "
            "(the Console's Config API is not enabled for this deployment)."
        )
    return service


# ---------------------------------------------------------------------------
# Mapping helper — S03 ActiveConfigEntry → API ConfigItemOut
# ---------------------------------------------------------------------------


def _to_item_out(entry: Any) -> ConfigItemOut:
    """Map an S03 ``ActiveConfigEntry`` to the API response model.

    Surfaces the bounds metadata when the registry declares it
    (``min`` / ``max`` for range-style entries, ``choices`` for
    enum-style entries) so the S11 Config view can render the
    edit form's constraint hints.  For non-range / non-enum
    entries (none in the current registry, but defensive for
    future keys) the bounds fields are ``None``.

    Locked keys' ``value`` carries whatever the service decided to
    surface (e.g. the redaction marker ``"***"`` for secret-shaped
    keys); the route does not re-shape it.

    Args:
        entry: The S03 :class:`~iw_rag.api.config_service.ActiveConfigEntry`
            (or any duck-typed object with the same field shape).

    Returns:
        A :class:`ConfigItemOut` ready to serialise.
    """
    min_v: float | int | None = None
    max_v: float | int | None = None
    choices: list[str] | None = None
    # Editable entries carry a registry entry with bounds; locked
    # entries do not (the service does not look up a registry entry
    # for a locked key — the registry is the editable allowlist).
    registry = get_registry_entry(entry.key) if entry.editable else None
    if registry is not None:
        bounds = registry.bounds
        if bounds.kind in ("int_range", "float_range"):
            min_v = bounds.min
            max_v = bounds.max
        elif bounds.kind == "enum":
            choices = list(bounds.choices)
    return ConfigItemOut(
        key=str(entry.key),
        value=entry.active_value,
        explanation=str(entry.description),
        editable=bool(entry.editable),
        locked=bool(entry.locked),
        pending_restart=bool(entry.pending_restart),
        min=min_v,
        max=max_v,
        choices=choices,
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@config_router.get("/api/config", response_model=ConfigResponse)
async def get_config(
    request: Request,
    ctx: AccessCtx,
) -> ConfigResponse:
    """Return the active config: per-key values, explanations, and edit flags.

    Open to any authenticated principal — the active config list
    reveals no sensitive content because the S03 service redacts
    secret-shaped locked keys (e.g. ``database.dsn`` is surfaced as
    ``"***"``; the live DSN is never echoed).  Editable keys' active
    values are the per-request ``get_settings()`` snapshot.

    The S03 service is the sole source of active-value /
    editable / locked / pending semantics (AC2; design Scope).
    Resolved from ``app.state.config_service`` exactly like
    :func:`iw_rag.api.benchmark_routes._benchmark_service` resolves
    the F-00093 report service.

    Args:
        request: The raw FastAPI request (for app state + trace id).
        ctx: The authenticated :class:`AccessContext` (any principal).

    Returns:
        A :class:`ConfigResponse` carrying one
        :class:`ConfigItemOut` per registered key.

    Raises:
        ValidationError: When ``config_service`` is not wired on
            ``app.state`` (→ 400 via the existing ``_ERROR_STATUS``
            table in :mod:`iw_rag.api.app`).
    """
    service = _config_service(request)
    entries = service.read_active()
    return ConfigResponse(items=[_to_item_out(e) for e in entries])


@config_router.put("/api/config/{key}", response_model=ConfigUpdateResponse)
async def put_config(
    key: str,
    body: ConfigUpdateRequest,
    request: Request,
    ctx: EvalOperatorCtx,
) -> ConfigUpdateResponse:
    """Apply one config edit: validate, persist, apply, audit (eval-operator only).

    Gated to the ``eval-operator`` group (Invariant 3 / AC2 / AC7).
    The S03 service's
    :meth:`~iw_rag.api.config_service.ConfigService.apply_edit` is
    the security keystone — it rejects locked keys, out-of-bounds
    values, and ``Settings``-validation failures, all as
    :class:`ValidationError` (no overlay/audit row written).  The
    route MUST NOT catch those — let the typed errors propagate to
    the existing ``_ERROR_STATUS`` handler in :mod:`iw_rag.api.app`
    so a locked key / out-of-bounds value maps to 4xx (AC3,
    Boundary row "out of declared bounds" / "locked key") and the
    audit-row contract is preserved.

    The ``principal_id`` is sourced from the resolved
    :class:`AccessContext` (``ctx.principal_id``), never from the
    request body — the body carries ONLY ``value`` (mirrors the
    :class:`QueryRequest` / :class:`ActivateRequest` keystone).

    Args:
        key: The dotted config key (path parameter).
        body: The :class:`ConfigUpdateRequest` carrying the new
            ``value`` (no acl/principal_id field).
        request: The raw FastAPI request (for app state + trace id).
        ctx: The authenticated :class:`AccessContext` (must contain
            ``eval-operator``, verified by the
            :data:`EvalOperatorCtx` dependency; re-uses
            :func:`iw_rag.api.benchmark_routes.require_eval_operator`).

    Returns:
        A :class:`ConfigUpdateResponse` echoing the applied
        ``applied_mode`` (``"hot"`` or ``"pending_restart"``) and
        the old / new values.

    Raises:
        AuthorizationError: When the caller's groups do not include
            ``"eval-operator"`` (→ 403, raised by the dependency).
        ValidationError: When the key is locked/unknown, the value
            is out of declared bounds, or ``Settings`` validation
            fails (→ 400, raised by the S03 service).
    """
    service = _config_service(request)
    trace_id = getattr(request.state, "trace_id", "")
    result = await service.apply_edit(
        key=key,
        new_value=body.value,
        principal_id=ctx.principal_id,
        trace_id=trace_id,
    )
    return ConfigUpdateResponse(
        key=str(result.config_key),
        applied_mode=str(result.applied_mode),
        old_value=result.old_value,
        new_value=result.new_value,
    )


__all__ = [
    "config_router",
]
