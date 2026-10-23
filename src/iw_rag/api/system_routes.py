"""HTTP routes for the System / Management view — the VRAM power switch.

Public surface for power control: a read-only status/VRAM pair open to any
authenticated principal, and a power flip gated to the configured power group.
The main API is a **thin facade** that proxies to the internal-only
``system-control`` sidecar (``:8091``) via
:class:`~iw_rag.serving.power_client.PowerClient`; the API never touches the
Docker socket.  The ACL gate runs **before** the proxy call (the sidecar is
unauthenticated, so this is the single choke point).

**Benchmark interlock.**  In benchmark mode llama-swap owns the GPU and the
swap-controller is the power authority — so ``POST /system/power`` refuses
(400) with a clear message and the idle monitor stands down.  The status route
still answers (so the console can render the "benchmark mode" banner).

**Graceful degradation.**  When power control is not wired (no sidecar) or the
sidecar is unreachable, the status/VRAM routes return HTTP 200 with
``enabled``/``available`` flags set false and ``unreachable`` markers — never a
5xx that blanks the console (mirrors ``GET /models`` Boundary row 3).
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from iw_rag.api.routes import AccessCtx
from iw_rag.api.system_power import resolve_mode
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, ValidationError
from iw_rag.serving.power_client import PowerCapacityError, PowerClient, PowerControlUnavailable

system_router = APIRouter()


# ---------------------------------------------------------------------------
# Request body
# ---------------------------------------------------------------------------


class SystemPowerRequest(BaseModel):
    """Body of ``POST /system/power``.

    Attributes:
        action: ``"on"`` to start, ``"off"`` to stop.
        components: Component keys to act on, or ``None`` for the master
            action (every managed component).
        force: Operator override — bypass the capacity pre-check (``on`` only).
    """

    action: Literal["on", "off"]
    components: list[str] | None = Field(default=None)
    force: bool = False


class SystemModeRequest(BaseModel):
    """Body of ``POST /system/mode``.

    Attributes:
        target: ``"serving"`` (chat/query, generator pinned) or
            ``"benchmark"`` (Benchmark Studio swap stack).
        force: Operator override — bypass the capacity pre-check.
    """

    target: Literal["serving", "benchmark"]
    force: bool = False


class SystemProfileRequest(BaseModel):
    """Body of ``POST /system/profile``.

    Attributes:
        name: Operating profile key (``normal`` / ``light`` / ``yield`` by default).
        force: Operator override — bypass the capacity pre-check.
    """

    name: str
    force: bool = False


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def _power_client(request: Request) -> PowerClient | None:
    """Return the wired power client, or ``None`` (feature not enabled)."""
    return getattr(request.app.state, "power_client", None)


def _power_group(request: Request) -> str:
    """Return ``settings.power.power_group`` (default ``"eval-operator"``)."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return "eval-operator"
    return getattr(getattr(settings, "power", None), "power_group", "eval-operator")


async def require_power_operator(request: Request, ctx: AccessCtx) -> AccessContext:
    """Require the configured power group in the caller's ACL (→ 403 otherwise).

    Mirrors :func:`iw_rag.api.models_routes.require_eval_operator`: groups come
    from the resolved :class:`AccessContext` only (auth keystone — never the
    request body).

    Raises:
        AuthorizationError: When the caller lacks the configured power group.
    """
    needed = _power_group(request)
    if needed not in ctx.principal_groups:
        raise AuthorizationError(f"'{needed}' group required to change system power")
    return ctx


PowerOperatorCtx = Annotated[AccessContext, Depends(require_power_operator)]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mode_block(request: Request) -> dict[str, Any]:
    """Return the mode block the console renders (mode badge + toggle state)."""
    settings = getattr(request.app.state, "settings", None)
    mode = resolve_mode(request.app.state)
    serving = mode == "serving"
    power = getattr(settings, "power", None) if settings is not None else None
    return {
        "mode": mode,
        # Power control works in BOTH modes now (the generator component
        # resolves the serving OR benchmark generator container).
        "power_control_active": True,
        "mode_switch_enabled": bool(getattr(power, "mode_switch_enabled", False)),
        # Idle auto-unload only applies in serving mode (in benchmark mode
        # llama-swap owns the generator lifecycle).
        "idle_unload_enabled": bool(getattr(power, "idle_unload_enabled", False)) and serving,
        "idle_unload_minutes": int(getattr(power, "idle_unload_minutes", 0)),
        "block_and_warm": bool(getattr(power, "block_and_warm", False)),
        "vram_ceiling_gb": float(getattr(power, "vram_ceiling_gb", 32.0)),
    }


def _power_cfg(request: Request) -> Any:
    """Return ``settings.power`` or ``None`` when settings are not wired."""
    settings = getattr(request.app.state, "settings", None)
    return getattr(settings, "power", None) if settings is not None else None


def _active_profile(request: Request) -> str:
    """Return the active operating profile key (runtime choice, else configured default)."""
    chosen = getattr(request.app.state, "power_profile", None)
    if chosen:
        return str(chosen)
    power = _power_cfg(request)
    return str(getattr(power, "active_profile", "normal"))


def _record_forced(request: Request, detail: str) -> None:
    """Log an operator capacity-pre-check bypass to the event log (if wired)."""
    events = getattr(request.app.state, "power_events", None)
    if events is not None:
        events.add("forced", detail=detail)


def _capacity_409(exc: PowerCapacityError) -> JSONResponse:
    """Render a sidecar capacity refusal as the console API's HTTP 409."""
    return JSONResponse(status_code=409, content={**exc.body, "error": "insufficient_vram"})


def _yield_block(request: Request) -> dict[str, Any]:
    """Return the reactive-yield settings + recent events for the System view."""
    power = _power_cfg(request)
    events = getattr(request.app.state, "power_events", None)
    return {
        "enabled": bool(getattr(power, "yield_enabled", False)),
        "threshold_gb": float(getattr(power, "yield_free_gb_threshold", 0.0)),
        "sustain_s": float(getattr(power, "yield_sustain_s", 0.0)),
        "margin_gb": float(getattr(power, "capacity_margin_gb", 0.0)),
        "recent_events": events.recent(20) if events is not None else [],
    }


def _disabled_payload(request: Request, reason: str) -> dict[str, Any]:
    """Status payload when power control is not wired (HTTP 200, not 5xx)."""
    return {
        "enabled": False,
        "available": False,
        "reason": reason,
        "state": "unknown",
        "components": [],
        "vram": {"available": False},
        **_mode_block(request),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@system_router.get("/system/status")
async def system_status(request: Request, ctx: AccessCtx) -> dict[str, Any]:
    """Return the full power snapshot + mode (open to any authenticated user).

    Degrades to ``enabled=false`` / ``available=false`` (HTTP 200) when the
    feature is off or the sidecar is unreachable, so the console always
    renders rather than erroring.
    """
    client = _power_client(request)
    if client is None:
        return _disabled_payload(request, "power control is not enabled for this deployment")
    try:
        status = await client.get_status()
    except PowerControlUnavailable as exc:
        payload = _disabled_payload(request, f"system-control unreachable: {exc}")
        payload["enabled"] = True  # feature IS enabled; the sidecar is just down
        return payload
    status["enabled"] = True
    status["available"] = True
    status.update(_mode_block(request))
    status["profile"] = _active_profile(request)
    status["yield"] = _yield_block(request)
    other_mb = status.get("estimated_other_mb")
    status["estimated_other_gb"] = None if other_mb is None else round(other_mb / 1024, 1)
    return status


@system_router.get("/system/vram")
async def system_vram(request: Request, ctx: AccessCtx) -> dict[str, Any]:
    """Return just the VRAM snapshot (cheap poll for the live gauge)."""
    client = _power_client(request)
    if client is None:
        return {"available": False, "reason": "power control is not enabled"}
    try:
        vram = await client.get_vram()
    except PowerControlUnavailable as exc:
        return {"available": False, "reason": f"system-control unreachable: {exc}"}
    vram["vram_ceiling_gb"] = _mode_block(request)["vram_ceiling_gb"]
    return vram


@system_router.post("/system/power")
async def system_power(
    body: SystemPowerRequest,
    request: Request,
    ctx: PowerOperatorCtx,
) -> Any:
    """Flip system power — gated to the configured power group.

    Works in BOTH modes: in serving mode it stops/starts the pinned
    ``iw-rag-vllm-1`` generator; in benchmark mode the generator component
    resolves (via the ``vllm-*`` pattern) to llama-swap's resident container,
    so the operator can free that VRAM too.  The ACL check runs **before** the
    proxy call.

    Raises:
        AuthorizationError: Caller lacks the power group (→ 403).
        ValidationError: Feature not wired (→ 400).
        PowerControlUnavailable: Sidecar unreachable (→ 503).

    A start that would not fit in the free VRAM is refused with HTTP 409
    (structured ``insufficient_vram`` body) unless the operator sets ``force``.
    """
    client = _power_client(request)
    if client is None:
        raise ValidationError("power control is not enabled for this deployment")
    if body.action == "on":
        try:
            if body.force:
                _record_forced(request, f"power on {body.components or 'all'} forced")
                status = await client.power_on(body.components, force=True)
            else:
                status = await client.power_on(body.components)
        except PowerCapacityError as exc:
            return _capacity_409(exc)
    else:
        status = await client.power_off(body.components)
    status["enabled"] = True
    status["available"] = True
    status.update(_mode_block(request))
    return status


@system_router.post("/system/mode")
async def system_mode(
    body: SystemModeRequest,
    request: Request,
    ctx: PowerOperatorCtx,
) -> Any:
    """Switch the box between serving (chat) and benchmark mode (gated).

    The in-UI equivalent of ``make serving-mode`` / ``make benchmark-mode``:
    the sidecar does the container dance (stop the outgoing stack, start the
    incoming one), then the API re-points its generator at the target mode's
    endpoint **in-process** (no container restart).  Switching to ``serving``
    starts the pinned generator — on a GPU already full from other workloads
    that start can OOM; the response then reflects the generator as not-ready.

    Raises:
        AuthorizationError: Caller lacks the power group (→ 403).
        ValidationError: Feature/mode-switch not wired, or unknown target (→ 400).
        PowerControlUnavailable: Sidecar unreachable (→ 503).
    """
    client = _power_client(request)
    if client is None:
        raise ValidationError("power control is not enabled for this deployment")
    settings = getattr(request.app.state, "settings", None)
    power = getattr(settings, "power", None) if settings is not None else None
    if power is None or not getattr(power, "mode_switch_enabled", False):
        raise ValidationError("mode switching is disabled for this deployment")
    plan = power.modes.get(body.target)
    if plan is None:
        raise ValidationError(f"no mode plan configured for '{body.target}'")

    # 1. Container dance (sidecar): stop the outgoing stack, start the incoming.
    try:
        if body.force:
            _record_forced(request, f"mode switch to {body.target} forced")
            status = await client.orchestrate(
                stop=list(plan.stop), start=list(plan.start), force=True
            )
        else:
            status = await client.orchestrate(stop=list(plan.stop), start=list(plan.start))
    except PowerCapacityError as exc:
        return _capacity_409(exc)
    # 2. Re-point the API generator in-process (no container restart).
    if plan.generator_endpoint:
        from iw_rag.api.app import rebuild_query_service_generator

        rebuild_query_service_generator(request.app, plan.generator_endpoint)
    request.app.state.runtime_mode = body.target

    status["enabled"] = True
    status["available"] = True
    status.update(_mode_block(request))
    return status


@system_router.get("/system/history")
async def system_history(
    request: Request,
    ctx: AccessCtx,
    window_s: Annotated[float, Query(gt=0, le=86_400)] = 3600.0,
) -> dict[str, Any]:
    """Return the VRAM time series merged with GPU-aware events (any auth).

    Degrades to an empty series (HTTP 200) when power control is off or the
    sidecar is unreachable, so the chart shows "no data yet" instead of erroring.
    """
    events = getattr(request.app.state, "power_events", None)
    recent = [e for e in (events.recent(100) if events is not None else [])]
    client = _power_client(request)
    if client is None:
        return {"available": False, "samples": [], "events": recent, "interval_s": None}
    try:
        hist = await client.get_history(window_s)
    except PowerControlUnavailable:
        return {"available": False, "samples": [], "events": recent, "interval_s": None}
    samples = hist.get("samples", [])
    if samples:
        start = samples[0]["ts"]
        recent = [e for e in recent if e["ts"] >= start]
    return {
        "available": True,
        "samples": samples,
        "events": recent,
        "interval_s": hist.get("interval_s"),
    }


@system_router.get("/system/profiles")
async def system_profiles(request: Request, ctx: AccessCtx) -> dict[str, Any]:
    """List the operating profiles with their VRAM budgets (any auth)."""
    power = _power_cfg(request)
    if power is None or _power_client(request) is None:
        return {"enabled": False, "active": None, "profiles": []}
    active = _active_profile(request)
    items = [
        {
            "name": name,
            "label": prof.label,
            "description": prof.description,
            "budget_gb": sum(power.components[k].vram_gb for k in prof.on if k in power.components),
            "on": list(prof.on),
            "off": list(prof.off),
            "active": name == active,
        }
        for name, prof in power.profiles.items()
    ]
    return {"enabled": True, "active": active, "profiles": items}


@system_router.post("/system/profile")
async def system_profile(
    body: SystemProfileRequest,
    request: Request,
    ctx: PowerOperatorCtx,
) -> Any:
    """Apply an operating profile — gated to the configured power group.

    The sidecar stops what the profile releases, runs the capacity pre-check
    against the projected free memory and starts the rest; a profile that
    would not fit is refused with HTTP 409 (unless ``force``).

    Raises:
        AuthorizationError: Caller lacks the power group (→ 403).
        ValidationError: Feature not wired, or unknown profile (→ 400).
        PowerControlUnavailable: Sidecar unreachable (→ 503).
    """
    client = _power_client(request)
    power = _power_cfg(request)
    if client is None or power is None:
        raise ValidationError("power control is not enabled for this deployment")
    if body.name not in power.profiles:
        raise ValidationError(f"unknown profile '{body.name}'; known: {sorted(power.profiles)}")
    try:
        if body.force:
            _record_forced(request, f"profile {body.name} forced")
            status = await client.apply_profile(body.name, force=True)
        else:
            status = await client.apply_profile(body.name)
    except PowerCapacityError as exc:
        return _capacity_409(exc)
    request.app.state.power_profile = body.name
    status["enabled"] = True
    status["available"] = True
    status.update(_mode_block(request))
    status["profile"] = body.name
    return status


@system_router.get("/system/capacity")
async def system_capacity(
    request: Request,
    ctx: AccessCtx,
    components: str | None = None,
) -> dict[str, Any]:
    """Dry-run the capacity pre-check for starting ``components`` (comma list; any auth).

    Degrades to ``{"ok": True, "skipped": "unavailable"}`` when power control is
    off or unreachable so the UI never blocks on a missing check.
    """
    client = _power_client(request)
    if client is None:
        return {"ok": True, "skipped": "unavailable"}
    keys = [k for k in components.split(",") if k] if components else None
    try:
        return await client.get_capacity(keys)
    except PowerControlUnavailable:
        return {"ok": True, "skipped": "unavailable"}


__all__ = ["system_router"]
