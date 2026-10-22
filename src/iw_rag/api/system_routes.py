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

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from iw_rag.api.routes import AccessCtx
from iw_rag.api.system_power import resolve_mode
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, ValidationError
from iw_rag.serving.power_client import PowerClient, PowerControlUnavailable

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
    """

    action: Literal["on", "off"]
    components: list[str] | None = Field(default=None)


class SystemModeRequest(BaseModel):
    """Body of ``POST /system/mode``.

    Attributes:
        target: ``"serving"`` (chat/query, generator pinned) or
            ``"benchmark"`` (Benchmark Studio swap stack).
    """

    target: Literal["serving", "benchmark"]


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
) -> dict[str, Any]:
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
    """
    client = _power_client(request)
    if client is None:
        raise ValidationError("power control is not enabled for this deployment")
    if body.action == "on":
        status = await client.power_on(body.components)
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
) -> dict[str, Any]:
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
    status = await client.orchestrate(stop=list(plan.stop), start=list(plan.start))
    # 2. Re-point the API generator in-process (no container restart).
    if plan.generator_endpoint:
        from iw_rag.api.app import rebuild_query_service_generator

        rebuild_query_service_generator(request.app, plan.generator_endpoint)
    request.app.state.runtime_mode = body.target

    status["enabled"] = True
    status["available"] = True
    status.update(_mode_block(request))
    return status


__all__ = ["system_router"]
