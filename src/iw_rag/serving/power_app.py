"""Thin FastAPI sidecar for the VRAM power switch (system-control).

This is the **only** process that mounts the host Docker socket and reads
NVML.  It owns the single :class:`~iw_rag.serving.power_controller.PowerController`
and exposes a tiny HTTP surface the main API proxies to:

* ``GET  /health``        — liveness (does NOT touch the daemon).
* ``GET  /power/status``  — full power snapshot (components + VRAM).
* ``GET  /power/vram``    — just the VRAM snapshot (cheap gauge poll).
* ``POST /power/on``      — start components (body ``{"components": [...] }``
  or ``{}`` / ``null`` for all) and wait until ready.
* ``POST /power/off``     — stop components (or all) → frees VRAM.

**Network exposure.**  Binds to ``:8091`` on the internal Compose network
only — NEVER published to the host.  It holds the Docker socket
(host-root-equivalent), so reachability is the security boundary; auth/ACL is
the public API's responsibility (the API gates ``POST /system/power`` on the
power group *before* proxying here).  Mirrors the swap-controller sidecar.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .docker_engine import DockerUnavailable, HttpDockerEngineClient
from .power_controller import ComponentStatus, PowerController, PowerState, PowerStatus
from .vram import NvmlVramReader, VramSnapshot


class PowerActionRequest(BaseModel):
    """Body of ``POST /power/on`` and ``POST /power/off``.

    Attributes:
        components: Component keys to act on, or ``None`` / omitted for the
            master action (every managed component).
    """

    components: list[str] | None = None


class OrchestrateRequest(BaseModel):
    """Body of ``POST /power/orchestrate`` (the container half of a mode switch).

    Attributes:
        stop: Container names to stop first.
        start: Container names to start + wait to ready.
    """

    stop: list[str] = []
    start: list[str] = []


def create_app(
    *,
    controller: PowerController | None = None,
) -> FastAPI:
    """Build the system-control sidecar app.

    Args:
        controller: Inject a pre-built controller (tests).  ``None`` defers
            construction to startup, where it is built from ``Settings.power``
            + the real Docker socket client + the NVML reader.

    Returns:
        A configured ``FastAPI`` owning the single :class:`PowerController`.
    """
    app = FastAPI(title="iw-rag-system-control", version="0.0.0")
    if controller is not None:
        app.state.controller = controller

    @app.on_event("startup")
    async def _init_controller() -> None:  # pragma: no cover - exercised via ASGI in prod
        if getattr(app.state, "controller", None) is not None:
            return
        from iw_rag.settings import get_settings

        power = get_settings().power
        docker = HttpDockerEngineClient(socket_path=power.docker_socket)
        vram = NvmlVramReader()
        app.state.controller = PowerController(
            components=dict(power.components),
            docker=docker,
            vram=vram,
            ready_timeout_s=power.ready_timeout_s,
            poll_interval_s=power.poll_interval_s,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe — does NOT touch the Docker daemon."""
        return {"status": "ok"}

    @app.get("/power/status")
    async def get_status() -> dict[str, Any]:
        """Return the full power snapshot (components + VRAM)."""
        controller = _get_controller(app)
        return _status_to_dict(await controller.status())

    @app.get("/power/vram")
    async def get_vram() -> dict[str, Any]:
        """Return just the VRAM snapshot (cheap; for the live gauge poll)."""
        controller = _get_controller(app)
        status = await controller.status()
        return _vram_to_dict(status.vram)

    @app.post("/power/on")
    async def power_on(body: PowerActionRequest) -> dict[str, Any]:
        """Start the requested components (or all) and wait until ready."""
        controller = _get_controller(app)
        try:
            status = await controller.power_on(body.components)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DockerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _status_to_dict(status)

    @app.post("/power/off")
    async def power_off(body: PowerActionRequest) -> dict[str, Any]:
        """Stop the requested components (or all) → frees VRAM."""
        controller = _get_controller(app)
        try:
            status = await controller.power_off(body.components)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except DockerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _status_to_dict(status)

    @app.post("/power/orchestrate")
    async def orchestrate(body: OrchestrateRequest) -> dict[str, Any]:
        """Stop one container set + start another (the container half of a mode switch)."""
        controller = _get_controller(app)
        try:
            status = await controller.orchestrate(body.stop, body.start)
        except DockerUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _status_to_dict(status)

    return app


def _get_controller(app: FastAPI) -> PowerController:
    """Return the controller stored on ``app.state`` or raise 500."""
    ctrl = getattr(app.state, "controller", None)
    if ctrl is None:
        raise HTTPException(status_code=500, detail="system-control not initialised")
    return ctrl  # type: ignore[no-any-return]


def _status_to_dict(status: PowerStatus) -> dict[str, Any]:
    """Serialise a :class:`PowerStatus` to a JSON-friendly dict."""
    return {
        "state": status.state.value if isinstance(status.state, PowerState) else str(status.state),
        "transition": status.transition,
        "transition_keys": list(status.transition_keys),
        "components": [_component_to_dict(c) for c in status.components],
        "vram": _vram_to_dict(status.vram),
    }


def _component_to_dict(c: ComponentStatus) -> dict[str, Any]:
    """Serialise a :class:`ComponentStatus` to a JSON-friendly dict."""
    return {
        "key": c.key,
        "label": c.label,
        "description": c.description,
        "idle_unload": c.idle_unload,
        "present": c.present,
        "running": c.running,
        "state": c.state,
        "containers": [
            {
                "name": s.name,
                "exists": s.exists,
                "running": s.running,
                "status": s.status,
                "health": s.health,
            }
            for s in c.containers
        ],
    }


def _vram_to_dict(v: VramSnapshot) -> dict[str, Any]:
    """Serialise a :class:`VramSnapshot` to a JSON-friendly dict."""
    return {
        "available": v.available,
        "used_mb": v.used_mb,
        "free_mb": v.free_mb,
        "total_mb": v.total_mb,
        "device_name": v.device_name,
        "processes": [{"pid": p.pid, "used_mb": p.used_mb, "name": p.name} for p in v.processes],
    }
