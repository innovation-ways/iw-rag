"""Unit tests for the system-control sidecar FastAPI app (power switch).

The sidecar owns the single :class:`PowerController`.  These tests drive its
routes through a ``TestClient`` against a controller built over in-memory
fakes — no Docker daemon, no GPU, no network.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from iw_rag.serving.docker_engine import ContainerState, DockerUnavailable
from iw_rag.serving.power_app import create_app
from iw_rag.serving.power_controller import PowerController
from iw_rag.serving.vram import VramSnapshot
from iw_rag.settings import PowerComponentCfg


class _FakeDocker:
    def __init__(self) -> None:
        self._state = {"vllm": True, "infinity": True}
        self.fail = False

    async def inspect(self, name: str) -> ContainerState:
        if self.fail:
            raise DockerUnavailable("down")
        if name not in self._state:
            return ContainerState(
                name=name, exists=False, running=False, status="absent", health=None
            )
        running = self._state[name]
        return ContainerState(
            name=name,
            exists=True,
            running=running,
            status="running" if running else "exited",
            health=None,
        )

    async def start(self, name: str) -> bool:
        if self.fail:
            raise DockerUnavailable("down")
        if name not in self._state:
            return False
        self._state[name] = True
        return True

    async def stop(self, name: str, *, timeout_s: int = 10) -> bool:
        if self.fail:
            raise DockerUnavailable("down")
        if name not in self._state:
            return False
        self._state[name] = False
        return True


class _FakeVram:
    def read(self) -> VramSnapshot:
        return VramSnapshot(
            available=True, used_mb=20000, free_mb=12607, total_mb=32607, device_name="RTX 5090"
        )


def _client(docker: _FakeDocker | None = None) -> TestClient:
    docker = docker or _FakeDocker()
    controller = PowerController(
        {
            "generator": PowerComponentCfg(containers=["vllm"], label="Gen"),
            "embedder": PowerComponentCfg(containers=["infinity"], label="Embed"),
        },
        docker,  # type: ignore[arg-type]
        _FakeVram(),
        ready_timeout_s=2,
        poll_interval_s=0.0,
    )
    return TestClient(create_app(controller=controller))


def test_health_does_not_touch_daemon() -> None:
    """/health is pure liveness — returns ok even with a dead daemon."""
    docker = _FakeDocker()
    docker.fail = True
    resp = _client(docker).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_returns_components_and_vram() -> None:
    """GET /power/status carries per-component state + the VRAM block."""
    resp = _client().get("/power/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "on"
    keys = {c["key"] for c in body["components"]}
    assert keys == {"generator", "embedder"}
    assert body["vram"]["used_mb"] == 20000
    assert body["vram"]["device_name"] == "RTX 5090"


def test_vram_endpoint_is_just_the_snapshot() -> None:
    """GET /power/vram returns the VRAM snapshot for the cheap gauge poll."""
    body = _client().get("/power/vram").json()
    assert body["available"] is True
    assert body["total_mb"] == 32607
    assert body["free_mb"] == 12607


def test_power_off_then_status_is_off() -> None:
    """POST /power/off stops the containers and the next status reads off."""
    client = _client()
    resp = client.post("/power/off", json={"components": None})
    assert resp.status_code == 200
    assert resp.json()["state"] == "off"


def test_power_on_single_component() -> None:
    """POST /power/on with one component starts only that component."""
    docker = _FakeDocker()
    docker._state = {"vllm": False, "infinity": False}
    client = _client(docker)
    resp = client.post("/power/on", json={"components": ["generator"]})
    assert resp.status_code == 200
    by_key = {c["key"]: c["state"] for c in resp.json()["components"]}
    assert by_key["generator"] == "on"
    assert by_key["embedder"] == "off"


def test_unknown_component_is_400() -> None:
    """An unknown component key maps to HTTP 400."""
    resp = _client().post("/power/off", json={"components": ["bogus"]})
    assert resp.status_code == 400


def test_daemon_unreachable_is_503() -> None:
    """A daemon outage mid-stop maps to HTTP 503 (not a 500 crash)."""
    docker = _FakeDocker()
    docker.fail = True
    resp = _client(docker).post("/power/off", json={})
    assert resp.status_code == 503


def test_orchestrate_stops_and_starts() -> None:
    """POST /power/orchestrate stops one set + starts another (mode-switch half)."""
    docker = _FakeDocker()
    docker._state = {"vllm": True, "infinity": True, "swap": False}
    client = _client(docker)
    resp = client.post("/power/orchestrate", json={"stop": ["vllm"], "start": ["swap"]})
    assert resp.status_code == 200
    assert docker._state["vllm"] is False
    assert docker._state["swap"] is True
