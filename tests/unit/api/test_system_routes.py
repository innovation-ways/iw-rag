"""API tests for the System / Management power-control routes.

A fake :class:`PowerClient` is injected via the ``power_client`` kwarg on
:func:`iw_rag.api.app.create_app` — no live system-control sidecar, no Docker,
no GPU.  Covers the read-only status/VRAM pair (open to any signed-in user,
graceful degradation) and the gated power flip (power group required, read
from auth only) plus the benchmark-mode interlock.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.serving.power_client import PowerControlUnavailable


class _StubQueryService:
    """Minimal query-service seam so create_app skips the live build."""

    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover - never called
        raise RuntimeError("not used in system-route tests")


class FakePowerClient:
    """In-process fake of :class:`PowerClient` (its methods are awaited)."""

    def __init__(self) -> None:
        self.status_payload: dict[str, Any] = {
            "state": "on",
            "transition": None,
            "transition_keys": [],
            "components": [
                {
                    "key": "generator",
                    "label": "Gen",
                    "state": "on",
                    "idle_unload": True,
                    "containers": [],
                },
                {
                    "key": "embedder",
                    "label": "Embed",
                    "state": "on",
                    "idle_unload": False,
                    "containers": [],
                },
            ],
            "vram": {"available": True, "used_mb": 20000, "total_mb": 32607, "free_mb": 12607},
        }
        self.vram_payload: dict[str, Any] = {
            "available": True,
            "used_mb": 20000,
            "total_mb": 32607,
            "free_mb": 12607,
        }
        self.status_raises: Exception | None = None
        self.power_on_calls: list[Any] = []
        self.power_off_calls: list[Any] = []
        self.orchestrate_calls: list[dict[str, Any]] = []

    async def get_status(self) -> dict[str, Any]:
        if self.status_raises is not None:
            raise self.status_raises
        return self.status_payload

    async def get_vram(self) -> dict[str, Any]:
        return self.vram_payload

    async def power_on(self, components: Any = None) -> dict[str, Any]:
        self.power_on_calls.append(components)
        return self.status_payload

    async def power_off(self, components: Any = None) -> dict[str, Any]:
        self.power_off_calls.append(components)
        return {**self.status_payload, "state": "off"}

    async def orchestrate(self, *, stop: Any, start: Any) -> dict[str, Any]:
        self.orchestrate_calls.append({"stop": stop, "start": start})
        return {**self.status_payload, "state": "partial"}


def _build() -> TestClient:
    auth = StaticTokenAuthenticator.from_config(
        {"user-token": ["finance"], "op-token": ["eval-operator"]},
        operator_tokens=[],
    )
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),
        power_client=FakePowerClient(),
    )
    return TestClient(app)


def _fake(client: TestClient) -> FakePowerClient:
    return client.app.state.power_client  # type: ignore[attr-defined,return-value]


# ---------------------------------------------------------------------------
# GET /system/status
# ---------------------------------------------------------------------------


def test_status_open_to_any_authenticated_user() -> None:
    """Any signed-in caller can read status; mode + components are present."""
    client = _build()
    resp = client.get("/system/status", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["available"] is True
    assert body["mode"] == "serving"
    assert {c["key"] for c in body["components"]} == {"generator", "embedder"}


def test_status_requires_auth() -> None:
    """No bearer → 401 (auth keystone still applies to the read route)."""
    assert _build().get("/system/status").status_code == 401


def test_status_degrades_when_feature_unwired() -> None:
    """power_client None → enabled=false, HTTP 200 (never a 5xx blanking)."""
    client = _build()
    client.app.state.power_client = None  # type: ignore[attr-defined]
    body = client.get("/system/status", headers={"authorization": "Bearer user-token"}).json()
    assert body["enabled"] is False
    assert body["available"] is False
    assert "reason" in body


def test_status_degrades_when_sidecar_unreachable() -> None:
    """Sidecar down → enabled=true (feature on) but available=false, 200."""
    client = _build()
    _fake(client).status_raises = PowerControlUnavailable("connect refused")
    body = client.get("/system/status", headers={"authorization": "Bearer user-token"}).json()
    assert body["enabled"] is True
    assert body["available"] is False


# ---------------------------------------------------------------------------
# GET /system/vram
# ---------------------------------------------------------------------------


def test_vram_returns_snapshot_with_ceiling() -> None:
    """The VRAM route returns the snapshot plus the configured ceiling."""
    client = _build()
    body = client.get("/system/vram", headers={"authorization": "Bearer user-token"}).json()
    assert body["used_mb"] == 20000
    assert body["vram_ceiling_gb"] == 32.0


# ---------------------------------------------------------------------------
# POST /system/power — ACL gate + interlock
# ---------------------------------------------------------------------------


def test_power_off_requires_power_group() -> None:
    """A caller lacking the power group gets 403 and no action is taken."""
    client = _build()
    resp = client.post(
        "/system/power",
        json={"action": "off"},
        headers={"authorization": "Bearer user-token"},  # finance, not eval-operator
    )
    assert resp.status_code == 403
    assert _fake(client).power_off_calls == []


def test_power_off_succeeds_for_power_operator() -> None:
    """An eval-operator can flip the switch; the master call sends components=null."""
    client = _build()
    resp = client.post(
        "/system/power",
        json={"action": "off", "components": None},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["state"] == "off"
    assert _fake(client).power_off_calls == [None]


def test_power_on_single_component() -> None:
    """A targeted power-on proxies the component list to the sidecar."""
    client = _build()
    resp = client.post(
        "/system/power",
        json={"action": "on", "components": ["generator"]},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 200
    assert _fake(client).power_on_calls == [["generator"]]


def test_power_works_in_benchmark_mode() -> None:
    """Power control is NOT interlocked in benchmark mode — the flip proceeds."""
    client = _build()
    with patch("iw_rag.api.system_routes.resolve_mode", return_value="benchmark"):
        resp = client.post(
            "/system/power",
            json={"action": "off", "components": ["generator"]},
            headers={"authorization": "Bearer op-token"},
        )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "benchmark"
    assert _fake(client).power_off_calls == [["generator"]]


# ---------------------------------------------------------------------------
# POST /system/mode — the benchmark↔running toggle
# ---------------------------------------------------------------------------


def test_mode_switch_requires_power_group() -> None:
    """A caller without the power group cannot switch modes (403)."""
    client = _build()
    resp = client.post(
        "/system/mode",
        json={"target": "benchmark"},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403
    assert _fake(client).orchestrate_calls == []


def test_mode_switch_to_benchmark_orchestrates_and_flips() -> None:
    """Switching to benchmark stops the serving stack, starts the swap stack, flips mode."""
    client = _build()
    resp = client.post(
        "/system/mode",
        json={"target": "benchmark"},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 200
    assert resp.json()["mode"] == "benchmark"
    call = _fake(client).orchestrate_calls[0]
    assert "iw-rag-llama-swap-1" in call["start"]
    assert "iw-rag-vllm-1" in call["stop"]
    # The runtime mode is now persisted on app.state for subsequent reads.
    assert client.app.state.runtime_mode == "benchmark"  # type: ignore[attr-defined]


def test_mode_switch_to_serving_orchestrates() -> None:
    """Switching to serving starts the generator/guards, stops the swap stack."""
    client = _build()
    resp = client.post(
        "/system/mode",
        json={"target": "serving"},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 200
    call = _fake(client).orchestrate_calls[0]
    assert "iw-rag-vllm-1" in call["start"]
    assert "iw-rag-llama-swap-1" in call["stop"]
    assert resp.json()["mode"] == "serving"


def test_mode_switch_invalid_target_is_422() -> None:
    """A target outside {serving, benchmark} is rejected by validation."""
    client = _build()
    resp = client.post(
        "/system/mode",
        json={"target": "turbo"},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 422


def test_power_invalid_action_is_422() -> None:
    """An action outside {on, off} is rejected by request validation."""
    client = _build()
    resp = client.post(
        "/system/power",
        json={"action": "explode"},
        headers={"authorization": "Bearer op-token"},
    )
    assert resp.status_code == 422
