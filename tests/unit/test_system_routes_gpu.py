"""API tests for the GPU-aware System routes and sidecar endpoints (F-00098).

Fake :class:`PowerClient` injected into ``create_app``; the sidecar app is driven
with a fake controller.  No Docker, no GPU, no sidecar.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.api.system_power import PowerEventLog
from iw_rag.serving.capacity import CapacityVerdict, InsufficientVram
from iw_rag.serving.power_app import create_app as create_sidecar
from iw_rag.serving.power_client import PowerCapacityError, PowerControlUnavailable
from iw_rag.serving.vram_history import VramHistory, VramSample
from iw_rag.settings import PowerControlSettings

REFUSAL = {
    "error": "insufficient_vram",
    "ok": False,
    "needed_mb": 18432,
    "free_mb": 3072,
    "projected_free_mb": 3072,
    "margin_mb": 3072,
    "shortfall_mb": 18432,
    "components": ["generator"],
}


class _Stub:
    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover
        raise RuntimeError("unused")


class FakeClient:
    """In-process fake of the new PowerClient surface."""

    def __init__(self) -> None:
        self.status_payload: dict[str, Any] = {
            "state": "on",
            "transition": None,
            "components": [{"key": "generator", "state": "on", "containers": []}],
            "vram": {"available": True, "used_mb": 20000, "total_mb": 32607, "free_mb": 12607},
            "estimated_held_mb": 18432,
            "estimated_other_mb": 1568,
        }
        self.refuse = False
        self.calls: list[tuple[str, Any]] = []
        self.history: dict[str, Any] = {"samples": [], "interval_s": 5.0}
        self.history_raises: Exception | None = None
        self.capacity: dict[str, Any] = {"ok": True, "shortfall_mb": 0}

    async def get_status(self) -> dict[str, Any]:
        return dict(self.status_payload)

    async def get_vram(self) -> dict[str, Any]:  # pragma: no cover - not used here
        return self.status_payload["vram"]

    async def power_on(self, components: Any = None, *, force: bool = False) -> dict[str, Any]:
        self.calls.append(("power_on", {"components": components, "force": force}))
        if self.refuse and not force:
            raise PowerCapacityError(REFUSAL)
        return dict(self.status_payload)

    async def power_off(self, components: Any = None) -> dict[str, Any]:  # pragma: no cover
        return dict(self.status_payload)

    async def orchestrate(self, *, stop: Any, start: Any, force: bool = False) -> dict[str, Any]:
        self.calls.append(("orchestrate", {"force": force}))
        if self.refuse and not force:
            raise PowerCapacityError(REFUSAL)
        return dict(self.status_payload)

    async def apply_profile(self, name: str, *, force: bool = False) -> dict[str, Any]:
        self.calls.append(("apply_profile", {"name": name, "force": force}))
        if self.refuse and not force:
            raise PowerCapacityError(REFUSAL)
        return dict(self.status_payload)

    async def get_history(self, window_s: float | None = None) -> dict[str, Any]:
        if self.history_raises is not None:
            raise self.history_raises
        return self.history

    async def get_capacity(self, components: Any = None) -> dict[str, Any]:
        return self.capacity


def _build(*, with_settings: bool = True) -> TestClient:
    auth = StaticTokenAuthenticator.from_config(
        {"user-token": ["finance"], "op-token": ["eval-operator"]}, operator_tokens=[]
    )
    app = create_app(
        authenticator=auth,
        query_service=_Stub(),
        power_client=FakeClient(),
    )
    if with_settings:
        # Slim wiring: only the ``power`` block the System routes read.
        app.state.settings = SimpleNamespace(power=PowerControlSettings())
    return TestClient(app)


def _fake(c: TestClient) -> FakeClient:
    return c.app.state.power_client  # type: ignore[attr-defined,no-any-return]


USER = {"authorization": "Bearer user-token"}
OP = {"authorization": "Bearer op-token"}

# ------------------------------- status -------------------------------


def test_status_gains_profile_yield_and_estimated_other() -> None:
    body = _build().get("/system/status", headers=USER).json()
    assert body["profile"] == "normal"
    assert body["estimated_other_gb"] == 1.5
    assert body["yield"]["enabled"] is True
    assert body["yield"]["threshold_gb"] == 2.0
    assert body["yield"]["margin_gb"] == 3.0
    assert body["yield"]["recent_events"] == []


def test_status_estimated_other_none_without_nvml() -> None:
    c = _build()
    _fake(c).status_payload["estimated_other_mb"] = None
    assert c.get("/system/status", headers=USER).json()["estimated_other_gb"] is None


# ------------------------------- profiles -------------------------------


def test_profiles_list_with_budgets_and_active_flag() -> None:
    body = _build().get("/system/profiles", headers=USER).json()
    by_name = {p["name"]: p for p in body["profiles"]}
    assert body["active"] == "normal" and by_name["normal"]["active"] is True
    assert by_name["yield"]["budget_gb"] == 4.0
    assert by_name["light"]["budget_gb"] == 22.0
    assert by_name["normal"]["budget_gb"] == 29.0


def test_profiles_disabled_when_unwired() -> None:
    c = _build()
    c.app.state.power_client = None  # type: ignore[attr-defined]
    assert c.get("/system/profiles", headers=USER).json() == {
        "enabled": False,
        "active": None,
        "profiles": [],
    }


def test_apply_profile_requires_power_group() -> None:
    c = _build()
    r = c.post("/system/profile", json={"name": "yield"}, headers=USER)
    assert r.status_code == 403
    assert _fake(c).calls == []


def test_apply_profile_requires_auth() -> None:
    assert _build().post("/system/profile", json={"name": "yield"}).status_code == 401


def test_apply_profile_ok_updates_active() -> None:
    c = _build()
    r = c.post("/system/profile", json={"name": "yield"}, headers=OP)
    assert r.status_code == 200 and r.json()["profile"] == "yield"
    assert _fake(c).calls == [("apply_profile", {"name": "yield", "force": False})]
    assert c.get("/system/profiles", headers=USER).json()["active"] == "yield"


def test_apply_unknown_profile_is_400_and_changes_nothing() -> None:
    c = _build()
    r = c.post("/system/profile", json={"name": "nope"}, headers=OP)
    assert r.status_code == 400
    assert _fake(c).calls == []


def test_apply_profile_refused_is_409_with_structured_body() -> None:
    c = _build()
    _fake(c).refuse = True
    r = c.post("/system/profile", json={"name": "normal"}, headers=OP)
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "insufficient_vram" and body["shortfall_mb"] == 18432
    assert c.get("/system/profiles", headers=USER).json()["active"] == "normal"  # unchanged


def test_apply_profile_force_bypasses_and_logs_event() -> None:
    c = _build()
    _fake(c).refuse = True
    r = c.post("/system/profile", json={"name": "normal", "force": True}, headers=OP)
    assert r.status_code == 200
    kinds = [e["kind"] for e in c.app.state.power_events.recent(5)]  # type: ignore[attr-defined]
    assert kinds == ["forced"]


# ------------------------------- power / mode 409 -------------------------------


def test_power_on_refused_is_409() -> None:
    c = _build()
    _fake(c).refuse = True
    r = c.post("/system/power", json={"action": "on", "components": ["generator"]}, headers=OP)
    assert r.status_code == 409 and r.json()["error"] == "insufficient_vram"


def test_power_on_force_is_passed_through() -> None:
    c = _build()
    _fake(c).refuse = True
    r = c.post(
        "/system/power",
        json={"action": "on", "components": ["generator"], "force": True},
        headers=OP,
    )
    assert r.status_code == 200
    assert _fake(c).calls[-1] == ("power_on", {"components": ["generator"], "force": True})


def test_power_force_requires_power_group() -> None:
    c = _build()
    r = c.post("/system/power", json={"action": "on", "force": True}, headers=USER)
    assert r.status_code == 403


def test_mode_switch_refused_is_409() -> None:
    c = _build()
    _fake(c).refuse = True
    r = c.post("/system/mode", json={"target": "benchmark"}, headers=OP)
    assert r.status_code == 409


# ------------------------------- history / capacity -------------------------------


def test_history_merges_samples_and_events() -> None:
    c = _build()
    _fake(c).history = {
        "samples": [{"ts": 100.0, "used_mb": 1, "free_mb": 2, "total_mb": 3}],
        "interval_s": 5.0,
    }
    log: PowerEventLog = c.app.state.power_events  # type: ignore[attr-defined]
    log.add("yield", ts=50.0, components=["generator"])  # before the window -> dropped
    log.add("yield", ts=150.0, components=["generator"])
    body = c.get("/system/history?window_s=600", headers=USER).json()
    assert body["available"] is True and len(body["samples"]) == 1
    assert [e["ts"] for e in body["events"]] == [150.0]


def test_history_empty_when_no_samples_yet() -> None:
    body = _build().get("/system/history", headers=USER).json()
    assert body["samples"] == [] and body["available"] is True


def test_history_degrades_when_sidecar_down() -> None:
    c = _build()
    _fake(c).history_raises = PowerControlUnavailable("down")
    body = c.get("/system/history", headers=USER).json()
    assert body["available"] is False and body["samples"] == []


def test_history_rejects_nonpositive_window() -> None:
    assert _build().get("/system/history?window_s=0", headers=USER).status_code == 422


def test_capacity_passthrough_and_degradation() -> None:
    c = _build()
    _fake(c).capacity = {"ok": False, "shortfall_mb": 5}
    assert c.get("/system/capacity?components=generator", headers=USER).json()["shortfall_mb"] == 5
    c.app.state.power_client = None  # type: ignore[attr-defined]
    assert c.get("/system/capacity", headers=USER).json() == {"ok": True, "skipped": "unavailable"}


# ------------------------------- sidecar app -------------------------------


class FakeController:
    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.calls: list[Any] = []

    def _verdict(self) -> CapacityVerdict:
        return CapacityVerdict(False, 100, 10, 10, 5, 95, None, ["generator"])

    async def power_on(self, keys: Any = None, *, force: bool = False) -> Any:
        self.calls.append(("power_on", force))
        if self.refuse and not force:
            raise InsufficientVram(self._verdict())
        return await self.status()

    async def apply_profile(self, name: str, profiles: Any, *, force: bool = False) -> Any:
        if name not in profiles:
            raise KeyError(name)
        if self.refuse and not force:
            raise InsufficientVram(self._verdict())
        return await self.status()

    async def check(self, keys: Any = None) -> CapacityVerdict:
        if keys and "nope" in keys:
            raise KeyError("nope")
        return self._verdict()

    async def status(self) -> Any:
        from iw_rag.serving.power_controller import PowerState, PowerStatus
        from iw_rag.serving.vram import VramSnapshot

        return PowerStatus(PowerState.ON, [], VramSnapshot(available=False))


def test_sidecar_409_body_and_force() -> None:
    sc = TestClient(create_sidecar(controller=FakeController(refuse=True)))  # type: ignore[arg-type]
    r = sc.post("/power/on", json={"components": ["generator"]})
    assert r.status_code == 409
    assert r.json()["error"] == "insufficient_vram" and r.json()["shortfall_mb"] == 95
    assert (
        sc.post("/power/on", json={"components": ["generator"], "force": True}).status_code == 200
    )


def test_sidecar_profile_endpoint() -> None:
    from iw_rag.settings import PowerProfileCfg

    app = create_sidecar(
        controller=FakeController(refuse=True),  # type: ignore[arg-type]
        profiles={"yield": PowerProfileCfg(label="y", on=["embedder"])},
    )
    sc = TestClient(app)
    assert sc.post("/power/profile", json={"name": "nope"}).status_code == 404
    assert sc.post("/power/profile", json={"name": "yield"}).status_code == 409
    assert sc.post("/power/profile", json={"name": "yield", "force": True}).status_code == 200


def test_sidecar_capacity_endpoint() -> None:
    sc = TestClient(create_sidecar(controller=FakeController()))  # type: ignore[arg-type]
    assert sc.get("/power/capacity?components=generator").json()["shortfall_mb"] == 95
    assert sc.get("/power/capacity?components=nope").status_code == 404


def test_sidecar_history_window_and_empty() -> None:
    hist = VramHistory(10)
    sc = TestClient(create_sidecar(controller=FakeController(), history=hist))  # type: ignore[arg-type]
    assert sc.get("/power/history").json()["samples"] == []
    import time

    now = time.time()
    hist.add(VramSample(now - 5000, 1, 2, 3))
    hist.add(VramSample(now - 10, 4, 5, 9))
    body = sc.get("/power/history?window_s=60").json()
    assert [s["used_mb"] for s in body["samples"]] == [4]
    assert len(sc.get("/power/history?window_s=99999").json()["samples"]) == 2
    assert sc.get("/power/history?window_s=0").status_code == 422
