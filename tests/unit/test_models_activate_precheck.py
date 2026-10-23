"""Capacity pre-check on ``POST /models/activate`` (F-00098)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.serving.power_client import PowerControlUnavailable
from iw_rag.settings import PowerControlSettings

GB = 1024


class _Stub:
    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover
        raise RuntimeError("unused")


class FakeSwap:
    def __init__(self, active: str | None = "qwen3-14b-awq") -> None:
        self.active = active
        self.activate_calls: list[str] = []

    async def get_catalog(self) -> dict[str, Any]:
        return {
            "models": [
                {
                    "id": "qwen3-8b-awq",
                    "served_name": "a",
                    "quantization": "awq",
                    "expected_vram_gb": 6.0,
                    "max_context": 1,
                    "state": "available",
                },
                {
                    "id": "qwen3-14b-awq",
                    "served_name": "b",
                    "quantization": "awq",
                    "expected_vram_gb": 9.0,
                    "max_context": 1,
                    "state": "active",
                },
                {
                    "id": "qwen3-32b-awq",
                    "served_name": "c",
                    "quantization": "awq",
                    "expected_vram_gb": 21.0,
                    "max_context": 1,
                    "state": "available",
                },
            ]
        }

    async def get_status(self) -> dict[str, Any]:
        return {"state": "ready", "active_model_id": self.active}

    async def activate(self, model_id: str) -> dict[str, Any]:
        self.activate_calls.append(model_id)
        return {"state": "ready", "active_model_id": model_id, "failure_reason": None}


class FakePower:
    def __init__(self, free_mb: int | None, *, raises: bool = False) -> None:
        self.free_mb = free_mb
        self.raises = raises

    async def get_vram(self) -> dict[str, Any]:
        if self.raises:
            raise PowerControlUnavailable("down")
        if self.free_mb is None:
            return {"available": False}
        return {"available": True, "free_mb": self.free_mb}


def _build(
    free_gb: float | None,
    *,
    active: str | None = "qwen3-14b-awq",
    power: bool = True,
    precheck: bool = True,
    raises: bool = False,
) -> tuple[TestClient, FakeSwap]:
    swap = FakeSwap(active)
    auth = StaticTokenAuthenticator.from_config({"op-token": ["eval-operator"]}, operator_tokens=[])
    app = create_app(
        authenticator=auth,
        query_service=_Stub(),
        swap_controller_client=swap,  # type: ignore[arg-type]
        power_client=FakePower(None if free_gb is None else int(free_gb * GB), raises=raises)
        if power
        else None,
    )
    cfg = PowerControlSettings(capacity_precheck_enabled=precheck)
    app.state.settings = SimpleNamespace(
        power=cfg, serving=SimpleNamespace(swap=SimpleNamespace(activation_group="eval-operator"))
    )
    return TestClient(app), swap


OP = {"authorization": "Bearer op-token"}


def _activate(c: TestClient, model_id: str) -> Any:
    return c.post("/models/activate", json={"model_id": model_id}, headers=OP)


def test_fits_proceeds_to_swap() -> None:
    c, swap = _build(free_gb=20)  # 20 + 9 freed >= 21 + 3
    r = _activate(c, "qwen3-32b-awq")
    assert r.status_code == 200 and swap.activate_calls == ["qwen3-32b-awq"]


def test_does_not_fit_is_409_and_swap_never_called() -> None:
    c, swap = _build(free_gb=5)  # 5 + 9 = 14 < 24
    r = _activate(c, "qwen3-32b-awq")
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "insufficient_vram" and body["components"] == ["qwen3-32b-awq"]
    assert body["shortfall_mb"] == (21 + 3 - 14) * GB
    assert swap.activate_calls == []


def test_resident_generator_memory_counts_as_freed() -> None:
    c, swap = _build(free_gb=15.5)  # 15.5 + 9 = 24.5 >= 24 -> ok only because 9 GB is freed
    assert _activate(c, "qwen3-32b-awq").status_code == 200
    c2, _ = _build(free_gb=15.5, active=None)  # nothing freed -> 15.5 < 24
    assert _activate(c2, "qwen3-32b-awq").status_code == 409


def test_reactivating_the_resident_model_is_never_blocked() -> None:
    c, swap = _build(free_gb=0.1)
    assert _activate(c, "qwen3-14b-awq").status_code == 200


def test_unknown_model_left_to_the_sidecar() -> None:
    c, swap = _build(free_gb=0.1)
    assert _activate(c, "no-such-model").status_code == 200 and swap.activate_calls == [
        "no-such-model"
    ]


def test_proceeds_when_nvml_unavailable_or_sidecar_down_or_disabled() -> None:
    assert _activate(_build(free_gb=None)[0], "qwen3-32b-awq").status_code == 200
    assert _activate(_build(free_gb=1, raises=True)[0], "qwen3-32b-awq").status_code == 200
    assert _activate(_build(free_gb=1, precheck=False)[0], "qwen3-32b-awq").status_code == 200


def test_proceeds_when_power_control_unwired() -> None:
    assert _activate(_build(free_gb=1, power=False)[0], "qwen3-32b-awq").status_code == 200


def test_non_operator_still_gets_403_before_any_check() -> None:
    c, swap = _build(free_gb=1)
    auth = StaticTokenAuthenticator.from_config({"u": ["finance"]}, operator_tokens=[])
    c.app.state.authenticator = auth  # type: ignore[attr-defined]
    r = c.post(
        "/models/activate",
        json={"model_id": "qwen3-32b-awq"},
        headers={"authorization": "Bearer u"},
    )
    assert r.status_code == 403 and swap.activate_calls == []
