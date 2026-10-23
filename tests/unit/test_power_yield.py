"""Unit tests for the reactive yield monitor and capacity-aware warm (F-00098)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException

from iw_rag.api.system_power import (
    PowerActivityTracker,
    PowerEventLog,
    maybe_warm_generator,
    power_yield_monitor,
    should_yield,
)
from iw_rag.serving.power_client import PowerCapacityError, PowerControlUnavailable

GB = 1024
YIELD_KEYS = ["generator", "judges"]


def _status(
    gen: str = "on", judges: str = "off", emb: str = "on", transition: str | None = None
) -> dict[str, Any]:
    return {
        "transition": transition,
        "components": [
            {"key": "generator", "state": gen},
            {"key": "embedder", "state": emb},
            {"key": "judges", "state": judges},
        ],
        "vram": {"available": True, "free_mb": 1 * GB, "used_mb": 31 * GB, "total_mb": 32 * GB},
    }


def _decide(**kw: Any) -> Any:
    args: dict[str, Any] = {
        "enabled": True,
        "free_mb": 1 * GB,
        "below_since": None,
        "now": 100.0,
        "threshold_mb": 2 * GB,
        "sustain_s": 20.0,
        "cooldown_s": 120.0,
        "last_yield_at": None,
        "status": _status(),
        "benchmark_active": False,
        "yield_components": YIELD_KEYS,
    }
    args.update(kw)
    return should_yield(**args)


def test_first_low_reading_starts_the_clock_without_yielding() -> None:
    d = _decide()
    assert d.action == "none" and d.below_since == 100.0


def test_sustain_not_reached_keeps_counter() -> None:
    d = _decide(below_since=90.0, now=100.0)
    assert d.action == "none" and d.below_since == 90.0


def test_recovery_resets_counter() -> None:
    d = _decide(below_since=90.0, free_mb=5 * GB)
    assert d.action == "none" and d.below_since is None


def test_sustained_low_yields_only_eligible_components_that_are_on() -> None:
    d = _decide(below_since=70.0, status=_status(gen="on", judges="on"))
    assert d.action == "yield" and d.components == ["generator", "judges"] and d.below_since is None
    d = _decide(below_since=70.0, status=_status(gen="on", judges="off"))
    assert d.components == ["generator"]


def test_embedder_is_never_yielded() -> None:
    d = _decide(below_since=70.0)
    assert "embedder" not in d.components


def test_nothing_eligible_on_is_noop() -> None:
    d = _decide(below_since=70.0, status=_status(gen="off", judges="off"))
    assert d.action == "none" and d.components == []


def test_transition_in_flight_skips() -> None:
    d = _decide(below_since=70.0, status=_status(transition="power_on"))
    assert d.action == "none"


def test_cooldown_blocks_second_yield() -> None:
    d = _decide(below_since=70.0, last_yield_at=50.0)
    assert d.action == "none"
    d = _decide(below_since=70.0, last_yield_at=-100.0)
    assert d.action == "yield"


def test_benchmark_active_defers() -> None:
    d = _decide(below_since=70.0, benchmark_active=True)
    assert d.action == "defer" and d.components == ["generator"]


def test_nvml_unavailable_does_nothing() -> None:
    d = _decide(free_mb=None, below_since=70.0)
    assert d.action == "none" and d.below_since is None


def test_disabled_does_nothing() -> None:
    assert _decide(enabled=False, below_since=70.0).action == "none"


def test_exactly_at_threshold_is_not_pressure() -> None:
    assert _decide(free_mb=2 * GB).below_since is None


def test_event_log_is_bounded_and_ordered() -> None:
    log = PowerEventLog(max_events=3)
    for i in range(5):
        log.add("yield", ts=float(i), free_mb=i, components=["generator"], detail=str(i))
    assert [e["ts"] for e in log.recent(10)] == [2.0, 3.0, 4.0]
    assert [e["ts"] for e in log.recent(2)] == [3.0, 4.0]


# ----------------------------- monitor loop -----------------------------


class _Client:
    def __init__(self, statuses: list[Any]) -> None:
        self.statuses = statuses
        self.i = 0
        self.off_calls: list[Any] = []

    async def get_status(self) -> dict[str, Any]:
        s = self.statuses[min(self.i, len(self.statuses) - 1)]
        self.i += 1
        if isinstance(s, Exception):
            raise s
        return s

    async def power_off(self, components: Any = None) -> dict[str, Any]:
        self.off_calls.append(components)
        return {}


def _app(client: Any, *, benchmark: bool = False, yield_enabled: bool = True) -> Any:
    power = SimpleNamespace(
        enabled=True,
        yield_enabled=yield_enabled,
        yield_free_gb_threshold=2.0,
        yield_sustain_s=0.01,
        yield_cooldown_s=0.0,
        yield_check_interval_s=0.01,
        components={
            "generator": SimpleNamespace(yield_unload=True),
            "embedder": SimpleNamespace(yield_unload=False),
            "judges": SimpleNamespace(yield_unload=True),
        },
    )
    settings = SimpleNamespace(
        power=power,
        models=SimpleNamespace(get=lambda _m: SimpleNamespace(endpoint="http://vllm:8001/v1")),
        generation=SimpleNamespace(model="x"),
    )
    manager = SimpleNamespace(is_active=lambda: benchmark)
    state = SimpleNamespace(
        settings=settings,
        power_client=client,
        power_events=PowerEventLog(),
        benchmark_run_manager=manager,
    )
    return SimpleNamespace(state=state)


async def _run(app: Any, seconds: float = 0.15) -> None:
    task = asyncio.create_task(power_yield_monitor(app))
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_monitor_yields_after_sustained_pressure_and_records_event() -> None:
    client = _Client([_status(gen="on", judges="on")])
    app = _app(client)
    await _run(app)
    assert client.off_calls and client.off_calls[0] == ["generator", "judges"]
    kinds = [e["kind"] for e in app.state.power_events.recent(10)]
    assert "yield" in kinds


async def test_monitor_defers_once_per_episode_while_benchmark_active() -> None:
    client = _Client([_status(gen="on", judges="on")])
    app = _app(client, benchmark=True)
    await _run(app)
    assert client.off_calls == []
    kinds = [e["kind"] for e in app.state.power_events.recent(10)]
    assert kinds.count("yield_deferred") == 1


async def test_monitor_survives_sidecar_outage_and_errors() -> None:
    client = _Client([PowerControlUnavailable("down"), RuntimeError("boom"), _status(gen="on")])
    app = _app(client)
    await _run(app)
    assert client.off_calls[0] == ["generator"]  # recovered and still yielded


async def test_monitor_exits_when_yield_disabled() -> None:
    client = _Client([_status()])
    app = _app(client, yield_enabled=False)
    await asyncio.wait_for(power_yield_monitor(app), timeout=1.0)
    assert client.i == 0


# --------------------------- capacity-aware warm ---------------------------


class _WarmClient:
    def __init__(self, verdict: dict[str, Any], *, on_raises: bool = False) -> None:
        self.verdict = verdict
        self.on_raises = on_raises
        self.power_on_calls: list[Any] = []

    async def get_status(self) -> dict[str, Any]:
        return {
            "transition": None,
            "components": [{"key": "generator", "state": "off"}],
        }

    async def get_capacity(self, components: Any = None) -> dict[str, Any]:
        return self.verdict

    async def power_on(self, components: Any = None) -> dict[str, Any]:
        self.power_on_calls.append(components)
        if self.on_raises:
            raise PowerCapacityError({"shortfall_mb": 1, "needed_mb": 2, "free_mb": 1})
        return {}


def _warm_request(client: Any) -> Any:
    power = SimpleNamespace(enabled=True, block_and_warm=True)
    settings = SimpleNamespace(
        power=power,
        models=SimpleNamespace(get=lambda _m: SimpleNamespace(endpoint="http://vllm:8001/v1")),
        generation=SimpleNamespace(model="x"),
    )
    state = SimpleNamespace(
        power_activity=PowerActivityTracker(),
        settings=settings,
        power_client=client,
        power_events=PowerEventLog(),
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


async def test_warm_refused_when_it_does_not_fit_returns_503_gpu_busy() -> None:
    client = _WarmClient(
        {"ok": False, "needed_mb": 18 * GB, "free_mb": 3 * GB, "shortfall_mb": 18 * GB}
    )
    req = _warm_request(client)
    with pytest.raises(HTTPException, match="503") as exc:
        await maybe_warm_generator(req)
    assert exc.value.status_code == 503
    assert exc.value.detail["error"] == "gpu_busy"  # type: ignore[index]
    assert exc.value.headers is not None and "Retry-After" in exc.value.headers
    assert client.power_on_calls == []
    assert [e["kind"] for e in req.app.state.power_events.recent(5)] == ["warm_refused"]


async def test_warm_proceeds_when_it_fits() -> None:
    client = _WarmClient({"ok": True, "needed_mb": 18 * GB, "free_mb": 30 * GB, "shortfall_mb": 0})
    await maybe_warm_generator(_warm_request(client))
    assert client.power_on_calls == [["generator"]]


async def test_warm_race_refusal_from_sidecar_is_also_503() -> None:
    client = _WarmClient({"ok": True}, on_raises=True)
    with pytest.raises(HTTPException, match="503") as exc:
        await maybe_warm_generator(_warm_request(client))
    assert exc.value.status_code == 503
