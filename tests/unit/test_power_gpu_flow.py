"""Cross-module flow tests for GPU-aware operation (F-00098).

A real :class:`PowerController` (fake Docker engine + a VRAM reader whose free
memory is *derived* from what is running plus a scriptable "other processes"
load) is driven through a loopback client by the real yield monitor, the
capacity-aware warm and profile apply.  No GPU, no container, no network.
"""

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
)
from iw_rag.serving.capacity import InsufficientVram
from iw_rag.serving.power_app import _status_to_dict
from iw_rag.serving.power_client import PowerCapacityError
from iw_rag.serving.power_controller import PowerController
from iw_rag.serving.vram import VramSnapshot
from iw_rag.settings import PowerControlSettings
from tests.unit.test_capacity_precheck import FakeDocker

GB = 1024
TOTAL = 32 * GB
SETTINGS = PowerControlSettings(
    yield_sustain_s=0.01, yield_cooldown_s=0.0, yield_check_interval_s=0.01
)
CONTAINERS = {k: f"{k}-1" for k in SETTINGS.components}


class DerivedVram:
    """Free VRAM = total - other processes - budgets of the running containers."""

    def __init__(self, docker: FakeDocker, other_mb: int) -> None:
        self.docker = docker
        self.other_mb = other_mb

    def read(self) -> VramSnapshot:
        held = sum(
            round(SETTINGS.components[k].vram_gb * GB)
            for k, name in CONTAINERS.items()
            if name in self.docker.running
        )
        used = min(TOTAL, self.other_mb + held)
        return VramSnapshot(available=True, used_mb=used, free_mb=TOTAL - used, total_mb=TOTAL)


class Loopback:
    """A PowerClient look-alike that talks straight to the controller."""

    def __init__(self, ctl: PowerController) -> None:
        self.ctl = ctl

    async def get_status(self) -> dict[str, Any]:
        return _status_to_dict(await self.ctl.status())

    async def get_capacity(self, components: Any = None) -> dict[str, Any]:
        return (await self.ctl.check(components)).to_dict()

    async def power_on(self, components: Any = None, *, force: bool = False) -> dict[str, Any]:
        try:
            return _status_to_dict(await self.ctl.power_on(components, force=force))
        except InsufficientVram as exc:
            raise PowerCapacityError(exc.verdict.to_dict()) from exc

    async def power_off(self, components: Any = None) -> dict[str, Any]:
        return _status_to_dict(await self.ctl.power_off(components))

    async def apply_profile(self, name: str, *, force: bool = False) -> dict[str, Any]:
        try:
            status = await self.ctl.apply_profile(name, SETTINGS.profiles, force=force)
        except InsufficientVram as exc:
            raise PowerCapacityError(exc.verdict.to_dict()) from exc
        return _status_to_dict(status)


def _box(
    running: set[str], other_gb: float
) -> tuple[FakeDocker, DerivedVram, PowerController, Loopback]:
    docker = FakeDocker(running={CONTAINERS[k] for k in running}, existing=set(CONTAINERS.values()))
    vram = DerivedVram(docker, int(other_gb * GB))
    comps = {
        k: cfg.model_copy(update={"containers": [CONTAINERS[k]]})
        for k, cfg in SETTINGS.components.items()
    }
    ctl = PowerController(
        comps,
        docker,  # type: ignore[arg-type]
        vram,
        poll_interval_s=0.001,
        ready_timeout_s=1,
        capacity_precheck_enabled=True,
        capacity_margin_gb=SETTINGS.capacity_margin_gb,
    )
    return docker, vram, ctl, Loopback(ctl)


def _app(client: Loopback, *, benchmark: bool = False) -> Any:
    state = SimpleNamespace(
        settings=SimpleNamespace(
            power=SETTINGS,
            models=SimpleNamespace(get=lambda _m: SimpleNamespace(endpoint="http://vllm:8001/v1")),
            generation=SimpleNamespace(model="x"),
        ),
        power_client=client,
        power_events=PowerEventLog(),
        power_activity=PowerActivityTracker(),
        benchmark_run_manager=SimpleNamespace(is_active=lambda: benchmark),
    )
    return SimpleNamespace(state=state, app=SimpleNamespace(state=state))


async def _monitor_for(app: Any, seconds: float = 0.2) -> None:
    task = asyncio.create_task(power_yield_monitor(app))
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# --------------------------- AC1 / AC2: pre-check ---------------------------


async def test_ac1_refused_start_changes_nothing_and_reports_shortfall() -> None:
    docker, _vram, _ctl, client = _box({"embedder"}, other_gb=25)  # free = 32-25-4 = 3 GB
    with pytest.raises(PowerCapacityError, match="insufficient VRAM") as exc:
        await client.power_on(["generator"])
    assert exc.value.body["shortfall_mb"] == (18 + 3 - 3) * GB
    assert docker.calls == []


async def test_ac2_profile_swap_allowed_only_because_stops_free_memory() -> None:
    # generator on (18) + embedder (4) + other 8 -> free 2 GB; "yield" frees the generator.
    docker, _vram, _ctl, client = _box({"generator", "embedder"}, other_gb=8)
    await client.apply_profile("yield")
    assert docker.running == {CONTAINERS["embedder"]}


# --------------------------- AC3 / AC4 / AC8: yield ---------------------------


async def test_ac3_sustained_pressure_releases_generator_and_judges_not_embedder() -> None:
    docker, vram, _ctl, client = _box({"generator", "judges", "embedder"}, other_gb=0)
    vram.other_mb = 6 * GB  # another app grows: free = 32 - 6 - 40 -> 0, well below 2 GB
    app = _app(client)
    await _monitor_for(app)
    assert docker.running == {CONTAINERS["embedder"]}
    events = app.state.power_events.recent(10)
    assert [e["kind"] for e in events] == ["yield"]  # exactly one yield; then nothing eligible
    assert set(events[0]["components"]) == {"generator", "judges"}


async def test_ac4_own_load_that_passed_precheck_never_trips_the_monitor() -> None:
    docker, _vram, _ctl, client = _box({"embedder"}, other_gb=2)  # free = 26 GB
    await client.power_on(["generator"])  # 26 - 18 = 8 GB left >= 3 GB margin
    app = _app(client)
    await _monitor_for(app)
    assert CONTAINERS["generator"] in docker.running
    assert app.state.power_events.recent(10) == []


async def test_ac8_benchmark_active_defers_and_stops_nothing() -> None:
    docker, vram, _ctl, client = _box({"generator", "judges", "embedder"}, other_gb=6)
    app = _app(client, benchmark=True)
    await _monitor_for(app)
    assert {CONTAINERS["generator"], CONTAINERS["judges"]} <= docker.running
    assert [e["kind"] for e in app.state.power_events.recent(10)] == ["yield_deferred"]


# --------------------------- AC5: capacity-aware re-warm ---------------------------


async def test_ac5_warm_after_yield_refused_until_room_returns() -> None:
    docker, vram, _ctl, client = _box({"generator", "embedder"}, other_gb=9)
    app = _app(client)
    await _monitor_for(app)  # yields the generator
    assert CONTAINERS["generator"] not in docker.running

    vram.other_mb = 27 * GB  # the other app is still hogging: free = 32-27-4 = 1 GB
    with pytest.raises(HTTPException, match="503") as exc:
        await maybe_warm_generator(app)
    assert exc.value.status_code == 503 and exc.value.detail["error"] == "gpu_busy"  # type: ignore[index]
    assert CONTAINERS["generator"] not in docker.running
    assert app.state.power_events.recent(10)[-1]["kind"] == "warm_refused"

    vram.other_mb = 2 * GB  # the other app let go: free = 26 GB
    await maybe_warm_generator(app)
    assert CONTAINERS["generator"] in docker.running


# --------------------------- AC6: profiles ---------------------------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        ("yield", {"embedder"}),
        ("light", {"generator", "embedder"}),
        ("normal", {"generator", "embedder", "guards"}),
    ],
)
async def test_ac6_each_profile_leaves_exactly_its_components_resident(
    profile: str, expected: set[str]
) -> None:
    docker, _vram, _ctl, client = _box({"generator", "embedder", "guards"}, other_gb=0)
    await client.apply_profile("yield")  # normalise to a known state
    await client.apply_profile(profile)
    assert docker.running == {CONTAINERS[k] for k in expected}


async def test_ac6_profile_that_does_not_fit_is_refused_and_force_applies_it() -> None:
    docker, _vram, _ctl, client = _box({"embedder"}, other_gb=24)  # free = 4 GB
    with pytest.raises(PowerCapacityError, match="insufficient VRAM"):
        await client.apply_profile("normal")
    assert docker.running == {CONTAINERS["embedder"]}
    await client.apply_profile("normal", force=True)
    assert CONTAINERS["generator"] in docker.running


async def test_profile_leaves_unnamed_judges_untouched() -> None:
    docker, _vram, _ctl, client = _box({"generator", "embedder", "judges"}, other_gb=0)
    await client.apply_profile("light")  # names generator/embedder/guards only
    assert CONTAINERS["judges"] in docker.running
