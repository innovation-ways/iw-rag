"""Unit tests for operating-profile planning and apply (F-00098)."""

from __future__ import annotations

import pytest

from iw_rag.serving.capacity import InsufficientVram
from iw_rag.serving.power_controller import PowerController
from iw_rag.serving.profiles import plan_profile, resolve_profile
from iw_rag.settings import PowerControlSettings, PowerProfileCfg
from tests.unit.test_capacity_precheck import FakeDocker, FakeVram

SETTINGS = PowerControlSettings()


def _states(**kw: str) -> dict[str, str]:
    base = {"generator": "off", "embedder": "off", "judges": "off", "guards": "off"}
    base.update(kw)
    return base


def test_resolve_known_and_unknown() -> None:
    assert resolve_profile("yield", SETTINGS).label == "Yield GPU"
    with pytest.raises(KeyError, match="unknown power profile"):
        resolve_profile("nope", SETTINGS)


def test_plan_yield_from_everything_on() -> None:
    plan = plan_profile(
        SETTINGS.profiles["yield"],
        SETTINGS.components,
        _states(generator="on", embedder="on", judges="on", guards="on"),
    )
    assert set(plan.stop) == {"generator", "judges", "guards"}
    assert plan.start == []
    assert plan.budget_gb == 4.0


def test_plan_light_drops_only_guards_and_leaves_judges_untouched() -> None:
    plan = plan_profile(
        SETTINGS.profiles["light"],
        SETTINGS.components,
        _states(generator="on", embedder="on", judges="on", guards="on"),
    )
    assert plan.stop == ["guards"]
    assert plan.start == []  # judges not named -> untouched (Invariant 3)
    assert plan.budget_gb == 22.0


def test_plan_normal_starts_what_is_off_and_skips_what_is_on() -> None:
    plan = plan_profile(SETTINGS.profiles["normal"], SETTINGS.components, _states(embedder="on"))
    assert plan.start == ["generator", "guards"]
    assert plan.stop == []


def test_plan_partial_component_counts_as_needing_nothing_to_stop_or_start() -> None:
    plan = plan_profile(
        SETTINGS.profiles["yield"],
        SETTINGS.components,
        _states(embedder="partial", generator="partial"),
    )
    assert plan.stop == ["generator"]
    assert plan.start == []


def _ctl(docker: FakeDocker, free_mb: int | None) -> PowerController:
    comps = {k: v for k, v in SETTINGS.components.items()}
    # Single fake container per component keeps the fake engine simple.
    for k, cfg in comps.items():
        comps[k] = cfg.model_copy(update={"containers": [f"{k}-1"]})
    return PowerController(
        comps,
        docker,  # type: ignore[arg-type]
        FakeVram(free_mb),
        poll_interval_s=0.001,
        ready_timeout_s=1,
        capacity_precheck_enabled=True,
        capacity_margin_gb=3.0,
    )


ALL = {"generator-1", "embedder-1", "judges-1", "guards-1"}


async def test_apply_yield_leaves_only_embedder() -> None:
    docker = FakeDocker(running=set(ALL), existing=set(ALL))
    ctl = _ctl(docker, free_mb=1024)
    await ctl.apply_profile("yield", SETTINGS.profiles)
    assert docker.running == {"embedder-1"}


async def test_apply_light_stops_guards_only() -> None:
    docker = FakeDocker(running=set(ALL), existing=set(ALL))
    ctl = _ctl(docker, free_mb=1024)
    await ctl.apply_profile("light", SETTINGS.profiles)
    assert docker.running == ALL - {"guards-1"}


async def test_apply_normal_refused_stops_and_starts_nothing() -> None:
    docker = FakeDocker(running={"embedder-1"}, existing=set(ALL))
    ctl = _ctl(docker, free_mb=2 * 1024)
    with pytest.raises(InsufficientVram, match="insufficient VRAM"):
        await ctl.apply_profile("normal", SETTINGS.profiles)
    assert docker.calls == []


async def test_apply_counts_memory_freed_by_the_profile_stops() -> None:
    profiles = {
        "swap": PowerProfileCfg(label="swap", on=["judges"], off=["generator"]),
    }
    docker = FakeDocker(running={"generator-1", "embedder-1"}, existing=set(ALL))
    ctl = _ctl(docker, free_mb=4 * 1024)  # 4 + 18 freed >= 18 + 3
    await ctl.apply_profile("swap", profiles)
    assert docker.running == {"embedder-1", "judges-1"}


async def test_apply_unknown_profile_raises_keyerror_and_changes_nothing() -> None:
    docker = FakeDocker(running=set(ALL), existing=set(ALL))
    ctl = _ctl(docker, free_mb=1024)
    with pytest.raises(KeyError, match="unknown power profile"):
        await ctl.apply_profile("nope", SETTINGS.profiles)
    assert docker.calls == []


async def test_apply_force_bypasses_capacity() -> None:
    docker = FakeDocker(running={"embedder-1"}, existing=set(ALL))
    ctl = _ctl(docker, free_mb=1024)
    await ctl.apply_profile("normal", SETTINGS.profiles, force=True)
    assert {"generator-1", "guards-1"} <= docker.running
