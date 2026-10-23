"""Unit tests for the capacity pre-check and its PowerController integration (F-00098)."""

from __future__ import annotations

import pytest

from iw_rag.serving.capacity import InsufficientVram, check_capacity, gb_to_mb
from iw_rag.serving.docker_engine import ContainerState
from iw_rag.serving.power_controller import PowerController
from iw_rag.serving.vram import VramSnapshot
from iw_rag.settings import PowerComponentCfg

MARGIN = 3 * 1024


def test_gb_to_mb() -> None:
    assert gb_to_mb(1.0) == 1024
    assert gb_to_mb(18) == 18432


def test_exact_limit_is_allowed() -> None:
    v = check_capacity(starts_mb=10_000, freed_mb=0, free_mb=10_000 + MARGIN, margin_mb=MARGIN)
    assert v.ok is True and v.shortfall_mb == 0 and v.skipped is None


def test_one_mib_short_is_blocked() -> None:
    v = check_capacity(starts_mb=10_000, freed_mb=0, free_mb=10_000 + MARGIN - 1, margin_mb=MARGIN)
    assert v.ok is False and v.shortfall_mb == 1


def test_freed_memory_counts_towards_projection() -> None:
    blocked = check_capacity(starts_mb=18_432, freed_mb=0, free_mb=2_048, margin_mb=MARGIN)
    allowed = check_capacity(starts_mb=18_432, freed_mb=18_432, free_mb=4_096, margin_mb=MARGIN)
    assert blocked.ok is False and blocked.shortfall_mb == 18_432 + MARGIN - 2_048
    assert allowed.ok is True and allowed.projected_free_mb == 22_528


def test_nothing_to_start_is_always_ok() -> None:
    v = check_capacity(starts_mb=0, freed_mb=0, free_mb=0, margin_mb=MARGIN)
    assert v.ok is True and v.needed_mb == 0


def test_nvml_unavailable_skips() -> None:
    v = check_capacity(starts_mb=99_999, freed_mb=0, free_mb=None, margin_mb=MARGIN)
    assert v.ok is True and v.skipped == "nvml_unavailable" and v.projected_free_mb is None


def test_disabled_never_blocks() -> None:
    v = check_capacity(starts_mb=99_999, freed_mb=0, free_mb=1, margin_mb=MARGIN, enabled=False)
    assert v.ok is True and v.skipped == "disabled"


def test_force_bypasses() -> None:
    v = check_capacity(starts_mb=99_999, freed_mb=0, free_mb=1, margin_mb=MARGIN, force=True)
    assert v.ok is True and v.skipped == "forced"


def test_insufficient_vram_message_and_verdict() -> None:
    v = check_capacity(
        starts_mb=5_000, freed_mb=0, free_mb=1_000, margin_mb=MARGIN, components=["generator"]
    )
    err = InsufficientVram(v)
    assert err.verdict is v
    assert "short by" in str(err) and "generator" in str(err)
    assert v.to_dict()["shortfall_mb"] == v.shortfall_mb


# --------------------------- controller integration ---------------------------


class FakeDocker:
    """Minimal fake Docker engine recording every mutating call."""

    def __init__(self, running: set[str], existing: set[str] | None = None) -> None:
        self.running = set(running)
        self.existing = set(existing) if existing is not None else set(running)
        self.calls: list[tuple[str, str]] = []

    async def inspect(self, name: str) -> ContainerState:
        exists = name in self.existing
        return ContainerState(
            name=name, exists=exists, running=name in self.running, status="running", health=None
        )

    async def start(self, name: str) -> bool:
        self.calls.append(("start", name))
        if name not in self.existing:
            return False
        self.running.add(name)
        return True

    async def stop(self, name: str, timeout_s: int = 10) -> bool:
        self.calls.append(("stop", name))
        self.running.discard(name)
        return True

    async def list_matching(self, prefix: str) -> list[str]:
        return sorted(n for n in self.running if n.startswith(prefix))


class FakeVram:
    def __init__(self, free_mb: int | None, total_mb: int = 32_768) -> None:
        self.free_mb = free_mb
        self.total_mb = total_mb

    def read(self) -> VramSnapshot:
        if self.free_mb is None:
            return VramSnapshot(available=False)
        return VramSnapshot(
            available=True,
            used_mb=self.total_mb - self.free_mb,
            free_mb=self.free_mb,
            total_mb=self.total_mb,
        )


def _components() -> dict[str, PowerComponentCfg]:
    return {
        "generator": PowerComponentCfg(containers=["gen-1"], vram_gb=18.0, yield_unload=True),
        "embedder": PowerComponentCfg(containers=["emb-1"], vram_gb=4.0),
        "judges": PowerComponentCfg(containers=["judge-1"], vram_gb=18.0, yield_unload=True),
    }


def _ctl(docker: FakeDocker, free_mb: int | None, *, enabled: bool = True) -> PowerController:
    return PowerController(
        _components(),
        docker,  # type: ignore[arg-type]
        FakeVram(free_mb),
        poll_interval_s=0.001,
        ready_timeout_s=1,
        capacity_precheck_enabled=enabled,
        capacity_margin_gb=3.0,
    )


async def test_power_on_refused_starts_nothing() -> None:
    docker = FakeDocker(running={"emb-1"}, existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=3 * 1024)
    with pytest.raises(InsufficientVram, match="insufficient VRAM") as exc:
        await ctl.power_on(["generator"])
    assert exc.value.verdict.components == ["generator"]
    assert exc.value.verdict.needed_mb == 18 * 1024
    assert docker.calls == []


async def test_power_on_allowed_when_it_fits() -> None:
    docker = FakeDocker(running={"emb-1"}, existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=21 * 1024)
    await ctl.power_on(["generator"])
    assert ("start", "gen-1") in docker.calls


async def test_component_already_on_contributes_zero() -> None:
    docker = FakeDocker(running={"gen-1", "emb-1"})
    ctl = _ctl(docker, free_mb=100)  # nothing free, but nothing to start
    await ctl.power_on(["generator", "embedder"])  # must not raise
    verdict = await ctl.check(["generator", "embedder"])
    assert verdict.ok is True and verdict.needed_mb == 0


async def test_force_bypasses_precheck() -> None:
    docker = FakeDocker(running={"emb-1"}, existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=1024)
    await ctl.power_on(["generator"], force=True)
    assert ("start", "gen-1") in docker.calls


async def test_precheck_disabled_never_blocks() -> None:
    docker = FakeDocker(running=set(), existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=1, enabled=False)
    await ctl.power_on(["generator"])
    assert ("start", "gen-1") in docker.calls


async def test_nvml_unavailable_proceeds() -> None:
    docker = FakeDocker(running=set(), existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=None)
    await ctl.power_on(["generator"])
    assert ("start", "gen-1") in docker.calls


async def test_orchestrate_counts_memory_freed_by_its_stops() -> None:
    docker = FakeDocker(running={"gen-1", "emb-1"}, existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=4 * 1024)
    # judges (18 GB) fit only because the generator (18 GB) is stopped first.
    await ctl.orchestrate(stop=["gen-1"], start=["judge-1"])
    assert docker.calls[0] == ("stop", "gen-1") and ("start", "judge-1") in docker.calls


async def test_orchestrate_refused_stops_nothing() -> None:
    docker = FakeDocker(running={"emb-1"}, existing={"gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=2 * 1024)
    with pytest.raises(InsufficientVram, match="insufficient VRAM"):
        await ctl.orchestrate(stop=["emb-1"], start=["gen-1", "judge-1"])
    assert docker.calls == []


async def test_unknown_container_has_zero_budget() -> None:
    docker = FakeDocker(running=set(), existing={"llama-swap", "gen-1", "emb-1", "judge-1"})
    ctl = _ctl(docker, free_mb=100)
    await ctl.orchestrate(stop=[], start=["llama-swap"])  # unmapped -> budget 0
    assert ("start", "llama-swap") in docker.calls


async def test_status_reports_estimated_split() -> None:
    docker = FakeDocker(running={"gen-1", "emb-1"})
    ctl = _ctl(docker, free_mb=4 * 1024)
    status = await ctl.status()
    held = (18 + 4) * 1024
    assert status.estimated_held_mb == held
    assert status.estimated_other_mb == (32_768 - 4 * 1024) - held


async def test_partial_component_is_prorated_in_the_held_estimate() -> None:
    comps = _components()
    comps["guards"] = PowerComponentCfg(containers=["g-gpu", "g-cpu"], vram_gb=8.0)
    docker = FakeDocker(running={"g-cpu"}, existing={"g-gpu", "g-cpu"})
    ctl = PowerController(comps, docker, FakeVram(16 * 1024))  # type: ignore[arg-type]
    status = await ctl.status()
    assert status.estimated_held_mb == 4 * 1024  # 1 of 2 containers running -> half of 8 GB
