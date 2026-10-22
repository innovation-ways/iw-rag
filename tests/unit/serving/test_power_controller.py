"""Unit tests for the :class:`PowerController` state machine.

No real Docker daemon, no GPU: a :class:`FakeDocker` tracks container
lifecycle in-memory and records start/stop calls; a :class:`FakeVram` returns
a canned snapshot.  Tests cover status derivation (on/off/partial/absent/
unreachable → overall ON/OFF/PARTIAL/UNKNOWN), power-off/on container actions,
readiness polling + timeout, the in-flight transition marker, and unknown-key
validation.
"""

from __future__ import annotations

import pytest

from iw_rag.serving.docker_engine import ContainerState, DockerUnavailable
from iw_rag.serving.power_controller import PowerController, PowerState
from iw_rag.serving.vram import VramSnapshot
from iw_rag.settings import PowerComponentCfg


class FakeDocker:
    """In-memory Docker stand-in: name → (running, health), records calls."""

    def __init__(self, present: dict[str, tuple[bool, str | None]]) -> None:
        # name -> (running, health); names not present here are "absent".
        self._present = dict(present)
        self.started: list[str] = []
        self.stopped: list[str] = []
        self.fail = False

    async def inspect(self, name: str) -> ContainerState:
        if self.fail:
            raise DockerUnavailable("daemon down")
        if name not in self._present:
            return ContainerState(
                name=name, exists=False, running=False, status="absent", health=None
            )
        running, health = self._present[name]
        return ContainerState(
            name=name,
            exists=True,
            running=running,
            status="running" if running else "exited",
            health=health,
        )

    async def start(self, name: str) -> bool:
        if self.fail:
            raise DockerUnavailable("daemon down")
        if name not in self._present:
            return False
        self.started.append(name)
        running, health = self._present[name]
        self._present[name] = (True, health)
        return True

    async def stop(self, name: str, *, timeout_s: int = 10) -> bool:
        if self.fail:
            raise DockerUnavailable("daemon down")
        if name not in self._present:
            return False
        self.stopped.append(name)
        _running, health = self._present[name]
        self._present[name] = (False, health)
        return True

    async def list_matching(self, prefix: str) -> list[str]:
        if self.fail:
            raise DockerUnavailable("daemon down")
        # Running containers only (mirrors the real running-only filter).
        return sorted(
            n for n, (running, _h) in self._present.items() if running and n.startswith(prefix)
        )


class FakeVram:
    def __init__(self, used_mb: int = 18000) -> None:
        self._used = used_mb

    def read(self) -> VramSnapshot:
        return VramSnapshot(
            available=True, used_mb=self._used, free_mb=32607 - self._used, total_mb=32607
        )


def _components() -> dict[str, PowerComponentCfg]:
    return {
        "generator": PowerComponentCfg(containers=["vllm"], label="Gen", idle_unload=True),
        "embedder": PowerComponentCfg(containers=["infinity"], label="Embed"),
        "guards": PowerComponentCfg(containers=["guard-a", "guard-b"], label="Guards"),
    }


def _controller(docker: FakeDocker, *, ready_timeout_s: int = 5) -> PowerController:
    return PowerController(
        _components(),
        docker,  # type: ignore[arg-type]
        FakeVram(),
        ready_timeout_s=ready_timeout_s,
        poll_interval_s=0.0,
    )


# ---------------------------------------------------------------------------
# status() derivation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_status_all_running_is_on() -> None:
    """Every present container running → overall ON, VRAM attached."""
    docker = FakeDocker(
        {
            "vllm": (True, "healthy"),
            "infinity": (True, None),
            "guard-a": (True, None),
            "guard-b": (True, None),
        }
    )
    status = await _controller(docker).status()
    assert status.state is PowerState.ON
    assert {c.key: c.state for c in status.components} == {
        "generator": "on",
        "embedder": "on",
        "guards": "on",
    }
    assert status.vram.used_mb == 18000


@pytest.mark.asyncio
async def test_status_all_stopped_is_off() -> None:
    """Every present container stopped → overall OFF."""
    docker = FakeDocker(
        {
            "vllm": (False, None),
            "infinity": (False, None),
            "guard-a": (False, None),
            "guard-b": (False, None),
        }
    )
    status = await _controller(docker).status()
    assert status.state is PowerState.OFF
    assert all(c.state == "off" for c in status.components)


@pytest.mark.asyncio
async def test_status_mixed_is_partial() -> None:
    """Generator up but embedder down → overall PARTIAL."""
    docker = FakeDocker(
        {
            "vllm": (True, "healthy"),
            "infinity": (False, None),
            "guard-a": (True, None),
            "guard-b": (False, None),
        }
    )
    status = await _controller(docker).status()
    assert status.state is PowerState.PARTIAL
    by_key = {c.key: c.state for c in status.components}
    assert by_key["generator"] == "on"
    assert by_key["embedder"] == "off"
    assert by_key["guards"] == "partial"  # one of two running


@pytest.mark.asyncio
async def test_status_absent_container_does_not_vote() -> None:
    """An absent (optional guard) container reports 'absent' and doesn't break ON."""
    docker = FakeDocker({"vllm": (True, "healthy"), "infinity": (True, None)})  # no guards
    status = await _controller(docker).status()
    by_key = {c.key: c.state for c in status.components}
    assert by_key["guards"] == "absent"
    # Only present components vote → still ON.
    assert status.state is PowerState.ON


@pytest.mark.asyncio
async def test_status_daemon_down_is_unknown() -> None:
    """A daemon outage maps overall to UNKNOWN with unreachable components."""
    docker = FakeDocker({"vllm": (True, "healthy")})
    docker.fail = True
    status = await _controller(docker).status()
    assert status.state is PowerState.UNKNOWN
    assert all(c.state == "unreachable" for c in status.components)


# ---------------------------------------------------------------------------
# power_off / power_on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_power_off_all_stops_every_container() -> None:
    """Master OFF stops all managed containers and lands OFF."""
    docker = FakeDocker(
        {
            "vllm": (True, "healthy"),
            "infinity": (True, None),
            "guard-a": (True, None),
            "guard-b": (True, None),
        }
    )
    status = await _controller(docker).power_off(None)
    assert set(docker.stopped) == {"vllm", "infinity", "guard-a", "guard-b"}
    assert status.state is PowerState.OFF


@pytest.mark.asyncio
async def test_power_off_single_component_only_stops_its_containers() -> None:
    """Powering off only 'generator' leaves the embedder running (→ PARTIAL)."""
    docker = FakeDocker({"vllm": (True, "healthy"), "infinity": (True, None)})
    ctrl = PowerController(
        {
            "generator": PowerComponentCfg(containers=["vllm"]),
            "embedder": PowerComponentCfg(containers=["infinity"]),
        },
        docker,  # type: ignore[arg-type]
        FakeVram(),
        poll_interval_s=0.0,
    )
    status = await ctrl.power_off(["generator"])
    assert docker.stopped == ["vllm"]
    assert "infinity" not in docker.stopped
    assert status.state is PowerState.PARTIAL


@pytest.mark.asyncio
async def test_power_on_starts_and_reaches_ready() -> None:
    """Master ON starts every container and polls each to ready → ON."""
    docker = FakeDocker(
        {
            "vllm": (False, "healthy"),
            "infinity": (False, None),
            "guard-a": (False, None),
            "guard-b": (False, None),
        }
    )
    status = await _controller(docker).power_on(None)
    assert set(docker.started) == {"vllm", "infinity", "guard-a", "guard-b"}
    assert status.state is PowerState.ON


@pytest.mark.asyncio
async def test_power_on_never_ready_times_out_without_hanging() -> None:
    """A container stuck 'starting' fails readiness within the budget (no hang)."""
    docker = FakeDocker({"vllm": (False, "starting")})
    ctrl = PowerController(
        {"generator": PowerComponentCfg(containers=["vllm"])},
        docker,  # type: ignore[arg-type]
        FakeVram(),
        ready_timeout_s=0,  # immediate timeout
        poll_interval_s=0.0,
    )
    status = await ctrl.power_on(["generator"])
    # start was issued, but health 'starting' never becomes ready → not ON.
    assert docker.started == ["vllm"]
    assert status.state is not PowerState.ON


@pytest.mark.asyncio
async def test_unknown_component_key_raises() -> None:
    """Requesting an unknown component key is a KeyError (→ 400 at the route)."""
    docker = FakeDocker({"vllm": (True, "healthy")})
    with pytest.raises(KeyError, match="nope"):
        await _controller(docker).power_off(["nope"])


# ---------------------------------------------------------------------------
# in-flight transition marker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generator_pattern_resolves_benchmark_container() -> None:
    """A 'vllm-*' pattern resolves llama-swap's dynamic generator in benchmark mode."""
    # Serving generator absent; a benchmark-style vllm-<id> is running.
    docker = FakeDocker({"vllm-qwen3-14b-awq": (True, None), "infinity": (True, None)})
    ctrl = PowerController(
        {
            "generator": PowerComponentCfg(containers=["iw-rag-vllm-1", "vllm-*"]),
            "embedder": PowerComponentCfg(containers=["infinity"]),
        },
        docker,  # type: ignore[arg-type]
        FakeVram(),
        poll_interval_s=0.0,
    )
    status = await ctrl.status()
    gen = next(c for c in status.components if c.key == "generator")
    # The dynamic container is present + running → generator is "on".
    assert gen.state == "on"
    assert any(c.name == "vllm-qwen3-14b-awq" and c.running for c in gen.containers)


@pytest.mark.asyncio
async def test_power_off_generator_stops_pattern_match() -> None:
    """Powering off the generator in benchmark mode stops the dynamic vllm-* container."""
    docker = FakeDocker({"vllm-qwen3-32b-awq": (True, None), "infinity": (True, None)})
    ctrl = PowerController(
        {"generator": PowerComponentCfg(containers=["iw-rag-vllm-1", "vllm-*"])},
        docker,  # type: ignore[arg-type]
        FakeVram(),
        poll_interval_s=0.0,
    )
    await ctrl.power_off(["generator"])
    assert docker.stopped == ["vllm-qwen3-32b-awq"]


@pytest.mark.asyncio
async def test_generator_pattern_never_matches_judge_container() -> None:
    """SAFETY: a 'vllm-qwen*' generator pattern must not touch a 'vllm-judge-*' container."""
    docker = FakeDocker({"vllm-qwen3-14b-awq": (True, None), "vllm-judge-selene": (True, None)})
    ctrl = PowerController(
        {"generator": PowerComponentCfg(containers=["iw-rag-vllm-1", "vllm-qwen*"])},
        docker,  # type: ignore[arg-type]
        FakeVram(),
        poll_interval_s=0.0,
    )
    await ctrl.power_off(["generator"])
    # Only the qwen generator was stopped; the judge container was untouched.
    assert docker.stopped == ["vllm-qwen3-14b-awq"]
    assert "vllm-judge-selene" not in docker.stopped


@pytest.mark.asyncio
async def test_orchestrate_stops_outgoing_starts_incoming() -> None:
    """orchestrate() stops the outgoing set and starts (+readies) the incoming set."""
    docker = FakeDocker(
        {
            "iw-rag-vllm-1": (True, "healthy"),
            "iw-rag-llama-swap-1": (False, None),
            "iw-rag-swap-controller-1": (False, None),
        }
    )
    ctrl = _controller(docker)
    await ctrl.orchestrate(
        stop=["iw-rag-vllm-1"],
        start=["iw-rag-llama-swap-1", "iw-rag-swap-controller-1"],
    )
    assert docker.stopped == ["iw-rag-vllm-1"]
    assert set(docker.started) == {"iw-rag-llama-swap-1", "iw-rag-swap-controller-1"}


@pytest.mark.asyncio
async def test_orchestrate_skips_absent_start_container() -> None:
    """orchestrate() does not fail when an incoming container does not exist."""
    docker = FakeDocker({"iw-rag-vllm-1": (True, "healthy")})
    ctrl = _controller(docker)
    # 'injguard' absent → start returns False → skipped, no error.
    status = await ctrl.orchestrate(stop=[], start=["injguard"])
    assert status.state is not PowerState.UNKNOWN
    assert docker.started == []


@pytest.mark.asyncio
async def test_inflight_marker_reports_powering_state() -> None:
    """While a transition is marked in-flight, status() reports POWERING_*."""
    docker = FakeDocker({"vllm": (True, "healthy")})
    ctrl = _controller(docker)
    ctrl._inflight = ("power_off", ["generator"])  # simulate mid-flip
    status = await ctrl.status()
    assert status.state is PowerState.POWERING_OFF
    assert status.transition == "power_off"
    assert status.transition_keys == ["generator"]
