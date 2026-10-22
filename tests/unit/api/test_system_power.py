"""Unit tests for the API-side power policy (idle decision + block-and-warm).

Pure-function and helper coverage — no app, no sidecar.  The idle-unload
decision and the block-and-warm warm-up are exercised directly with fakes.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from iw_rag.api.system_power import (
    PowerActivityTracker,
    _generator_needs_warm,
    is_serving_mode,
    maybe_warm_generator,
    resolve_mode,
    should_idle_unload,
)

# ---------------------------------------------------------------------------
# PowerActivityTracker
# ---------------------------------------------------------------------------


def test_activity_tracker_touch_advances_timestamp() -> None:
    """touch() moves last_activity forward (monotonic)."""
    t = PowerActivityTracker()
    t._last_activity = t.last_activity - 100.0  # pretend last activity was 100s ago
    before = t.last_activity
    t.touch()
    assert t.last_activity > before


# ---------------------------------------------------------------------------
# is_serving_mode
# ---------------------------------------------------------------------------


def _settings_with_endpoint(endpoint: str) -> Any:
    entry = SimpleNamespace(endpoint=endpoint)
    models = SimpleNamespace(get=lambda _name: entry)
    return SimpleNamespace(models=models, generation=SimpleNamespace(model="generator"))


def test_is_serving_mode_true_for_direct_vllm() -> None:
    """A direct vLLM generator endpoint is serving mode."""
    assert is_serving_mode(_settings_with_endpoint("http://vllm:8001/v1")) is True


def test_is_serving_mode_false_for_llama_swap() -> None:
    """A llama-swap generator endpoint is benchmark mode (power control off)."""
    assert is_serving_mode(_settings_with_endpoint("http://llama-swap:8080/v1")) is False


def test_is_serving_mode_defaults_true_when_unreadable() -> None:
    """A None/unreadable settings fails toward serving (switch stays usable)."""
    assert is_serving_mode(None) is True


def test_resolve_mode_prefers_runtime_override() -> None:
    """resolve_mode honours app.state.runtime_mode set by the toggle, over the endpoint."""
    settings = _settings_with_endpoint("http://vllm:8001/v1")  # endpoint says serving
    state = SimpleNamespace(runtime_mode="benchmark", settings=settings)
    assert resolve_mode(state) == "benchmark"  # the runtime override wins


def test_resolve_mode_falls_back_to_endpoint() -> None:
    """With no runtime override, resolve_mode derives mode from the generator endpoint."""
    state = SimpleNamespace(
        runtime_mode=None, settings=_settings_with_endpoint("http://llama-swap:8080/v1")
    )
    assert resolve_mode(state) == "benchmark"


# ---------------------------------------------------------------------------
# should_idle_unload — the pure decision
# ---------------------------------------------------------------------------


def _status(generator_state: str = "on", transition: Any = None) -> dict[str, Any]:
    return {
        "transition": transition,
        "components": [
            {"key": "generator", "state": generator_state},
            {"key": "embedder", "state": "on"},
        ],
    }


def _base_kwargs(**over: Any) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "last_activity": 0.0,
        "now": 1000.0,  # 1000s idle
        "threshold_s": 900.0,  # 15 min
        "idle_enabled": True,
        "serving_mode": True,
        "status": _status("on"),
        "idle_components": ["generator"],
    }
    kw.update(over)
    return kw


def test_idle_unload_fires_when_idle_and_on() -> None:
    """Past the threshold, in serving mode, with the generator on → unload it."""
    assert should_idle_unload(**_base_kwargs()) == ["generator"]


def test_idle_unload_skips_when_disabled() -> None:
    """Idle-unload disabled → never acts."""
    assert should_idle_unload(**_base_kwargs(idle_enabled=False)) == []


def test_idle_unload_skips_in_benchmark_mode() -> None:
    """Benchmark mode → never acts (llama-swap owns the GPU)."""
    assert should_idle_unload(**_base_kwargs(serving_mode=False)) == []


def test_idle_unload_skips_before_threshold() -> None:
    """Within the idle window → never acts."""
    assert should_idle_unload(**_base_kwargs(now=100.0)) == []


def test_idle_unload_skips_when_already_off() -> None:
    """A component already off is not re-unloaded."""
    assert should_idle_unload(**_base_kwargs(status=_status("off"))) == []


def test_idle_unload_skips_during_transition() -> None:
    """A transition already in flight → stand down."""
    assert should_idle_unload(**_base_kwargs(status=_status("on", transition="power_off"))) == []


def test_idle_unload_skips_when_status_unknown() -> None:
    """No status (sidecar down) → do nothing."""
    assert should_idle_unload(**_base_kwargs(status=None)) == []


# ---------------------------------------------------------------------------
# _generator_needs_warm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "state,expected",
    [("off", True), ("partial", True), ("on", False)],
)
def test_generator_needs_warm(state: str, expected: bool) -> None:
    """A generator that is off/partial needs warming; on does not."""
    assert _generator_needs_warm(_status(state)) is expected


def test_generator_needs_warm_false_during_transition() -> None:
    """Another warm already in flight → do not start a second one."""
    assert _generator_needs_warm(_status("off", transition="power_on")) is False


# ---------------------------------------------------------------------------
# maybe_warm_generator — block-and-warm
# ---------------------------------------------------------------------------


class _FakePowerClient:
    def __init__(self, status: dict[str, Any]) -> None:
        self._status = status
        self.power_on_calls: list[Any] = []

    async def get_status(self) -> dict[str, Any]:
        return self._status

    async def power_on(self, components: Any = None) -> dict[str, Any]:
        self.power_on_calls.append(components)
        return self._status


def _request(*, client: Any, serving: bool = True, enabled: bool = True) -> Any:
    endpoint = "http://vllm:8001/v1" if serving else "http://llama-swap:8080/v1"
    power = SimpleNamespace(enabled=enabled, block_and_warm=True)
    settings = _settings_with_endpoint(endpoint)
    settings.power = power
    state = SimpleNamespace(
        power_activity=PowerActivityTracker(),
        settings=settings,
        power_client=client,
    )
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.asyncio
async def test_warm_powers_on_when_generator_off() -> None:
    """A query with the generator off triggers power_on(['generator'])."""
    client = _FakePowerClient(_status("off"))
    await maybe_warm_generator(_request(client=client))
    assert client.power_on_calls == [["generator"]]


@pytest.mark.asyncio
async def test_warm_noop_when_generator_on() -> None:
    """A query with the generator already up does not power on again."""
    client = _FakePowerClient(_status("on"))
    await maybe_warm_generator(_request(client=client))
    assert client.power_on_calls == []


@pytest.mark.asyncio
async def test_warm_noop_in_benchmark_mode() -> None:
    """Benchmark mode skips warm entirely (interlock)."""
    client = _FakePowerClient(_status("off"))
    await maybe_warm_generator(_request(client=client, serving=False))
    assert client.power_on_calls == []


@pytest.mark.asyncio
async def test_warm_touches_activity_even_without_client() -> None:
    """Warm always resets the idle clock, even when power control is unwired."""
    req = _request(client=None)
    before = req.app.state.power_activity.last_activity
    req.app.state.power_activity._last_activity = before - 100.0
    await maybe_warm_generator(req)
    assert req.app.state.power_activity.last_activity > before - 100.0
