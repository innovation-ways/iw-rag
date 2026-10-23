"""Unit tests for the ``PowerControlSettings`` block on ``Settings``.

Pins the documented defaults (idle threshold, the serving-mode component →
container mapping, the power group) and verifies the ``IW_RAG__POWER__*`` env
mapping, including the wholesale JSON component override.
"""

from __future__ import annotations

import pytest

from iw_rag.settings import PowerControlSettings


def _fresh_settings(monkeypatch: pytest.MonkeyPatch) -> object:
    """Build a fresh ``Settings`` with the required env stubs (mirrors serving)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    return settings_mod.get_settings()


def test_power_model_defaults() -> None:
    """The standalone model carries the documented defaults + serving components."""
    p = PowerControlSettings()
    assert p.enabled is True
    assert p.base_url == "http://system-control:8091"
    assert p.power_group == "eval-operator"
    assert p.idle_unload_enabled is True
    assert p.idle_unload_minutes == 15
    assert p.block_and_warm is True
    assert p.vram_ceiling_gb == 32.0
    # The generator matches the serving pin AND the benchmark vendor families,
    # but NEVER the judges (vllm-judge-*) — a safety invariant on a shared box.
    gen = p.components["generator"].containers
    assert "iw-rag-vllm-1" in gen
    assert "vllm-qwen*" in gen
    assert all(not c.startswith("vllm-judge") for c in gen)
    assert p.components["generator"].idle_unload is True
    assert p.components["embedder"].containers == ["iw-rag-infinity-1"]
    assert p.components["embedder"].idle_unload is False
    # Judges are their own component so a resident judge's VRAM is freeable.
    assert set(p.components["judges"].containers) == {"vllm-judge-lynx", "vllm-judge-selene"}
    assert "iw-rag-vllm-guard" in p.components["guards"].containers


def test_power_mode_switch_plans() -> None:
    """The serving/benchmark mode plans mirror the make targets."""
    p = PowerControlSettings()
    assert p.mode_switch_enabled is True
    serving = p.modes["serving"]
    benchmark = p.modes["benchmark"]
    assert serving.generator_endpoint == "http://vllm:8001/v1"
    assert "iw-rag-vllm-1" in serving.start
    assert "iw-rag-llama-swap-1" in serving.stop
    assert benchmark.generator_endpoint == "http://llama-swap:8080/v1"
    assert "iw-rag-llama-swap-1" in benchmark.start
    assert "iw-rag-vllm-1" in benchmark.stop
    assert benchmark.guards is False


def test_power_defaults_resolve_on_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Settings.power`` resolves from config/default.yaml with the defaults."""
    s = _fresh_settings(monkeypatch)
    assert s.power.idle_unload_minutes == 15  # type: ignore[attr-defined]
    assert s.power.base_url == "http://system-control:8091"  # type: ignore[attr-defined]
    assert set(s.power.components) == {"generator", "embedder", "judges", "guards"}  # type: ignore[attr-defined]


def test_power_idle_minutes_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IW_RAG__POWER__IDLE_UNLOAD_MINUTES=30`` overrides the default 15."""
    monkeypatch.setenv("IW_RAG__POWER__IDLE_UNLOAD_MINUTES", "30")
    s = _fresh_settings(monkeypatch)
    assert s.power.idle_unload_minutes == 30  # type: ignore[attr-defined]


def test_power_enabled_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IW_RAG__POWER__ENABLED=false`` disables the feature."""
    monkeypatch.setenv("IW_RAG__POWER__ENABLED", "false")
    s = _fresh_settings(monkeypatch)
    assert s.power.enabled is False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# F-00098 — capacity pre-check, reactive yield, history and profile knobs
# ---------------------------------------------------------------------------


def test_power_gpu_aware_defaults() -> None:
    """The F-00098 knobs carry the documented defaults and budgets."""
    p = PowerControlSettings()
    assert p.capacity_precheck_enabled is True
    assert p.capacity_margin_gb == 3.0
    assert p.yield_enabled is True
    assert p.yield_free_gb_threshold == 2.0
    assert p.yield_sustain_s == 20.0
    assert p.yield_cooldown_s == 120.0
    assert p.yield_check_interval_s == 5.0
    assert p.history_enabled is True
    assert p.history_interval_s == 5.0
    assert p.history_max_points == 720
    assert p.active_profile == "normal"
    assert {k: c.vram_gb for k, c in p.components.items()} == {
        "generator": 18.0,
        "embedder": 4.0,
        "judges": 18.0,
        "guards": 7.0,
    }
    # Only the big, cheaply-reloaded holders may be yielded automatically.
    assert {k for k, c in p.components.items() if c.yield_unload} == {"generator", "judges"}


def test_power_default_profiles() -> None:
    """normal / light / yield are defined and manage only the components they name."""
    p = PowerControlSettings()
    assert set(p.profiles) == {"normal", "light", "yield"}
    assert p.profiles["normal"].on == ["generator", "embedder", "guards"]
    assert p.profiles["light"].on == ["generator", "embedder"]
    assert p.profiles["light"].off == ["guards"]
    assert p.profiles["yield"].on == ["embedder"]
    assert set(p.profiles["yield"].off) == {"generator", "guards", "judges"}


def test_power_margin_must_exceed_yield_threshold() -> None:
    """Invariant 1: a start that passed the pre-check cannot trip the yield monitor."""
    with pytest.raises(ValueError, match="capacity_margin_gb"):
        PowerControlSettings(capacity_margin_gb=2.0, yield_free_gb_threshold=2.0)
    with pytest.raises(ValueError, match="capacity_margin_gb"):
        PowerControlSettings(capacity_margin_gb=1.0, yield_free_gb_threshold=2.0)


def test_power_profile_rejects_unknown_component() -> None:
    """A profile naming a component that is not configured fails at load."""
    from iw_rag.settings import PowerProfileCfg

    with pytest.raises(ValueError, match="unknown component"):
        PowerControlSettings(
            profiles={"x": PowerProfileCfg(label="x", on=["nope"])},
            active_profile="x",
        )


def test_power_profile_on_off_must_be_disjoint() -> None:
    """``on`` and ``off`` overlapping is a config error."""
    from iw_rag.settings import PowerProfileCfg

    with pytest.raises(ValueError, match="both on and off"):
        PowerProfileCfg(label="x", on=["generator"], off=["generator"])


def test_power_active_profile_must_exist() -> None:
    """``active_profile`` must be one of the configured profiles."""
    with pytest.raises(ValueError, match="active_profile"):
        PowerControlSettings(active_profile="does-not-exist")


def test_power_gpu_knobs_resolve_from_yaml_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """config/default.yaml supplies the knobs; ``IW_RAG__POWER__*`` overrides them."""
    monkeypatch.setenv("IW_RAG__POWER__YIELD_FREE_GB_THRESHOLD", "1.5")
    s = _fresh_settings(monkeypatch)
    assert s.power.yield_free_gb_threshold == 1.5  # type: ignore[attr-defined]
    assert s.power.capacity_margin_gb == 3.0  # type: ignore[attr-defined]
    assert set(s.power.profiles) == {"normal", "light", "yield"}  # type: ignore[attr-defined]
    assert s.power.components["generator"].vram_gb == 18.0  # type: ignore[attr-defined]


def test_default_yaml_profiles_are_not_mangled_by_yaml_booleans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: bare ``on:``/``off:`` keys in YAML 1.1 become booleans and empty the profile."""
    s = _fresh_settings(monkeypatch)
    profiles = s.power.profiles  # type: ignore[attr-defined]
    assert profiles["normal"].on == ["generator", "embedder", "guards"]
    assert profiles["light"].off == ["guards"]
    assert profiles["yield"].on == ["embedder"]
    assert set(profiles["yield"].off) == {"generator", "guards", "judges"}


def test_power_profile_with_no_components_is_rejected() -> None:
    """An empty profile (what a YAML-boolean mishap produces) fails loudly at load."""
    from iw_rag.settings import PowerProfileCfg

    with pytest.raises(ValueError, match="names no components"):
        PowerProfileCfg(label="x")
