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
