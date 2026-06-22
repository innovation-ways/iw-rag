"""Unit tests for the F-00094 ``ServingSwapSettings`` block on ``Settings``.

TDD RED→GREEN: this test was written alongside the ``ServingSwapSettings``
addition.  The behavioural tests for the catalog loader / activation state
machine live in S02 + S05; this file is the minimal settings-field guard so
the defaults documented in the design are pinned and the env mapping works.
"""

from __future__ import annotations

import pytest


def _fresh_settings(monkeypatch: pytest.MonkeyPatch) -> object:
    """Build a fresh ``Settings`` with the required env stubs in place.

    Args:
        monkeypatch: pytest env fixture.

    Returns:
        A freshly built ``Settings`` instance.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    return settings_mod.get_settings()


# ---------------------------------------------------------------------------
# 1. Documented defaults land on every attribute
# ---------------------------------------------------------------------------


def test_serving_defaults_match_design(monkeypatch: pytest.MonkeyPatch) -> None:
    """``Settings.serving`` resolves with the design's documented defaults."""
    s = _fresh_settings(monkeypatch)
    assert s.serving.catalog_path == "config/model_catalog.yaml"
    assert s.serving.llama_swap_base_url == "http://llama-swap:8080"
    assert s.serving.swap_controller_base_url == "http://swap-controller:8090"
    assert s.serving.health_timeout_s == 300
    assert s.serving.activation_group == "eval-operator"


# ---------------------------------------------------------------------------
# 2. Env mapping — IW_RAG__SERVING__* overrides the defaults
# ---------------------------------------------------------------------------


def test_serving_health_timeout_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IW_RAG__SERVING__HEALTH_TIMEOUT_S=120`` overrides the default 300."""
    monkeypatch.setenv("IW_RAG__SERVING__HEALTH_TIMEOUT_S", "120")
    s = _fresh_settings(monkeypatch)
    assert s.serving.health_timeout_s == 120


def test_serving_activation_group_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IW_RAG__SERVING__ACTIVATION_GROUP=benchmark-ops`` overrides the default."""
    monkeypatch.setenv("IW_RAG__SERVING__ACTIVATION_GROUP", "benchmark-ops")
    s = _fresh_settings(monkeypatch)
    assert s.serving.activation_group == "benchmark-ops"


def test_serving_catalog_path_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``IW_RAG__SERVING__CATALOG_PATH`` overrides the default catalog path."""
    monkeypatch.setenv("IW_RAG__SERVING__CATALOG_PATH", "config/alt_catalog.yaml")
    s = _fresh_settings(monkeypatch)
    assert s.serving.catalog_path == "config/alt_catalog.yaml"


# ---------------------------------------------------------------------------
# 3. default.yaml lands its `serving:` values when env is silent
# ---------------------------------------------------------------------------


def test_serving_defaults_match_default_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """``config/default.yaml`` under ``serving:`` is the lowest-priority source.

    The defaults in :class:`ServingSwapSettings` MUST equal the values in
    ``config/default.yaml`` so the YAML and the class stay in sync
    (lld-config §4).
    """
    import yaml

    s = _fresh_settings(monkeypatch)

    with open("config/default.yaml") as fh:
        raw = yaml.safe_load(fh)
    yaml_serving = raw["serving"]

    assert s.serving.catalog_path == yaml_serving["catalog_path"]
    assert s.serving.llama_swap_base_url == yaml_serving["llama_swap_base_url"]
    assert s.serving.swap_controller_base_url == yaml_serving["swap_controller_base_url"]
    assert s.serving.health_timeout_s == yaml_serving["health_timeout_s"]
    assert s.serving.activation_group == yaml_serving["activation_group"]
