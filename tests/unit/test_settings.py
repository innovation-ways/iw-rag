"""Unit tests for Settings, ModelRegistry, and config precedence.

TDD RED-first: these tests were written before the implementation existed;
they verify the full settings contract: precedence layers, env nesting,
extra="forbid" rejection, ConfigError on missing dsn / unknown model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(env: dict[str, str], **kwargs: Any) -> Any:
    """Build a Settings object with the given env vars injected.

    Args:
        env: Environment variable overrides.
        **kwargs: Additional keyword arguments passed to Settings().

    Returns:
        A fresh Settings instance built with the given env.
    """
    import iw_rag.settings as settings_mod

    # Clear the lru_cache so each call gets a fresh load
    settings_mod.get_settings.cache_clear()

    with patch.dict("os.environ", env, clear=False):
        return settings_mod.Settings(**kwargs)


# ---------------------------------------------------------------------------
# 1. Env nesting — IW_RAG__SECTION__KEY maps to the right attribute
# ---------------------------------------------------------------------------


def test_retrieval_top_k_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """IW_RAG__RETRIEVAL__TOP_K=10 lands on Settings.retrieval.top_k."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__RETRIEVAL__TOP_K", "10")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.retrieval.top_k == 10


def test_rrf_k_default_is_60(monkeypatch: pytest.MonkeyPatch) -> None:
    """rrf_k defaults to exactly 60 when not set."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.retrieval.rrf_k == 60


# ---------------------------------------------------------------------------
# 2. extra="forbid" — unknown env keys must be rejected
# ---------------------------------------------------------------------------


def test_extra_forbid_rejects_unknown_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """extra="forbid" raises ValidationError for an unknown Settings field."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()

    with pytest.raises(ValidationError, match="extra_forbidden|Extra inputs are not permitted"):
        settings_mod.Settings(
            database=settings_mod.DatabaseSettings(dsn="postgresql+psycopg://u:p@h/db"),
            unknown_field_xyz="bad",  # type: ignore[call-arg]
        )


# ---------------------------------------------------------------------------
# 3. Missing dsn → ConfigError naming the key
# ---------------------------------------------------------------------------


def test_missing_dsn_raises_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing IW_RAG__DATABASE__DSN raises ConfigError at startup."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    # Neutralise the local .env (gitignored, may carry a DSN) so the "missing
    # DSN" condition is genuinely simulated and the test is hermetic.
    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    # Ensure DSN is NOT set
    monkeypatch.delenv("IW_RAG__DATABASE__DSN", raising=False)
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()

    with pytest.raises(ConfigError, match="IW_RAG__DATABASE__DSN"):
        settings_mod.get_settings()


# ---------------------------------------------------------------------------
# 4. ModelRegistry.get — known resolves, unknown → ConfigError
# ---------------------------------------------------------------------------


def test_model_registry_get_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelRegistry.get('generator') resolves to the generator ModelEntry."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    entry = s.models.get("generator")
    assert entry is not None
    assert entry.role == "generation"
    assert entry.engine == "vllm"


def test_model_registry_unknown_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ModelRegistry.get with an unknown logical name raises ConfigError."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()

    with pytest.raises(ConfigError, match="nonexistent_model_xyz"):
        s.models.get("nonexistent_model_xyz")


# ---------------------------------------------------------------------------
# 5. Endpoint env interpolation
# ---------------------------------------------------------------------------


def test_endpoint_interpolation_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Endpoint ${IW_RAG__MODELS__GENERATOR__ENDPOINT} is resolved from env."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://custom-vllm:9001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()

    entry = s.models.get("generator")
    assert entry.endpoint == "http://custom-vllm:9001/v1"


# ---------------------------------------------------------------------------
# 6. get_settings is cached (same instance on repeated calls)
# ---------------------------------------------------------------------------


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_settings() returns the same object on repeated calls (lru_cache)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s1 = settings_mod.get_settings()
    s2 = settings_mod.get_settings()
    assert s1 is s2


# ---------------------------------------------------------------------------
# 7. Precedence — default.yaml < env
# ---------------------------------------------------------------------------


def test_env_overrides_default_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Process env overrides default.yaml values (precedence: default < env)."""
    import iw_rag.settings as settings_mod

    # top_k default in default.yaml is 8; override via env
    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__RETRIEVAL__TOP_K", "99")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.retrieval.top_k == 99


# ---------------------------------------------------------------------------
# 8. ModelEntry.engine includes "infinity"
# ---------------------------------------------------------------------------


def test_embedder_engine_can_be_infinity(monkeypatch: pytest.MonkeyPatch) -> None:
    """ModelEntry.engine accepts 'infinity' (the Blackwell fallback for TEI)."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()

    # Both embedder and reranker use engine=infinity
    embedder = s.models.get("embedder")
    assert embedder.engine == "infinity"
    reranker = s.models.get("reranker")
    assert reranker.engine == "infinity"


# ---------------------------------------------------------------------------
# 9. _interpolate_env — ${VAR:-default} branch and bare-unset → ConfigError
# ---------------------------------------------------------------------------


def test_interpolate_env_default_substitution_when_var_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """${VAR:-default} resolves to the literal default when VAR is not in env."""
    import iw_rag.settings as settings_mod

    monkeypatch.delenv("MY_TEST_VAR_UNSET_XYZ", raising=False)
    result = settings_mod._interpolate_env("prefix-${MY_TEST_VAR_UNSET_XYZ:-fallback_val}-suffix")
    assert result == "prefix-fallback_val-suffix"


def test_interpolate_env_default_substitution_when_var_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """${VAR:-default} resolves to the env value when VAR is set."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("MY_TEST_VAR_SET_XYZ", "real_val")
    result = settings_mod._interpolate_env("${MY_TEST_VAR_SET_XYZ:-fallback_val}")
    assert result == "real_val"


def test_interpolate_env_bare_unset_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ${VAR} placeholder with no default raises ConfigError when VAR is unset."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    monkeypatch.delenv("MY_BARE_UNSET_VAR_XYZ", raising=False)
    with pytest.raises(ConfigError, match="MY_BARE_UNSET_VAR_XYZ"):
        settings_mod._interpolate_env("${MY_BARE_UNSET_VAR_XYZ}")


# ---------------------------------------------------------------------------
# 10. _load_model_registry — missing file and missing top-level key
# ---------------------------------------------------------------------------


def test_load_model_registry_missing_file_raises_config_error() -> None:
    """_load_model_registry raises ConfigError when the registry file does not exist."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    with pytest.raises(ConfigError, match="models.yaml not found"):
        settings_mod._load_model_registry("/nonexistent/path/models_zzz.yaml")


def test_load_model_registry_missing_models_key_raises_config_error(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """_load_model_registry raises ConfigError when the YAML has no top-level 'models:' key."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    bad_yaml = tmp_path / "models_bad.yaml"  # type: ignore[operator]
    bad_yaml.write_text("roles:\n  generator: {}\n")
    with pytest.raises(ConfigError, match="top-level 'models:' key"):
        settings_mod._load_model_registry(str(bad_yaml))


# ---------------------------------------------------------------------------
# 11. LongContextModeSettings._reject_yarn — settings-level YaRN guard
# ---------------------------------------------------------------------------


def test_long_context_mode_settings_enable_yarn_raises_config_error() -> None:
    """LongContextModeSettings(enable_yarn=True) raises ConfigError at startup."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    with pytest.raises(ConfigError, match="ENABLE_YARN"):
        settings_mod.LongContextModeSettings(enable_yarn=True)


def test_long_context_mode_settings_enable_yarn_false_is_accepted() -> None:
    """LongContextModeSettings(enable_yarn=False) constructs without error."""
    import iw_rag.settings as settings_mod

    cfg = settings_mod.LongContextModeSettings(enable_yarn=False)
    assert cfg.enable_yarn is False


def test_long_context_mode_settings_via_env_raises_config_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IW_RAG__MODES__LONG_CONTEXT__ENABLE_YARN=true via Settings raises ConfigError at startup."""
    import iw_rag.settings as settings_mod
    from iw_rag.domain.errors import ConfigError

    monkeypatch.setattr(settings_mod, "_load_dotenv_safe", lambda: None)
    monkeypatch.setenv("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    monkeypatch.setenv("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    monkeypatch.setenv("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://vllm-judge:8002/v1")
    monkeypatch.setenv("IW_RAG__MODES__LONG_CONTEXT__ENABLE_YARN", "true")
    settings_mod.get_settings.cache_clear()

    with pytest.raises((ConfigError, Exception), match="ENABLE_YARN"):
        settings_mod.get_settings()
