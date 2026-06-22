"""Shared pytest fixtures and the ``--run-live`` opt-in gate.

``@pytest.mark.live`` tests hit real vLLM/TEI endpoints and are skipped unless
``--run-live`` is passed, so the spike's live calls never run in plain ``pytest``.
The repo root is placed on ``sys.path`` so tests can import ``scripts.*`` (a
namespace package) — e.g. the spike's pure threshold helpers.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# --- Hermetic settings for the test suite -----------------------------------
# A developer's local .env must never leak into pytest (it bit us: a stray
# IW_RAG__GUARDRAILS__ENABLED=false in .env silently flipped guard tests). We
# skip the on-disk .env and provide fixed baseline values for the settings that
# get_settings() strictly requires (hot-path endpoints + DSN). Optional settings
# (guardrails.enabled, judge/guard endpoints) keep their code defaults, so tests
# exercise the real defaults regardless of the environment.
os.environ.setdefault("IW_RAG_SKIP_DOTENV", "1")

_BASELINE_ENV = {
    "IW_RAG__DATABASE__DSN": "postgresql+psycopg://iw_rag:test@localhost:5432/iw_rag",
    "IW_RAG__MODELS__GENERATOR__ENDPOINT": "http://localhost:8001/v1",
    "IW_RAG__MODELS__EMBEDDER__ENDPOINT": "http://localhost:7997",
    "IW_RAG__MODELS__RERANKER__ENDPOINT": "http://localhost:7997",
}
for _k, _v in _BASELINE_ENV.items():
    os.environ.setdefault(_k, _v)

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _settings_isolation() -> object:
    """Isolate get_settings() across tests (no .env, no cross-test cache bleed).

    Integration fixtures legitimately set + later POP env vars (DSN, endpoints)
    in their teardown, relying — historically — on a per-call .env reload to put
    them back. With .env skipped, a popped DSN would make the next test's
    get_settings() raise ConfigError. So before each test we re-assert the
    baseline required vars (setdefault — never clobbering a test's own value) and
    clear the lru_cache; we clear again after so no Settings object leaks forward.
    """
    import iw_rag.settings as _settings

    for _bk, _bv in _BASELINE_ENV.items():
        os.environ.setdefault(_bk, _bv)
    _settings.get_settings.cache_clear()
    yield
    _settings.get_settings.cache_clear()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run @pytest.mark.live tests against real vLLM/TEI endpoints",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="needs --run-live (real vLLM/TEI endpoints)")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
