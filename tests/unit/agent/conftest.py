"""Shared fixtures for the agent loop unit tests.

These tests exercise the LangGraph loop *mechanics* (router, corrective
retrieval, generation grading, fail-closed) with fake LLM clients. Phase 3.1
inserted ``input_guard`` as the graph entrypoint, which by default scans the
turn with the local Prompt Guard 2 / Granite Guardian clients. Those models are
not served in a plain ``pytest`` run, so — left enabled — the guard would
fail-closed and short-circuit every loop test to the reject terminal.

The input guard's own behaviour (allow/block/fail-closed, node wiring, reject
terminal) is covered by ``tests/unit/test_input_guard.py`` and
``tests/integration/test_input_guard_node.py``. Here we disable the master
guardrails switch so the entrypoint allows through and the loop mechanics under
test are reached. Disabling is via the canonical ``IW_RAG__GUARDRAILS__ENABLED``
env mapping (not by patching internals), and the settings cache is cleared
around each test so the override is isolated.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def _guardrails_disabled(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Disable the master guardrails switch for agent loop mechanics tests.

    The ``input_guard`` entrypoint then returns an ``allow`` verdict without
    calling any (unavailable) guard model, so the router/retrieve/grade loop is
    reached. The guard layer itself is tested directly elsewhere.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__GUARDRAILS__ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    yield
    settings_mod.get_settings.cache_clear()
