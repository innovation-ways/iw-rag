"""Unit tests for the no-op-degrading tracing module."""

from __future__ import annotations

import pytest

from iw_rag.observability import tracing


@pytest.fixture(autouse=True)
def _reset() -> None:
    tracing.reset_tracing_for_tests()


def test_disabled_is_noop() -> None:
    assert tracing.configure_tracing(endpoint=None, enabled=True) is False
    with tracing.span("iw_rag.retrieval.dense"):
        pass  # must not raise
    assert tracing.current_trace_ref() == {}


def test_enabled_false_is_noop() -> None:
    assert tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=False) is False


def test_configure_is_idempotent_when_disabled() -> None:
    assert tracing.configure_tracing(endpoint=None) is False
    # second call returns the same (cached) disabled state without raising
    assert tracing.configure_tracing(endpoint="http://phoenix:4317") is False


def test_span_noop_yields_when_inactive() -> None:
    """tracing.span yields its body when tracing is inactive (no-op path)."""
    entered = False
    with tracing.span("iw_rag.generation.answer"):
        entered = True
    assert entered is True, (
        "tracing.span must yield its body even when tracing is disabled (no-op contract)"
    )


def test_hash_groups_is_stable_and_not_raw() -> None:
    a = tracing.hash_groups(("finance", "hr"))
    b = tracing.hash_groups(("hr", "finance"))  # order-independent
    assert a == b
    assert "finance" not in a and "hr" not in a
    assert len(a) == 16
