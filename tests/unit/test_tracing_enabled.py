"""Unit tests for the ENABLED path of iw_rag.observability.tracing.

Covers the success branches that require the OpenTelemetry SDK
(``observability`` dep group). The entire module is skipped cleanly when the
SDK is absent — ``pytest.importorskip("opentelemetry.sdk")`` at module level
intercepts the missing import and marks all tests as skipped.

Tests (ENABLED path):
- ``configure_tracing(endpoint=..., enabled=True)`` returns True and installs
  a TracerProvider.
- ``configure_tracing`` is idempotent: a second call with the same args returns
  the already-cached result without installing a second provider.
- ``current_trace_ref()`` returns ``{"trace_id": ..., "span_id": ...}`` when
  called inside an active span, with exact 32-char and 16-char hex format.
- ``current_trace_ref()`` with an invalid (non-recording) span context returns
  ``{}`` (the ctx.is_valid false sub-branch).
- ``hash_groups(())`` on an empty tuple returns the exact SHA-256 digest of the
  empty string, truncated to 16 hex chars.

Isolation: every test resets the module-level globals via
``tracing.reset_tracing_for_tests()``.

References:
- src/iw_rag/observability/tracing.py — source of truth.
- ADR-017 — OTel/GenAI conventions.
"""

from __future__ import annotations

import hashlib

import pytest

# Guard: skip the entire module if the OTel SDK is not installed.
# ``observability`` dep group is NOT a dev dependency, so CI without it is fine.
pytest.importorskip("opentelemetry.sdk")

from iw_rag.observability import tracing  # noqa: E402 (after importorskip guard)


@pytest.fixture(autouse=True)
def _reset() -> object:
    """Reset tracing module globals before and after each test.

    Returns:
        A pytest fixture generator that resets the OTel provider state.
    """
    tracing.reset_tracing_for_tests()
    yield
    tracing.reset_tracing_for_tests()


# ---------------------------------------------------------------------------
# configure_tracing — ENABLED / success branch
# ---------------------------------------------------------------------------


class TestConfigureTracingEnabled:
    """configure_tracing returns True and installs a provider when SDK is available."""

    def test_configure_tracing_with_endpoint_returns_true(self) -> None:
        """configure_tracing with a non-None endpoint and enabled=True returns True."""
        from opentelemetry.sdk.trace import TracerProvider

        result = tracing.configure_tracing(
            endpoint="http://phoenix:4317",
            service_name="iw-rag-test",
            enabled=True,
        )
        assert result is True, (
            "configure_tracing must return True when endpoint is set and enabled=True"
        )
        # The internal _PROVIDER must be a real TracerProvider instance.
        assert isinstance(tracing._PROVIDER, TracerProvider), (  # type: ignore[attr-defined]
            "configure_tracing must install a TracerProvider instance"
        )

    def test_configure_tracing_sets_enabled_flag(self) -> None:
        """After a successful configure_tracing call _ENABLED is True."""
        tracing.configure_tracing(
            endpoint="http://phoenix:4317",
            service_name="iw-rag-test",
            enabled=True,
        )
        assert tracing._ENABLED is True, "_ENABLED must be True after successful configuration"  # type: ignore[attr-defined]

    def test_configure_tracing_is_idempotent_when_provider_already_set(self) -> None:
        """A second configure_tracing call returns the cached _ENABLED without re-installing."""
        first = tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        assert first is True
        # A second call with different args returns the cached state.
        second = tracing.configure_tracing(endpoint=None, enabled=False)
        assert second is True, (
            "configure_tracing must be idempotent: returns cached True when provider already set"
        )

    def test_configure_tracing_service_name_set_on_resource(self) -> None:
        """The TracerProvider resource carries the requested service.name."""
        tracing.configure_tracing(
            endpoint="http://phoenix:4317",
            service_name="my-test-service",
            enabled=True,
        )
        provider = tracing._PROVIDER  # type: ignore[attr-defined]
        resource_attrs = provider.resource.attributes
        assert resource_attrs.get("service.name") == "my-test-service", (
            "service_name must be stored in the TracerProvider resource as service.name"
        )


# ---------------------------------------------------------------------------
# current_trace_ref — ENABLED branch
# ---------------------------------------------------------------------------


class TestCurrentTraceRefEnabled:
    """current_trace_ref returns the active span ids or {} when no valid span."""

    def test_current_trace_ref_outside_span_returns_empty(self) -> None:
        """current_trace_ref returns {} when no span is active (ctx.is_valid false)."""
        tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        # No active span — the default INVALID_SPAN context is not valid.
        result = tracing.current_trace_ref()
        assert result == {}, "current_trace_ref must return {} when no active span is present"

    def test_current_trace_ref_inside_span_returns_exact_keys(self) -> None:
        """Inside a tracing.span context, current_trace_ref returns {trace_id, span_id}."""
        tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        with tracing.span("iw_rag.test.probe"):
            ref = tracing.current_trace_ref()
        assert set(ref.keys()) == {"trace_id", "span_id"}, (
            "current_trace_ref must return exactly {trace_id, span_id} inside an active span"
        )

    def test_current_trace_ref_trace_id_is_32_hex_chars(self) -> None:
        """trace_id in the returned ref is a 32-character lowercase hex string."""
        tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        with tracing.span("iw_rag.test.probe"):
            ref = tracing.current_trace_ref()
        trace_id = ref["trace_id"]
        assert len(trace_id) == 32, (
            f"trace_id must be 32 hex chars, got {len(trace_id)}: {trace_id!r}"
        )
        assert all(c in "0123456789abcdef" for c in trace_id), (
            f"trace_id must be lowercase hex, got: {trace_id!r}"
        )

    def test_current_trace_ref_span_id_is_16_hex_chars(self) -> None:
        """span_id in the returned ref is a 16-character lowercase hex string."""
        tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        with tracing.span("iw_rag.test.probe"):
            ref = tracing.current_trace_ref()
        span_id = ref["span_id"]
        assert len(span_id) == 16, f"span_id must be 16 hex chars, got {len(span_id)}: {span_id!r}"
        assert all(c in "0123456789abcdef" for c in span_id), (
            f"span_id must be lowercase hex, got: {span_id!r}"
        )

    def test_current_trace_ref_is_valid_false_path_returns_empty(self) -> None:
        """current_trace_ref returns {} when ctx.is_valid is False (invalid span context).

        This exercises the ``if not ctx.is_valid: return {}`` sub-branch inside
        the ENABLED path of current_trace_ref.
        """
        from opentelemetry import trace as otel_trace

        tracing.configure_tracing(endpoint="http://phoenix:4317", enabled=True)
        # Retrieve the INVALID_SPAN context directly — is_valid is False.
        invalid_ctx = otel_trace.INVALID_SPAN.get_span_context()
        assert invalid_ctx.is_valid is False, "Precondition: INVALID_SPAN must have is_valid=False"
        # Outside any span the active span is the INVALID_SPAN.
        result = tracing.current_trace_ref()
        assert result == {}, "current_trace_ref must return {} for ctx.is_valid=False sub-branch"


# ---------------------------------------------------------------------------
# hash_groups — empty-tuple edge case
# ---------------------------------------------------------------------------


class TestHashGroupsEdgeCases:
    """hash_groups handles edge cases including the empty-tuple input."""

    def test_hash_groups_empty_tuple_exact_digest(self) -> None:
        """hash_groups(()) returns the first 16 hex chars of sha256('').hexdigest()."""
        expected = hashlib.sha256(b"").hexdigest()[:16]
        result = tracing.hash_groups(())
        assert result == expected, (
            f"hash_groups(()) must equal sha256('')[:16]={expected!r}, got {result!r}"
        )

    def test_hash_groups_empty_tuple_length_is_16(self) -> None:
        """hash_groups(()) always returns exactly 16 characters."""
        result = tracing.hash_groups(())
        assert len(result) == 16, f"hash_groups digest must be 16 chars, got {len(result)}"

    def test_hash_groups_empty_tuple_is_deterministic(self) -> None:
        """hash_groups(()) returns the same value on repeated calls (stable digest)."""
        first = tracing.hash_groups(())
        second = tracing.hash_groups(())
        assert first == second, "hash_groups(()) must be deterministic"

    def test_hash_groups_empty_differs_from_nonempty(self) -> None:
        """hash_groups(()) produces a different digest than hash_groups(('group',))."""
        empty_hash = tracing.hash_groups(())
        nonempty_hash = tracing.hash_groups(("group",))
        assert empty_hash != nonempty_hash, (
            "hash_groups(()) must differ from hash_groups(('group',))"
        )
