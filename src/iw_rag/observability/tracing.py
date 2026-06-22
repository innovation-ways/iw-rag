"""Observability — OpenTelemetry/OpenInference tracing with graceful no-op.

Instruments the spine/generation pieces as ``iw_rag.<layer>.<op>`` spans on the
OTel GenAI conventions (ADR-017). Designed to **degrade to a no-op** when the
collector is down, tracing is disabled, or the OTel packages aren't installed
(the ``observability`` dep group) — so importing this module and calling
``span``/``current_trace_ref`` is always safe. Span attributes never carry raw
chunk content or raw principal groups (groups are hashed).

``configure_tracing`` is idempotent (one TracerProvider). The real OTLP export +
auto-instrumentors live behind a lazy import; everything here works without them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_PROVIDER: Any | None = None
_ENABLED = False


def configure_tracing(
    *, endpoint: str | None, service_name: str = "iw-rag", enabled: bool = True
) -> bool:
    """Configure a single global TracerProvider (idempotent).

    No-ops (returns False) when disabled, when ``endpoint`` is unset, or when the
    OTel packages are not installed — tracing then silently does nothing.

    Args:
        endpoint: OTLP gRPC endpoint (e.g. ``http://phoenix:4317``); None disables.
        service_name: The service resource name.
        enabled: Master switch.

    Returns:
        True if tracing is now active, else False.
    """
    global _PROVIDER, _ENABLED
    if _PROVIDER is not None:
        return _ENABLED
    if not enabled or not endpoint:
        _ENABLED = False
        return False
    try:
        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
    except ImportError:
        _ENABLED = False
        return False

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    _PROVIDER = provider
    _ENABLED = True
    return True


@contextmanager
def span(
    name: str,
    attributes: dict[str, Any] | None = None,
    *,
    span_kind: str | None = None,
) -> Iterator[None]:
    """Start a span named ``name`` (no-op export when tracing is inactive).

    Always times the wrapped block and records it to the active per-request
    timeline (:mod:`iw_rag.observability.timeline`) so the UI's timing breakdown
    works even when OpenTelemetry/Phoenix is disabled or unreachable.  The OTel
    span export is additionally emitted only when tracing is configured.

    Args:
        name: Span name, conventionally ``iw_rag.<layer>.<op>``.
        attributes: Span attributes (MUST NOT include raw content/principal groups).
        span_kind: Optional OpenInference span kind, set as the
            ``openinference.span.kind`` attribute on the OTel span AND folded
            into the offline timeline ``StepRecord`` attributes so the per-stage
            view (RERANKER, RETRIEVER, LLM, GUARDRAIL, CHAIN, …) works without
            Phoenix.  When ``None`` (the default) the attribute is not added —
            existing callers are unchanged.

    Yields:
        None.
    """
    import time  # noqa: PLC0415

    from iw_rag.observability.timeline import record_span  # noqa: PLC0415

    start = time.perf_counter()
    span_attrs: dict[str, Any] = dict(attributes) if attributes else {}
    if span_kind is not None:
        span_attrs.setdefault("openinference.span.kind", span_kind)
    if not _ENABLED:
        try:
            yield
        finally:
            record_span(name, start, time.perf_counter(), span_attrs)
        return
    from opentelemetry import trace  # noqa: PLC0415

    tracer = trace.get_tracer("iw_rag")
    with tracer.start_as_current_span(name) as current:
        for key, value in span_attrs.items():
            current.set_attribute(key, value)
        try:
            yield
        finally:
            record_span(name, start, time.perf_counter(), span_attrs)


def current_trace_ref() -> dict[str, str]:
    """Return ``{trace_id, span_id}`` of the active span, or ``{}`` if none.

    Used as ``eval_results.trace_ref`` and the frontend's "view trace" link.
    """
    if not _ENABLED:
        return {}
    from opentelemetry import trace  # noqa: PLC0415

    ctx = trace.get_current_span().get_span_context()
    if not ctx.is_valid:
        return {}
    return {"trace_id": format(ctx.trace_id, "032x"), "span_id": format(ctx.span_id, "016x")}


def hash_groups(groups: tuple[str, ...]) -> str:
    """Stable, non-reversible digest of principal groups (for span attributes).

    Args:
        groups: The principal's groups.

    Returns:
        A short sha256 hex digest; never the raw group names.
    """
    joined = ",".join(sorted(groups))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def reset_tracing_for_tests() -> None:
    """Reset module state (test-only helper)."""
    global _PROVIDER, _ENABLED
    _PROVIDER = None
    _ENABLED = False
