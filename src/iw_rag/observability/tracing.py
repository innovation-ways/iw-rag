"""Observability — OpenTelemetry/OpenInference tracing with graceful no-op.

Instruments the spine/generation pieces as ``iw_rag.<layer>.<op>`` spans on the
OTel GenAI conventions (ADR-017). Designed to **degrade to a no-op** when the
collector is down, tracing is disabled, or the OTel packages aren't installed
(the ``observability`` dep group) — so importing this module and calling
``span``/``current_trace_ref`` is always safe. Raw principal groups never reach a
span (groups are hashed).

**Content capture is opt-in** (lld-eval-observability §PII-in-traces, OTel GenAI
conventions): structural metadata — model, token counts, route, verdicts,
document ids/scores — is always attached via :func:`annotate`; raw text —
questions, prompts, completions, chunk content — only via
:func:`annotate_content`, which is a no-op unless ``configure_tracing`` was
called with ``capture_content=True`` (``IW_RAG__OBSERVABILITY__CAPTURE_CONTENT``).

``configure_tracing`` is idempotent (one TracerProvider). The real OTLP export +
auto-instrumentors live behind a lazy import; everything here works without them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

_PROVIDER: Any | None = None
_ENABLED = False
_CAPTURE_CONTENT = False

# Cap on any single captured text attribute — keeps a huge long-context prompt
# from producing a multi-MB span (Phoenix/OTLP choke on those).
_MAX_CONTENT_CHARS = 32_000

# OpenInference span kind inferred from the span name when the caller passes
# none, so Phoenix renders the tree with proper kinds (CHAIN / RETRIEVER / …)
# instead of "unknown".  First matching prefix wins.  OTel export only — the
# offline timeline attributes are unchanged.
_KIND_BY_PREFIX: tuple[tuple[str, str], ...] = (
    ("iw_rag.generation.llm", "LLM"),
    ("iw_rag.retrieval.encode", "EMBEDDING"),
    ("iw_rag.retrieval.rerank", "RERANKER"),
    ("iw_rag.retrieval.", "RETRIEVER"),
    ("iw_rag.agent.retrieve", "RETRIEVER"),
    ("iw_rag.agent.visual_retrieve", "RETRIEVER"),
    ("iw_rag.agent.graph_retrieve", "RETRIEVER"),
    ("iw_rag.agent.long_context_retrieve", "RETRIEVER"),
    ("iw_rag.guardrails.", "GUARDRAIL"),
    ("iw_rag.agent.input_guard", "GUARDRAIL"),
    ("iw_rag.", "CHAIN"),
)


def _infer_span_kind(name: str) -> str | None:
    for prefix, kind in _KIND_BY_PREFIX:
        if name.startswith(prefix):
            return kind
    return None


def configure_tracing(
    *,
    endpoint: str | None,
    service_name: str = "iw-rag",
    enabled: bool = True,
    capture_content: bool = False,
) -> bool:
    """Configure a single global TracerProvider (idempotent).

    No-ops (returns False) when disabled, when ``endpoint`` is unset, or when the
    OTel packages are not installed — tracing then silently does nothing.

    Args:
        endpoint: OTLP gRPC endpoint (e.g. ``http://phoenix:4317``); None disables.
        service_name: The service resource name.
        enabled: Master switch.
        capture_content: Opt-in raw-content capture (prompts, completions,
            questions, chunk text) via :func:`annotate_content`.  Default off.

    Returns:
        True if tracing is now active, else False.
    """
    global _PROVIDER, _ENABLED, _CAPTURE_CONTENT
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
    _CAPTURE_CONTENT = capture_content
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
    otel_attrs = dict(span_attrs)
    if "openinference.span.kind" not in otel_attrs:
        inferred = _infer_span_kind(name)
        if inferred is not None:
            otel_attrs["openinference.span.kind"] = inferred
    with tracer.start_as_current_span(name) as current:
        for key, value in otel_attrs.items():
            current.set_attribute(key, value)
        try:
            yield
        finally:
            record_span(name, start, time.perf_counter(), span_attrs)


def _coerce(value: Any) -> Any:
    """Coerce a value to an OTel-legal attribute type (or ``None`` to skip)."""
    if value is None:
        return None
    if isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
        return list(value)
    if isinstance(value, list | tuple) and all(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value
    ):
        return list(value)
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _set_on_current(attrs: Mapping[str, Any], *, truncate: bool) -> None:
    from opentelemetry import trace  # noqa: PLC0415

    current = trace.get_current_span()
    if not current.is_recording():
        return
    for key, value in attrs.items():
        coerced = _coerce(value)
        if coerced is None:
            continue
        if truncate and isinstance(coerced, str) and len(coerced) > _MAX_CONTENT_CHARS:
            coerced = coerced[:_MAX_CONTENT_CHARS] + "…[truncated]"
        current.set_attribute(key, coerced)


def annotate(attrs: Mapping[str, Any]) -> None:
    """Attach **structural** attributes (no raw text) to the current span.

    For results only known after the work ran — chosen route, verdicts, token
    counts, document ids/scores.  No-op when tracing is inactive or no span is
    recording.  ``None`` values are skipped; non-scalar values are JSON-encoded.
    MUST NOT carry raw content or raw principal groups — use
    :func:`annotate_content` for text.

    Args:
        attrs: Attribute key → value.
    """
    if not _ENABLED:
        return
    _set_on_current(attrs, truncate=False)


def annotate_content(attrs: Mapping[str, Any]) -> None:
    """Attach **raw content** (prompts, completions, questions, chunk text).

    Opt-in: a no-op unless tracing was configured with ``capture_content=True``.
    Long strings are truncated to ``_MAX_CONTENT_CHARS``.

    Args:
        attrs: Attribute key → value (e.g. ``input.value``, ``output.value``,
            ``llm.input_messages.0.message.content``).
    """
    if not (_ENABLED and _CAPTURE_CONTENT):
        return
    _set_on_current(attrs, truncate=True)


def trace_verdict(verdict: Any) -> None:
    """Attach an agent ``Verdict`` (kind/value/score/attempt) to the current span.

    The verdict ``reason`` can paraphrase user or document text (e.g. the
    router's classifier reason), so it is recorded only under content capture.

    Args:
        verdict: Any object with ``kind``/``value`` (+ optional ``score``,
            ``reason``, ``attempt``) — duck-typed to keep this module free of
            agent imports.
    """
    if not _ENABLED:
        return
    annotate(
        {
            "iw_rag.verdict.kind": getattr(verdict, "kind", None),
            "iw_rag.verdict.value": getattr(verdict, "value", None),
            "iw_rag.verdict.score": getattr(verdict, "score", None),
            "iw_rag.verdict.attempt": getattr(verdict, "attempt", None),
        }
    )
    annotate_content({"iw_rag.verdict.reason": getattr(verdict, "reason", None)})


def content_capture_enabled() -> bool:
    """Whether raw content is being captured (lets callers skip building it)."""
    return _ENABLED and _CAPTURE_CONTENT


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
    global _PROVIDER, _ENABLED, _CAPTURE_CONTENT
    _PROVIDER = None
    _ENABLED = False
    _CAPTURE_CONTENT = False
