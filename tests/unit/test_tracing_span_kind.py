"""Unit tests for the ``span_kind`` parameter on ``observability.tracing.span``.

Adds the OpenInference ``openinference.span.kind`` attribute to OTel spans
(``"RERANKER"``, ``"RETRIEVER"``, ``"LLM"``, ``"GUARDRAIL"``, ``"CHAIN"`` …) and
mirrors the same value on the offline timeline ``StepRecord.attributes`` so the
per-stage latency view works without Phoenix.

RED: the current ``span()`` context manager takes only ``(name, attributes)``;
calling it with the new keyword-only ``span_kind`` parameter raises
``TypeError``.  The tests below also pin the no-op contract when tracing is
disabled (no provider configured).
"""

from __future__ import annotations

import pytest

from iw_rag.observability import tracing
from iw_rag.observability.timeline import record_span, start_timeline


@pytest.fixture(autouse=True)
def _reset_tracing() -> None:
    """Reset tracing state between tests (the OTel tracer provider is global, so
    it can only be set ONCE per process — the OTel-export test installs the
    real provider and is therefore the LAST test to run)."""
    tracing.reset_tracing_for_tests()
    yield
    tracing.reset_tracing_for_tests()


def test_span_accepts_span_kind_keyword() -> None:
    """``span(..., span_kind=...)`` must be accepted and records the kind in the timeline."""
    with (
        start_timeline() as timeline,
        tracing.span("iw_rag.retrieval.rerank", span_kind="RERANKER"),
    ):
        pass  # must not raise
    assert len(timeline.records) == 1, (
        "span with span_kind must produce exactly one timeline record"
    )
    assert timeline.records[0].attributes.get("openinference.span.kind") == "RERANKER", (
        "span_kind='RERANKER' must be recorded as openinference.span.kind attribute"
    )


def test_span_kind_default_is_none_and_callers_unchanged() -> None:
    """Existing callers (no ``span_kind``) still work and inject no openinference attr."""
    with start_timeline() as timeline:
        with tracing.span("iw_rag.retrieval.dense", {"candidate_k": 40}):
            pass
        with tracing.span("iw_rag.retrieval.dense"):
            pass
    assert len(timeline.records) == 2, (
        "Both no-span_kind calls must each produce exactly one timeline record"
    )
    for record in timeline.records:
        assert "openinference.span.kind" not in record.attributes, (
            "openinference.span.kind must NOT be set when span_kind is not provided"
        )


def test_span_kind_emits_openinference_attribute_on_otel_span() -> None:
    """When tracing is active, ``span_kind`` becomes ``openinference.span.kind``.

    The OTel tracer provider is a process-wide global, so we install a fresh
    ``InMemorySpanExporter`` for the duration of the test by swapping the
    provider's tracer; the real ``configure_tracing()`` path is covered by
    integration tests.
    """
    pytest.importorskip("opentelemetry")
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    # Replace the global provider in-place so this test is independent of any
    # earlier test that may have already installed one (OTel's public
    # ``set_tracer_provider`` can only be called once per process).
    original_provider = trace._TRACER_PROVIDER  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    try:
        tracing._ENABLED = True  # type: ignore[attr-defined]
        with tracing.span("iw_rag.retrieval.rerank", span_kind="RERANKER"):
            pass
    finally:
        tracing.reset_tracing_for_tests()
        trace._TRACER_PROVIDER = original_provider  # type: ignore[attr-defined]

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes or {})
    assert attrs.get("openinference.span.kind") == "RERANKER"


def test_span_kind_mirrors_into_timeline_attributes() -> None:
    """``span_kind`` is also folded into the offline timeline ``StepRecord`` attrs."""
    tracing.reset_tracing_for_tests()  # no-op export path
    with (
        start_timeline() as timeline,
        tracing.span("iw_rag.retrieval.rerank", span_kind="RERANKER"),
    ):
        pass
    assert len(timeline.records) == 1
    record = timeline.records[0]
    assert record.name == "iw_rag.retrieval.rerank"
    assert record.attributes.get("openinference.span.kind") == "RERANKER"


def test_span_kind_not_set_when_none() -> None:
    """``span_kind=None`` (default) must NOT add the attribute to the record."""
    tracing.reset_tracing_for_tests()  # no-op export path
    with start_timeline() as timeline, tracing.span("iw_rag.retrieval.dense", {"candidate_k": 40}):
        pass
    record = timeline.records[0]
    assert "openinference.span.kind" not in record.attributes


def test_disabled_path_still_records_and_yields_with_span_kind() -> None:
    """When tracing is disabled, ``span_kind`` is still folded into the timeline and the
    block yields without error (graceful no-op contract preserved)."""
    tracing.reset_tracing_for_tests()  # ensures disabled
    with start_timeline() as timeline:
        entered = False
        with tracing.span("iw_rag.retrieval.rerank", span_kind="RERANKER"):
            entered = True
        assert entered
    assert len(timeline.records) == 1
    assert timeline.records[0].attributes.get("openinference.span.kind") == "RERANKER"


def test_user_attributes_preserved_when_span_kind_set() -> None:
    """Caller-supplied attributes are kept verbatim alongside the span_kind marker."""
    tracing.reset_tracing_for_tests()
    with (
        start_timeline() as timeline,
        tracing.span(
            "iw_rag.retrieval.rerank",
            {"candidates": 12, "top_k": 8},
            span_kind="RERANKER",
        ),
    ):
        pass
    attrs = timeline.records[0].attributes
    assert attrs.get("candidates") == 12
    assert attrs.get("top_k") == 8
    assert attrs.get("openinference.span.kind") == "RERANKER"


def test_record_span_back_compat_with_non_dict() -> None:
    """``record_span`` (called internally by ``span``) must still tolerate non-dict input.

    The timeline implementation already handles ``None``; this pins that the
    span_kind change does not break the call contract when the user supplies
    something falsy.
    """
    with start_timeline() as timeline:
        record_span("iw_rag.retrieval.dense", 0.0, 0.01, None)
        record_span("iw_rag.retrieval.sparse", 0.01, 0.02, {})
    assert len(timeline.records) == 2
