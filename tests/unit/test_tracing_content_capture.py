"""Tracing richness + the opt-in content-capture policy.

Structural metadata (model, tokens, verdicts, ids/scores, span kinds) must
always reach the span; raw text (prompts, completions, questions, chunk text)
only when ``capture_content`` is opted in (lld-eval-observability
§PII-in-traces).  Uses an in-memory exporter swapped into the global provider
(same pattern as ``test_tracing_span_kind.py``).
"""

from __future__ import annotations

import types
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from iw_rag.generation.llm import GenerationParams, VllmClient  # noqa: E402
from iw_rag.observability import tracing  # noqa: E402
from iw_rag.observability.timeline import start_timeline  # noqa: E402


@pytest.fixture
def exporter() -> Iterator[InMemorySpanExporter]:
    exp = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exp))
    original = trace._TRACER_PROVIDER  # type: ignore[attr-defined]
    trace._TRACER_PROVIDER = provider  # type: ignore[attr-defined]
    tracing.reset_tracing_for_tests()
    tracing._ENABLED = True  # type: ignore[attr-defined]
    try:
        yield exp
    finally:
        tracing.reset_tracing_for_tests()
        trace._TRACER_PROVIDER = original  # type: ignore[attr-defined]


def _attrs(exp: InMemorySpanExporter, name: str) -> dict[str, Any]:
    spans = [s for s in exp.get_finished_spans() if s.name == name]
    assert len(spans) == 1, [s.name for s in exp.get_finished_spans()]
    return dict(spans[0].attributes or {})


@pytest.mark.parametrize("capture", [False, True])
def test_annotate_always_content_only_when_opted_in(
    exporter: InMemorySpanExporter, capture: bool
) -> None:
    tracing._CAPTURE_CONTENT = capture  # type: ignore[attr-defined]
    with tracing.span("iw_rag.agent.route"):
        tracing.annotate({"iw_rag.route.actual": "retrieve", "skipped": None})
        tracing.annotate_content({"input.value": "My name is Sérgio"})

    attrs = _attrs(exporter, "iw_rag.agent.route")
    assert attrs["iw_rag.route.actual"] == "retrieve"
    assert "skipped" not in attrs  # None values are not written
    assert ("input.value" in attrs) is capture
    if capture:
        assert attrs["input.value"] == "My name is Sérgio"


def test_annotate_content_truncates_huge_strings(exporter: InMemorySpanExporter) -> None:
    tracing._CAPTURE_CONTENT = True  # type: ignore[attr-defined]
    with tracing.span("iw_rag.generation.answer"):
        tracing.annotate_content({"input.value": "x" * 100_000})
    value = _attrs(exporter, "iw_rag.generation.answer")["input.value"]
    assert value.endswith("…[truncated]")
    assert len(value) < 40_000


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("iw_rag.query", "CHAIN"),
        ("iw_rag.agent.grade_documents", "CHAIN"),
        ("iw_rag.agent.retrieve", "RETRIEVER"),
        ("iw_rag.retrieval.dense", "RETRIEVER"),
        ("iw_rag.retrieval.encode", "EMBEDDING"),
        ("iw_rag.guardrails.output_guard", "GUARDRAIL"),
        ("iw_rag.agent.input_guard", "GUARDRAIL"),
    ],
)
def test_span_kind_inferred_for_otel_but_not_timeline(
    exporter: InMemorySpanExporter, name: str, kind: str
) -> None:
    with start_timeline() as timeline, tracing.span(name):
        pass
    assert _attrs(exporter, name)["openinference.span.kind"] == kind
    # The offline timeline (console breakdown) is unchanged by the inference.
    assert "openinference.span.kind" not in timeline.records[0].attributes


@pytest.mark.parametrize("capture", [False, True])
def test_trace_verdict_reason_is_content_gated(
    exporter: InMemorySpanExporter, capture: bool
) -> None:
    tracing._CAPTURE_CONTENT = capture  # type: ignore[attr-defined]
    verdict = types.SimpleNamespace(
        kind="router", value="skip", score=None, reason="user greets with name", attempt=0
    )
    with tracing.span("iw_rag.agent.route"):
        tracing.trace_verdict(verdict)
    attrs = _attrs(exporter, "iw_rag.agent.route")
    assert attrs["iw_rag.verdict.kind"] == "router"
    assert attrs["iw_rag.verdict.value"] == "skip"
    assert ("iw_rag.verdict.reason" in attrs) is capture


@pytest.mark.asyncio
@pytest.mark.parametrize("capture", [False, True])
async def test_vllm_call_emits_llm_span(exporter: InMemorySpanExporter, capture: bool) -> None:
    """Every generator call is an LLM span: model + params + tokens always,
    prompt messages + completion only under content capture."""
    tracing._CAPTURE_CONTENT = capture  # type: ignore[attr-defined]
    client = VllmClient("http://vllm:8000/v1", enable_thinking=False)

    async def _fake_post(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "Olá, Sérgio!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 5, "total_tokens": 47},
        }

    client._post = _fake_post  # type: ignore[method-assign]
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "O meu nome é Sérgio"},
    ]
    resp = await client.complete(messages, GenerationParams(model="qwen", max_tokens=64))
    assert resp.text == "Olá, Sérgio!"

    attrs = _attrs(exporter, "iw_rag.generation.llm")
    assert attrs["openinference.span.kind"] == "LLM"
    assert attrs["llm.model_name"] == "qwen"
    assert attrs["llm.token_count.prompt"] == 42
    assert attrs["llm.token_count.completion"] == 5
    assert attrs["llm.token_count.total"] == 47
    assert attrs["llm.finish_reason"] == "stop"
    assert '"max_tokens": 64' in attrs["llm.invocation_parameters"]

    content_keys = {
        "llm.input_messages.0.message.content",
        "llm.input_messages.1.message.content",
        "llm.output_messages.0.message.content",
        "input.value",
        "output.value",
    }
    if capture:
        assert attrs["llm.input_messages.0.message.role"] == "system"
        assert attrs["llm.input_messages.1.message.content"] == "O meu nome é Sérgio"
        assert attrs["llm.output_messages.0.message.content"] == "Olá, Sérgio!"
        assert attrs["output.value"] == "Olá, Sérgio!"
    else:
        assert content_keys.isdisjoint(attrs)


def test_configure_tracing_threads_capture_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """``capture_content`` defaults off and is honoured when passed."""
    tracing.reset_tracing_for_tests()
    assert tracing.content_capture_enabled() is False
    monkeypatch.setattr(trace, "set_tracer_provider", lambda _p: None)
    assert tracing.configure_tracing(endpoint="http://localhost:4317", capture_content=True)
    try:
        assert tracing.content_capture_enabled() is True
    finally:
        tracing.reset_tracing_for_tests()


def test_settings_capture_content_defaults_off() -> None:
    from iw_rag.settings import ObservabilitySettings

    assert ObservabilitySettings().capture_content is False
