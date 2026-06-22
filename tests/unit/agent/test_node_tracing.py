"""Unit tests for loop tracing — span wrapping in agent nodes.

TDD RED phase: these tests define the expected behaviour:
- Tracing disabled (default) is a no-op; the loop still produces the same
  terminal state.
- current_trace_ref() returns {} when disabled.
- Span calls do not raise or mutate state.

These tests use the fake spine/assembler/LLM already established in the agent
test suite; no live models required.
"""

from __future__ import annotations

import pytest

from iw_rag.agent.state import GraphState
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer, RetrievalResult, RetrievedChunk
from iw_rag.observability import tracing

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-trace-test",
)

_CHUNK = RetrievedChunk(
    chunk_id="c1",
    document_id="doc-1",
    content="leave policy text",
    score=0.9,
    source_uri="corpus://hr/policy.md",
)
_RESULT = RetrievalResult(chunks=[_CHUNK], retriever="spine", acl_applied=True)
_ANSWER = GeneratedAnswer(
    answer="You have 20 days [1].",
    grounded=True,
    citations=[
        Citation(
            marker=1,
            chunk_id="c1",
            document_id="doc-1",
            source_uri="corpus://hr/policy.md",
            quote="20 days",
        )
    ],
    used_chunk_ids=["c1"],
)


class _FakeSpine:
    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        """Return canned result."""
        return _RESULT


class _FakeAssembler:
    async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
        """Return canned answer."""
        return _ANSWER


@pytest.fixture(autouse=True)
def _reset_tracing() -> None:
    """Reset tracing module to disabled state before each test."""
    tracing.reset_tracing_for_tests()


@pytest.mark.asyncio
async def test_tracing_disabled_loop_produces_terminal_state() -> None:
    """With tracing disabled (default), the graph runs without error and returns terminal state."""
    from iw_rag.agent.graph import answer_question, build_graph

    # Tracing is disabled by default (reset_tracing_for_tests called above)
    assert tracing.current_trace_ref() == {}

    graph = build_graph(_FakeSpine(), _FakeAssembler())
    state = await answer_question(graph, "what is the leave policy?", _CTX)

    # The loop still produced a valid terminal state with no tracing side-effects
    assert state.get("disposition") in ("answered", "refused")
    # current_trace_ref remains {} since tracing is still disabled
    assert tracing.current_trace_ref() == {}


@pytest.mark.asyncio
async def test_span_context_manager_noop_does_not_raise() -> None:
    """span() context manager is a no-op when tracing is disabled."""
    # Direct check: span must not raise or modify any external state
    with tracing.span("iw_rag.agent.router", {"node": "router", "attempt": 0}):
        pass  # must not raise

    # Still no trace ref
    assert tracing.current_trace_ref() == {}


@pytest.mark.asyncio
async def test_router_node_with_tracing_disabled_does_not_raise() -> None:
    """The router node returns route='retrieve' and one verdict when tracing is off."""
    from unittest.mock import AsyncMock

    from iw_rag.agent.nodes import make_router
    from iw_rag.generation.llm import LlmResponse
    from iw_rag.settings import AgentSettings

    fake_llm = AsyncMock()
    fake_llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text='{"route": "retrieve", "query_type": "factual", "reason": "test"}',
        )
    )

    settings = AgentSettings()
    router_node = make_router(fake_llm, settings)

    state: GraphState = {
        "question": "test question",
        "original_question": "test question",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    # Should not raise even with spans in node body
    result = await router_node(state)
    assert "route" in result
    assert "verdicts" in result
    # Exact value checks: the LLM returned route='retrieve'
    assert result["route"] == "retrieve", (
        f"Expected route='retrieve' from the fake LLM response, got {result['route']!r}"
    )
    # The router appends exactly one verdict per call
    verdicts = result["verdicts"]
    assert len(verdicts) == 1, f"Expected exactly 1 verdict from router, got {len(verdicts)}"
    assert verdicts[0].kind == "router", f"Expected verdict.kind='router', got {verdicts[0].kind!r}"


@pytest.mark.asyncio
async def test_fail_closed_node_with_tracing_disabled() -> None:
    """fail_closed node returns the correct terminal state even with span wrapping."""
    from iw_rag.agent.nodes import fail_closed

    state: GraphState = {
        "question": "Q",
        "original_question": "Q",
        "access_context": _CTX,
        "retries": 2,
        "max_retries": 2,
        "verdicts": [],
        "doc_verdict": "incorrect",
        "gen_verdict": "unsupported",
    }

    result = await fail_closed(state)
    assert result["disposition"] == "refused"
    assert result["answer"] is None
    assert "fail_closed" in result["verdicts"][0].kind


@pytest.mark.asyncio
async def test_done_node_with_tracing_disabled() -> None:
    """done node returns answered disposition with no tracing interference."""
    from iw_rag.agent.nodes import done

    state: GraphState = {
        "question": "Q",
        "original_question": "Q",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "draft": "This is the grounded answer.",
        "gen_verdict": "grounded",
    }

    result = await done(state)
    assert result["disposition"] == "answered"
    assert result["answer"] == "This is the grounded answer."


def test_span_attributes_never_include_raw_groups() -> None:
    """hash_groups must produce a non-reversible digest that doesn't include group names."""
    groups = ("finance", "hr", "exec")
    hashed = tracing.hash_groups(groups)
    for g in groups:
        assert g not in hashed, f"Group name {g!r} leaked into hash output"
    # Must be the canonical 16-char hex
    assert len(hashed) == 16
    # Must be stable
    assert hashed == tracing.hash_groups(groups)
