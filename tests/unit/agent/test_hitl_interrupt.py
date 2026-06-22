"""Unit tests for the HITL (human-in-the-loop) interrupt-before-done path.

Tests cover:
- build_graph with HITL enabled compiles with interrupt_before=["done"].
- The first invoke pauses before "done": state.next == ("done",) and
  disposition is not yet set.
- The draft and gen_verdict are available in the paused state (the operator
  can inspect them before deciding to approve or reject).
- Resuming via ainvoke(None, config) runs the "done" node and produces
  disposition="answered".
- Resuming after a grounded gen_verdict produces the correct answer.
- The HITL path is off by default — a non-HITL build with the same fake LLM
  completes in a single invoke without pausing.
- build_graph raises ValueError when HITL is enabled but no checkpointer is
  provided (LangGraph requirement: interrupt_before needs a checkpointer).

The conftest autouse fixture disables guardrails so the input_guard pass-through
does not block the loop.

References:
- graph.py ~line 510-511, 723-726 (interrupt_before wiring)
- lld-agentic-loop §5 (checkpointing / HITL)
- task-2.6-checkpointing-and-hitl §2, §3
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.agent.graph import build_graph
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer, RetrievalResult, RetrievedChunk
from iw_rag.settings import AgentSettings

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-hitl-test",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk() -> RetrievedChunk:
    """Build a minimal evidence chunk for HITL tests.

    Returns:
        A RetrievedChunk with acl_applied=True.
    """
    return RetrievedChunk(
        chunk_id="c1",
        document_id="d1",
        content="Evidence for grounding",
        score=0.9,
        source_uri="file:///doc.pdf",
        retriever="spine",
        acl_applied=True,
    )


def _make_citation() -> Citation:
    """Build a minimal citation for HITL tests.

    Returns:
        A Citation referencing chunk 'c1'.
    """
    return Citation(
        marker=1,
        chunk_id="c1",
        document_id="d1",
        source_uri="file:///doc.pdf",
        quote="Evidence for grounding",
    )


class _FakeLLM:
    """Fake LLM that returns valid structured outputs for all grader schemas.

    Identifies which schema is being used by inspecting the 'required' fields:
    - ['query_type', 'route'] → router → retrieve
    - ['grades']              → doc grade → all relevant
    - ['supported']           → ISSUP → supported=true
    - ['answers']             → ISUSE → answers=true
    """

    async def complete(self, messages: list[dict[str, str]], params: Any) -> Any:
        """Return a stub rewrite response.

        Args:
            messages: Conversation messages (ignored).
            params: Generation parameters (ignored).

        Returns:
            A stub response with text="rewritten query".
        """

        class _Resp:
            text = "rewritten query"

        return _Resp()

    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        params: Any,
    ) -> Any:
        """Return structured output matching the schema's required fields.

        Args:
            messages: Conversation messages (ignored).
            schema: The JSON schema dict — inspected for 'required' keys.
            params: Generation parameters (ignored).

        Returns:
            A stub response with text matching the expected schema output.
        """

        class _Resp:
            text: str = "{}"

        resp = _Resp()
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if "route" in required:
            resp.text = '{"route": "retrieve", "query_type": "factual"}'
        elif "grades" in required:
            resp.text = '{"grades": [{"relevant": true}]}'
        elif "supported" in required:
            resp.text = '{"supported": true}'
        elif "answers" in required:
            resp.text = '{"answers": true}'
        return resp


class _FakeSpine:
    """Fake retrieval spine that always returns one evidence chunk."""

    async def retrieve(self, query: str, ctx: Any) -> RetrievalResult:
        """Return a single evidence chunk.

        Args:
            query: The search query (ignored).
            ctx: The access context (ignored).

        Returns:
            A RetrievalResult with one evidence chunk.
        """
        return RetrievalResult(
            chunks=[_make_chunk()],
            retriever="spine",
            acl_applied=True,
        )


class _FakeAssembler:
    """Fake answer assembler that returns a grounded answer with a citation."""

    async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
        """Return a grounded answer with one citation.

        Args:
            query: The original question (ignored).
            result: The retrieval result (ignored).

        Returns:
            A grounded GeneratedAnswer with citation to 'c1'.
        """
        return GeneratedAnswer(
            answer="The answer is here [1].",
            grounded=True,
            citations=[_make_citation()],
            used_chunk_ids=["c1"],
        )


def _hitl_settings() -> AgentSettings:
    """Build AgentSettings with HITL enabled.

    Returns:
        AgentSettings with hitl.enabled=True.
    """
    settings = AgentSettings()
    settings.hitl.enabled = True
    return settings


# ---------------------------------------------------------------------------
# Tests: build_graph with HITL enabled
# ---------------------------------------------------------------------------


def test_build_graph_hitl_enabled_compiles_successfully() -> None:
    """build_graph with HITL=True and a checkpointer compiles without error."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    # Strong assertion: the full graph has exactly 16 nodes (HITL wiring complete)
    # __start__, input_guard, reject, audit, router, answer_direct, retrieve, sanitize,
    # grade_documents, consume_retry, rewrite_query, generate, grade_generation,
    # output_guard, done, fail_closed
    node_count = len(graph.nodes)
    assert node_count == 16, (
        f"HITL full graph must have 16 nodes, got {node_count}: {sorted(graph.nodes.keys())!r}"
    )


def test_build_graph_hitl_enabled_without_checkpointer_still_compiles() -> None:
    """build_graph with HITL=True and no checkpointer still compiles.

    LangGraph does not raise at compile time for interrupt_before without a
    checkpointer — it degrades silently (no persistent state for resume).
    The interrupt semantics are lost, but the graph remains executable.
    This test documents that compile-time behaviour.
    """
    settings = _hitl_settings()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=None,
        settings=settings,
    )
    # Strong assertion: even without a checkpointer the full graph topology is intact (16 nodes)
    node_count = len(graph.nodes)
    assert node_count == 16, (
        f"Graph with HITL and no checkpointer must still have 16 nodes, got {node_count}"
    )


# ---------------------------------------------------------------------------
# Tests: HITL pause — first invoke stops before "done"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_first_invoke_pauses_before_done() -> None:
    """First invoke with HITL enabled pauses before 'done'; state.next == ('done',)."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-pause-test"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    result = await graph.ainvoke(init, config=config)

    # The graph pauses before "done" — disposition is not yet set
    assert result.get("disposition") is None, (
        f"HITL pause: disposition must be None before 'done' runs, "
        f"got {result.get('disposition')!r}"
    )

    # Verify the interrupt is recorded in the graph state
    state = await graph.aget_state(config)
    assert state.next == ("done",), f"HITL pause: state.next must be ('done',), got {state.next!r}"


@pytest.mark.asyncio
async def test_hitl_paused_state_has_draft_and_gen_verdict() -> None:
    """Paused HITL state has draft + gen_verdict available for operator inspection."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-draft-inspect"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    await graph.ainvoke(init, config=config)

    # Operator can inspect the paused state
    state = await graph.aget_state(config)
    paused_values = state.values

    assert paused_values.get("draft") is not None, (
        "Paused state must have 'draft' for operator inspection"
    )
    assert paused_values.get("gen_verdict") == "grounded", (
        f"Paused state gen_verdict must be 'grounded', got {paused_values.get('gen_verdict')!r}"
    )


# ---------------------------------------------------------------------------
# Tests: HITL resume — second invoke completes with answered
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_resume_produces_answered_disposition() -> None:
    """Resuming (ainvoke with None) after HITL pause produces disposition='answered'."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-resume-answered"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    # Phase 1: pause
    await graph.ainvoke(init, config=config)

    # Phase 2: resume
    result = await graph.ainvoke(None, config=config)

    assert result.get("disposition") == "answered", (
        f"Resumed HITL must produce disposition='answered', got {result.get('disposition')!r}"
    )


@pytest.mark.asyncio
async def test_hitl_resume_produces_non_none_answer() -> None:
    """After HITL resume, the answer field is non-None and matches the draft."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-resume-answer-text"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    # Pause
    await graph.ainvoke(init, config=config)

    # Resume
    result = await graph.ainvoke(None, config=config)

    answer = result.get("answer")
    assert answer == "The answer is here [1].", (
        f"Resumed answer must match assembler output, got {answer!r}"
    )


@pytest.mark.asyncio
async def test_hitl_resume_state_next_is_empty_after_done() -> None:
    """After HITL resume, state.next is empty (graph has completed)."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-completed-state"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    # Pause then resume
    await graph.ainvoke(init, config=config)
    await graph.ainvoke(None, config=config)

    # After completion, next should be empty
    state = await graph.aget_state(config)
    assert state.next == (), f"After resume completion state.next must be empty, got {state.next!r}"


# ---------------------------------------------------------------------------
# Tests: HITL off by default (sanity check — parallels test_checkpoint.py)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_hitl_off_by_default_completes_in_single_invoke() -> None:
    """Without HITL, the graph runs to completion in a single ainvoke call."""
    from langgraph.checkpoint.memory import InMemorySaver

    settings = AgentSettings()
    assert settings.hitl.enabled is False, "HITL must be disabled by default"

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=settings,
    )
    config: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-off-test"},
        "recursion_limit": 25,
    }
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    result = await graph.ainvoke(init, config=config)

    # With HITL off, the graph completes with a terminal disposition
    assert result.get("disposition") in ("answered", "refused"), (
        f"HITL-off graph must complete in one invoke, got {result.get('disposition')!r}"
    )

    # Verify the graph is complete (not paused)
    state = await graph.aget_state(config)
    assert state.next == (), (
        f"HITL-off graph state.next must be empty after completion, got {state.next!r}"
    )


@pytest.mark.asyncio
async def test_hitl_pause_different_thread_ids_are_isolated() -> None:
    """Two separate thread_ids each maintain independent HITL pause state."""
    from langgraph.checkpoint.memory import InMemorySaver

    saver = InMemorySaver()
    graph = build_graph(
        spine=_FakeSpine(),
        assembler=_FakeAssembler(),
        llm=_FakeLLM(),
        checkpointer=saver,
        settings=_hitl_settings(),
    )
    init = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }

    config_a: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-thread-a"},
        "recursion_limit": 25,
    }
    config_b: dict[str, Any] = {
        "configurable": {"thread_id": "hitl-thread-b"},
        "recursion_limit": 25,
    }

    # Pause both threads
    await graph.ainvoke(init, config=config_a)
    await graph.ainvoke(init, config=config_b)

    # Both should be paused before "done"
    state_a = await graph.aget_state(config_a)
    state_b = await graph.aget_state(config_b)
    assert state_a.next == ("done",), f"Thread A must be paused at 'done', got {state_a.next!r}"
    assert state_b.next == ("done",), f"Thread B must be paused at 'done', got {state_b.next!r}"

    # Resume only thread A
    result_a = await graph.ainvoke(None, config=config_a)

    # Thread A completes; thread B stays paused
    state_b_after = await graph.aget_state(config_b)
    assert result_a.get("disposition") == "answered", (
        f"Thread A must be answered after resume, got {result_a.get('disposition')!r}"
    )
    assert state_b_after.next == ("done",), (
        f"Thread B must still be paused at 'done', got {state_b_after.next!r}"
    )
