"""Unit tests for GraphState, Verdict, AgentBudgets, build_graph, and answer_question.

Covers:
- The `verdicts` Annotated[list, add] reducer appends (never overwrites).
- `build_graph` compiles with and without a checkpointer.
- `answer_question` initialises defaults (retries==0, original_question==question, etc.).
- AgentBudgets frozen-dataclass defaults.
"""

from __future__ import annotations

import operator

import pytest

from iw_rag.domain.acl import AccessContext

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def test_verdict_dataclass_fields() -> None:
    """Verifies Verdict can be instantiated with minimal required fields."""
    from iw_rag.agent.state import Verdict

    v = Verdict(kind="router", value="retrieve")
    assert v.kind == "router"
    assert v.value == "retrieve"
    assert v.score is None
    assert v.reason == ""
    assert v.attempt == 0


def test_verdict_with_all_fields() -> None:
    """Verifies Verdict accepts all optional fields."""
    from iw_rag.agent.state import Verdict

    v = Verdict(kind="grade_documents", value="relevant", score=0.9, reason="good", attempt=1)
    assert v.score == 0.9
    assert v.reason == "good"
    assert v.attempt == 1


# ---------------------------------------------------------------------------
# GraphState type annotation check
# ---------------------------------------------------------------------------


def test_graphstate_verdicts_uses_add_reducer() -> None:
    """Verifies the `verdicts` field annotation uses operator.add as the reducer."""
    import typing

    from iw_rag.agent.state import GraphState

    hints = typing.get_type_hints(GraphState, include_extras=True)
    assert "verdicts" in hints
    ann = hints["verdicts"]
    # Annotated[list[Verdict], operator.add] — check metadata contains operator.add
    metadata = getattr(ann, "__metadata__", ())
    assert operator.add in metadata, f"Expected operator.add in metadata, got {metadata}"


def test_graphstate_chunks_field_exists() -> None:
    """Verifies chunks: list[RetrievedChunk] is declared (not candidates)."""
    import typing

    from iw_rag.agent.state import GraphState

    hints = typing.get_type_hints(GraphState, include_extras=True)
    assert "chunks" in hints
    assert "candidates" not in hints


def test_graphstate_required_fields_present() -> None:
    """Verifies GraphState declares all required fields."""
    import typing

    from iw_rag.agent.state import GraphState

    hints = typing.get_type_hints(GraphState, include_extras=True)
    required = {
        "question",
        "original_question",
        "access_context",
        "chunks",
        "graded",
        "verdicts",
        "retries",
        "route",
        "answer",
        "citations",
        "used_chunk_ids",
        "disposition",
        "diagnostics",
    }
    for field in required:
        assert field in hints, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# AgentBudgets
# ---------------------------------------------------------------------------


def test_agent_budgets_defaults() -> None:
    """Verifies AgentBudgets defaults match spec: max_retries=2, recursion_limit=25."""
    from iw_rag.agent.budgets import AgentBudgets

    b = AgentBudgets()
    assert b.max_retries == 2
    assert b.recursion_limit == 25


def test_agent_budgets_frozen() -> None:
    """Verifies AgentBudgets is frozen (immutable)."""
    from iw_rag.agent.budgets import AgentBudgets

    b = AgentBudgets()
    with pytest.raises((AttributeError, TypeError), match="cannot assign to field 'max_retries'"):
        b.max_retries = 5  # type: ignore[misc]


def test_budget_constants_match() -> None:
    """Verifies module-level budget constants match AgentBudgets defaults."""
    from iw_rag.agent.budgets import DEFAULT_MAX_RETRIES, RECURSION_LIMIT, AgentBudgets

    assert AgentBudgets().max_retries == DEFAULT_MAX_RETRIES
    assert AgentBudgets().recursion_limit == RECURSION_LIMIT


# ---------------------------------------------------------------------------
# build_graph
# ---------------------------------------------------------------------------


def test_build_graph_compiles_without_checkpointer() -> None:
    """Verifies build_graph compiles a runnable StateGraph without a checkpointer."""
    from iw_rag.agent.graph import build_graph

    # build_graph receives spine + assembler; use None stubs as per spec
    # In Task 2.0, the single 'answer' node is all we need.
    graph = build_graph(None, None)
    # A compiled LangGraph graph must have get_graph()
    assert hasattr(graph, "invoke") or hasattr(graph, "ainvoke")


def test_build_graph_compiles_with_memory_saver() -> None:
    """Verifies build_graph compiles successfully when given an InMemorySaver."""
    from langgraph.checkpoint.memory import MemorySaver

    from iw_rag.agent.graph import build_graph

    saver = MemorySaver()
    graph = build_graph(None, None, checkpointer=saver)
    assert hasattr(graph, "ainvoke")


# ---------------------------------------------------------------------------
# answer_question defaults
# ---------------------------------------------------------------------------


_CTX = AccessContext(
    principal_id="user1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-test",
)


class _FakeSpine:
    """Minimal spine stub returning an empty RetrievalResult."""

    async def retrieve(self, query: str, ctx: AccessContext):  # type: ignore[no-untyped-def]
        from iw_rag.domain.models import RetrievalResult

        return RetrievalResult(chunks=[], retriever="spine", acl_applied=True)


class _FakeAssembler:
    """Minimal assembler stub returning a grounded answer."""

    async def generate(self, query: str, result):  # type: ignore[no-untyped-def]
        from iw_rag.domain.models import GeneratedAnswer

        return GeneratedAnswer(answer="stub answer", grounded=True)


@pytest.mark.asyncio
async def test_answer_question_initialises_defaults() -> None:
    """Verifies answer_question returns a GraphState with correct initial defaults."""
    from iw_rag.agent.graph import answer_question, build_graph

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)

    state = await answer_question(graph, "hello world", _CTX)
    assert state["original_question"] == "hello world"
    assert state["retries"] == 0
    assert state["verdicts"] is not None


@pytest.mark.asyncio
async def test_answer_question_returns_graph_state() -> None:
    """Verifies answer_question returns a dict-like GraphState with exact field values."""
    from iw_rag.agent.graph import answer_question, build_graph

    graph = build_graph(_FakeSpine(), _FakeAssembler())
    state = await answer_question(graph, "test question", _CTX)
    # GraphState is a TypedDict — state should be dict-like
    assert isinstance(state, dict)
    assert "original_question" in state
    # Exact value check: original_question must equal the query passed in
    assert state["original_question"] == "test question"
    # disposition must be one of the two terminal values (refused on stub path)
    assert state.get("disposition") in ("answered", "refused")


# ---------------------------------------------------------------------------
# verdicts reducer appends (via a tiny graph test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verdicts_reducer_appends() -> None:
    """Verifies the verdicts add-reducer appends across node writes in a graph."""
    from langgraph.graph import END, START, StateGraph

    from iw_rag.agent.state import GraphState, Verdict

    def node_a(state: GraphState) -> dict:  # type: ignore[type-arg]
        return {"verdicts": [Verdict(kind="router", value="retrieve")]}

    def node_b(state: GraphState) -> dict:  # type: ignore[type-arg]
        return {"verdicts": [Verdict(kind="grade_documents", value="relevant")]}

    g = StateGraph(GraphState)
    g.add_node("a", node_a)
    g.add_node("b", node_b)
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    compiled = g.compile()

    result = await compiled.ainvoke(
        {"question": "q", "original_question": "q", "retries": 0, "verdicts": []}
    )
    # Both nodes' verdicts should be appended (2 items), not one overwriting
    assert len(result["verdicts"]) == 2
    assert result["verdicts"][0].kind == "router"
    assert result["verdicts"][1].kind == "grade_documents"
