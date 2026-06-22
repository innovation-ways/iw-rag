"""Unit tests for the router node, answer_direct node, and graph wiring.

All LLM calls are mocked — no live models required.

Covers:
- router node: skip/retrieve/graph/long_context routing with config gating.
- router_enabled=False → retrieve with zero LLM calls.
- Unknown/garbage label → retrieve.
- graph/long_context collapse to retrieve, but intended route is in the verdict.
- Exactly one router verdict appended per call.
- answer_direct produces a draft.
- End-to-end: router → skip → answer_direct → END with fake llm/assembler.

References:
- task-2.1-router-node §2, §3, §5 (Tests to write)
- lld-agentic-loop §3.1 (router table)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from iw_rag.agent.state import GraphState, Verdict
from iw_rag.domain.acl import AccessContext
from iw_rag.generation.llm import GenerationParams, LlmResponse
from iw_rag.settings import AgentSettings

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-router",
)

_BASE_STATE: GraphState = {
    "question": "What is the leave policy?",
    "original_question": "What is the leave policy?",
    "access_context": _CTX,
    "retries": 0,
    "max_retries": 2,
    "verdicts": [],
}


def _make_llm_response(route: str, query_type: str = "factual", reason: str = "") -> LlmResponse:
    """Build a canned LlmResponse with a router JSON payload.

    Args:
        route: The route value to encode.
        query_type: The query_type value to encode.
        reason: Optional reason string.

    Returns:
        An LlmResponse whose text is a JSON-encoded router verdict.
    """
    import json

    payload: dict[str, Any] = {"query_type": query_type, "route": route}
    if reason:
        payload["reason"] = reason
    return LlmResponse(text=json.dumps(payload), finish_reason="stop")


# ---------------------------------------------------------------------------
# Router node — config-gated behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_disabled_returns_retrieve_without_llm_call() -> None:
    """When router_enabled=False, route='retrieve' is returned with zero LLM calls.

    The LLM mock must NOT be called; the router short-circuits immediately.
    """
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    settings = AgentSettings(router_enabled=False, skip_enabled=True)
    router = make_router(llm, settings)

    state = dict(_BASE_STATE)
    result = await router(state)  # type: ignore[arg-type]

    assert result["route"] == "retrieve", f"Expected 'retrieve', got {result['route']!r}"
    # Critically: no LLM call was made
    llm.complete_structured.assert_not_called()
    # Exactly one verdict
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0].kind == "router"
    assert verdicts[0].value == "retrieve"


@pytest.mark.asyncio
async def test_router_skip_label_with_skip_enabled() -> None:
    """When LLM returns route='skip' and skip_enabled=True, route='skip'."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=_make_llm_response("skip", "factual"))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "skip"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0].kind == "router"
    assert verdicts[0].value == "skip"


@pytest.mark.asyncio
async def test_router_skip_label_with_skip_disabled_returns_retrieve() -> None:
    """When LLM returns route='skip' but skip_enabled=False, route falls back to 'retrieve'."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=_make_llm_response("skip", "factual"))
    settings = AgentSettings(router_enabled=True, skip_enabled=False)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "retrieve", f"Expected 'retrieve', got {result['route']!r}"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0].kind == "router"
    assert verdicts[0].value == "retrieve"


@pytest.mark.asyncio
async def test_router_unknown_label_falls_back_to_retrieve() -> None:
    """An unknown/garbage route label falls back to 'retrieve' and is recorded."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    # Return a garbage label
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(text='{"query_type": "factual", "route": "banana"}')
    )
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "retrieve"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0].kind == "router"
    assert verdicts[0].value == "retrieve"


@pytest.mark.asyncio
async def test_router_unparseable_response_falls_back_to_retrieve() -> None:
    """Unparseable JSON in the LLM response falls back to 'retrieve'."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=LlmResponse(text="not json at all!!!"))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "retrieve"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1


# ---------------------------------------------------------------------------
# Router parse hardening — recover complete-but-dirty JSON, fall back on the
# unrecoverable (truncation) cases.  See _parse_router_response.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_recovers_json_wrapped_in_think_tags() -> None:
    """Qwen3 ``<think>…</think>`` leakage around the JSON is stripped, not dropped.

    The reasoning block must not defeat the parse; the classified ``graph``
    route is recovered and preserved for the edge to dispatch.
    """
    from iw_rag.agent.nodes import make_router

    leaked = (
        "<think>The question is relational; entities are linked across the "
        "corpus, so graph is appropriate.</think>"
        '{"query_type": "relational", "route": "graph"}'
    )
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=LlmResponse(text=leaked))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "graph", f"think-wrapped JSON not recovered: {result['route']!r}"
    assert result["query_type"] == "relational"


@pytest.mark.asyncio
async def test_router_recovers_json_in_markdown_code_fence() -> None:
    """A ```json fenced block is unwrapped before parsing."""
    from iw_rag.agent.nodes import make_router

    fenced = '```json\n{"query_type": "factual", "route": "skip"}\n```'
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=LlmResponse(text=fenced))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "skip", f"fenced JSON not recovered: {result['route']!r}"


@pytest.mark.asyncio
async def test_router_recovers_json_with_surrounding_prose() -> None:
    """Leading/trailing prose around a complete object is tolerated."""
    from iw_rag.agent.nodes import make_router

    chatty = 'Here is the classification: {"query_type": "factual", "route": "visual"} — done.'
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=LlmResponse(text=chatty))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "visual", f"prose-wrapped JSON not recovered: {result['route']!r}"


@pytest.mark.asyncio
async def test_router_truncated_json_falls_back_to_vector_not_partial_route() -> None:
    """The real incident: a reason field overruns max_tokens, truncating the JSON.

    The object is genuinely incomplete (no closing brace).  The hardened parser
    MUST NOT honour the partially-emitted ``route="graph"`` — it falls back to
    the safe vector route (``retrieve``).  Trusting a truncated route is exactly
    how a cheap lookup got dispatched to the heaviest graph path.
    """
    from iw_rag.agent.nodes import make_router

    truncated = (
        '{\n\n\n"query_type": "relational"\n,\n\n\n"route": "graph"\n,\n\n\n'
        '"reason": "The question is asking about the specific mode in which the '
        "SMILE 5 UPS interface displays real-time information, including voltage, "
        "current, power, and frequency. This requires understanding the UPS"
    )
    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=LlmResponse(text=truncated))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "retrieve", (
        f"truncated JSON must fall back to vector, not honour partial route: {result['route']!r}"
    )


def test_parse_router_response_recovers_and_rejects() -> None:
    """Direct unit coverage of the tolerant parser helper."""
    from iw_rag.agent.nodes import _parse_router_response

    # Clean JSON — fast path.
    assert _parse_router_response('{"route": "retrieve"}') == {"route": "retrieve"}
    # Dirty-but-complete — recovered.
    assert _parse_router_response('<think>x</think>{"route": "graph"}')["route"] == "graph"
    assert _parse_router_response('prefix {"route": "skip"} suffix')["route"] == "skip"
    # Truncated / unparseable — empty dict so the caller applies the fallback.
    assert _parse_router_response('{"route": "graph", "reason": "unterminated') == {}
    assert _parse_router_response("not json at all!!!") == {}
    # A JSON non-object (array/scalar) is not a verdict → empty dict.
    assert _parse_router_response('["graph"]') == {}


@pytest.mark.asyncio
async def test_router_graph_route_preserved_for_edge_dispatch() -> None:
    """route='graph' is PRESERVED by the node; the edge (route_after_router) gates it.

    Since WI-028/029/031/032 every Phase-4 route (graph/visual/long_context) is an
    implemented route. The router node only has ``AgentSettings`` and cannot read
    ``modes.*.enabled``, so it records the classified route verbatim (for
    router-accuracy metrics) and ``route_after_router`` collapses it to
    ``retrieve`` when the mode is disabled — covered by
    ``test_route_after_router_graph_disabled_collapses_to_retrieve``.
    """
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=_make_llm_response("graph", "multihop", "multi-hop entities")
    )
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    # The node preserves the classified route; the edge handles dispatch/collapse.
    assert result["route"] == "graph"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.kind == "router"
    assert v.value == "graph"


@pytest.mark.asyncio
async def test_router_long_context_route_preserved_for_edge_dispatch() -> None:
    """route='long_context' is preserved by the node; the edge gates dispatch/collapse.

    See ``test_router_graph_route_preserved_for_edge_dispatch`` — the node records
    the classified route verbatim; ``route_after_router`` collapses to ``retrieve``
    when ``modes.long_context.enabled=false`` (covered by the edge tests).
    """
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=_make_llm_response("long_context", "wholedoc", "whole-doc question")
    )
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "long_context"
    verdicts: list[Verdict] = result["verdicts"]
    assert len(verdicts) == 1
    assert verdicts[0].value == "long_context"


@pytest.mark.asyncio
async def test_router_appends_exactly_one_verdict_per_call() -> None:
    """Exactly one router verdict is appended per invocation."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(return_value=_make_llm_response("retrieve", "factual"))
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert len(result["verdicts"]) == 1


@pytest.mark.asyncio
async def test_router_retrieve_label_routed_correctly() -> None:
    """route='retrieve' is passed through directly."""
    from iw_rag.agent.nodes import make_router

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=_make_llm_response("retrieve", "factual", "needs document grounding")
    )
    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    router = make_router(llm, settings)

    result = await router(dict(_BASE_STATE))  # type: ignore[arg-type]

    assert result["route"] == "retrieve"


# ---------------------------------------------------------------------------
# answer_direct node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_answer_direct_produces_draft() -> None:
    """answer_direct generates a draft answer without calling the retrieval spine."""
    from iw_rag.agent.nodes import make_answer_direct

    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LlmResponse(text="42 days of annual leave.", finish_reason="stop")
    )
    settings = AgentSettings()
    # make_answer_direct takes llm + settings and returns the node
    answer_direct = make_answer_direct(llm, settings)

    state: GraphState = {
        "question": "How many leave days?",
        "original_question": "How many leave days?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    result = await answer_direct(state)

    # Must produce a draft (answer text)
    assert result.get("answer") or result.get("draft"), (
        "answer_direct must set 'answer' or 'draft' in the returned dict"
    )
    draft_text = result.get("answer") or result.get("draft") or ""
    assert "42" in draft_text, f"Expected LLM response text in draft; got {draft_text!r}"
    # Citations must be empty (no retrieval)
    assert result.get("citations") == [] or result.get("citations") is None
    # Exactly one verdict
    verdicts: list[Verdict] = result.get("verdicts") or []
    assert len(verdicts) == 1
    assert verdicts[0].kind in ("answer_direct", "router")


# ---------------------------------------------------------------------------
# End-to-end: router → skip → answer_direct → END
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_skip_route_end_to_end() -> None:
    """With router returning 'skip', the graph routes through answer_direct to END.

    Uses fake llm + fake assembler; no live models.
    """
    from iw_rag.agent.graph import answer_question, build_graph
    from iw_rag.domain.models import GeneratedAnswer, RetrievalResult

    # LLM always returns 'skip' for router; also handles ISSUP/ISUSE for grade_generation.
    # Note: answer_direct sets graded=[] and citations=[], so grade_generation takes
    # the deterministic no-citations path (ISSUP=False → unsupported) and only calls
    # ISUSE. With budget=0 the graph then routes to done (fail-closed seam).
    class _FakeLlm:
        async def complete(self, messages: Any, params: GenerationParams) -> LlmResponse:
            return LlmResponse(text="The capital of France is Paris.", finish_reason="stop")

        async def complete_structured(
            self, messages: Any, json_schema: Any, params: GenerationParams
        ) -> LlmResponse:
            import json

            # Dispatch by schema to handle all node types
            required = json_schema.get("required", [])
            if "answers" in required:
                # ISUSE: answer addresses the question
                return LlmResponse(
                    text=json.dumps({"answers": True, "reason": "useful"}),
                    finish_reason="stop",
                )
            if "supported" in required:
                # ISSUP: grounded (though skip path bypasses this via no-citations rule)
                return LlmResponse(
                    text=json.dumps({"supported": True, "reason": "grounded"}),
                    finish_reason="stop",
                )
            # Default: router → skip
            return _make_llm_response("skip", "factual", "trivial parametric")

        async def complete_tools(
            self, messages: Any, tools: Any, params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="", finish_reason="stop")

        async def health(self) -> bool:
            return True

    class _FakeSpine:
        async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
            return RetrievalResult(chunks=[], retriever="spine", acl_applied=True)

    class _FakeAssembler:
        async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
            return GeneratedAnswer(answer="stub answer", grounded=True)

    fake_llm = _FakeLlm()
    settings = AgentSettings(router_enabled=True, skip_enabled=True)

    graph = build_graph(
        _FakeSpine(),
        _FakeAssembler(),
        fake_llm,
        settings=settings,
    )
    state = await answer_question(graph, "What is the capital of France?", _CTX)

    # The graph must complete without error and produce some terminal state
    assert isinstance(state, dict)
    # disposition must be set (either answered or refused — not None from an unfinished run)
    assert state.get("disposition") is not None or state.get("answer") is not None, (
        f"Expected terminal state with disposition or answer; got {dict(state)!r}"
    )
    # Verdicts must include a router verdict
    verdicts = state.get("verdicts") or []
    assert any(v.kind == "router" for v in verdicts), f"Expected a router verdict in {verdicts!r}"


# ---------------------------------------------------------------------------
# ROUTER_SCHEMA shape check
# ---------------------------------------------------------------------------


def test_router_schema_shape() -> None:
    """Verifies ROUTER_SCHEMA has the required fields and enums."""
    from iw_rag.agent.prompts import ROUTER_SCHEMA

    assert ROUTER_SCHEMA["type"] == "object"
    props = ROUTER_SCHEMA["properties"]
    assert "query_type" in props
    assert "route" in props
    assert set(props["query_type"]["enum"]) == {"factual", "multihop", "wholedoc", "relational"}
    assert set(props["route"]["enum"]) == {"skip", "retrieve", "graph", "long_context", "visual"}
    assert "required" in ROUTER_SCHEMA
    assert "query_type" in ROUTER_SCHEMA["required"]
    assert "route" in ROUTER_SCHEMA["required"]
    # ``reason`` is intentionally NOT in the schema: an unbounded free-text field
    # overran ROUTER_MAX_TOKENS and truncated the JSON mid-emit, forcing the
    # vector fallback.  The router only needs the two routing labels.
    assert "reason" not in props, "drop 'reason' — it truncates the constrained decode"


def test_router_system_prompt_exists() -> None:
    """Verifies ROUTER_SYSTEM is a non-empty string."""
    from iw_rag.agent.prompts import ROUTER_SYSTEM

    assert isinstance(ROUTER_SYSTEM, str)
    assert len(ROUTER_SYSTEM) > 20  # noqa: PLR2004 — sanity check, not a magic number
    assert "retrieve" in ROUTER_SYSTEM.lower()


def test_router_schema_decodes_route_before_query_type() -> None:
    """ROUTER_SCHEMA emits ``route`` before ``query_type`` so the route is decoded first.

    guided_json follows the schema's property order. Deciding the well-defined
    ``route`` first (instead of an under-specified ``query_type``) stops the model
    from committing to a bogus ``relational`` type that drags the route to
    ``graph`` (live finding 2026-06-14).
    """
    from iw_rag.agent.prompts import ROUTER_SCHEMA

    keys = list(ROUTER_SCHEMA["properties"].keys())
    assert keys.index("route") < keys.index("query_type"), (
        "route must precede query_type in the schema so guided_json decodes it first"
    )


def test_router_system_prompt_defines_query_type_and_constrains_graph() -> None:
    """ROUTER_SYSTEM must define the query_type labels and reserve ``graph`` for multi-hop.

    The original prompt described only the route labels, so the model had no
    guidance for query_type and defaulted to ``relational`` → ``graph`` for every
    question. The prompt must (a) mention each query_type label and (b) say graph
    is only for relational/multi-entity/multi-hop questions.
    """
    from iw_rag.agent.prompts import ROUTER_SYSTEM

    low = ROUTER_SYSTEM.lower()
    for label in ("factual", "multihop", "wholedoc", "relational"):
        assert label in low, f"ROUTER_SYSTEM must define query_type label {label!r}"
    assert "multi-hop" in low or "multi-entity" in low or "relationship" in low, (
        "ROUTER_SYSTEM must constrain graph to relational/multi-hop questions"
    )
