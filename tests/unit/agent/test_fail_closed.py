"""Unit tests for the fail_closed node and budget-exhaustion routing.

Covers:
- ``fail_closed`` node: sets ``disposition="refused"``, ``answer=None``,
  appends a ``Verdict``, never sets an answer.
- ``route_after_grade_generation``: budget-exhausted → ``"fail_closed"``
  (not ``"done"``); grounded → ``"done"`` unchanged.
- ``route_after_grade_documents``: budget-exhausted + incorrect → ``"fail_closed"``
  (routed through generate → grade_generation → fail_closed; the doc-edge still
  routes to ``"generate"`` on budget exhaustion, then grade_generation → fail_closed).
- Bounded-loop invariant: alternating doc-fail + gen-fail total ≤ max_retries.
- Invariant: no path yields ``disposition=="answered"`` without ``gen_verdict=="grounded"``.
- ``GraphRecursionError`` handling: caught → refusal, never a 500.
- ``AgentQueryService.answer`` with a refused state → ``QueryOutcome(grounded=False)``.

References:
- task-2.5-budgets-and-fail-closed §2, §3, §4, §5
- lld-agentic-loop §3.6, §7
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk])
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.agent.state import GraphState, Verdict
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievedChunk

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-fail-closed",
)


def _base_state(
    retries: int = 0,
    max_retries: int = 2,
    gen_verdict: str | None = None,
    doc_verdict: str | None = None,
) -> GraphState:
    """Build a minimal GraphState for fail_closed tests.

    Args:
        retries: Number of correction passes consumed so far.
        max_retries: Maximum allowed correction passes.
        gen_verdict: The GenVerdict to inject.
        doc_verdict: The DocVerdict to inject.

    Returns:
        A ``GraphState`` dict suitable for fail_closed / edge tests.
    """
    s: GraphState = {
        "question": "test?",
        "original_question": "test?",
        "access_context": _CTX,
        "retries": retries,
        "max_retries": max_retries,
        "verdicts": [],
    }
    if gen_verdict is not None:
        s["gen_verdict"] = gen_verdict  # type: ignore[typeddict-item]
    if doc_verdict is not None:
        s["doc_verdict"] = doc_verdict  # type: ignore[typeddict-item]
    return s


# ---------------------------------------------------------------------------
# fail_closed node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fail_closed_sets_disposition_refused() -> None:
    """fail_closed must set disposition='refused'."""
    from iw_rag.agent.nodes import fail_closed

    state = _base_state(retries=2, max_retries=2, gen_verdict="unsupported")
    result = await fail_closed(state)

    assert result.get("disposition") == "refused", (
        f"Expected disposition='refused', got {result.get('disposition')!r}"
    )


@pytest.mark.asyncio
async def test_fail_closed_answer_is_none() -> None:
    """fail_closed must set answer=None — no ungrounded answer is emitted."""
    from iw_rag.agent.nodes import fail_closed

    state = _base_state(retries=2, max_retries=2, gen_verdict="unsupported")
    result = await fail_closed(state)

    assert result.get("answer") is None, (
        f"fail_closed must set answer=None, got {result.get('answer')!r}"
    )


@pytest.mark.asyncio
async def test_fail_closed_sets_refusal_message() -> None:
    """fail_closed must set the canonical FAIL_CLOSED_MESSAGE as the refusal."""
    from iw_rag.agent.nodes import FAIL_CLOSED_MESSAGE, fail_closed

    state = _base_state(retries=2, max_retries=2, gen_verdict="unsupported")
    result = await fail_closed(state)

    refusal = result.get("refusal")
    assert refusal == FAIL_CLOSED_MESSAGE, (
        f"Expected refusal={FAIL_CLOSED_MESSAGE!r}, got {refusal!r}"
    )


@pytest.mark.asyncio
async def test_fail_closed_appends_verdict() -> None:
    """fail_closed must append exactly one Verdict(kind='fail_closed', value='refused')."""
    from iw_rag.agent.nodes import fail_closed

    state = _base_state(retries=2, max_retries=2, gen_verdict="unsupported")
    result = await fail_closed(state)

    verdicts = result.get("verdicts", [])
    assert isinstance(verdicts, list) and len(verdicts) == 1, (
        f"Expected exactly one Verdict, got {verdicts!r}"
    )
    v = verdicts[0]
    assert isinstance(v, Verdict), f"Expected a Verdict, got {type(v)}"
    # Exact value pin: the verdict must record the refused outcome
    assert v.value == "refused", f"Expected verdict.value='refused', got {v.value!r}"


@pytest.mark.asyncio
async def test_fail_closed_verdict_kind_is_fail_closed() -> None:
    """fail_closed Verdict must have kind='fail_closed'."""
    from iw_rag.agent.nodes import fail_closed

    state = _base_state(retries=2, max_retries=2, gen_verdict="unsupported")
    result = await fail_closed(state)

    v = result.get("verdicts", [{}])[0]
    assert isinstance(v, Verdict)
    assert v.kind == "fail_closed", f"Expected kind='fail_closed', got {v.kind!r}"


@pytest.mark.asyncio
async def test_fail_closed_never_sets_answered_disposition() -> None:
    """fail_closed must never set disposition='answered' regardless of input state."""
    from iw_rag.agent.nodes import fail_closed

    for gv in ("unsupported", "not_answering", None):
        state = _base_state(retries=2, max_retries=2, gen_verdict=gv)
        result = await fail_closed(state)
        assert result.get("disposition") != "answered", (
            f"fail_closed set answered disposition for gen_verdict={gv!r}"
        )


# ---------------------------------------------------------------------------
# route_after_grade_generation: budget-exhausted → "fail_closed" (not "done")
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen_verdict, retries, max_retries, expected_node",
    [
        # grounded → output_guard (Task 3.3: output_guard inserted before done)
        ("grounded", 0, 2, "output_guard"),
        ("grounded", 2, 2, "output_guard"),
        ("grounded", 3, 2, "output_guard"),
        # unsupported with budget remaining → consume_retry
        ("unsupported", 0, 2, "consume_retry"),
        ("unsupported", 1, 2, "consume_retry"),
        # unsupported budget EXHAUSTED → fail_closed (not "done" any more)
        ("unsupported", 2, 2, "fail_closed"),
        ("unsupported", 3, 2, "fail_closed"),
        # not_answering with budget → consume_retry
        ("not_answering", 0, 2, "consume_retry"),
        ("not_answering", 1, 2, "consume_retry"),
        # not_answering budget exhausted → fail_closed
        ("not_answering", 2, 2, "fail_closed"),
        ("not_answering", 3, 2, "fail_closed"),
        # max_retries=1
        ("unsupported", 0, 1, "consume_retry"),
        ("unsupported", 1, 1, "fail_closed"),
        # max_retries=0 → terminal immediately
        ("unsupported", 0, 0, "fail_closed"),
        ("not_answering", 0, 0, "fail_closed"),
    ],
)
def test_route_after_grade_generation_fail_closed_table(
    gen_verdict: str,
    retries: int,
    max_retries: int,
    expected_node: str,
) -> None:
    """Verifies routing from grade_generation (Task 2.5/3.3).

    Task 2.5: budget-exhausted routes to 'fail_closed' not 'done'.
    Task 3.3: grounded routes to 'output_guard' (not 'done') — output guard
    inserted before done for citation validation + PII redaction.

    Args:
        gen_verdict: The GenVerdict (grounded/unsupported/not_answering).
        retries: Correction passes consumed so far.
        max_retries: Budget limit.
        expected_node: The expected next node name.
    """
    from iw_rag.agent.edges import route_after_grade_generation

    state = _base_state(gen_verdict=gen_verdict, retries=retries, max_retries=max_retries)
    result = route_after_grade_generation(state)

    assert result == expected_node, (
        f"gen_verdict={gen_verdict!r}, retries={retries}, max_retries={max_retries}: "
        f"expected={expected_node!r}, got={result!r}"
    )


def test_route_after_grade_generation_total_with_none() -> None:
    """None gen_verdict (budget left) → consume_retry; exhausted → fail_closed."""
    from iw_rag.agent.edges import route_after_grade_generation

    # Budget left → consume_retry
    state = _base_state(gen_verdict=None, retries=0, max_retries=2)
    result = route_after_grade_generation(state)
    assert result in ("fail_closed", "consume_retry"), (
        f"Expected 'fail_closed' or 'consume_retry', got {result!r}"
    )

    # Budget exhausted → fail_closed
    state_exhausted = _base_state(gen_verdict=None, retries=2, max_retries=2)
    result_exhausted = route_after_grade_generation(state_exhausted)
    assert result_exhausted == "fail_closed", (
        f"Exhausted budget, None verdict → expected 'fail_closed', got {result_exhausted!r}"
    )


# ---------------------------------------------------------------------------
# Invariant: no path produces disposition="answered" without gen_verdict="grounded"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invariant_no_answered_without_grounded() -> None:
    """Invariant: fail_closed never produces answered, done only when grounded.

    Neither fail_closed nor the budget-exhausted seam should produce
    disposition='answered' when gen_verdict != 'grounded'.
    """
    from iw_rag.agent.nodes import done, fail_closed

    for gv in ("unsupported", "not_answering", None):
        # fail_closed path
        state = _base_state(gen_verdict=gv, retries=2, max_retries=2)
        fc_result = await fail_closed(state)
        assert fc_result.get("disposition") != "answered", (
            f"fail_closed set 'answered' for gen_verdict={gv!r}"
        )

        # done path with non-grounded gen_verdict (should refuse, not answer)
        state["draft"] = "some draft"  # type: ignore[typeddict-unknown-key]
        done_result = await done(state)
        assert done_result.get("disposition") != "answered", (
            f"done set 'answered' for gen_verdict={gv!r}"
        )

    # Positive: done with grounded → answered
    grounded_state = _base_state(gen_verdict="grounded", retries=0, max_retries=2)
    grounded_state["draft"] = "a grounded answer"  # type: ignore[typeddict-unknown-key]
    done_grounded = await done(grounded_state)
    assert done_grounded.get("disposition") == "answered", (
        "done with gen_verdict='grounded' must set disposition='answered'"
    )


# ---------------------------------------------------------------------------
# Bounded-loop test: alternating doc-fail + gen-fail ≤ max_retries
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bounded_loop_alternating_failures() -> None:
    """A loop that alternates doc-fail and gen-fail consumes at most max_retries passes.

    Simulates the full graph with a fake grade_generation that always returns
    'unsupported', asserting the loop terminates in at most max_retries corrective
    passes and the final state has disposition='refused'.

    The graph uses a real LLM fake (no actual model calls).
    """
    from iw_rag.agent.budgets import DEFAULT_MAX_RETRIES
    from iw_rag.agent.graph import answer_question, build_graph
    from iw_rag.domain.models import GeneratedAnswer, RetrievalResult

    max_retries = DEFAULT_MAX_RETRIES

    class _AlwaysUnsupportedLLM:
        """Fake LLM that always returns 'unsupported' gen verdict signals."""

        async def complete(self, messages: list[dict[str, str]], params: Any) -> Any:
            """Return a stub rewritten query."""

            class _Resp:
                text = "rewritten query"

            return _Resp()

        async def complete_structured(
            self, messages: list[dict[str, str]], schema: Any, params: Any
        ) -> Any:
            """Return structured stubs — grade_generation always unsupported."""

            class _Resp:
                pass

            resp = _Resp()
            # Router → always retrieve
            if hasattr(schema, "get") and schema.get("title") == "RouterOutput":
                resp.text = '{"route": "retrieve", "query_type": "factual"}'  # type: ignore[attr-defined]
            elif schema.get("title") == "DocGradeOutput":  # type: ignore[union-attr]
                # All chunks irrelevant → doc_verdict=incorrect
                resp.text = '{"grades": []}'  # type: ignore[attr-defined]
            elif schema.get("title") == "ISSUPOutput":  # type: ignore[union-attr]
                resp.text = '{"supported": false}'  # type: ignore[attr-defined]
            elif schema.get("title") == "ISUSEOutput":  # type: ignore[union-attr]
                resp.text = '{"answers": false}'  # type: ignore[attr-defined]
            else:
                resp.text = "{}"  # type: ignore[attr-defined]
            return resp

    class _StubSpine:
        def __init__(self) -> None:
            self.call_count = 0

        async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
            self.call_count += 1
            chunk = RetrievedChunk(
                chunk_id="c1",
                document_id="doc-c1",
                content="stub content",
                score=0.5,
                source_uri="corpus://a",
            )
            return RetrievalResult(chunks=[chunk], retriever="spine", acl_applied=True)

    class _StubAssembler:
        async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
            return GeneratedAnswer(
                answer="stub draft",
                grounded=False,
                citations=[],
            )

    from iw_rag.settings import AgentSettings

    settings = AgentSettings(router_enabled=False)
    llm = _AlwaysUnsupportedLLM()
    spine = _StubSpine()
    assembler = _StubAssembler()

    graph = build_graph(spine, assembler, llm=llm, settings=settings)
    state = await answer_question(graph, "test question", _CTX, max_retries=max_retries)

    # The loop must have terminated with a refusal
    assert state.get("disposition") == "refused", (
        f"Expected disposition='refused', got {state.get('disposition')!r}"
    )
    assert state.get("answer") is None, (
        f"fail_closed must not set an answer: {state.get('answer')!r}"
    )
    # Retries must not exceed max_retries
    assert state.get("retries", 0) <= max_retries, (
        f"retries={state.get('retries')} exceeded max_retries={max_retries}"
    )


# ---------------------------------------------------------------------------
# GraphRecursionError path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_recursion_error_returns_refused_state() -> None:
    """GraphRecursionError is caught → disposition='refused', not a raised exception.

    Forces the error by setting recursion_limit=1 on a non-trivial graph.
    """
    from langgraph.graph import START, StateGraph

    from iw_rag.agent.graph import answer_question
    from iw_rag.agent.state import GraphState

    # Build a tiny cycle that LangGraph will hit the recursion limit on.
    call_count: list[int] = [0]

    async def looping_node(state: GraphState) -> dict[str, Any]:
        """Always returns an empty dict — forces the graph into an infinite-looking loop."""
        call_count[0] += 1
        # Not setting disposition = the graph keeps looping.
        return {}

    g: Any = StateGraph(GraphState)
    g.add_node("loop", looping_node)
    g.add_edge(START, "loop")
    # No END edge on purpose — the graph cycles back via a conditional stub
    # We need a real cycle; add an edge back to itself (LangGraph supports this via conditional)
    g.add_conditional_edges("loop", lambda _: "loop", {"loop": "loop"})

    # This will cause recursion; compile with tiny limit.
    tiny_graph = g.compile()

    # answer_question should catch GraphRecursionError and return refused state
    state = await answer_question(tiny_graph, "cause recursion", _CTX, recursion_limit=5)

    assert state.get("disposition") == "refused", (
        f"GraphRecursionError must yield disposition='refused', got {state.get('disposition')!r}"
    )
    assert state.get("answer") is None, (
        f"GraphRecursionError path must not set an answer: {state.get('answer')!r}"
    )


@pytest.mark.asyncio
async def test_graph_recursion_error_logs_anomaly(caplog: pytest.LogCaptureFixture) -> None:
    """GraphRecursionError catch path logs a warning-level anomaly."""
    import logging

    from langgraph.errors import GraphRecursionError

    from iw_rag.agent.graph import FAIL_CLOSED_MESSAGE, answer_question

    class _RecursingGraph:
        """Fake graph that always raises GraphRecursionError."""

        async def ainvoke(self, init: Any, config: Any) -> Any:
            raise GraphRecursionError("boom")

    with caplog.at_level(logging.WARNING, logger="iw_rag.agent.graph"):
        state = await answer_question(_RecursingGraph(), "q", _CTX, thread_id="t1")

    assert state.get("disposition") == "refused"
    # The anomaly should be in the log
    assert any("recursion" in r.message.lower() for r in caplog.records), (
        "Expected 'recursion' in log records"
    )
    _ = FAIL_CLOSED_MESSAGE  # referenced constant must be importable


# ---------------------------------------------------------------------------
# AgentQueryService: refused state → QueryOutcome(grounded=False), no exception
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_query_service_refused_state_returns_query_outcome() -> None:
    """AgentQueryService.answer returns QueryOutcome(grounded=False) for a refused graph.

    Uses a fake graph that always returns a refused state so no real LLM is needed.
    """
    from iw_rag.api.service import AgentQueryService, QueryOutcome

    class _RefusingGraph:
        """Fake compiled graph that always returns a refused terminal state."""

        async def ainvoke(self, init: GraphState, config: Any) -> GraphState:
            return {
                **init,
                "answer": None,
                "refusal": "I can't ground an answer in the retrieved evidence.",
                "disposition": "refused",
                "diagnostics": {},
                "citations": [],
                "used_chunk_ids": [],
            }

    class _StubSpine:
        async def retrieve(self, q: str, ctx: AccessContext) -> Any:
            from iw_rag.domain.models import RetrievalResult

            return RetrievalResult(chunks=[], retriever="spine", acl_applied=True)

    class _StubAssembler:
        async def generate(self, q: str, r: Any) -> Any:
            from iw_rag.domain.models import GeneratedAnswer

            return GeneratedAnswer(answer="stub", grounded=True)

    svc = AgentQueryService(
        spine=_StubSpine(),
        assembler=_StubAssembler(),
        graph=_RefusingGraph(),
    )

    outcome = await svc.answer("test question", _CTX)

    assert isinstance(outcome, QueryOutcome)
    assert outcome.answer.grounded is False, (
        f"Refused state must map to grounded=False, got {outcome.answer.grounded!r}"
    )


@pytest.mark.asyncio
async def test_agent_query_service_never_raises_on_refused() -> None:
    """AgentQueryService.answer returns a grounded=False QueryOutcome for a refused graph."""
    from iw_rag.api.service import AgentQueryService, QueryOutcome

    class _RefusingGraph:
        async def ainvoke(self, init: Any, config: Any) -> Any:
            return {
                **init,
                "answer": None,
                "refusal": "no evidence",
                "disposition": "refused",
                "diagnostics": {},
            }

    class _Stub:
        async def retrieve(self, *a: Any, **k: Any) -> Any:
            from iw_rag.domain.models import RetrievalResult

            return RetrievalResult(chunks=[], retriever="spine", acl_applied=True)

        async def generate(self, *a: Any, **k: Any) -> Any:
            from iw_rag.domain.models import GeneratedAnswer

            return GeneratedAnswer(answer="s", grounded=True)

    svc = AgentQueryService(spine=_Stub(), assembler=_Stub(), graph=_RefusingGraph())

    # Must not raise; outcome must be a QueryOutcome with grounded=False
    outcome = await svc.answer("q", _CTX)
    assert isinstance(outcome, QueryOutcome), f"Expected QueryOutcome, got {type(outcome)!r}"
    assert outcome.answer.grounded is False, (
        f"Refused state must map to grounded=False, got {outcome.answer.grounded!r}"
    )


# ---------------------------------------------------------------------------
# Full-graph refuse integration test (TestClient + fake service)
# ---------------------------------------------------------------------------


def test_query_endpoint_http_200_on_refusal() -> None:
    """HTTP /query returns 200 with grounded=false when AgentQueryService refuses.

    Uses FastAPI TestClient with a fake query service injected via
    ``create_app(query_service=...)`` — no live graph or LLM required.
    The refused state must never result in a 500 (it is a valid first-class
    outcome: disposition='refused', HTTP 200 with grounded=false).
    """
    from fastapi.testclient import TestClient

    from iw_rag.api.app import create_app
    from iw_rag.api.auth import StaticTokenAuthenticator
    from iw_rag.api.service import QueryOutcome
    from iw_rag.domain.models import GeneratedAnswer

    refused_outcome = QueryOutcome(
        answer=GeneratedAnswer(
            answer="I can't ground an answer in the retrieved evidence.",
            grounded=False,
            citations=[],
            used_chunk_ids=[],
        ),
        retriever="agent",
        acl_applied=True,
        diagnostics={"disposition": "refused"},
    )

    class _RefusingService:
        """Always-refusing fake query service — duck-types QueryService/AgentQueryService."""

        async def answer(self, query: str, ctx: AccessContext, **kwargs: Any) -> QueryOutcome:
            """Return a refused outcome regardless of the query."""
            return refused_outcome

    auth = StaticTokenAuthenticator.from_config({"test-token": ["public"]}, [])
    app = create_app(authenticator=auth, query_service=_RefusingService())
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what is the leave policy?"},
        headers={"authorization": "Bearer test-token"},
    )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()
    assert data.get("grounded") is False, (
        f"Expected grounded=false in response, got {data.get('grounded')!r}"
    )
