"""Unit tests for AgentQueryService.

Verifies:
- AgentQueryService.answer returns a QueryOutcome with the assembler's answer/grounded/citations.
- The loop=off escape hatch bypasses the graph and calls spine+assembler directly.
- AgentQueryService satisfies the same interface as QueryService (duck-typed by tests).
"""

from __future__ import annotations

import pytest

from iw_rag.api.service import QueryOutcome
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer, RetrievalResult, RetrievedChunk

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-svc",
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
    answer="You have 20 days of annual leave [1].",
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
    """Spine stub returning a canned RetrievalResult."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, AccessContext]] = []

    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        """Record call and return canned result."""
        self.calls.append((query, ctx))
        return _RESULT


class _FakeAssembler:
    """Assembler stub returning a canned GeneratedAnswer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, RetrievalResult]] = []

    async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
        """Record call and return canned answer."""
        self.calls.append((query, result))
        return _ANSWER


@pytest.mark.asyncio
async def test_agent_query_service_answer_returns_query_outcome() -> None:
    """Verifies AgentQueryService.answer returns a valid QueryOutcome."""
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)
    svc = AgentQueryService(spine=spine, assembler=assembler, graph=graph)

    outcome = await svc.answer("what is the leave policy?", _CTX)

    assert isinstance(outcome, QueryOutcome)
    # The outcome's answer should be grounded via the spine→assembler path
    assert outcome.answer.grounded is True
    assert outcome.answer.answer == _ANSWER.answer


@pytest.mark.asyncio
async def test_agent_query_service_citations_flow_through() -> None:
    """Verifies citations from the assembler are accessible in the outcome."""
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)
    svc = AgentQueryService(spine=spine, assembler=assembler, graph=graph)

    outcome = await svc.answer("leave days?", _CTX)
    assert len(outcome.answer.citations) == 1
    assert outcome.answer.citations[0].chunk_id == "c1"


@pytest.mark.asyncio
async def test_agent_query_service_loop_off_calls_spine_directly() -> None:
    """Verifies loop=off bypasses the graph and calls spine+assembler directly."""
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)
    svc = AgentQueryService(spine=spine, assembler=assembler, graph=graph)

    outcome = await svc.answer("direct query", _CTX, loop=False)

    assert isinstance(outcome, QueryOutcome)
    # Spine must have been called directly
    assert len(spine.calls) == 1
    assert spine.calls[0][0] == "direct query"
    # Assembler must have been called with the spine's result
    assert len(assembler.calls) == 1
    assert assembler.calls[0][1] is _RESULT


@pytest.mark.asyncio
async def test_agent_query_service_loop_on_calls_graph() -> None:
    """Verifies loop=on (default) routes through the graph."""
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)
    svc = AgentQueryService(spine=spine, assembler=assembler, graph=graph)

    # loop=True is the default — graph path
    outcome = await svc.answer("graph query", _CTX, loop=True)

    assert isinstance(outcome, QueryOutcome)
    assert outcome.answer.grounded is True


@pytest.mark.asyncio
async def test_agent_query_service_surfaces_full_trajectory() -> None:
    """Loop-path diagnostics expose the full ordered trajectory + route/retries.

    The trajectory is the append-only verdict trail (every attempt, in order) the
    trajectory-eval harness scores — distinct from ``guard_trail`` (latest verdict
    per kind for the UI). See docs/build/03-tasks/trajectory-eval/.
    """
    from iw_rag.agent.graph import build_graph
    from iw_rag.api.service import AgentQueryService

    spine = _FakeSpine()
    assembler = _FakeAssembler()
    graph = build_graph(spine, assembler)
    svc = AgentQueryService(spine=spine, assembler=assembler, graph=graph)

    outcome = await svc.answer("what is the leave policy?", _CTX, loop=True)
    diag = outcome.diagnostics

    # The full trajectory is an ordered list of verdict dicts with the keys the
    # scorer reads (kind/value/attempt) — never empty on the loop path.
    trajectory = diag["trajectory"]
    assert isinstance(trajectory, list)
    assert trajectory, "loop path must surface a non-empty trajectory"
    for entry in trajectory:
        assert set(entry) == {"kind", "value", "score", "attempt"}
    # The input guard always runs first, so it heads the trail.
    assert trajectory[0]["kind"] == "input_guard"

    # Route + budget are surfaced for the router-accuracy / retry metrics
    # (route may be None when the turn is rejected before the router runs).
    assert "route" in diag
    assert diag["retries"] == 0
    assert diag["max_retries"] >= 0

    # guard_trail is unchanged (latest-per-kind, still carries reason).
    guard_trail = diag["guard_trail"]
    assert isinstance(guard_trail, list)
    kinds = [v["kind"] for v in guard_trail]
    assert len(kinds) == len(set(kinds)), "guard_trail keeps only the latest verdict per kind"
    assert all("reason" in v for v in guard_trail)
