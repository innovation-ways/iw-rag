"""Unit tests for the retrieve node (WI-016 / Task 2.2).

All tests use fakes — no DB, no live model, no network.

Covers:
- ``make_retrieve_node`` maps ``RetrievalResult.chunks → state["chunks"]``.
- ``access_context`` passed to the spine unchanged (security assertion — query
  text / model output must never reach the ACL predicate).
- ``acl_applied is False`` → ``RetrievalError`` raised (ACL invariant guard).
- Empty result (``chunks=[]``, ``acl_applied=True``) → ``chunks=[]``, no error.
- ``RetrievalError`` from the spine propagates (node does not swallow it).
- Integration-ish end-to-end with a decomposed graph: ``router → retrieve →
  grade_documents → generate`` produces a terminal state with ``answer`` and
  ``chunks``.

References:
- task-2.2-retrieve-toolnode §1 (make_retrieve_node), §2 (access_context),
  §5 (error propagation), Tests to write.
- lld-agentic-loop §6.2, §7 (error handling)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate), §6 (acl_applied)
"""

from __future__ import annotations

from typing import Any

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import RetrievalError
from iw_rag.domain.models import (
    Citation,
    GeneratedAnswer,
    RetrievalResult,
    RetrievedChunk,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-retrieve-test",
)

_CHUNK = RetrievedChunk(
    chunk_id="c1",
    document_id="doc-1",
    content="Leave policy: 20 days per year.",
    score=0.95,
    source_uri="corpus://hr/leave_policy.md",
)


def _make_base_state(**overrides: Any) -> dict[str, Any]:
    """Build a minimal GraphState dict for tests.

    Args:
        **overrides: Keys to override/add in the state dict.

    Returns:
        A dict compatible with ``GraphState``.
    """
    from iw_rag.agent.state import GraphState

    state: GraphState = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fake collaborators
# ---------------------------------------------------------------------------


class _FakeSpine:
    """Spine stub that records calls and returns a configurable result."""

    def __init__(
        self,
        result: RetrievalResult | None = None,
        raise_error: Exception | None = None,
    ) -> None:
        """Configure the fake.

        Args:
            result: The ``RetrievalResult`` to return (or None for default).
            raise_error: If set, ``retrieve`` raises this instead.
        """
        self.calls: list[tuple[str, AccessContext]] = []
        self._result = result
        self._raise = raise_error

    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        """Record the call and return the configured result.

        Args:
            query: The query string.
            ctx: The access context.

        Returns:
            The configured RetrievalResult.

        Raises:
            Exception: If ``raise_error`` was set in the constructor.
        """
        self.calls.append((query, ctx))
        if self._raise is not None:
            raise self._raise
        if self._result is not None:
            return self._result
        return RetrievalResult(chunks=[_CHUNK], retriever="spine", acl_applied=True)


class _FakeAssembler:
    """Assembler stub returning a canned grounded answer."""

    def __init__(self, grounded: bool = True, answer: str = "stub answer [1].") -> None:
        """Configure the fake.

        Args:
            grounded: Whether the returned answer is grounded.
            answer: The answer text.
        """
        self.calls: list[tuple[str, RetrievalResult]] = []
        self._grounded = grounded
        self._answer = answer

    async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
        """Record the call and return a canned answer.

        Args:
            query: The query string.
            result: The retrieval result passed to the assembler.

        Returns:
            A canned GeneratedAnswer.
        """
        self.calls.append((query, result))
        return GeneratedAnswer(
            answer=self._answer,
            grounded=self._grounded,
            citations=[
                Citation(
                    marker=1,
                    chunk_id="c1",
                    document_id="doc-1",
                    source_uri="corpus://hr/leave_policy.md",
                )
            ],
            used_chunk_ids=["c1"],
        )


# ---------------------------------------------------------------------------
# Test: retrieve node maps RetrievalResult.chunks → state["chunks"]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_maps_chunks_to_state() -> None:
    """Verifies the retrieve node sets state["chunks"] from result.chunks.

    The node must map ``RetrievalResult.chunks`` (a ``list[RetrievedChunk]``)
    into the returned dict under the key ``"chunks"``.  No Candidate/passages.
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine = _FakeSpine(
        result=RetrievalResult(
            chunks=[_CHUNK],
            retriever="spine",
            acl_applied=True,
        )
    )
    node = make_retrieve_node(spine)
    state = _make_base_state()

    result = await node(state)  # type: ignore[arg-type]

    assert "chunks" in result, "retrieve node must set 'chunks' in returned dict"
    assert result["chunks"] == [_CHUNK], f"Expected chunks=[_CHUNK], got {result['chunks']!r}"


# ---------------------------------------------------------------------------
# Test: access_context passed unchanged (security invariant)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_passes_access_context_unchanged() -> None:
    """Verifies the spine receives the exact access_context object from state.

    The ACL predicate must NEVER be derived from question text or model output.
    The node must pass ``state["access_context"]`` through unmodified.
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine = _FakeSpine()
    node = make_retrieve_node(spine)
    custom_ctx = AccessContext(
        principal_id="specific-user",
        principal_groups=("finance", "hr"),
        is_operator=False,
        request_id="req-acl-security",
    )
    state = _make_base_state(access_context=custom_ctx)

    await node(state)  # type: ignore[arg-type]

    assert len(spine.calls) == 1, f"Expected exactly 1 spine call, got {len(spine.calls)}"
    _, received_ctx = spine.calls[0]
    assert received_ctx is custom_ctx, (
        "Spine must receive the exact same access_context object from state — "
        f"got {received_ctx!r} instead of {custom_ctx!r}"
    )


# ---------------------------------------------------------------------------
# Test: acl_applied is False → RetrievalError raised
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_raises_on_acl_not_applied() -> None:
    """Verifies the node raises RetrievalError with ACL violation message when acl_applied is False.

    This is the ACL invariant guard: the hot path must NEVER serve chunks
    unless the ACL pre-filter confirmed it ran (``acl_applied=True``).
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine = _FakeSpine(
        result=RetrievalResult(
            chunks=[_CHUNK],
            retriever="spine",
            acl_applied=False,  # ACL NOT applied!
        )
    )
    node = make_retrieve_node(spine)
    state = _make_base_state()

    with pytest.raises(RetrievalError, match="acl_applied=False"):
        await node(state)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test: empty result with acl_applied=True → chunks=[], no error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_empty_result_is_legitimate() -> None:
    """Verifies empty chunks with acl_applied=True is not an error.

    An empty ACL-filtered result is legitimate (all chunks filtered out).
    The node must set ``chunks=[]`` and NOT raise.
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine = _FakeSpine(
        result=RetrievalResult(
            chunks=[],  # empty — filtered by ACL
            retriever="spine",
            acl_applied=True,
        )
    )
    node = make_retrieve_node(spine)
    state = _make_base_state()

    result = await node(state)  # type: ignore[arg-type]

    assert result["chunks"] == [], f"Expected empty chunks list, got {result['chunks']!r}"


# ---------------------------------------------------------------------------
# Test: RetrievalError from spine propagates (not swallowed)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_propagates_retrieval_error_from_spine() -> None:
    """Verifies a RetrievalError raised by the spine propagates out of the node.

    The node must NOT catch and swallow retrieval errors; they must propagate
    to the caller (answer_question maps them to 503).
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine_error = RetrievalError("DB connection lost")
    spine = _FakeSpine(raise_error=spine_error)
    node = make_retrieve_node(spine)
    state = _make_base_state()

    with pytest.raises(RetrievalError, match="DB connection lost"):
        await node(state)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test: verdicts include retrieval diagnostic entry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_node_appends_verdicts_entry() -> None:
    """Verifies the node appends at least one verdict/diagnostic entry.

    The verdict trail must capture the retrieval pass for audit purposes.
    """
    from iw_rag.agent.tools import make_retrieve_node

    spine = _FakeSpine(
        result=RetrievalResult(
            chunks=[_CHUNK],
            retriever="spine",
            acl_applied=True,
            candidate_count=10,
        )
    )
    node = make_retrieve_node(spine)
    state = _make_base_state()

    result = await node(state)  # type: ignore[arg-type]

    verdicts = result.get("verdicts") or []
    assert len(verdicts) >= 1, (
        f"Expected at least 1 verdict entry from the retrieve node, got {verdicts!r}"
    )
    # The verdict must record the node name
    assert any("retrieve" in str(v) for v in verdicts), (
        f"Expected a verdict mentioning 'retrieve', got {verdicts!r}"
    )


# ---------------------------------------------------------------------------
# Integration-ish: build_graph with decomposed retrieve → grade_documents →
# generate path; fake spine ≥1 chunk; fake assembler → terminal state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_decomposed_graph_retrieve_path_end_to_end() -> None:
    """End-to-end with decomposed graph: router → retrieve → grade_documents → generate.

    Uses a fake LLM (returns 'retrieve' route), fake spine (returns 1 chunk),
    and fake assembler.  Verifies the terminal state contains ``answer`` +
    ``chunks`` after traversing the new node topology.
    """
    from iw_rag.agent.graph import answer_question, build_graph
    from iw_rag.generation.llm import GenerationParams, LlmResponse
    from iw_rag.settings import AgentSettings

    class _FakeLlm:
        async def complete(self, messages: Any, params: GenerationParams) -> LlmResponse:
            return LlmResponse(text="stub answer", finish_reason="stop")

        async def complete_structured(
            self, messages: Any, json_schema: Any, params: GenerationParams
        ) -> LlmResponse:
            import json

            # Dispatch based on the schema's required fields.
            # Router schema requires "route"; ISREL requires "grades";
            # ISSUP requires "supported"; ISUSE requires "answers".
            required = json_schema.get("required", [])
            if "route" in required:
                return LlmResponse(
                    text=json.dumps({"query_type": "factual", "route": "retrieve"}),
                    finish_reason="stop",
                )
            if "grades" in required:
                # ISREL grade_documents: all chunks relevant
                return LlmResponse(
                    text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
                    finish_reason="stop",
                )
            if "supported" in required:
                # ISSUP grader: answer is supported
                return LlmResponse(
                    text=json.dumps({"supported": True, "reason": "grounded"}),
                    finish_reason="stop",
                )
            if "answers" in required:
                # ISUSE grader: answer addresses the question
                return LlmResponse(
                    text=json.dumps({"answers": True, "reason": "useful"}),
                    finish_reason="stop",
                )
            # Default: route to retrieve
            return LlmResponse(
                text=json.dumps({"query_type": "factual", "route": "retrieve"}),
                finish_reason="stop",
            )

        async def complete_tools(
            self, messages: Any, tools: Any, params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="", finish_reason="stop")

        async def health(self) -> bool:
            return True

    spine = _FakeSpine(
        result=RetrievalResult(
            chunks=[_CHUNK],
            retriever="spine",
            acl_applied=True,
            candidate_count=1,
        )
    )
    assembler = _FakeAssembler(grounded=True, answer="You have 20 days of leave [1].")

    fake_llm = _FakeLlm()
    settings = AgentSettings(router_enabled=True, skip_enabled=True)

    graph = build_graph(spine, assembler, fake_llm, settings=settings)
    state = await answer_question(graph, "What is the leave policy?", _CTX)

    assert isinstance(state, dict), f"Expected dict state, got {type(state)!r}"
    # Terminal answer must be present
    assert state.get("answer") is not None, (
        f"Expected 'answer' in terminal state, got {dict(state)!r}"
    )
    # Chunks must have been set (from the retrieve node)
    assert state.get("chunks") is not None, (
        f"Expected 'chunks' in terminal state, got {dict(state)!r}"
    )
    chunks = state.get("chunks") or []
    assert len(chunks) >= 1, f"Expected ≥1 chunk in terminal state, got {chunks!r}"
    # disposition must be answered
    assert state.get("disposition") == "answered", (
        f"Expected disposition='answered', got {state.get('disposition')!r}"
    )
    # Verdicts must contain router + retrieve entries
    verdicts = state.get("verdicts") or []
    kinds = [v.kind if hasattr(v, "kind") else str(v) for v in verdicts]
    assert any("router" in k for k in kinds), f"Expected router verdict in {kinds!r}"
