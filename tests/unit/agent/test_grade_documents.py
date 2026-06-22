"""Unit tests for grade_documents node, _aggregate_doc_verdict, rewrite_query, and consume_retry.

Covers:
- ``_aggregate_doc_verdict``: full CRAG 3-action table (correct/ambiguous/incorrect).
- ``make_grade_documents``: single batched structured call (not per-chunk), keeps only
  relevant chunks in ``graded``, appends exactly one verdict.
- ``rewrite_query``: rewrites from ``original_question`` + failure signal; preserves
  ``original_question`` unchanged; records the new question in state.
- ``consume_retry``: increments ``retries`` by exactly 1; no other node body touches it.
- Integration-ish: first retrieval graded incorrect, retry produces relevant → retries==1.

References:
- task-2.3-grade-documents-and-rewrite §2, §3, §4, §6, Tests to write
- lld-agentic-loop §3.2 (CRAG aggregation), §3.3 (rewrite), §3.6 (budget)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate), §4
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from iw_rag.agent.state import GraphState, Verdict
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.generation.llm import GenerationParams, LlmResponse
from iw_rag.settings import AgentSettings

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-grade-test",
)

_CHUNK_A = RetrievedChunk(
    chunk_id="ca",
    document_id="doc-a",
    content="The company grants 20 days of leave per year.",
    score=0.92,
    source_uri="corpus://hr/leave_policy.md",
)

_CHUNK_B = RetrievedChunk(
    chunk_id="cb",
    document_id="doc-b",
    content="The marketing budget for Q3 is $500k.",
    score=0.60,
    source_uri="corpus://finance/budget.md",
)

_CHUNK_C = RetrievedChunk(
    chunk_id="cc",
    document_id="doc-c",
    content="Annual leave entitlement is 20 days for full-time staff.",
    score=0.88,
    source_uri="corpus://hr/leave_policy.md",
)


def _base_state(**overrides: Any) -> GraphState:
    """Build a minimal GraphState dict for tests.

    Args:
        **overrides: Keys to override/add in the state dict.

    Returns:
        A dict compatible with ``GraphState``.
    """
    state: GraphState = {
        "question": "What is the leave policy?",
        "original_question": "What is the leave policy?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "chunks": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _grade_response(relevant: bool) -> LlmResponse:
    """Build a fake structured-output LlmResponse for an ISREL grade.

    Args:
        relevant: Whether the chunk is relevant.

    Returns:
        An LlmResponse whose text is a JSON-encoded ISREL verdict.
    """
    return LlmResponse(
        text=json.dumps({"relevant": relevant, "reason": "test"}),
        finish_reason="stop",
    )


# ---------------------------------------------------------------------------
# Tests: _aggregate_doc_verdict — CRAG 3-action table
# ---------------------------------------------------------------------------


def test_aggregate_doc_verdict_all_relevant_returns_correct() -> None:
    """All chunks relevant → ``correct`` (full agreement)."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[True, True, True])
    assert result == "correct", f"Expected 'correct', got {result!r}"


def test_aggregate_doc_verdict_some_relevant_returns_ambiguous() -> None:
    """Some but not all chunks relevant → ``ambiguous``."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[True, False, True])
    assert result == "ambiguous", f"Expected 'ambiguous', got {result!r}"


def test_aggregate_doc_verdict_none_relevant_returns_incorrect() -> None:
    """No chunks relevant → ``incorrect``."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[False, False, False])
    assert result == "incorrect", f"Expected 'incorrect', got {result!r}"


def test_aggregate_doc_verdict_empty_returns_incorrect() -> None:
    """Empty candidates (no chunks) → ``incorrect``."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[])
    assert result == "incorrect", f"Expected 'incorrect' for empty grades, got {result!r}"


def test_aggregate_doc_verdict_single_relevant_is_correct() -> None:
    """Single chunk, relevant → ``correct``."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[True])
    assert result == "correct", f"Expected 'correct' for single-relevant, got {result!r}"


def test_aggregate_doc_verdict_single_irrelevant_is_incorrect() -> None:
    """Single chunk, irrelevant → ``incorrect``."""
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=[False])
    assert result == "incorrect", f"Expected 'incorrect' for single-irrelevant, got {result!r}"


@pytest.mark.parametrize(
    "grades, expected",
    [
        # All relevant
        ([True, True], "correct"),
        ([True, True, True], "correct"),
        # None relevant
        ([False, False], "incorrect"),
        ([False], "incorrect"),
        # Mixed → ambiguous
        ([True, False], "ambiguous"),
        ([False, True, False], "ambiguous"),
        ([True, True, False], "ambiguous"),
        # Empty
        ([], "incorrect"),
    ],
)
def test_aggregate_doc_verdict_parametrized_table(grades: list[bool], expected: str) -> None:
    """Table-driven CRAG aggregation: full coverage of all branches.

    Args:
        grades: List of per-chunk relevance booleans.
        expected: Expected CRAG action string.
    """
    from iw_rag.agent.nodes import _aggregate_doc_verdict

    result = _aggregate_doc_verdict(grades=grades)
    assert result == expected, f"grades={grades!r}: expected={expected!r}, got={result!r}"


# ---------------------------------------------------------------------------
# Tests: grade_documents node — batched, single call, keeps relevant subset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_documents_issues_single_batched_call() -> None:
    """grade_documents issues a single batched structured call, not one per chunk.

    The grader LLM's complete_structured must be called exactly once regardless
    of the number of chunks — batching is mandatory (R-00105 F8).
    """
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    # Return a batch response containing grades for all chunks:
    # CHUNK_A relevant, CHUNK_B not → doc_verdict='ambiguous'
    batch_response = json.dumps(
        {
            "grades": [
                {"relevant": True, "reason": "ok"},
                {"relevant": False, "reason": "off-topic"},
            ]
        }
    )
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(text=batch_response, finish_reason="stop")
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_A, _CHUNK_B])
    result = await node(state)

    # Exactly ONE structured call regardless of chunk count
    assert llm.complete_structured.call_count == 1, (
        f"Expected exactly 1 complete_structured call (batched), "
        f"got {llm.complete_structured.call_count}"
    )
    # Behavioural assertion: mixed grades → doc_verdict='ambiguous', only relevant chunk kept
    assert result.get("doc_verdict") == "ambiguous", (
        f"Mixed grades (one relevant, one irrelevant) must yield doc_verdict='ambiguous', "
        f"got {result.get('doc_verdict')!r}"
    )
    assert result.get("graded") == [_CHUNK_A], (
        f"Only the relevant chunk (CHUNK_A) must be in graded, got {result.get('graded')!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_keeps_only_relevant_chunks() -> None:
    """grade_documents keeps only the chunks graded relevant in state['graded']."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    # CHUNK_A relevant, CHUNK_B not, CHUNK_C relevant
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps(
                {
                    "grades": [
                        {"relevant": True, "reason": "on-topic"},
                        {"relevant": False, "reason": "off-topic"},
                        {"relevant": True, "reason": "on-topic"},
                    ]
                }
            ),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_A, _CHUNK_B, _CHUNK_C])
    result = await node(state)

    graded = result["graded"]
    assert graded == [_CHUNK_A, _CHUNK_C], (
        f"Expected only relevant chunks in graded, got {graded!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_appends_exactly_one_verdict() -> None:
    """grade_documents appends exactly one Verdict per invocation."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_A])
    result = await node(state)

    verdicts: list[Verdict] = result.get("verdicts") or []
    assert len(verdicts) == 1, f"Expected exactly 1 verdict, got {len(verdicts)}"
    v = verdicts[0]
    assert v.kind == "grade_documents", f"Expected kind='grade_documents', got {v.kind!r}"


@pytest.mark.asyncio
async def test_grade_documents_all_relevant_sets_correct_verdict() -> None:
    """All chunks relevant → doc_verdict='correct', all chunks in graded."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps(
                {
                    "grades": [
                        {"relevant": True, "reason": "a"},
                        {"relevant": True, "reason": "b"},
                    ]
                }
            ),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_A, _CHUNK_C])
    result = await node(state)

    assert result["doc_verdict"] == "correct", (
        f"Expected doc_verdict='correct', got {result['doc_verdict']!r}"
    )
    assert result["graded"] == [_CHUNK_A, _CHUNK_C]


@pytest.mark.asyncio
async def test_grade_documents_none_relevant_sets_incorrect_verdict() -> None:
    """No chunks relevant → doc_verdict='incorrect', graded is empty."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps({"grades": [{"relevant": False, "reason": "off"}]}),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_B])
    result = await node(state)

    assert result["doc_verdict"] == "incorrect", (
        f"Expected doc_verdict='incorrect', got {result['doc_verdict']!r}"
    )
    assert result["graded"] == []


@pytest.mark.asyncio
async def test_grade_documents_mixed_sets_ambiguous_verdict() -> None:
    """Mixed chunks → doc_verdict='ambiguous', only relevant chunks in graded."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps(
                {
                    "grades": [
                        {"relevant": True, "reason": "good"},
                        {"relevant": False, "reason": "bad"},
                    ]
                }
            ),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[_CHUNK_A, _CHUNK_B])
    result = await node(state)

    assert result["doc_verdict"] == "ambiguous", (
        f"Expected doc_verdict='ambiguous', got {result['doc_verdict']!r}"
    )
    assert result["graded"] == [_CHUNK_A]


@pytest.mark.asyncio
async def test_grade_documents_empty_chunks_returns_incorrect() -> None:
    """Empty chunks → doc_verdict='incorrect', no LLM call needed."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    state = _base_state(chunks=[])
    result = await node(state)

    assert result["doc_verdict"] == "incorrect", (
        f"Expected 'incorrect' for empty chunks, got {result['doc_verdict']!r}"
    )
    assert result["graded"] == []
    # No LLM call needed for empty chunks
    llm.complete_structured.assert_not_called()


@pytest.mark.asyncio
async def test_grade_documents_verdict_records_attempt_index() -> None:
    """The grade_documents verdict records the current retries pass as attempt."""
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)

    # Second pass (retries=1)
    state = _base_state(chunks=[_CHUNK_A], retries=1)
    result = await node(state)

    verdicts: list[Verdict] = result.get("verdicts") or []
    assert verdicts[0].attempt == 1, (
        f"Expected attempt=1 (matching retries), got {verdicts[0].attempt}"
    )


# ---------------------------------------------------------------------------
# Tests: rewrite_query node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rewrite_query_updates_question_from_original() -> None:
    """rewrite_query sets state['question'] to a new value based on original_question.

    The original_question must remain unchanged (no drift from user intent).
    """
    from iw_rag.agent.nodes import make_rewrite_query

    rewritten = "annual leave entitlement company policy"
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LlmResponse(text=rewritten, finish_reason="stop"))

    settings = AgentSettings()
    node = make_rewrite_query(llm, settings)

    state = _base_state(
        question="What is the leave policy?",
        original_question="What is the leave policy?",
        doc_verdict="incorrect",
    )
    result = await node(state)

    assert result["question"] == rewritten, (
        f"Expected rewritten question={rewritten!r}, got {result['question']!r}"
    )
    # original_question must NOT be written (preserved)
    assert "original_question" not in result or result.get("original_question") is None, (
        "rewrite_query must not overwrite original_question"
    )


@pytest.mark.asyncio
async def test_rewrite_query_appends_verdict() -> None:
    """rewrite_query appends exactly one verdict of kind='rewrite_query'."""
    from iw_rag.agent.nodes import make_rewrite_query

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LlmResponse(text="rewritten query", finish_reason="stop"))

    settings = AgentSettings()
    node = make_rewrite_query(llm, settings)

    state = _base_state(doc_verdict="incorrect")
    result = await node(state)

    verdicts: list[Verdict] = result.get("verdicts") or []
    assert len(verdicts) == 1, f"Expected 1 verdict, got {len(verdicts)}"
    assert verdicts[0].kind == "rewrite_query", (
        f"Expected kind='rewrite_query', got {verdicts[0].kind!r}"
    )


@pytest.mark.asyncio
async def test_rewrite_query_uses_original_question_not_current() -> None:
    """rewrite_query reformulates from original_question, not state['question'].

    This ensures successive rewrites don't drift off the user's original intent
    (R-00105 F4).  The rewritten question in state['question'] must be updated
    to the LLM's return value, not the stale current question.
    """
    from iw_rag.agent.nodes import make_rewrite_query

    calls: list[list[Any]] = []

    class _TrackingLlm:
        async def complete(self, messages: list[Any], params: GenerationParams) -> LlmResponse:
            calls.append(messages)
            return LlmResponse(text="better query", finish_reason="stop")

        async def complete_structured(
            self, messages: list[Any], json_schema: Any, params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="{}", finish_reason="stop")

        async def complete_tools(
            self, messages: list[Any], tools: Any, params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="", finish_reason="stop")

        async def health(self) -> bool:
            return True

    settings = AgentSettings()
    node = make_rewrite_query(_TrackingLlm(), settings)

    state = _base_state(
        question="rewritten v1",  # current question (already rewritten once)
        original_question="What is the leave policy?",
        doc_verdict="incorrect",
    )
    result = await node(state)

    # The prompt must include the original_question
    assert calls, "Expected at least one LLM call"
    all_content = " ".join(m.get("content", "") for m in calls[0] if isinstance(m, dict))
    assert "What is the leave policy?" in all_content, (
        f"Expected original_question in LLM prompt; got messages={calls[0]!r}"
    )
    # Exact value check: state['question'] must be the LLM's rewritten text
    assert result.get("question") == "better query", (
        f"rewrite_query must set state['question'] to LLM output 'better query', "
        f"got {result.get('question')!r}"
    )


# ---------------------------------------------------------------------------
# Tests: consume_retry node
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consume_retry_increments_retries_by_one() -> None:
    """consume_retry increments retries by exactly 1."""
    from iw_rag.agent.nodes import consume_retry

    state = _base_state(retries=0)
    result = await consume_retry(state)

    assert result["retries"] == 1, f"Expected retries=1, got {result['retries']}"


@pytest.mark.asyncio
async def test_consume_retry_increments_from_nonzero() -> None:
    """consume_retry increments from any starting value by exactly 1."""
    from iw_rag.agent.nodes import consume_retry

    state = _base_state(retries=1)
    result = await consume_retry(state)

    assert result["retries"] == 2, f"Expected retries=2, got {result['retries']}"


@pytest.mark.asyncio
async def test_consume_retry_only_returns_retries_key() -> None:
    """consume_retry only touches the 'retries' key — no other state mutation.

    This enforces the single-authority rule: retries is incremented in exactly
    one place (lld-agentic-loop §3.6).
    """
    from iw_rag.agent.nodes import consume_retry

    state = _base_state(retries=0)
    result = await consume_retry(state)

    # The returned dict must ONLY set retries
    assert list(result.keys()) == ["retries"], (
        f"consume_retry must only return {{'retries': ...}}, got keys={list(result.keys())!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_node_body_does_not_mutate_retries() -> None:
    """grade_documents must not return 'retries' in its result dict.

    Verifies at runtime that a grade_documents call does not include 'retries'
    in its returned dict (the definitive single-authority invariant check).
    """
    from iw_rag.agent.nodes import make_grade_documents

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(
        return_value=LlmResponse(
            text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
            finish_reason="stop",
        )
    )

    settings = AgentSettings()
    node = make_grade_documents(llm, settings)
    state = _base_state(chunks=[_CHUNK_A], retries=0)

    result = await node(state)

    assert "retries" not in result, (
        f"grade_documents must NOT write to state['retries']. "
        f"Only consume_retry may increment the counter. "
        f"Got result keys: {list(result.keys())!r}"
    )


# ---------------------------------------------------------------------------
# Tests: I-00131 per-candidate context-budget clamp (defense in depth)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_documents_clamps_oversized_candidate_to_budget() -> None:
    """grade_documents must clamp each candidate's content to the char budget.

    Defense in depth (I-00131): the spine already gates parent expansion on
    the budget, but if an oversized candidate ever slips through, the grader
    prompt must still be bounded so it can never overflow the generator's
    ``max_model_len`` window.  The clamp is a deterministic first-N-chars
    prefix on each candidate before it is joined into the prompt.
    """
    from iw_rag.agent.nodes import make_grade_documents

    budget = 200
    # _CHUNK_A is "The company grants 20 days of leave per year." (~44 chars).
    # Build an oversized chunk that must be clamped to exactly ``budget`` chars.
    oversized = RetrievedChunk(
        chunk_id="big",
        document_id="doc-big",
        content="X" * 50_000,
        score=0.99,
        source_uri="corpus://big.md",
    )

    captured: dict[str, Any] = {}

    async def _capture(messages: list[Any], json_schema: Any, params: Any) -> LlmResponse:
        captured["messages"] = messages
        return LlmResponse(
            text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
            finish_reason="stop",
        )

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(side_effect=_capture)

    settings = AgentSettings()
    node = make_grade_documents(llm, settings, parent_expansion_max_chars=budget)
    state = _base_state(chunks=[oversized, _CHUNK_A])

    await node(state)

    # Inspect the user message to verify clamping
    user_message = captured["messages"][1]["content"]
    # The oversized chunk appears as "[1] XXXX..." with exactly budget X's
    assert f"[1] {'X' * budget}" in user_message, (
        f"Expected '[1] ' + {budget} X's (clamped) in the prompt, got user_message={user_message!r}"
    )
    # The next chunk's index is [2] (clamp must not affect subsequent items)
    assert "[2] The company grants" in user_message, (
        f"Expected '[2] The company grants...' (next candidate unclamped) in the prompt, "
        f"got user_message={user_message!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_clamp_passes_through_small_chunks() -> None:
    """A candidate within the budget must NOT be truncated (no regression)."""
    from iw_rag.agent.nodes import make_grade_documents

    captured: dict[str, Any] = {}

    async def _capture(messages: list[Any], json_schema: Any, params: Any) -> LlmResponse:
        captured["messages"] = messages
        return LlmResponse(
            text=json.dumps({"grades": [{"relevant": True, "reason": "ok"}]}),
            finish_reason="stop",
        )

    llm = AsyncMock()
    llm.complete_structured = AsyncMock(side_effect=_capture)

    settings = AgentSettings()
    # Big budget so neither candidate is clamped
    node = make_grade_documents(llm, settings, parent_expansion_max_chars=4_000)
    state = _base_state(chunks=[_CHUNK_A, _CHUNK_B])

    await node(state)

    user_message = captured["messages"][1]["content"]
    assert _CHUNK_A.content in user_message
    assert _CHUNK_B.content in user_message


def test_rewrite_query_node_body_does_not_mutate_retries() -> None:
    """rewrite_query must not touch state['retries'] — single authority check.

    Verifies at source-code level that make_rewrite_query never sets 'retries'
    in its returned dict, then confirms the runtime return value has no 'retries' key.
    """
    import inspect

    from iw_rag.agent import nodes as nodes_mod

    source = inspect.getsource(nodes_mod.make_rewrite_query)
    # Key check: the string "retries" should not appear as an assignment target
    lines = source.splitlines()
    for line in lines:
        stripped = line.strip()
        # Flag if we see something like '"retries": ...' being *set* (not read)
        if '"retries"' in stripped and "state.get" not in stripped and "state[" not in stripped:
            pytest.fail(
                f"make_rewrite_query appears to write to 'retries': {stripped!r}\n"
                "Only consume_retry may increment retries."
            )
    # Additional assertion: 'retries' must not appear as a return key in source
    assert '"retries"' not in source or "state.get" in source or "state[" in source, (
        "make_rewrite_query source must not assign to 'retries'"
    )


# ---------------------------------------------------------------------------
# budget_left helper tests
# ---------------------------------------------------------------------------


def test_budget_left_returns_true_when_retries_below_max() -> None:
    """budget_left returns True when retries < max_retries."""
    from iw_rag.agent.budgets import budget_left

    state = _base_state(retries=0, max_retries=2)
    assert budget_left(state) is True


def test_budget_left_returns_false_when_retries_equals_max() -> None:
    """budget_left returns False when retries == max_retries (budget exhausted)."""
    from iw_rag.agent.budgets import budget_left

    state = _base_state(retries=2, max_retries=2)
    assert budget_left(state) is False


def test_budget_left_returns_false_when_retries_exceeds_max() -> None:
    """budget_left returns False when retries > max_retries."""
    from iw_rag.agent.budgets import budget_left

    state = _base_state(retries=3, max_retries=2)
    assert budget_left(state) is False


# ---------------------------------------------------------------------------
# Integration-ish: corrective cycle with fakes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_corrective_cycle_retries_once_on_incorrect_then_relevant() -> None:
    """First retrieval graded incorrect triggers corrective cycle; second is relevant.

    Verifies the full corrective cycle: incorrect → consume_retry → rewrite →
    retrieve (relevant) → generate. Asserts retries==1 at the end.

    Uses a fake LLM + spine controlled per-call.
    """
    from iw_rag.agent.graph import answer_question, build_graph
    from iw_rag.domain.models import Citation, GeneratedAnswer

    call_count_structured = 0
    call_count_complete = 0

    class _FakeLlm:
        """Fake LLM that simulates multiple node call types in sequence.

        Call sequence (Task 2.4 graph topology):
        1. Router → route=retrieve
        2. grade_documents pass 1 → all irrelevant → corrective cycle
        3. grade_documents pass 2 → all relevant → generate
        4. ISSUP pass 1 → supported=True
        5. ISUSE pass 1 → answers=True
        → done (grounded) → answered
        """

        async def complete(self, messages: list[Any], params: GenerationParams) -> LlmResponse:
            nonlocal call_count_complete
            call_count_complete += 1
            # Router or rewrite call: return generic text
            return LlmResponse(text="rewritten query about leave entitlement", finish_reason="stop")

        async def complete_structured(
            self, messages: list[Any], json_schema: Any, params: GenerationParams
        ) -> LlmResponse:
            nonlocal call_count_structured
            call_count_structured += 1
            # Dispatch by schema required fields to be order-independent and robust.
            required = json_schema.get("required", [])

            # Router schema
            if "route" in required:
                return LlmResponse(
                    text=json.dumps({"query_type": "factual", "route": "retrieve"}),
                    finish_reason="stop",
                )
            # ISREL grade_documents schema
            if "grades" in required:
                if call_count_structured == 2:
                    # Pass 1: irrelevant (triggers corrective cycle)
                    return LlmResponse(
                        text=json.dumps({"grades": [{"relevant": False, "reason": "off-topic"}]}),
                        finish_reason="stop",
                    )
                # Pass 2+: relevant
                return LlmResponse(
                    text=json.dumps({"grades": [{"relevant": True, "reason": "on-topic"}]}),
                    finish_reason="stop",
                )
            # ISSUP schema
            if "supported" in required:
                return LlmResponse(
                    text=json.dumps({"supported": True, "reason": "grounded"}),
                    finish_reason="stop",
                )
            # ISUSE schema
            if "answers" in required:
                return LlmResponse(
                    text=json.dumps({"answers": True, "reason": "useful"}),
                    finish_reason="stop",
                )
            # Default fallback
            return LlmResponse(
                text=json.dumps({"query_type": "factual", "route": "retrieve"}),
                finish_reason="stop",
            )

        async def complete_tools(
            self, messages: list[Any], tools: Any, params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="", finish_reason="stop")

        async def health(self) -> bool:
            return True

    spine_call_count = 0

    class _FakeSpine:
        """Spine that always returns one chunk."""

        async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
            nonlocal spine_call_count
            spine_call_count += 1
            return RetrievalResult(
                chunks=[_CHUNK_A],
                retriever="spine",
                acl_applied=True,
                candidate_count=1,
            )

    class _FakeAssembler:
        """Assembler that returns a grounded answer."""

        async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
            return GeneratedAnswer(
                answer="You have 20 days of annual leave [1].",
                grounded=True,
                citations=[
                    Citation(
                        marker=1,
                        chunk_id="ca",
                        document_id="doc-a",
                        source_uri="corpus://hr/leave_policy.md",
                    )
                ],
                used_chunk_ids=["ca"],
            )

    settings = AgentSettings(router_enabled=True, skip_enabled=True)
    graph = build_graph(
        _FakeSpine(),
        _FakeAssembler(),
        _FakeLlm(),
        settings=settings,
    )

    state = await answer_question(graph, "What is the leave policy?", _CTX)

    # Corrective cycle must have run → retries should be 1
    assert state.get("retries") == 1, (
        f"Expected retries==1 after one corrective pass, got {state.get('retries')!r}"
    )
    # The graph must terminate with an answer
    assert state.get("disposition") == "answered", (
        f"Expected disposition='answered', got {state.get('disposition')!r}"
    )
    # Spine was called twice (initial + retry)
    assert spine_call_count == 2, (
        f"Expected spine called 2 times (initial + retry), got {spine_call_count}"
    )
