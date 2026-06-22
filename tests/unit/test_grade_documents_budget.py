"""S05 — Reproduction + regression tests for the ``grade_documents`` per-candidate
context-budget clamp (I-00131, defense in depth).

The runtime half of I-00131 is enforced at two layers:

1. **Spine gate** (``spine._expand_parents``) — refuses to substitute a parent
   that exceeds ``parent_expansion_max_chars`` and keeps the leaf content.
2. **Grader clamp** (``nodes._clamp_to_budget``, applied in
   ``make_grade_documents``) — deterministically truncates each candidate's
   ``content`` to at most ``parent_expansion_max_chars`` chars before it is
   joined into the assembled grader prompt.  This is the *defence in depth*
   layer: even if a future retrieval path bypasses the spine (long_context,
   visual, graphrag, etc.) and a candidate slips through oversize, the grader
   prompt can never exceed the per-candidate budget.

These tests assert the *semantic* invariant — **no candidate's contribution
to the assembled grader prompt exceeds the budget** — not merely the
structural shape of the response.  Each test uses a deterministic fake
:class:`LlmClient` that records the ``messages`` it is called with, then
inspects the recorded user message's actual string length to verify the
invariant.  No network calls, no live models, no testcontainers.

The S01 file ``tests/unit/agent/test_grade_documents.py`` contains two
*primary* clamp tests (oversize clamped, within-budget passthrough); this
file is the S05-spec reproduction suite — additional semantic tests that
lock in the design's *assembled-prompt* and *per-candidate-verdict* invariants
and would fail against the pre-fix code (which concatenated candidate
content with no per-candidate cap, so an oversized candidate would push
the prompt to ``top_k * parent_size`` chars and overflow the 16,384-token
generator window).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import pytest

from iw_rag.agent.state import GraphState
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievedChunk
from iw_rag.generation.llm import GenerationParams, LlmClient, LlmResponse, Message
from iw_rag.settings import AgentSettings

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u-budget",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-budget-clamp",
)

_CHUNK_SMALL = RetrievedChunk(
    chunk_id="c-small",
    document_id="doc-small",
    content="The company grants 20 days of leave per year.",
    score=0.92,
    source_uri="corpus://hr/leave_policy.md",
)


def _oversized_chunk(
    *, marker: str = "X", size: int = 50_000, chunk_id: str = "c-big"
) -> RetrievedChunk:
    """Build a candidate whose ``content`` is far larger than the budget."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-big",
        content=marker * size,
        score=0.99,
        source_uri="corpus://uploaded/smile5.pdf",
    )


def _base_state(**overrides: Any) -> GraphState:
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


# ---------------------------------------------------------------------------
# Fake LLM client (records every complete_structured call)
# ---------------------------------------------------------------------------


class _RecordingLlm:
    """Deterministic, no-network :class:`LlmClient` fake.

    Records every ``complete_structured`` call's ``messages`` and ``params``
    so tests can inspect the actual string the grader built.  A list of
    per-call verdicts (``[True, False, True, ...]``) may be provided; the
    i-th call returns a batched ``{"grades": [{"relevant": v}, ...]}``
    response with one entry per chunk.  ``complete``/``complete_tools``/
    ``health`` are no-ops so the fake satisfies the protocol.
    """

    def __init__(self, verdicts_per_call: list[list[bool]] | None = None) -> None:
        self.records: list[dict[str, Any]] = []
        self._verdicts = verdicts_per_call or []
        self._call_idx = 0

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        return LlmResponse(text="", finish_reason="stop")

    async def complete_tools(
        self, messages: Sequence[Message], tools: Any, params: GenerationParams
    ) -> LlmResponse:
        return LlmResponse(text="", finish_reason="stop")

    async def complete_structured(
        self, messages: Sequence[Message], json_schema: dict[str, Any], params: GenerationParams
    ) -> LlmResponse:
        # Snapshot the messages (deep enough — they're dict[str, str])
        self.records.append(
            {
                "messages": list(messages),
                "params": params,
                "json_schema": json_schema,
            }
        )
        # If the caller provided a per-call verdict list, use it; otherwise
        # mark all chunks as relevant (the safe default for the no-fail path).
        if self._call_idx < len(self._verdicts):
            grades = self._verdicts[self._call_idx]
        else:
            grades = [True] * 4  # safe default, unused by the tests below
        self._call_idx += 1
        return LlmResponse(
            text=json.dumps({"grades": [{"relevant": g} for g in grades]}),
            finish_reason="stop",
        )

    async def health(self) -> bool:
        return True


def _is_llm_client_compliant(llm: _RecordingLlm) -> bool:
    """Smoke check: the fake honours the structural :class:`LlmClient` protocol.

    ``LlmClient`` is ``@runtime_checkable`` so an ``isinstance`` check would
    work, but a method-existence check is clearer and not coupled to the
    runtime-checkable decoration.
    """
    return all(
        callable(getattr(llm, name, None))
        for name in ("complete", "complete_tools", "complete_structured", "health")
    )


# ---------------------------------------------------------------------------
# Helpers — extract per-candidate content from the recorded user message
# ---------------------------------------------------------------------------


def _user_message(llm: _RecordingLlm, *, call_index: int = 0) -> str:
    """Return the user message recorded on the given ``complete_structured`` call.

    The grader issues exactly one structured call (batched) — see
    ``tests/unit/agent/test_grade_documents.py::test_grade_documents_issues_single_batched_call``.
    Tests can therefore safely inspect ``records[0]``.
    """
    record = llm.records[call_index]
    messages = record["messages"]
    # Defensive: the grader always emits a [system, user] pair
    assert len(messages) >= 2, (
        f"Expected at least 2 messages (system + user); got {len(messages)}: {messages!r}"
    )
    user = messages[1]
    role = user.get("role")
    assert role == "user", f"Expected second message role='user'; got role={role!r}"
    content = user.get("content")
    assert isinstance(content, str), (
        f"User message content must be str; got {type(content).__name__}"
    )
    return content


def _extract_candidate_blocks(user_msg: str) -> list[str]:
    """Return the list of ``[i] ...`` candidate blocks from the grader's user message.

    The grader's user message is::

        Question: <q>
        <blank>
        Documents to grade (one per number):
        [1] <clamped content>
        <blank>
        [2] <clamped content>
        ...

    This helper pulls out the content after each ``[<n>] `` prefix, in order.
    The split is whitespace-tolerant (the candidate blocks are separated by
    blank lines, but the helper does not assume the exact separator).
    """
    blocks: list[str] = []
    lines = user_msg.splitlines()
    current: list[str] = []
    for line in lines:
        if line.startswith("[") and "]" in line:
            if current:
                blocks.append("\n".join(current).rstrip("\n"))
                current = []
            # Drop the "[N] " prefix
            current.append(line.split("]", 1)[1].lstrip())
        else:
            if current:
                current.append(line)
    if current:
        blocks.append("\n".join(current).rstrip("\n"))
    return blocks


# ---------------------------------------------------------------------------
# Tests — grader clamp (I-00131 S05)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_grade_documents_clamp_records_user_message_and_clamps_oversized() -> None:
    """RED before fix: the user message contains the full 50,000-char content
    of the oversized candidate, so the assembled prompt can exceed the
    generator's 16,384-token window.

    GREEN after: each candidate's content in the user message is clamped to
    exactly ``budget`` chars; the recorded user message has a bounded length
    regardless of how many oversized candidates are passed in.
    """
    budget = 500
    oversized = _oversized_chunk(marker="X", size=50_000, chunk_id="big-1")

    llm = _RecordingLlm(verdicts_per_call=[[True]])
    assert _is_llm_client_compliant(llm), "Fake must satisfy the LlmClient protocol"

    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=budget)
    state = _base_state(chunks=[oversized])

    result = await node(state)

    # ---- Semantic invariant: recorded message is bounded --------------------
    user_msg = _user_message(llm)
    blocks = _extract_candidate_blocks(user_msg)
    assert len(blocks) == 1, f"Expected 1 candidate block, got {len(blocks)}: {blocks!r}"

    candidate_text = blocks[0]
    # The clamp is a deterministic first-N-chars prefix
    assert candidate_text == "X" * budget, (
        f"Oversized candidate must be clamped to exactly {budget} chars; "
        f"got len={len(candidate_text)} candidate_text={candidate_text[:80]!r}..."
    )
    # And, crucially, the actual recorded user-message length is bounded
    assert len(user_msg) <= budget + 200, (
        f"Recorded user message must stay within budget + small overhead; "
        f"got len(user_msg)={len(user_msg)} for budget={budget}. "
        f"An oversize candidate slipped into the prompt."
    )

    # ---- Grader still returns per-candidate verdicts correctly --------------
    assert "doc_verdict" in result
    assert result["doc_verdict"] == "correct"  # the fake returned [True]
    assert result["graded"] == [oversized]
    assert len(result["verdicts"]) == 1
    assert result["verdicts"][0].kind == "grade_documents"
    assert result["verdicts"][0].value == "correct"


@pytest.mark.asyncio
async def test_grade_documents_token_budget_bounds_prompt_for_eight_oversized() -> None:
    """Real-world repro (I-CTXOVERFLOW): 8 oversized candidates (the design's
    ``top_k=8``) must NOT multiply into an over-the-window prompt.

    The char clamp alone could not guarantee this: 8 candidates each clamped to
    4000 chars still summed past the 16,384-token window once guards-on
    datamarking inflated the *token* count ~12x — which is exactly how the live
    grader 400'd at 34k input tokens.  The token-budgeting upgrade fits the
    candidates to ``max_model_len`` minus the grader's output ceiling and the
    fixed-overhead headroom: it keeps the highest-ranked candidates that fit,
    drops the lowest-ranked overflow tail, and only truncates a lone oversized
    candidate.  The contract this test pins is the strong one — **the assembled
    grader prompt provably fits the window** — not the old "keep all 8, clamp
    each" shape.
    """
    from iw_rag.agent.prompts import DOC_GRADE_MAX_TOKENS
    from iw_rag.generation.context_budget import estimate_tokens
    from iw_rag.settings import GenerationSettings

    gen = GenerationSettings()  # max_model_len=16384, max_tokens=1024, headroom=2048
    budget = 4_000  # the design's recommended per-candidate char ceiling
    top_k = 8
    chunks = [_oversized_chunk(marker="A", size=50_000, chunk_id=f"big-{i}") for i in range(top_k)]

    llm = _RecordingLlm(verdicts_per_call=[[True] * top_k])
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(
        llm,
        AgentSettings(),
        parent_expansion_max_chars=budget,
        max_model_len=gen.max_model_len,
        context_headroom_tokens=gen.context_headroom_tokens,
    )
    state = _base_state(chunks=chunks)

    result = await node(state)

    user_msg = _user_message(llm)
    blocks = _extract_candidate_blocks(user_msg)

    # 1) The number of candidates is a bounded PREFIX (never multiplied past
    #    top_k; the overflow tail is dropped rather than blowing the window).
    assert 0 < len(blocks) <= top_k, f"Expected 1..{top_k} candidate blocks, got {len(blocks)}"

    # 2) Each kept candidate still respects the per-candidate char clamp.
    for i, block in enumerate(blocks):
        assert len(block) <= budget, (
            f"Candidate [{i + 1}] must stay within the {budget}-char clamp; got len={len(block)}"
        )

    # 3) THE invariant: the assembled grader prompt provably fits the window —
    #    estimated input tokens stay under max_model_len even by the deliberately
    #    conservative (high-biased) estimator.  This is what the live 400 violated.
    est = estimate_tokens(user_msg)
    assert est <= gen.max_model_len - DOC_GRADE_MAX_TOKENS, (
        f"Assembled grader prompt must fit the window: estimate={est} tokens, "
        f"ceiling={gen.max_model_len - DOC_GRADE_MAX_TOKENS}"
    )

    # 4) The kept candidates are the highest-ranked prefix, and the graded subset
    #    is drawn only from them (so generation never sees the dropped tail).
    kept = result["graded"]
    assert kept == chunks[: len(kept)], "Kept candidates must be the ranked prefix"
    assert result["doc_verdict"] == "correct"  # every kept candidate graded relevant


@pytest.mark.asyncio
async def test_grade_documents_clamp_passes_through_within_budget_candidate() -> None:
    """AC3 — no regression: a candidate within the budget is passed through
    verbatim (the original content survives in the prompt, no truncation).
    """
    budget = 4_000
    small = RetrievedChunk(
        chunk_id="c1",
        document_id="doc-1",
        content="leave policy: 20 days per year.",
        score=0.9,
        source_uri="corpus://hr/leave.md",
    )
    llm = _RecordingLlm(verdicts_per_call=[[True]])
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=budget)
    state = _base_state(chunks=[small])

    await node(state)

    user_msg = _user_message(llm)
    assert small.content in user_msg, (
        f"Within-budget candidate must be passed through verbatim; "
        f"content={small.content!r} not in user_msg={user_msg!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_clamp_preserves_question_in_prompt() -> None:
    """The clamp must not interfere with the question that anchors the prompt.

    The user's question must still be present (it identifies what the grader
    is judging relevance *to*).  This is a defensive check: the clamp only
    touches candidate content, not the question.
    """
    budget = 100
    question = "What is the target group for the SMILE5 installation manual?"
    oversized = _oversized_chunk(marker="Y", size=10_000, chunk_id="b1")

    llm = _RecordingLlm(verdicts_per_call=[[True]])
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=budget)
    state = _base_state(chunks=[oversized], question=question)

    await node(state)

    user_msg = _user_message(llm)
    assert question in user_msg, (
        f"Question must remain in the prompt (clamp must not touch it); "
        f"question={question!r} not in user_msg={user_msg!r}"
    )


@pytest.mark.asyncio
async def test_grade_documents_clamp_handles_per_candidate_verdicts_for_mix() -> None:
    """Mixed relevance — the grader must still parse per-candidate verdicts
    correctly after the clamp is applied.  The clamp only changes what the
    *model sees*, not the model's *response*; the aggregation logic
    (relevant/irrelevant → correct/ambiguous/incorrect) must be unaffected.
    """
    budget = 300
    chunks = [
        _oversized_chunk(marker="A", size=5_000, chunk_id="big-1"),
        _CHUNK_SMALL,
        _oversized_chunk(marker="C", size=5_000, chunk_id="big-3"),
    ]
    # Per-candidate verdicts: [True, False, True] → 2/3 relevant → 'ambiguous'
    verdicts = [[True, False, True]]
    llm = _RecordingLlm(verdicts_per_call=verdicts)
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=budget)
    state = _base_state(chunks=chunks)

    result = await node(state)

    # Per-candidate verdict aggregation is intact despite the clamp
    assert result["doc_verdict"] == "ambiguous", (
        f"Mixed relevance → 'ambiguous'; got {result['doc_verdict']!r}"
    )
    assert result["graded"] == [chunks[0], chunks[2]], (
        f"Only the relevant chunks (indices 0 and 2) should remain in 'graded'; "
        f"got {result['graded']!r}"
    )

    # And the clamp still bounded each oversized candidate
    user_msg = _user_message(llm)
    blocks = _extract_candidate_blocks(user_msg)
    assert len(blocks) == 3
    for i, block in enumerate(blocks):
        # The small chunk (index 1) is within budget and unchanged
        if i == 1:
            assert _CHUNK_SMALL.content in block, (
                f"Small chunk must be passed through; got block={block!r}"
            )
        else:
            # Oversized chunks clamped to exactly ``budget`` chars
            marker = "A" if i == 0 else "C"
            assert block == marker * budget, (
                f"Candidate [{i + 1}] must be clamped to exactly {budget} '{marker}'s; "
                f"got len={len(block)}"
            )


@pytest.mark.asyncio
async def test_grade_documents_clamp_with_zero_budget_yields_empty_candidate() -> None:
    """``parent_expansion_max_chars <= 0`` collapses each candidate to the empty
    string.  This is the pathological case the helper handles: the spine
    should have already filtered such a candidate, but the clamp does not
    crash — it returns ``""`` for every candidate, the prompt is still well
    formed (numbered), and the LLM is still called once.  The grader's
    fail-open-to-relevant default then keeps the chunk; downstream nodes
    will see an empty-text chunk and surface the "no answer" path.  This is
    documented behaviour, not a silent bug.
    """
    chunks = [_CHUNK_SMALL]
    llm = _RecordingLlm(verdicts_per_call=[[True]])
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=0)
    state = _base_state(chunks=chunks)

    result = await node(state)

    user_msg = _user_message(llm)
    blocks = _extract_candidate_blocks(user_msg)
    assert len(blocks) == 1
    assert blocks[0] == "", f"Zero budget must collapse content to ''; got {blocks[0]!r}"
    # The numbered prefix "[1] " must still be present so the prompt is well formed
    assert "[1] " in user_msg, f"Numbered prefix must remain; user_msg={user_msg!r}"
    # LLM is still called once (batched)
    assert len(llm.records) == 1
    # Grader still returns a verdict; fail-open default marks the chunk relevant
    assert result["doc_verdict"] == "correct"


@pytest.mark.asyncio
async def test_grade_documents_clamp_does_not_mutate_chunk_in_state() -> None:
    """The clamp is a prompt-time concern — the candidate ``RetrievedChunk``
    objects passed in via ``state["chunks"]`` are NOT mutated.  The
    ``graded`` field in the result contains the *original* chunk objects
    (with the original content), not truncated copies.
    """
    budget = 200
    original = _oversized_chunk(marker="Z", size=10_000, chunk_id="orig")
    original_content_id = id(original.content)

    llm = _RecordingLlm(verdicts_per_call=[[True]])
    from iw_rag.agent.nodes import make_grade_documents

    node = make_grade_documents(llm, AgentSettings(), parent_expansion_max_chars=budget)
    state = _base_state(chunks=[original])

    result = await node(state)

    # The original chunk's content object is untouched
    assert id(original.content) == original_content_id
    assert len(original.content) == 10_000
    # The graded list contains the same chunk (referential equality)
    assert result["graded"] == [original]
    assert result["graded"][0] is original
    assert len(result["graded"][0].content) == 10_000


# ---------------------------------------------------------------------------
# LlmClient protocol smoke check
# ---------------------------------------------------------------------------


def test_recording_llm_satisfies_llm_client_protocol() -> None:
    """The fake is duck-typed to the :class:`LlmClient` protocol so it can be
    passed to ``make_grade_documents`` (which does not have a type check
    that would reject it).  This is a structural sanity check — the
    downstream assertions above depend on it.
    """
    llm = _RecordingLlm()
    # Protocol is runtime-checkable; explicit isinstance is the canonical check
    assert isinstance(llm, LlmClient)
    # All four protocol methods must be present and callable.
    required_methods = ("complete", "complete_tools", "complete_structured", "health")
    missing = [m for m in required_methods if not callable(getattr(llm, m, None))]
    assert missing == [], f"_RecordingLlm is missing LlmClient methods: {missing}"
