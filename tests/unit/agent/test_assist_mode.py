"""Unit tests for assist mode (``allow_ungrounded``).

Assist mode relaxes the *groundedness* gate so the system can answer from general
knowledge (clearly marked) instead of fail-closing — while keeping every other
safeguard (fabricated-citation block, PII redaction, input/injection/safety
guards) fully enforced. These tests pin the four coordinated pieces:

1. ``route_after_grade_generation`` — budget-exhausted + ungrounded routes to
   ``output_guard`` in assist mode (vs ``fail_closed`` in strict mode).
2. ``output_guard_node`` — groundedness is informational in assist mode; a
   *fabricated* citation still hard-blocks.
3. ``AnswerAssembler.generate(allow_ungrounded=True)`` — general-knowledge
   fallback when evidence is thin.
4. ``done`` — emits an ungrounded answer the guard allowed in assist mode.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import patch

import pytest

from iw_rag.agent.edges import route_after_grade_generation
from iw_rag.agent.nodes import done, output_guard_node
from iw_rag.agent.state import GraphState
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, RetrievalResult, RetrievedChunk
from iw_rag.generation.answer import AnswerAssembler, strip_reasoning
from iw_rag.generation.llm import GenerationParams, LlmResponse, Message
from iw_rag.generation.prompts import GENERAL_KNOWLEDGE_DISCLAIMER, REFUSAL_TEXT, CitationPrompt

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-assist",
)

_CHUNK = RetrievedChunk(
    chunk_id="chunk-1",
    document_id="doc-1",
    content="The total is $100.",
    score=0.9,
    source_uri="corpus://finance/x",
)
_VALID_CITATION = Citation(
    marker=1, chunk_id="chunk-1", document_id="doc-1", source_uri="corpus://finance/x"
)
_FABRICATED_CITATION = Citation(
    marker=2, chunk_id="NOPE", document_id="doc-1", source_uri="corpus://nope"
)


# ---------------------------------------------------------------------------
# 1. Edge routing
# ---------------------------------------------------------------------------


def _gen_state(*, allow_ungrounded: bool, retries: int = 2, max_retries: int = 2) -> GraphState:
    s: GraphState = {
        "question": "q?",
        "original_question": "q?",
        "access_context": _CTX,
        "retries": retries,
        "max_retries": max_retries,
        "gen_verdict": "unsupported",
        "allow_ungrounded": allow_ungrounded,
        "verdicts": [],
    }
    return s


def test_budget_exhausted_ungrounded_routes_to_fail_closed_in_strict_mode() -> None:
    """Strict mode keeps the no-ungrounded-answer invariant: → fail_closed."""
    assert route_after_grade_generation(_gen_state(allow_ungrounded=False)) == "fail_closed"


def test_budget_exhausted_ungrounded_routes_to_output_guard_in_assist_mode() -> None:
    """Assist mode emits the (marked) ungrounded answer via output_guard → done."""
    assert route_after_grade_generation(_gen_state(allow_ungrounded=True)) == "output_guard"


def test_assist_mode_is_single_pass_emits_via_output_guard_without_retrying() -> None:
    """Assist mode is single-pass for responsiveness: ungrounded → output_guard,
    even with budget remaining (no corrective retry loop)."""
    state = _gen_state(allow_ungrounded=True, retries=0, max_retries=2)
    assert route_after_grade_generation(state) == "output_guard"


def test_strict_mode_retries_first_when_budget_remains() -> None:
    """Strict mode still spends its correction budget before fail-closing."""
    state = _gen_state(allow_ungrounded=False, retries=0, max_retries=2)
    assert route_after_grade_generation(state) == "consume_retry"


# ---------------------------------------------------------------------------
# 2. output_guard_node — groundedness informational, fabricated citation blocks
# ---------------------------------------------------------------------------

_GUARDS_ON = types.SimpleNamespace(guardrails=types.SimpleNamespace(enabled=True))


async def _identity_redact(text: str, _ctx: Any) -> str:
    return text


def _guard_state(*, allow_ungrounded: bool, citations: list[Citation]) -> GraphState:
    s: GraphState = {
        "question": "q?",
        "original_question": "q?",
        "access_context": _CTX,
        "retries": 0,
        "draft": "An answer.",
        "citations": citations,
        "graded": [_CHUNK],
        "gen_verdict": "unsupported",  # ungrounded
        "allow_ungrounded": allow_ungrounded,
        "verdicts": [],
    }
    return s


@pytest.mark.asyncio
@patch("iw_rag.agent.nodes._presidio_redact", _identity_redact)
@patch("iw_rag.agent.nodes.get_settings", lambda: _GUARDS_ON)
async def test_output_guard_allows_ungrounded_in_assist_mode() -> None:
    """Assist mode: an ungrounded, uncited answer is ALLOWED (groundedness informational)."""
    result = await output_guard_node(_guard_state(allow_ungrounded=True, citations=[]))
    assert result["output_guard_decision"] == "allow"
    assert result["diagnostics"]["output_guard_assist_mode"] is True
    assert result["diagnostics"]["output_guard_grounded"] is False


@pytest.mark.asyncio
@patch("iw_rag.agent.nodes._presidio_redact", _identity_redact)
@patch("iw_rag.agent.nodes.get_settings", lambda: _GUARDS_ON)
async def test_output_guard_blocks_fabricated_citation_even_in_assist_mode() -> None:
    """Anti-hallucination invariant holds in every mode: a fabricated citation blocks."""
    state = _guard_state(allow_ungrounded=True, citations=[_FABRICATED_CITATION])
    result = await output_guard_node(state)
    assert result["output_guard_decision"] == "block"


@pytest.mark.asyncio
@patch("iw_rag.agent.nodes._presidio_redact", _identity_redact)
@patch("iw_rag.agent.nodes.get_settings", lambda: _GUARDS_ON)
async def test_output_guard_blocks_ungrounded_in_strict_mode() -> None:
    """Strict mode (default): ungrounded → block (fail-closed)."""
    result = await output_guard_node(_guard_state(allow_ungrounded=False, citations=[]))
    assert result["output_guard_decision"] == "block"


# ---------------------------------------------------------------------------
# 3. AnswerAssembler general-knowledge fallback
# ---------------------------------------------------------------------------


class _FakeLlm:
    def __init__(self, text: str) -> None:
        self._text = text

    async def complete(self, messages: list[Message], params: GenerationParams) -> LlmResponse:
        return LlmResponse(text=self._text, usage={"total_tokens": 5})

    async def complete_tools(self, *a: Any, **k: Any) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")

    async def complete_structured(self, *a: Any, **k: Any) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")


def _assembler(text: str) -> AnswerAssembler:
    return AnswerAssembler(_FakeLlm(text), CitationPrompt(), GenerationParams(model="m"))


@pytest.mark.asyncio
async def test_no_chunks_strict_mode_refuses_without_llm_call() -> None:
    """Strict mode + no evidence → fail-closed refusal, no general-knowledge answer."""
    empty = RetrievalResult(chunks=[], retriever="spine", acl_applied=True)
    out = await _assembler("Paris is the capital of France.").generate("q", empty)
    assert out.grounded is False
    assert out.answer == REFUSAL_TEXT


@pytest.mark.asyncio
async def test_no_chunks_assist_mode_answers_from_general_knowledge() -> None:
    """Assist mode + no evidence → marked general-knowledge answer."""
    empty = RetrievalResult(chunks=[], retriever="spine", acl_applied=True)
    out = await _assembler("Paris is the capital of France.").generate(
        "What is the capital of France?", empty, allow_ungrounded=True
    )
    assert out.grounded is False
    assert out.diagnostics.get("general_knowledge") is True
    assert GENERAL_KNOWLEDGE_DISCLAIMER in out.answer
    assert "Paris" in out.answer
    assert out.citations == []


@pytest.mark.asyncio
async def test_grounded_refusal_falls_back_to_general_knowledge_in_assist_mode() -> None:
    """When the documents exist but can't ground, assist mode still answers (marked)."""
    result = RetrievalResult(chunks=[_CHUNK], retriever="spine", acl_applied=True)
    # The grounded attempt returns the refusal phrase → fall back to general knowledge.
    out = await _assembler(REFUSAL_TEXT).generate("q", result, allow_ungrounded=True)
    assert out.grounded is False
    assert out.diagnostics.get("general_knowledge") is True
    assert out.diagnostics.get("reason") == "grounded_refusal"


# ---------------------------------------------------------------------------
# 3b. Reasoning-block stripping (no <think> leak, incl. unterminated blocks)
# ---------------------------------------------------------------------------


def test_strip_reasoning_removes_closed_think_block() -> None:
    assert strip_reasoning("<think>pondering</think>Hello!") == "Hello!"


def test_strip_reasoning_removes_unterminated_think_block() -> None:
    """An overthought, token-truncated <think> with no closing tag must not leak."""
    leaked = "<think>\nOkay, the user said hi. I should respond. Let me ma"
    assert strip_reasoning(leaked) == ""


@pytest.mark.asyncio
async def test_empty_generation_falls_back_to_general_knowledge_in_assist_mode() -> None:
    """A generation that strips to empty (pure reasoning) → general-knowledge in assist."""
    result = RetrievalResult(chunks=[_CHUNK], retriever="spine", acl_applied=True)
    llm = _FakeLlm("<think>thinking forever with no close tag and no answer")
    out = await AnswerAssembler(llm, CitationPrompt(), GenerationParams(model="m")).generate(
        "q", result, allow_ungrounded=True
    )
    assert out.diagnostics.get("general_knowledge") is True
    assert out.diagnostics.get("reason") == "empty_generation"


# ---------------------------------------------------------------------------
# 4. done finaliser emits the assist-mode answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_done_emits_ungrounded_answer_when_guard_allowed_in_assist_mode() -> None:
    """done promotes an ungrounded draft to answer when assist guard allowed it."""
    state: GraphState = {
        "gen_verdict": "unsupported",
        "draft": "General-knowledge answer.",
        "retries": 0,
        "allow_ungrounded": True,
        "output_guard_decision": "allow",
    }
    out = await done(state)
    assert out["disposition"] == "answered"
    assert out["answer"] == "General-knowledge answer."
    assert out["diagnostics"]["answer_ungrounded"] is True


@pytest.mark.asyncio
async def test_done_refuses_ungrounded_answer_in_strict_mode() -> None:
    """Strict mode: an ungrounded draft is refused, never emitted."""
    state: GraphState = {
        "gen_verdict": "unsupported",
        "draft": "Should not be emitted.",
        "retries": 0,
        "allow_ungrounded": False,
        "output_guard_decision": "block",
    }
    out = await done(state)
    assert out["disposition"] == "refused"
    assert out["answer"] is None
