"""Unit tests for eval/judges.py — TDD RED-first.

Tests cover: JudgeVerdict/GradeVerdict shape, FaithfulnessJudge (Lynx-8B),
GradingJudge (Selene-Mini-8B) with position-swap bias control, JudgeUnavailable
fail-fast, and the different-family contamination guard.  All use faked judge
clients — no live models, no ragas install required.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.errors import ConfigError, JudgeUnavailable
from iw_rag.eval.judges import (
    FaithfulnessJudge,
    GradeSample,
    GradeVerdict,
    GradingJudge,
    JudgeVerdict,
    Rubric,
)
from iw_rag.generation.llm import GenerationParams, LlmResponse

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _faithful_response() -> LlmResponse:
    """A fake LLM response containing a FAITHFUL verdict."""
    return LlmResponse(
        text='{"verdict": "FAITHFUL", "score": 1.0, "reason": "Every claim is supported."}',
        usage={},
    )


def _unfaithful_response() -> LlmResponse:
    """A fake LLM response containing an UNFAITHFUL verdict."""
    return LlmResponse(
        text='{"verdict": "UNFAITHFUL", "score": 0.0, "reason": "Claim not in context."}',
        usage={},
    )


def _grade_response(score: float = 4.0) -> LlmResponse:
    """A fake LLM response for a rubric grade."""
    return LlmResponse(
        text=f'{{"score": {score}, "reason": "Meets criteria."}}',
        usage={},
    )


def _make_fake_client(responses: list[LlmResponse]) -> Any:
    """Build an async LLM client that replays a list of responses in order."""
    call_count = 0
    responses_copy = list(responses)

    async def _complete(messages: Sequence[Any], params: GenerationParams) -> LlmResponse:
        nonlocal call_count
        resp = responses_copy[call_count % len(responses_copy)]
        call_count += 1
        return resp

    client = MagicMock()
    client.complete = AsyncMock(side_effect=_complete)
    return client, lambda: call_count


def _make_erroring_client() -> Any:
    """Build a client that always raises an httpx-like connection error."""
    from iw_rag.domain.errors import GenerationError

    client = MagicMock()
    client.complete = AsyncMock(side_effect=GenerationError("Connection refused"))
    return client


# ---------------------------------------------------------------------------
# 1. JudgeVerdict shape
# ---------------------------------------------------------------------------


def test_judge_verdict_is_dataclass() -> None:
    """JudgeVerdict has passed, score, and reason fields."""
    v = JudgeVerdict(passed=True, score=1.0, reason="ok")
    assert v.passed is True
    assert v.score == 1.0
    assert v.reason == "ok"


def test_grade_verdict_is_dataclass() -> None:
    """GradeVerdict has score, reason, and stable fields."""
    v = GradeVerdict(score=3.5, reason="Good", stable=True)
    assert v.score == pytest.approx(3.5)
    assert v.stable is True


# ---------------------------------------------------------------------------
# 2. FaithfulnessJudge — Lynx-8B (faithful path)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_faithful_answer_passes() -> None:
    """FaithfulnessJudge returns passed=True when the LLM says FAITHFUL."""
    client, _ = _make_fake_client([_faithful_response()])
    judge = FaithfulnessJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    verdict = await judge.judge(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        answer="The capital of France is Paris.",
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed is True
    assert verdict.score == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_unfaithful_answer_fails() -> None:
    """FaithfulnessJudge returns passed=False when the LLM says UNFAITHFUL."""
    client, _ = _make_fake_client([_unfaithful_response()])
    judge = FaithfulnessJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    verdict = await judge.judge(
        question="What is the revenue?",
        context="Revenue was $10M in Q3.",
        answer="Revenue was $50M in Q3.",
    )
    assert verdict.passed is False
    assert verdict.score == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_faithfulness_judge_verdict_has_reason() -> None:
    """FaithfulnessJudge verdict always carries a non-empty reason."""
    client, _ = _make_fake_client([_faithful_response()])
    judge = FaithfulnessJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    verdict = await judge.judge(question="q", context="ctx", answer="ans")
    assert verdict.reason != ""


# ---------------------------------------------------------------------------
# 3. FaithfulnessJudge — JudgeUnavailable fail-fast
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_unavailable_raised_on_endpoint_error() -> None:
    """FaithfulnessJudge raises JudgeUnavailable when the client errors — no fallback."""
    client = _make_erroring_client()
    judge = FaithfulnessJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    with pytest.raises(JudgeUnavailable, match="Faithfulness judge endpoint unreachable"):
        await judge.judge(question="q", context="ctx", answer="ans")


@pytest.mark.asyncio
async def test_grading_judge_unavailable_on_endpoint_error() -> None:
    """GradingJudge raises JudgeUnavailable when the client errors — no fallback."""
    client = _make_erroring_client()
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="Is the answer relevant?", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    with pytest.raises(JudgeUnavailable, match="Grading judge endpoint unreachable"):
        await judge.grade(rubric=rubric, sample=sample)


# ---------------------------------------------------------------------------
# 4. Different-family contamination guard
# ---------------------------------------------------------------------------


def test_different_family_guard_raises_when_judge_equals_generator() -> None:
    """FaithfulnessJudge raises ConfigError when judge == generator (self-preference)."""
    with pytest.raises(ConfigError, match="Contamination"):
        FaithfulnessJudge(
            judge_client=MagicMock(),
            generator_model="Qwen/Qwen3-14B-AWQ",
            judge_model="Qwen/Qwen3-14B-AWQ",  # same as generator!
        )


def test_grading_judge_different_family_guard_raises() -> None:
    """GradingJudge raises ConfigError when judge == generator."""
    with pytest.raises(ConfigError, match="Contamination"):
        GradingJudge(
            judge_client=MagicMock(),
            generator_model="Qwen/Qwen3-14B-AWQ",
            judge_model="Qwen/Qwen3-14B-AWQ",  # same family!
        )


def test_different_family_passes_with_different_model() -> None:
    """FaithfulnessJudge construction succeeds with genuinely different judge."""
    # Should not raise
    judge = FaithfulnessJudge(
        judge_client=MagicMock(),
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="PatronusAI/Lynx-8B-Instruct",
    )
    # Construction succeeded (no contamination guard fired) and the distinct
    # judge model is retained.
    assert judge._judge_model == "PatronusAI/Lynx-8B-Instruct"


# ---------------------------------------------------------------------------
# 5. GradingJudge — position swap
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_position_swap_calls_client_twice() -> None:
    """GradingJudge with position_swap=True calls the client exactly twice."""
    client, get_count = _make_fake_client([_grade_response(4.0), _grade_response(4.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="Is the answer complete?", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    verdict = await judge.grade(rubric=rubric, sample=sample, position_swap=True)
    assert get_count() == 2
    assert isinstance(verdict, GradeVerdict)


@pytest.mark.asyncio
async def test_position_swap_default_is_true() -> None:
    """GradingJudge.grade default calls client twice (position_swap default=True)."""
    client, get_count = _make_fake_client([_grade_response(3.0), _grade_response(3.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="Rubric", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    # Call with NO position_swap arg — default should be True → 2 calls
    await judge.grade(rubric=rubric, sample=sample)
    assert get_count() == 2


@pytest.mark.asyncio
async def test_position_swap_false_calls_once() -> None:
    """GradingJudge with position_swap=False calls client exactly once."""
    client, get_count = _make_fake_client([_grade_response(3.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="Rubric", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    await judge.grade(rubric=rubric, sample=sample, position_swap=False)
    assert get_count() == 1


@pytest.mark.asyncio
async def test_position_swap_grade_is_average() -> None:
    """GradingJudge position-swap merges both scores by averaging them."""
    # Two calls return different scores; expect average
    client, _ = _make_fake_client([_grade_response(4.0), _grade_response(2.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="R", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    verdict = await judge.grade(rubric=rubric, sample=sample, position_swap=True)
    assert verdict.score == pytest.approx(3.0)


@pytest.mark.asyncio
async def test_position_swap_flags_unstable_grade() -> None:
    """GradingJudge marks grade as unstable when scores diverge beyond tolerance."""
    # Scores 0.0 and 5.0 — maximum divergence; should flag stable=False
    client, _ = _make_fake_client([_grade_response(0.0), _grade_response(5.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="R", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    verdict = await judge.grade(rubric=rubric, sample=sample, position_swap=True)
    assert verdict.stable is False


@pytest.mark.asyncio
async def test_position_swap_stable_when_scores_agree() -> None:
    """GradingJudge marks grade as stable when both swap scores agree closely."""
    client, _ = _make_fake_client([_grade_response(4.0), _grade_response(4.0)])
    judge = GradingJudge(
        judge_client=client,
        generator_model="Qwen/Qwen3-14B-AWQ",
        judge_model="AtlaAI/Selene-Mini-8B-Instruct",
    )
    rubric = Rubric(criteria="R", max_score=5.0)
    sample = GradeSample(question="q", answer="ans", context="ctx")
    verdict = await judge.grade(rubric=rubric, sample=sample, position_swap=True)
    assert verdict.stable is True


# ---------------------------------------------------------------------------
# _parse_faithfulness_verdict — tolerant of real judge output (live-validated)
# ---------------------------------------------------------------------------


def test_parse_faithfulness_verdict_near_json_unquoted_values() -> None:
    """Lynx-8B emits near-JSON with UNQUOTED values; the parser must still read it.

    Live validation showed Lynx returns e.g.
    {"verdict": FAITHFUL, "score": 1.0, "reason": <prose>} — invalid JSON — so the
    tolerant fallback extracts the verdict token + score.
    """
    from iw_rag.eval.judges import _parse_faithfulness_verdict

    v_ok = _parse_faithfulness_verdict(
        '{"verdict": FAITHFUL, "score": 1.0, "reason": fully supported by the context}'
    )
    assert v_ok.passed is True
    assert v_ok.score == 1.0

    v_bad = _parse_faithfulness_verdict(
        '{"verdict": UNFAITHFUL, "score": 0.0, "reason": the claim is not in the context}'
    )
    assert v_bad.passed is False
    assert v_bad.score == 0.0


def test_parse_faithfulness_verdict_strict_json_still_works() -> None:
    """Well-formed JSON (other judges) keeps working."""
    from iw_rag.eval.judges import _parse_faithfulness_verdict

    v = _parse_faithfulness_verdict('{"verdict": "FAITHFUL", "score": 1.0, "reason": "ok"}')
    assert v.passed is True and v.score == 1.0
