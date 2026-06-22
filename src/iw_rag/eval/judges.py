"""Offline judge layer — FaithfulnessJudge (Lynx-8B) + GradingJudge (Selene-Mini-8B).

OFFLINE / batch only.  These judges MUST NOT be imported by any hot-path module
(api/, agent/, retrieval/, generation/).  The two-clocks rule (ADR-006,
CONVENTIONS §10, lld-eval-observability §5) places all LLM-judging in the
offline/eval path.

Judge specialisation (R-00110/F2, ADR-013):
- **FaithfulnessJudge** → Patronus Lynx-8B (Llama-3 family) — binary
  FAITHFUL/UNFAITHFUL + reasoning for groundedness checks.
- **GradingJudge** → Atla Selene-Mini-8B (Llama-3.1 family) — rubric-anchored
  pointwise grading with position-swap bias control.

Both are a **different model family** from the Qwen3-14B generator, enforced at
construction time by ``assert_generator_not_judge`` (self-preference de-bias,
CONVENTIONS §4).

The judge client is injected (``LlmClient``), so tests supply a fake.  Endpoints
come from ``config/models.yaml`` via ``Settings`` — never hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from iw_rag.domain.errors import JudgeUnavailable
from iw_rag.eval.golden_set import assert_generator_not_judge
from iw_rag.generation.llm import GenerationParams, LlmClient

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class JudgeVerdict:
    """The output of a :class:`FaithfulnessJudge` call.

    Attributes:
        passed: ``True`` for FAITHFUL, ``False`` for UNFAITHFUL.
        score: Numeric confidence in [0.0, 1.0].
        reason: Judge's textual reasoning (non-empty).
    """

    passed: bool
    score: float
    reason: str


@dataclass(frozen=True, slots=True)
class GradeVerdict:
    """The output of a :class:`GradingJudge` call.

    Attributes:
        score: Rubric score (e.g. 0–5).
        reason: Judge's textual reasoning.
        stable: ``True`` if both position-swap calls agreed within tolerance;
            ``False`` if the score flipped substantially on option order alone
            (signals unreliable grading).
    """

    score: float
    reason: str
    stable: bool = True


@dataclass(frozen=True, slots=True)
class Rubric:
    """A rubric that anchors a :class:`GradingJudge` evaluation.

    Attributes:
        criteria: Human-readable grading criteria.
        max_score: Maximum achievable score.
    """

    criteria: str
    max_score: float = 5.0


@dataclass(frozen=True, slots=True)
class GradeSample:
    """The input to :class:`GradingJudge.grade`.

    Attributes:
        question: The question asked.
        answer: The answer to grade.
        context: The retrieved context available to the answerer.
    """

    question: str
    answer: str
    context: str


# ---------------------------------------------------------------------------
# Prompt templates (rubric-anchored, pointwise — bias control §5)
# ---------------------------------------------------------------------------

_FAITHFULNESS_SYSTEM = (
    "You are an expert judge evaluating whether an answer is grounded in the "
    "provided context. Respond ONLY with a JSON object matching this schema: "
    '{"verdict": "FAITHFUL" | "UNFAITHFUL", "score": 0.0 | 1.0, "reason": "<string>"}. '
    "FAITHFUL means every factual claim in the answer is directly supported by "
    "the context. UNFAITHFUL means any claim is unsupported or contradicted."
)

_FAITHFULNESS_USER = (
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluate the answer for faithfulness to the context."
)

_GRADING_SYSTEM = (
    "You are an expert evaluator. Apply the rubric strictly and respond ONLY with "
    'a JSON object: {{"score": <float>, "reason": "<string>"}}. '
    "Score on the given scale (0 to {max_score}), anchored to the rubric criteria. "
    "Do not reward verbosity; penalise unsupported claims."
)

_GRADING_USER = (
    "Rubric:\n{criteria}\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:\n{answer}\n\n"
    "Grade the answer according to the rubric."
)

# Tolerance for position-swap stability: |score1 - score2| / max_score
_SWAP_INSTABILITY_THRESHOLD = 0.4  # 40% of max_score divergence → unstable


# ---------------------------------------------------------------------------
# FaithfulnessJudge — Patronus Lynx-8B
# ---------------------------------------------------------------------------


class FaithfulnessJudge:
    """Faithfulness/groundedness judge using Patronus Lynx-8B (Llama-3 family).

    Performs pointwise, rubric-anchored evaluation: given (question, context,
    answer) returns FAITHFUL/UNFAITHFUL + reasoning.  The judge MUST be a
    different model family from the generator (self-preference de-bias,
    CONVENTIONS §4).

    The client is injected so tests can supply a fake without any network.
    Endpoints come from ``config/models.yaml`` — never hardcoded.

    Attributes:
        _client: The injected LLM client pointing at the offline judge endpoint.
        _judge_model: The served model name for the judge endpoint.
        _params: Default generation parameters for judge calls.
    """

    def __init__(
        self,
        judge_client: LlmClient,
        generator_model: str,
        judge_model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the FaithfulnessJudge and enforce family separation.

        Args:
            judge_client: An ``LlmClient`` pointing at the offline judge endpoint
                (``VLLM_JUDGE_URL``), distinct from the hot-path generator.
            generator_model: The served generator model name (for contamination guard).
            judge_model: The served judge model name (must differ from generator).
            temperature: Sampling temperature (0.0 for deterministic grading).
            max_tokens: Max tokens for the judge's response.

        Raises:
            ConfigError: If ``judge_model == generator_model`` (contamination).
        """
        assert_generator_not_judge(generator_model, [judge_model])
        self._client = judge_client
        self._judge_model = judge_model
        self._params = GenerationParams(
            model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def judge(self, question: str, context: str, answer: str) -> JudgeVerdict:
        """Evaluate whether an answer is faithful to the retrieved context.

        Sends (question, context, answer) to the Lynx-8B judge and parses the
        structured JSON verdict.  Pointwise — one call per (question, context,
        answer) triple.

        Args:
            question: The original query.
            context: The retrieved context used to ground the answer.
            answer: The generated answer to evaluate.

        Returns:
            A :class:`JudgeVerdict` with ``passed``, ``score``, and ``reason``.

        Raises:
            JudgeUnavailable: If the judge endpoint is unreachable or returns an
                unusable response.  The run MUST fail fast — never fall back to
                the generator (invariant: generator ≠ judge).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": _FAITHFULNESS_SYSTEM},
            {
                "role": "user",
                "content": _FAITHFULNESS_USER.format(
                    question=question, context=context, answer=answer
                ),
            },
        ]
        try:
            resp = await self._client.complete(messages, self._params)
        except Exception as exc:
            raise JudgeUnavailable(
                f"Faithfulness judge endpoint unreachable or failed: {exc}"
            ) from exc

        return _parse_faithfulness_verdict(resp.text)


def _parse_faithfulness_verdict(text: str) -> JudgeVerdict:
    """Parse the structured JSON verdict from a Lynx-8B response.

    Args:
        text: Raw response text from the judge (expected JSON).

    Returns:
        A :class:`JudgeVerdict`.

    Raises:
        JudgeUnavailable: If the text cannot be parsed as a valid verdict.
    """
    import re

    # Strip any surrounding markdown code fences
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    # 1) Strict JSON (well-behaved models / valid output).
    data: dict[str, object] | None = None
    try:
        parsed = json.loads(cleaned)
        data = parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                data = parsed if isinstance(parsed, dict) else None
            except (json.JSONDecodeError, ValueError):
                data = None

    if data is not None and "verdict" in data:
        verdict_str = str(data.get("verdict", "")).upper()
        passed = "UNFAITHFUL" not in verdict_str and "FAITHFUL" in verdict_str
        score = float(data.get("score", 1.0 if passed else 0.0))  # type: ignore[arg-type]
        reason = str(data.get("reason", "")) or verdict_str
        return JudgeVerdict(passed=passed, score=score, reason=reason)

    # 2) Tolerant fallback: real judges (e.g. Lynx) often emit *near*-JSON with
    # UNQUOTED values — `{"verdict": FAITHFUL, "score": 1.0, "reason": ...}` —
    # which json.loads rejects. Extract the verdict token + score directly.
    up = cleaned.upper()
    if "UNFAITHFUL" in up:
        passed = False
    elif "FAITHFUL" in up:
        passed = True
    else:
        raise JudgeUnavailable(f"Could not parse judge verdict (no FAITHFUL/UNFAITHFUL): {text!r}")
    score_m = re.search(r"score\W{0,3}([01](?:\.\d+)?)", cleaned, re.IGNORECASE)
    score = float(score_m.group(1)) if score_m else (1.0 if passed else 0.0)
    reason_m = re.search(r"reason\W{0,3}(.+)", cleaned, re.IGNORECASE | re.DOTALL)
    reason = (
        reason_m.group(1).strip().rstrip("}").strip().strip('"').strip("'")[:500]
        if reason_m
        else ("FAITHFUL" if passed else "UNFAITHFUL")
    )
    return JudgeVerdict(passed=passed, score=score, reason=reason)


# ---------------------------------------------------------------------------
# GradingJudge — Atla Selene-Mini-8B
# ---------------------------------------------------------------------------


class GradingJudge:
    """General rubric grader using Atla Selene-Mini-8B (Llama-3.1 family).

    Performs rubric-anchored **pointwise** grading.  When ``position_swap=True``
    (the default), the judge is called **twice** with the context/answer in
    swapped positions and the scores are averaged.  A large divergence between
    the two calls is flagged as ``stable=False`` (unreliable grade).

    This mitigates position bias (lld-eval-observability §5 bias table).

    Attributes:
        _client: Injected LLM client for the judge endpoint.
        _judge_model: Served model name for generation params.
        _params: Default generation parameters.
    """

    def __init__(
        self,
        judge_client: LlmClient,
        generator_model: str,
        judge_model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        """Initialise the GradingJudge and enforce family separation.

        Args:
            judge_client: An ``LlmClient`` pointing at the offline judge endpoint.
            generator_model: Served generator model name (contamination guard).
            judge_model: Served judge model name (must differ from generator).
            temperature: Sampling temperature.
            max_tokens: Max tokens in the judge's response.

        Raises:
            ConfigError: If ``judge_model == generator_model``.
        """
        assert_generator_not_judge(generator_model, [judge_model])
        self._client = judge_client
        self._judge_model = judge_model
        self._params = GenerationParams(
            model=judge_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def grade(
        self,
        rubric: Rubric,
        sample: GradeSample,
        *,
        position_swap: bool = True,
    ) -> GradeVerdict:
        """Grade a sample against a rubric.

        When ``position_swap=True`` (the default) the judge is invoked twice —
        once with the context/answer in the normal order and once with them
        swapped — and the results are merged by averaging.  This controls
        position bias (lld-eval-observability §5).

        Args:
            rubric: The grading rubric (criteria + max_score).
            sample: The (question, answer, context) to grade.
            position_swap: Whether to call twice with swapped option order and
                merge (default ``True``).

        Returns:
            A :class:`GradeVerdict` with averaged ``score``, merged ``reason``,
            and ``stable=False`` if the two swap calls diverged substantially.

        Raises:
            JudgeUnavailable: If the judge endpoint fails.
        """
        score1, reason1 = await self._single_grade(rubric, sample, swapped=False)

        if not position_swap:
            return GradeVerdict(score=score1, reason=reason1, stable=True)

        # Second call with context/answer order swapped (position-swap bias control)
        score2, reason2 = await self._single_grade(rubric, sample, swapped=True)

        avg_score = (score1 + score2) / 2.0
        divergence = abs(score1 - score2) / rubric.max_score
        stable = divergence < _SWAP_INSTABILITY_THRESHOLD
        reason = reason1  # primary ordering's reason is the canonical one
        return GradeVerdict(score=avg_score, reason=reason, stable=stable)

    async def _single_grade(
        self, rubric: Rubric, sample: GradeSample, *, swapped: bool
    ) -> tuple[float, str]:
        """Perform one judge call; swap context/answer ordering when requested.

        Args:
            rubric: Grading rubric.
            sample: Sample to grade.
            swapped: If ``True``, swap the context and answer fields in the prompt
                so the judge sees them in a different order (position-swap).

        Returns:
            ``(score, reason)`` tuple.

        Raises:
            JudgeUnavailable: If the judge endpoint fails.
        """
        if swapped:
            # Present answer before context (position swap)
            user_content = _GRADING_USER.format(
                criteria=rubric.criteria,
                question=sample.question,
                context=sample.answer,  # swapped
                answer=sample.context,  # swapped
                max_score=rubric.max_score,
            )
        else:
            user_content = _GRADING_USER.format(
                criteria=rubric.criteria,
                question=sample.question,
                context=sample.context,
                answer=sample.answer,
                max_score=rubric.max_score,
            )

        system_content = _GRADING_SYSTEM.format(max_score=rubric.max_score)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        try:
            resp = await self._client.complete(messages, self._params)
        except Exception as exc:
            raise JudgeUnavailable(f"Grading judge endpoint unreachable or failed: {exc}") from exc

        return _parse_grade_response(resp.text, rubric.max_score)


def _parse_grade_response(text: str, max_score: float) -> tuple[float, str]:
    """Parse a grading JSON response from the judge.

    Args:
        text: Raw response text (expected JSON with ``score`` and ``reason``).
        max_score: The rubric's max score (used as fallback).

    Returns:
        ``(score, reason)`` tuple.

    Raises:
        JudgeUnavailable: If the text cannot be parsed.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as exc:
        import re

        match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except (json.JSONDecodeError, ValueError):
                raise JudgeUnavailable(f"Could not parse grading verdict JSON: {text!r}") from exc
        else:
            raise JudgeUnavailable(f"Could not parse grading verdict JSON: {text!r}") from exc

    score = float(data.get("score", max_score / 2))
    reason = str(data.get("reason", ""))
    return score, reason
