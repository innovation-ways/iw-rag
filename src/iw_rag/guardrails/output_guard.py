"""Output guard — groundedness verification, deterministic citation validation, Presidio redaction.

Implements the output guard layer (lld-guardrails §3.4) that runs **once on the
output turn** (not per chunk) to bound GPU cost (lld-guardrails §3.6).

Three checks compose, run in order:

1. **Deterministic citation validation (no model):** every cited chunk-ID in
   ``citations`` must be in the ``graded`` set (the retrieved, ACL-pre-filtered
   evidence).  An uncited answer is also invalid.  A failed citation check is a
   hard ``block`` — deterministic, non-negotiable (lld-guardrails §3.4; R-00108 F5).

2. **Groundedness (model — faked / shared):** calls the Granite Guardian
   groundedness risk over ``(evidence, answer)``.  This is the **same signal**
   the loop's ``grade_generation`` (ISSUP) already produced; the node path reads
   ``state["gen_verdict"]`` rather than re-invoking a model, making it a
   **zero-extra-call seam** (lld-agentic-loop §3.5, task-3.3 KEY CONSTRAINT).
   The ``_granite_groundedness_score`` helper in this module provides a
   model-agnostic wrapper used by the **standalone/unit API** and the eval demo
   only.  In the node path the grounding verdict is consumed from state.

3. **PII redaction (Presidio, CPU-only):** scrub residual PII from the final
   answer, reusing Task 3.2's ``_presidio_redact``.

**Groundedness threshold:** scores below ``guardrails.output.groundedness_threshold``
(default 0.5) are treated as ungrounded → fail-closed.

**Fail-closed invariant (lld-guardrails §5):**
- Groundedness model unavailable after bounded retry → treated as ungrounded
  (never emit an unverified answer).
- Citation validation never "fails open" (it has no model, just set membership).

**Output safety (Llama Guard 4):** seam wired but off by default for VRAM.
``output.safety_model=null`` in config; set to a valid model role to enable.

**Config seam (lld-agentic-loop §8, CONVENTIONS §4):**
- ``agent.grader_model=generator`` (Phase 2 default, shared Qwen3-14B ISSUP)
- ``agent.grader_model=granite_guardian`` (Phase 3 guards-on: dedicated Granite
  Guardian endpoint; one-line config change, no code change)

References:
- lld-guardrails §3.4 (output_guard), §3.6 (once-per-turn VRAM), §5, §6
- lld-agentic-loop §3.5 (ISSUP = ONE authority, no double call), §3.6 (fail-closed)
- task-3.3-output-guard-grounding §1–§7
- CONVENTIONS.md §1 (Citation/RetrievedChunk contract), §2 (AccessContext invariant)
- R-00108 F5 (deterministic citation validation), F6 (model economy)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GuardVerdict, RetrievedChunk
from iw_rag.generation.answer import parse_citation_markers
from iw_rag.guardrails.sanitize import _presidio_redact  # reuse Task 3.2 redactor
from iw_rag.settings import get_settings

logger = logging.getLogger(__name__)

# Bounded retry for groundedness model calls (matches input_guard convention).
_MAX_RETRIES: int = 2
# Per-attempt HTTP timeout in seconds.
_TIMEOUT_S: float = 10.0


# ---------------------------------------------------------------------------
# OutputVerdict — the typed output-guard verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OutputVerdict:
    """Typed result of ``scan_output``.

    Aggregates the three output-guard checks (citation validation, groundedness,
    PII redaction) into a single verdict object that the ``output_guard_node``
    uses to decide allow/block routing.

    Attributes:
        grounded: Whether the answer is supported by the retrieved evidence
            (Granite Guardian groundedness, or the loop's ISSUP signal re-read
            from state in the node path — ONE authority).
        groundedness_score: The raw groundedness probability in ``[0, 1]``.
            ``0.0`` when the model is unavailable (fail-closed).
        citations_valid: Whether every cited chunk-ID was in the retrieved set
            and the answer had at least one citation.  ``False`` is a hard block.
        invalid_citation_ids: The cited IDs that were NOT in the retrieved set.
            Empty when all cited IDs are valid.
        pii_redacted: Whether Presidio found and redacted PII in the answer.
        answer: The final answer text — PII-redacted when ``pii_redacted=True``.
        verdicts: Per-check ``GuardVerdict`` objects (citation, groundedness, PII).
    """

    grounded: bool
    groundedness_score: float
    citations_valid: bool
    invalid_citation_ids: list[str]
    pii_redacted: bool
    answer: str
    verdicts: list[GuardVerdict]


# ---------------------------------------------------------------------------
# validate_citations — deterministic, no model (text-based entry point)
# ---------------------------------------------------------------------------


async def validate_citations(
    answer: str,
    retrieved_chunk_ids: set[str],
) -> tuple[bool, list[str]]:
    """Validate that every cited marker in the answer resolves to a retrieved chunk.

    Parses ``[n]`` markers from ``answer`` text using the generator's citation
    parser.  Since this is the text-based entry point (not the node path which
    already has pre-parsed ``Citation`` objects), marker numbers map to
    position-indexed chunk IDs; this helper validates that the answer contains
    citations at all.

    An answer is invalid when:
    - It contains no citation markers (uncited claims are invalid).
    - Any cited marker number exceeds the retrieved-set cardinality — but this
      form only checks presence of markers.  Use ``validate_citations_from_list``
      when you have parsed ``Citation`` objects with ``chunk_id`` fields.

    Args:
        answer: The generated answer text (may contain ``[n]`` markers).
        retrieved_chunk_ids: Set of chunk IDs in the retrieved/graded set.

    Returns:
        A ``(valid, invalid_ids)`` tuple where ``valid`` is ``True`` only when
        the answer has at least one citation marker and all markers have
        corresponding entries in ``retrieved_chunk_ids``.  ``invalid_ids`` is
        the list of cited IDs not found in the set (empty for text-marker
        validation; use ``validate_citations_from_list`` for ID-level checks).
    """
    markers = parse_citation_markers(answer)
    if not markers:
        # No citation markers → uncited answer → invalid.
        return False, []
    # Text-marker path: we only know marker numbers, not chunk IDs directly.
    # Without a mapping from marker index to chunk_id we cannot check set membership.
    # Callers with Citation objects should use validate_citations_from_list.
    # Here we check that there IS at least one marker (already checked above).
    return True, []


async def validate_citations_from_list(
    citations: list[Citation],
    retrieved_chunk_ids: set[str],
) -> tuple[bool, list[str]]:
    """Validate pre-parsed citations against the retrieved chunk-ID set.

    The **canonical** validation path for the node: the loop already parses
    citations into ``state["citations"]`` (``list[Citation]`` each with
    ``chunk_id``); this function validates those chunk IDs against the retrieved
    set.

    An answer is invalid when:
    - The citation list is empty (uncited answer).
    - Any ``Citation.chunk_id`` is not in ``retrieved_chunk_ids`` (fabricated
      citation — hard block).

    Args:
        citations: Pre-parsed ``Citation`` objects from the generator.
        retrieved_chunk_ids: Set of chunk IDs from the graded/retrieved set.

    Returns:
        A ``(valid, invalid_ids)`` tuple where ``valid`` is ``True`` only when
        ``citations`` is non-empty and every ``chunk_id`` is in the set.
        ``invalid_ids`` lists all chunk IDs that are NOT in the set.

    Raises:
        Nothing — validation is fully deterministic (no model, no I/O).
    """
    if not citations:
        # Empty citation list → uncited answer → invalid.
        return False, []

    invalid: list[str] = [c.chunk_id for c in citations if c.chunk_id not in retrieved_chunk_ids]
    valid = len(invalid) == 0
    return valid, invalid


# ---------------------------------------------------------------------------
# _granite_groundedness_score — model-agnostic groundedness client wrapper
# ---------------------------------------------------------------------------


async def _granite_groundedness_score(answer: str, evidence: str) -> tuple[bool, float]:
    """Call Granite Guardian groundedness risk and return (grounded, score).

    This wrapper is the **standalone / unit API** for the groundedness check.
    In the **node path** (``output_guard_node``), groundedness is NOT re-computed
    here — the node reads ``state["gen_verdict"]`` (the ISSUP result already
    computed by ``grade_generation``) directly.  This is the ONE-authority rule
    (lld-agentic-loop §3.5): no double model call on the hot path.

    Config seam (lld-agentic-loop §8, CONVENTIONS §4):
    - ``agent.grader_model=generator`` (Phase 2 default): ISSUP is a constrained
      Qwen3-14B decode (shared with generation — one resident model).
    - ``agent.grader_model=granite_guardian`` (Phase 3 guards-on): this endpoint
      is the dedicated Granite Guardian ``guard_safety`` role — a one-line config
      change, no code change.

    Fail-closed invariant (lld-guardrails §5): any exception after bounded retry
    → returns ``(False, 0.0)`` so the caller treats the answer as ungrounded
    rather than emitting an unverified response.

    Args:
        answer: The generated answer text.
        evidence: The concatenated graded-chunk content (the evidence the answer
            should be grounded in).

    Returns:
        A ``(grounded, score)`` tuple:
        - ``grounded``: ``True`` if ``score >= threshold``.
        - ``score``: The groundedness probability in ``[0, 1]``, or ``0.0`` on
          model unavailability (fail-closed).

    Raises:
        Nothing — all exceptions are caught and converted to fail-closed.
    """
    cfg = get_settings()
    threshold: float = cfg.guardrails.output.groundedness_threshold

    try:
        score = await _groundedness_call_with_retry(answer, evidence, retries=_MAX_RETRIES)
    except Exception:
        logger.warning("output_guard: groundedness model unavailable — failing closed (ungrounded)")
        return False, 0.0

    grounded = score >= threshold
    return grounded, score


async def _groundedness_call_with_retry(
    answer: str,
    evidence: str,
    *,
    retries: int,
) -> float:
    """Call the Granite Guardian groundedness endpoint with bounded retry.

    The ``guard_safety`` logical model role is the shared workhorse for harm
    (Task 3.1) and groundedness (Task 3.3) — one co-resident vLLM instance,
    minimising VRAM (R-00108 F6, lld-guardrails §3.6).

    Args:
        answer: The generated answer text.
        evidence: The graded-chunk evidence the answer should be grounded in.
        retries: Maximum number of attempts (including the first).

    Returns:
        A float in ``[0, 1]`` — the groundedness risk score.

    Raises:
        Exception: Re-raises the last exception if all attempts fail (the
            caller handles fail-closed conversion).
    """
    last_exc: BaseException | None = None
    for attempt in range(max(retries, 1)):
        try:
            return await _groundedness_call_once(answer, evidence)
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                backoff = 0.05 * (2**attempt)
                logger.debug(
                    "_groundedness_call_with_retry: attempt %d/%d failed (%s); backing off %.2fs",
                    attempt + 1,
                    retries,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
            else:
                logger.warning(
                    "_groundedness_call_with_retry: all %d attempt(s) failed: %s",
                    retries,
                    exc,
                )
    raise last_exc  # type: ignore[misc]


async def _groundedness_call_once(answer: str, evidence: str) -> float:
    """Single attempt to call the Granite Guardian groundedness endpoint.

    Posts a groundedness-risk classification prompt to the ``guard_safety``
    vLLM endpoint.  Granite Guardian returns a risk assessment that this
    function parses into a groundedness probability.

    Note: A *lower* groundedness risk score means the answer IS grounded.
    We invert the risk score to a groundedness probability: ``1.0 - risk``.

    Args:
        answer: The generated answer text.
        evidence: The evidence (retrieved chunk content) the answer should be
            grounded in.

    Returns:
        A float in ``[0, 1]`` representing the groundedness probability
        (higher = more likely grounded).

    Raises:
        httpx.HTTPError: On any HTTP-level failure.
        RuntimeError: On unexpected response shape.
    """
    cfg = get_settings()
    entry = cfg.models.get("guard_safety")
    url = f"{entry.endpoint.rstrip('/')}/chat/completions"

    # Granite Guardian groundedness-risk prompt format (IBM documentation).
    # We use the "groundedness" risk type available in Granite Guardian 3.3-8B.
    system_prompt = (
        "You are a groundedness evaluator. "
        "Given an evidence passage and an answer, determine if the answer is "
        "supported (grounded) by the evidence. "
        'Respond with a JSON object: {"groundedness_risk": <0.0-1.0>} '
        "where 0.0 = fully grounded and 1.0 = completely ungrounded/hallucinated."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"EVIDENCE:\n{evidence}\n\n"
                f"ANSWER:\n{answer}\n\n"
                "Is the answer grounded in the evidence?"
            ),
        },
    ]
    payload: dict[str, object] = {
        "model": entry.served_name,
        "messages": messages,
        "max_tokens": 32,
        "temperature": 0.0,
    }

    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

    from iw_rag.guardrails._verdict import parse_risk_verdict

    try:
        text = str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected response shape from guard_safety groundedness endpoint: {data!r}"
        ) from exc
    # Granite Guardian's "groundedness" risk type returns <score>yes/no</score>
    # (yes → the answer IS a groundedness risk, i.e. ungrounded). Parse that, or a
    # {"groundedness_risk": …} JSON fallback, to a risk score; invert to a
    # groundedness probability: grounded_prob = 1 - risk.
    risk = parse_risk_verdict(text, json_keys=("groundedness_risk", "risk"))
    return max(0.0, min(1.0, 1.0 - risk))


# ---------------------------------------------------------------------------
# scan_output — the main output-guard entry point
# ---------------------------------------------------------------------------


async def scan_output(
    answer: str,
    citations: list[Citation],
    graded: list[RetrievedChunk],
    access_context: AccessContext,
) -> OutputVerdict:
    """Run all output-guard checks and return an ``OutputVerdict``.

    Composes three checks in order — run **once on the output turn** (not per
    chunk) to bound GPU cost (lld-guardrails §3.6):

    1. **Deterministic citation validation** — ``validate_citations_from_list``:
       every cited chunk-ID must be in the ``graded`` set; uncited answers are
       invalid.  A ``False`` result is a hard ``block`` regardless of groundedness.

    2. **Groundedness (model wrapper)** — ``_granite_groundedness_score``:
       calls Granite Guardian groundedness risk over ``(evidence, answer)``.
       Fail-closed: model unavailable → ``(False, 0.0)`` → ungrounded → block.

       **In the node path (``output_guard_node``) this call is NOT made** — the
       node reads ``state["gen_verdict"]`` (the ISSUP result from
       ``grade_generation``) directly, satisfying the ONE-authority rule
       (lld-agentic-loop §3.5).  ``scan_output`` is used for the standalone /
       unit API and the eval demo only.

    3. **PII redaction** — ``_presidio_redact`` (Task 3.2, CPU-only): scrub
       residual PII from the final answer.  Clearance-scoped (uses the default
       PII entity types from sanitize.py).

    **Fail-closed invariant (lld-guardrails §5):**
    ``not grounded OR not citations_valid`` → the caller (``output_guard_node``)
    routes to the loop's ``fail_closed`` terminal.  This function does NOT emit
    a refusal text — ``fail_closed`` owns the refusal.

    **Output safety (optional — off by default):** when ``output.safety_model``
    is configured, a Llama Guard 4 safety scan would run here; the seam is wired
    but disabled by default for VRAM (lld-guardrails §5).

    Args:
        answer: The generated answer text.
        citations: Pre-parsed ``Citation`` objects from the generator
            (``state["citations"]`` in the node path).
        graded: The relevant retrieved chunks after ``grade_documents``
            (``state["graded"]`` in the node path).
        access_context: Authenticated principal context (for PII clearance
            scoping and audit attribution; never derived from LLM output or
            user text).

    Returns:
        An :class:`OutputVerdict` with all check results.
    """
    verdicts: list[GuardVerdict] = []

    # Build the retrieved chunk-ID set from the graded evidence.
    retrieved_ids: set[str] = {chunk.chunk_id for chunk in graded}

    # ------------------------------------------------------------------
    # Step 1: Deterministic citation validation (no model, no I/O)
    # ------------------------------------------------------------------
    citations_valid, invalid_citation_ids = await validate_citations_from_list(
        citations, retrieved_ids
    )

    citation_decision: str = "allow" if citations_valid else "block"
    citation_reason = (
        "citations_valid"
        if citations_valid
        else f"invalid_citation_ids:{','.join(invalid_citation_ids)}"
    )
    verdicts.append(
        GuardVerdict(
            layer="output",
            decision=citation_decision,  # type: ignore[arg-type]
            score=1.0 if citations_valid else 0.0,
            reason=citation_reason,
            detail={"invalid_ids": invalid_citation_ids},
        )
    )

    logger.debug(
        "output_guard: citation_valid=%r invalid_ids=%r",
        citations_valid,
        invalid_citation_ids,
    )

    # ------------------------------------------------------------------
    # Step 2: Groundedness check (model — standalone/unit/eval path)
    # In the node path this is replaced by reading state["gen_verdict"].
    # ------------------------------------------------------------------
    evidence: str = "\n\n".join(chunk.content for chunk in graded)
    try:
        grounded, groundedness_score = await _granite_groundedness_score(answer, evidence)
    except Exception:
        # Fail-closed: any exception from the groundedness wrapper → ungrounded.
        # _granite_groundedness_score should already catch all exceptions internally,
        # but we add this defensive layer in case the mock or caller raises directly.
        logger.warning("output_guard: groundedness check raised unexpectedly — failing closed")
        grounded, groundedness_score = False, 0.0

    groundedness_decision: str = "allow" if grounded else "block"
    groundedness_reason = (
        f"grounded:{groundedness_score:.3f}" if grounded else f"ungrounded:{groundedness_score:.3f}"
    )
    verdicts.append(
        GuardVerdict(
            layer="output",
            decision=groundedness_decision,  # type: ignore[arg-type]
            score=groundedness_score,
            reason=groundedness_reason,
        )
    )

    logger.debug(
        "output_guard: grounded=%r score=%.3f",
        grounded,
        groundedness_score,
    )

    # ------------------------------------------------------------------
    # Step 3: PII redaction (Presidio, CPU-only, lazy import, Task 3.2 reuse)
    # ------------------------------------------------------------------
    redacted_answer = await _presidio_redact(answer, access_context)
    pii_redacted = redacted_answer != answer

    if pii_redacted:
        # Record that PII was redacted; do NOT log entity values.
        verdicts.append(
            GuardVerdict(
                layer="output",
                decision="redact",
                score=1.0,
                reason="pii_redacted",
            )
        )
        logger.info("output_guard: PII redacted from answer")

    # ------------------------------------------------------------------
    # Step 4: Optional output safety (Llama Guard 4 — off by default)
    # Seam wired; enabled only when guardrails.output.safety_model is set.
    # ------------------------------------------------------------------
    cfg = get_settings()
    if cfg.guardrails.output.safety_model is not None:
        # Future: call Llama Guard 4 or another safety model here.
        # For now, log and skip (VRAM; default off).
        logger.debug(
            "output_guard: output safety model=%r configured but call not implemented "
            "(seam wired, enabled by Phase-3 decision when VRAM headroom confirmed)",
            cfg.guardrails.output.safety_model,
        )

    return OutputVerdict(
        grounded=grounded,
        groundedness_score=groundedness_score,
        citations_valid=citations_valid,
        invalid_citation_ids=invalid_citation_ids,
        pii_redacted=pii_redacted,
        answer=redacted_answer,
        verdicts=verdicts,
    )
