"""Fail-closed-refusal demo (lld-guardrails §8.3 demo 3).

Demonstrates that an answer which is NOT supported by the retrieved evidence is
never emitted: the output guard's groundedness check (or a fabricated citation)
forces the loop's fail-closed seam, and the refusal is recorded for audit.

Three contrasts, all model-mocked (a pass/fail harness, not a live LLM run):
(a) ``scan_output`` on an ungrounded answer → ``OutputVerdict.grounded is False``
    → block (the loop would route to ``fail_closed``).
(b) ``scan_output`` on a grounded answer that cites a chunk NOT in the retrieved
    set → ``citations_valid is False`` → block wins deterministically even though
    the model says "grounded" (a fabricated citation is caught with certainty).
(c) Graph-level: an unsupported generation → ``grade_generation`` unsupported →
    loop refuses via ``fail_closed`` → ``disposition == "refused"``, no answer
    emitted, and the refusal/groundedness is visible in the trail/diagnostics
    (the durable audit row itself is Task 3.4).

References:
- lld-guardrails §3.4 (output_guard), §8.3 demo 3 (fail-closed refusal)
- lld-agentic-loop §3.5 (one groundedness authority), §3.6 (fail-closed seam)
- task-3.3-output-guard-grounding (acceptance: demo passes)
- R-00108 F5 (deterministic citation validation backstops the model check)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer, RetrievalResult, RetrievedChunk

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-eval-fail-closed",
)

_QUESTION = "How many paid leave days does the policy grant?"

# The retrieved evidence — note it does NOT actually answer the question.
_EVIDENCE = [
    RetrievedChunk(
        chunk_id="hr_chunk",
        document_id="doc-hr_chunk",
        content="The expense reimbursement form must be filed within 30 days.",
        score=0.61,
        source_uri="corpus://hr_expenses",
    )
]


# ---------------------------------------------------------------------------
# Demo (a): ungrounded answer → scan_output blocks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ungrounded_answer_is_blocked() -> None:
    """An answer unsupported by the evidence → grounded is False → block."""
    from iw_rag.guardrails import output_guard

    answer = "The policy grants 25 paid leave days per year [1]."
    citations = [
        Citation(
            marker=1,
            chunk_id="hr_chunk",
            document_id="doc-hr_chunk",
            source_uri="corpus://hr_expenses",
        )
    ]

    with (
        # Groundedness model says NOT grounded (the evidence does not support it).
        patch.object(
            output_guard,
            "_granite_groundedness_score",
            new=AsyncMock(return_value=(False, 0.12)),
        ),
        patch.object(output_guard, "_presidio_redact", new=AsyncMock(side_effect=lambda t, c: t)),
    ):
        verdict = await output_guard.scan_output(answer, citations, _EVIDENCE, _CTX)

    assert verdict.grounded is False, "Unsupported answer must be graded ungrounded"
    # Citations are structurally valid; the block is driven by groundedness.
    assert verdict.citations_valid is True


# ---------------------------------------------------------------------------
# Demo (b): fabricated citation → deterministic block wins over "grounded"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fabricated_citation_blocks_even_if_model_says_grounded() -> None:
    """A cited id not in the retrieved set is caught with certainty (no model)."""
    from iw_rag.guardrails import output_guard

    answer = "The policy grants 20 paid leave days per year [1]."
    # Cites a chunk id that was NEVER retrieved — a fabricated citation.
    citations = [
        Citation(
            marker=1,
            chunk_id="ghost_chunk",
            document_id="doc-ghost",
            source_uri="corpus://nowhere",
        )
    ]

    with (
        # Even if the model is fooled into "grounded"...
        patch.object(
            output_guard,
            "_granite_groundedness_score",
            new=AsyncMock(return_value=(True, 0.97)),
        ),
        patch.object(output_guard, "_presidio_redact", new=AsyncMock(side_effect=lambda t, c: t)),
    ):
        verdict = await output_guard.scan_output(answer, citations, _EVIDENCE, _CTX)

    assert verdict.citations_valid is False, "A fabricated citation id must be invalid"
    assert "ghost_chunk" in verdict.invalid_citation_ids


# ---------------------------------------------------------------------------
# Demo (c): graph-level — unsupported generation → fail-closed refusal
# ---------------------------------------------------------------------------


class _UnsupportedAssembler:
    """Assembler that drafts an answer the evidence does not support."""

    async def generate(self, query: str, result: RetrievalResult) -> GeneratedAnswer:
        """Return a (cited) draft regardless — grounding is judged downstream."""
        if not result.chunks:
            return GeneratedAnswer(answer="", grounded=False, citations=[], used_chunk_ids=[])
        return GeneratedAnswer(
            answer="The policy grants 25 paid leave days [1].",
            grounded=True,
            citations=[
                Citation(
                    marker=1,
                    chunk_id="hr_chunk",
                    document_id="doc-hr_chunk",
                    source_uri="corpus://hr_expenses",
                )
            ],
            used_chunk_ids=["hr_chunk"],
        )


class _UnsupportedGraderLlm:
    """Stub LLM: routes to retrieve, grades docs relevant, but ISSUP=unsupported."""

    async def complete(self, messages: object, params: object) -> object:
        class _R:
            text = "rewritten query"

        return _R()

    async def complete_structured(self, messages: object, schema: object, params: object) -> object:
        class _R:
            pass

        r = _R()
        required = schema.get("required", []) if isinstance(schema, dict) else []
        if "answers" in required and "supported" in required:
            r.text = '{"supported": false, "answers": true, "reason": "not grounded"}'  # type: ignore[attr-defined]
        elif "answers" in required:
            r.text = '{"answers": true, "reason": "addresses"}'  # type: ignore[attr-defined]
        elif "supported" in required:
            # ISSUP: the draft is NOT supported by the evidence.
            r.text = '{"supported": false, "reason": "evidence does not support claim"}'  # type: ignore[attr-defined]
        elif "grades" in required:
            r.text = '{"grades": [{"relevant": true, "reason": "topical"}]}'  # type: ignore[attr-defined]
        elif "route" in required:
            r.text = '{"route": "retrieve", "query_type": "factual"}'  # type: ignore[attr-defined]
        else:
            r.text = "{}"  # type: ignore[attr-defined]
        return r

    async def health(self) -> bool:
        return True


class _EvidenceSpine:
    """Spine returning the (non-answering) evidence."""

    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        return RetrievalResult(chunks=list(_EVIDENCE), retriever="spine", acl_applied=True)


@pytest.mark.asyncio
async def test_graph_refuses_when_answer_unsupported() -> None:
    """Unsupported generation → loop refuses via fail_closed; no answer emitted.

    Uses the default in-memory checkpointer (no Postgres needed for this demo).
    """
    import iw_rag.settings as settings_mod
    from iw_rag.agent.graph import answer_question, build_graph
    from iw_rag.settings import AgentSettings

    settings_mod.get_settings.cache_clear()

    graph = build_graph(
        _EvidenceSpine(),
        _UnsupportedAssembler(),
        llm=_UnsupportedGraderLlm(),
        settings=AgentSettings(router_enabled=True, skip_enabled=False, max_retries=1),
    )

    with (
        patch(
            "iw_rag.guardrails.input_guard.prompt_guard_score",
            new=AsyncMock(return_value=0.05),
        ),
        patch(
            "iw_rag.guardrails.input_guard.granite_harm_score",
            new=AsyncMock(return_value=0.05),
        ),
        patch(
            "iw_rag.guardrails.sanitize.prompt_guard_score",
            new=AsyncMock(return_value=0.05),  # nothing poisoned
        ),
        patch(
            "iw_rag.guardrails.sanitize._presidio_redact",
            new=AsyncMock(side_effect=lambda t, c: t),
        ),
    ):
        state = await answer_question(graph, _QUESTION, _CTX)

    assert state.get("disposition") == "refused", (
        f"Unsupported answer must fail closed; got {state.get('disposition')!r}"
    )
    assert state.get("answer") is None, "No unsupported answer may be emitted"
    # The refusal is recorded in the verdict trail (durable audit row is Task 3.4).
    kinds = [(v.kind, v.value) for v in (state.get("verdicts") or [])]
    assert any(k == "grade_generation" for k, _ in kinds), (
        f"grade_generation verdict must be in the trail; got {kinds!r}"
    )
