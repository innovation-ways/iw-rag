"""Integration tests for the long-context route (WI-032 / Task 4c).

Tests cover (per spec §Tests to write):
1. Whole-doc fixture under lc_max_input → entry eligible → assembled →
   valid RetrievalResult with document-level citations, retriever=="long_context".
2. Doc over the ceiling → eligible=False → router uses spine.
3. ACL-scoped doc → forced to RAG regardless of intent.
4. Contract test: build_context() returns a structurally valid RetrievalResult
   the generator accepts in document-level citation mode.
5. Router test: route=="long_context" + eligible → long-context node →
   grade_generation (bypassing grade_documents); ineligible/disabled → falls
   back to "retrieve".

Uses in-memory fakes; no Postgres required.

References:
- task-4c-long-context-route §Tests to write
- lld-long-context.md §2 (contract), §7 (test strategy)
- CONVENTIONS.md §6 (acl_applied=False carve-out)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievedChunk

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_DOC_A = uuid.uuid4()
_DOC_B = uuid.uuid4()

_PUBLIC_CTX = AccessContext(
    principal_id="u1",
    principal_groups=(),
    is_operator=False,
    request_id="req-lc-integ-test",
)

_SCOPED_CTX = AccessContext(
    principal_id="u2",
    principal_groups=("hr",),  # ACL-scoped
    is_operator=False,
    request_id="req-lc-acl-integ",
)


def _make_chunk(doc_id: uuid.UUID, content: str = "Document content.") -> RetrievedChunk:
    """Build a document-level RetrievedChunk.

    Args:
        doc_id: The document UUID.
        content: Chunk text content.

    Returns:
        A RetrievedChunk with document_id in metadata.
    """
    return RetrievedChunk(
        document_id="doc-chunk",
        chunk_id=str(doc_id),
        content=content,
        score=1.0,
        source_uri=f"file:///doc_{doc_id}.md",
        metadata={"document_id": str(doc_id)},
        retriever="long_context",
        acl_applied=False,
    )


def _make_assembler_for_config(
    *,
    token_estimate: int = 5000,
    hard_ceiling: int = 30000,
) -> Any:
    """Build a fake WholeDocAssembler.

    Args:
        token_estimate: The token count returned by estimate_tokens and
            assemble().
        hard_ceiling: Not used by the assembler itself; passed here for
            documentation.

    Returns:
        A mock with async estimate_tokens and assemble methods.
    """
    from iw_rag.modes.long_context.assembler import AssembledContext

    assembler = MagicMock()
    assembler.estimate_tokens = AsyncMock(return_value=token_estimate)

    _SYSTEM_PREFIX = "SYSTEM: You are a document analyst.\n\n"

    async def _assemble(doc_ids: Any) -> AssembledContext:
        chunks = [_make_chunk(d) for d in doc_ids]
        full_text = _SYSTEM_PREFIX + "\n\n".join(c.content for c in chunks)
        return AssembledContext(
            full_text=full_text,
            token_count=token_estimate,
            system_prefix=_SYSTEM_PREFIX,
            chunks=chunks,
        )

    assembler.assemble = AsyncMock(side_effect=_assemble)
    return assembler


# ---------------------------------------------------------------------------
# 1. Whole-doc under lc_max_input → eligible → assembled → valid RetrievalResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whole_doc_under_budget_eligible_and_assembled() -> None:
    """Whole-doc fixture under lc_max_input → eligible, assembled, valid RetrievalResult.

    Validates the happy path:
    - evaluate_entry returns eligible=True.
    - build_context returns a RetrievalResult with retriever="long_context".
    - acl_applied=False (the single sanctioned carve-out, CONVENTIONS §6).
    - document-level citations present in chunk metadata.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000, hard_ceiling=30000)
    assembler = _make_assembler_for_config(token_estimate=5000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    # Entry check
    decision = await route.evaluate_entry(
        "Summarise the leave policy document",
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
    )
    assert decision.eligible, f"Expected eligible, got reason={decision.reason!r}"

    # Build context
    result = await route.build_context("Summarise the leave policy document", decision, _PUBLIC_CTX)

    assert result.retriever == "long_context"
    assert result.acl_applied is False  # sanctioned carve-out
    assert len(result.chunks) > 0
    for chunk in result.chunks:
        assert "document_id" in chunk.metadata


@pytest.mark.asyncio
async def test_whole_doc_result_has_correct_diagnostics() -> None:
    """build_context() diagnostics include route='long_context' and input_tokens."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler_for_config(token_estimate=8000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A, _DOC_B],
        estimated_input_tokens=8000,
    )
    result = await route.build_context("Compare these docs", decision, _PUBLIC_CTX)

    assert result.diagnostics.get("route") == "long_context"
    assert "input_tokens" in result.diagnostics
    assert result.diagnostics["prefix_cache_friendly"] is True


# ---------------------------------------------------------------------------
# 2. Doc over the ceiling → eligible=False → router uses spine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_over_lc_max_input_ineligible() -> None:
    """Token count > lc_max_input → eligible=False, reason='over_budget'.

    The router should fall through to the spine retrieve node.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=1000)
    assembler = _make_assembler_for_config(token_estimate=25000)  # over budget
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this large document corpus",
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A, _DOC_B],
        query_type="wholedoc",
    )

    assert not decision.eligible
    assert decision.reason == "over_budget"
    assert decision.estimated_input_tokens == 25000


@pytest.mark.asyncio
async def test_doc_over_hard_ceiling_raises() -> None:
    """Token count > hard_ceiling post-assembly → LongContextOverBudget raised.

    Never silently enable YaRN (lld-long-context §5).
    """
    from iw_rag.modes.long_context.route import LongContextOverBudget, LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000, hard_ceiling=500)
    assembler = _make_assembler_for_config(token_estimate=1000)  # > hard_ceiling
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A],
        estimated_input_tokens=400,
    )

    with pytest.raises(LongContextOverBudget, match="hard_ceiling"):
        await route.build_context("Summarise", decision, _PUBLIC_CTX)


# ---------------------------------------------------------------------------
# 3. ACL-scoped doc → forced to RAG regardless of intent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acl_scoped_context_forced_to_rag() -> None:
    """ACL-scoped access → eligible=False, reason='acl_required' (T4 structural rule).

    The long-context route NEVER serves ACL-scoped queries. This is the primary
    structural exclusion: there's no per-document ACL when a whole blob is stuffed.
    Forced to RAG regardless of intent.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler_for_config(token_estimate=5000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this HR policy document",
        _SCOPED_CTX,  # ACL-scoped principal
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
    )

    assert not decision.eligible
    assert decision.reason == "acl_required"


@pytest.mark.asyncio
async def test_acl_scoped_regardless_of_whole_doc_intent() -> None:
    """Even a whole-doc query with correct token budget is refused if ACL-scoped.

    The ACL check happens BEFORE the token budget check (structural, not
    configurable). T4: ACL-scoped and attribution-strict queries are structurally
    RAG-only.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler_for_config(token_estimate=100)  # tiny, clearly within budget
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise the entire document",
        _SCOPED_CTX,
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
    )

    assert not decision.eligible
    assert decision.reason == "acl_required"
    # The assembler should NOT have been called (ACL check short-circuits)
    assembler.estimate_tokens.assert_not_called()


# ---------------------------------------------------------------------------
# 4. Contract test — RetrievalResult accepted by generator in document-level mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_context_result_generator_compatible() -> None:
    """build_context() returns a RetrievalResult compatible with document-level citation mode.

    The generator switches to document-level mode when route="long_context" is in
    diagnostics. This test verifies the result has all required fields.
    """
    from iw_rag.domain.models import RetrievalResult
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler_for_config(token_estimate=6000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A, _DOC_B],
        estimated_input_tokens=6000,
    )
    result = await route.build_context("Compare these two documents", decision, _PUBLIC_CTX)

    # Must be a valid RetrievalResult
    assert isinstance(result, RetrievalResult)
    # Required fields for the generator
    assert result.retriever == "long_context"
    assert isinstance(result.chunks, list)
    assert len(result.chunks) > 0
    # acl_applied=False is the sanctioned carve-out for long-context
    assert result.acl_applied is False
    # Each chunk has document_id for document-level citation
    for chunk in result.chunks:
        assert chunk.chunk_id, "chunk_id must be non-empty"
        assert chunk.content, "content must be non-empty"
        assert "document_id" in chunk.metadata


# ---------------------------------------------------------------------------
# 5. Router wiring tests (integration-level, using the graph state pattern)
# ---------------------------------------------------------------------------


def test_route_after_router_long_context_enabled_dispatches() -> None:
    """route_after_router dispatches 'long_context' → 'long_context' node when enabled.

    Mirrors the visual/graph dispatch pattern: when modes.long_context.enabled=True
    AND the node is wired, the edge dispatches to 'long_context'.
    """
    # Temporarily enable long_context mode via env
    import os

    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    old = os.environ.get("IW_RAG__MODES__LONG_CONTEXT__ENABLED")
    os.environ["IW_RAG__MODES__LONG_CONTEXT__ENABLED"] = "true"
    settings_mod.get_settings.cache_clear()

    try:
        state: GraphState = {  # type: ignore[typeddict-item]
            "route": "long_context",
            "question": "Summarise this doc",
            "original_question": "Summarise this doc",
            "verdicts": [],
            "retries": 0,
            "max_retries": 2,
        }
        result = route_after_router(state)
        assert result == "long_context"
    finally:
        if old is None:
            os.environ.pop("IW_RAG__MODES__LONG_CONTEXT__ENABLED", None)
        else:
            os.environ["IW_RAG__MODES__LONG_CONTEXT__ENABLED"] = old
        settings_mod.get_settings.cache_clear()


def test_route_after_router_long_context_disabled_collapses_to_retrieve() -> None:
    """route_after_router collapses 'long_context' → 'retrieve' when mode disabled.

    Default behaviour: modes.long_context.enabled=False → spine fallback.
    """
    import os

    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    old = os.environ.get("IW_RAG__MODES__LONG_CONTEXT__ENABLED")
    os.environ["IW_RAG__MODES__LONG_CONTEXT__ENABLED"] = "false"
    settings_mod.get_settings.cache_clear()

    try:
        state: GraphState = {  # type: ignore[typeddict-item]
            "route": "long_context",
            "question": "Summarise this doc",
            "original_question": "Summarise this doc",
            "verdicts": [],
            "retries": 0,
            "max_retries": 2,
        }
        result = route_after_router(state)
        assert result == "retrieve"
    finally:
        if old is None:
            os.environ.pop("IW_RAG__MODES__LONG_CONTEXT__ENABLED", None)
        else:
            os.environ["IW_RAG__MODES__LONG_CONTEXT__ENABLED"] = old
        settings_mod.get_settings.cache_clear()
