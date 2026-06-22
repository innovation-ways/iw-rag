"""Unit tests for the long-context route (WI-032 / Task 4c).

Tests cover (per spec §Tests to write):
1. Entry-predicate truth table — all four conditions × pass/fail → correct
   EntryDecision with the right eligible flag and reason.
2. Token estimator correctness.
3. YaRN-rejection guard: enable_yarn=True config → startup rejection.
4. Exit predicate over synthetic GenerationSignal strings (unsupported /
   refusal / summarize-not-answer → fallback; grounded → no fallback).
5. Assembler ordering: the stable system/instruction prefix is byte-identical
   across calls for the same doc set (prefix-cache-hit precondition).
6. Contract test: build_context() returns a structurally valid RetrievalResult.

References:
- task-4c-long-context-route §Tests to write
- lld-long-context.md §2 (public interface), §3 (algorithms)
- CONVENTIONS.md §6 (acl_applied=False carve-out, only sanctioned case)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult
from iw_rag.modes.long_context.types import LongContextConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PUBLIC_CTX = AccessContext(
    principal_id="u1",
    principal_groups=(),  # no groups → non-ACL-scoped
    is_operator=False,
    request_id="req-lc-test",
)

_SCOPED_CTX = AccessContext(
    principal_id="u2",
    principal_groups=("finance",),  # has groups → ACL-scoped
    is_operator=False,
    request_id="req-lc-acl",
)

_DOC_A = uuid.uuid4()
_DOC_B = uuid.uuid4()


def _make_config(**overrides: Any) -> LongContextConfig:
    """Build a LongContextConfig with test-friendly defaults."""
    return LongContextConfig(**overrides)


def _make_assembler(
    *,
    token_estimate: int = 1000,
    fail_estimate: bool = False,
) -> Any:
    """Build a fake WholeDocAssembler.

    Args:
        token_estimate: The token count that estimate_tokens returns.
        fail_estimate: When True, estimate_tokens raises RuntimeError.

    Returns:
        A mock with async estimate_tokens and assemble methods.
    """
    from iw_rag.modes.long_context.assembler import AssembledContext

    assembler = AsyncMock()
    if fail_estimate:
        assembler.estimate_tokens = AsyncMock(side_effect=RuntimeError("store unavailable"))
    else:
        assembler.estimate_tokens = AsyncMock(return_value=token_estimate)

    # Build a minimal AssembledContext for assemble() calls
    _SYSTEM_PREFIX = "SYSTEM: You are a document summariser.\n\n"

    async def _assemble(doc_ids: Any) -> AssembledContext:
        doc_blocks = "\n\n".join(f"[DOC {i}] content for doc {d}" for i, d in enumerate(doc_ids))
        full_text = _SYSTEM_PREFIX + doc_blocks
        from iw_rag.domain.models import RetrievedChunk

        chunks = [
            RetrievedChunk(
                document_id="doc-chunk",
                chunk_id=str(d),
                content=f"content for doc {d}",
                score=1.0,
                source_uri=f"file:///doc_{d}.md",
                metadata={"document_id": str(d)},
                retriever="long_context",
                acl_applied=False,
            )
            for d in doc_ids
        ]
        return AssembledContext(
            full_text=full_text,
            token_count=token_estimate,
            system_prefix=_SYSTEM_PREFIX,
            chunks=chunks,
        )

    assembler.assemble = AsyncMock(side_effect=_assemble)
    return assembler


# ---------------------------------------------------------------------------
# 1. YaRN-rejection guard — enable_yarn=True at startup must raise
# ---------------------------------------------------------------------------


def test_yarn_rejection_guard_raises_on_enable_yarn_true() -> None:
    """LongContextConfig(enable_yarn=True) must raise ConfigError at instantiation.

    This is the hard invariant: the route NEVER enables the 131K YaRN window
    (R-00109/F3). Any attempt to set enable_yarn=True must be caught at
    startup.
    """
    from pydantic import ValidationError

    from iw_rag.modes.long_context.types import LongContextConfig

    with pytest.raises((ValueError, ValidationError), match="enable_yarn=True is forbidden"):
        LongContextConfig(enable_yarn=True)


def test_yarn_rejection_guard_default_false_ok() -> None:
    """LongContextConfig() with default enable_yarn=False is accepted."""
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assert cfg.enable_yarn is False


# ---------------------------------------------------------------------------
# 2. Entry-predicate truth table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entry_predicate_no_candidate_doc_ids_ineligible() -> None:
    """Condition: no candidate_doc_ids → eligible=False, reason='no_bounded_corpus'."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this document",
        _PUBLIC_CTX,
        candidate_doc_ids=None,
    )
    assert not decision.eligible
    assert decision.reason == "no_bounded_corpus"
    assert decision.candidate_doc_ids == []
    assert decision.estimated_input_tokens == 0


@pytest.mark.asyncio
async def test_entry_predicate_non_wholedoc_intent_ineligible() -> None:
    """Condition: query_type != wholedoc → eligible=False, reason='not_wholedoc'.

    The route evaluates intent via the query — if the query is clearly pinpoint-
    fact (not whole-doc / global), the entry is refused.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    # A pinpoint-fact query should not be eligible
    decision = await route.evaluate_entry(
        "What is the specific revenue number in Q3?",  # pinpoint, not wholedoc
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A],
        query_type="factual",
    )
    assert not decision.eligible
    assert decision.reason == "not_wholedoc"


@pytest.mark.asyncio
async def test_entry_predicate_acl_scoped_ineligible() -> None:
    """Condition: ACL-scoped access → eligible=False, reason='acl_required' (T4).

    ACL-scoped queries are structurally excluded from the long-context route.
    This is the first structural exclusion: the route never serves ACL-scoped
    queries (CONVENTIONS §6, T4, lld-long-context §2 invariant).
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this finance document",
        _SCOPED_CTX,  # ACL-scoped (has principal_groups)
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
    )
    assert not decision.eligible
    assert decision.reason == "acl_required"


@pytest.mark.asyncio
async def test_entry_predicate_per_claim_attribution_ineligible() -> None:
    """Condition: per-claim attribution required → eligible=False, reason='attribution_strict'.

    The second structural exclusion: no per-claim citation when a whole blob is
    stuffed (CONVENTIONS §6, T4).
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(
        lc_max_input=24000,
        require_document_level_attribution_only=True,
    )
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this document",
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
        require_per_claim_attribution=True,  # caller signals per-claim need
    )
    assert not decision.eligible
    assert decision.reason == "attribution_strict"


@pytest.mark.asyncio
async def test_entry_predicate_over_budget_ineligible() -> None:
    """Condition: estimated_input_tokens > lc_max_input → eligible=False, reason='over_budget'."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=1000)
    # The assembler estimates more tokens than the budget allows
    assembler = _make_assembler(token_estimate=5000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this document",
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A],
        query_type="wholedoc",
    )
    assert not decision.eligible
    assert decision.reason == "over_budget"
    assert decision.estimated_input_tokens == 5000


@pytest.mark.asyncio
async def test_entry_predicate_all_pass_eligible() -> None:
    """All four conditions pass → eligible=True, reason='eligible'."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = await route.evaluate_entry(
        "Summarise this document for me",
        _PUBLIC_CTX,
        candidate_doc_ids=[_DOC_A, _DOC_B],
        query_type="wholedoc",
    )
    assert decision.eligible
    assert decision.reason == "eligible"
    assert set(decision.candidate_doc_ids) == {_DOC_A, _DOC_B}
    assert decision.estimated_input_tokens == 1000


# ---------------------------------------------------------------------------
# 3. Token estimator correctness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_token_estimator_called_with_doc_ids() -> None:
    """evaluate_entry delegates token estimation to the assembler."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=8000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    doc_ids = [_DOC_A, _DOC_B]
    decision = await route.evaluate_entry(
        "Summarise this document",
        _PUBLIC_CTX,
        candidate_doc_ids=doc_ids,
        query_type="wholedoc",
    )

    # Token estimate should be reflected in the decision
    assert decision.estimated_input_tokens == 8000
    # Assembler was called with the doc_ids
    assembler.estimate_tokens.assert_called_once()
    call_args = assembler.estimate_tokens.call_args[0][0]
    assert list(call_args) == list(doc_ids)


# ---------------------------------------------------------------------------
# 4. Exit predicate — should_fallback over synthetic signals
# ---------------------------------------------------------------------------


def test_should_fallback_grounded_no_fallback() -> None:
    """grounded signal → should_fallback=False (route keeps control)."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assembler = _make_assembler()
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    assert not route.should_fallback("grounded")


def test_should_fallback_unsupported_fallback() -> None:
    """unsupported signal → should_fallback=True (demote to RAG)."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assembler = _make_assembler()
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    assert route.should_fallback("unsupported")


def test_should_fallback_refusal_fallback() -> None:
    """refusal signal (model declined to answer) → should_fallback=True."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assembler = _make_assembler()
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    assert route.should_fallback("not_answering")


def test_should_fallback_summarize_not_answer_fallback() -> None:
    """summarize_not_answer signal (Databricks failure mode) → should_fallback=True.

    The summarize-instead-of-answer failure mode (R-00109/F6) is the long-context
    model collapsing instruction-following for large inputs — caught by the exit
    predicate.
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assembler = _make_assembler()
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    assert route.should_fallback("summarize_not_answer")


def test_should_fallback_none_signal_fallback() -> None:
    """None / unknown signal → should_fallback=True (fail-closed default)."""
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assembler = _make_assembler()
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    assert route.should_fallback(None)
    assert route.should_fallback("unexpected_value")


# ---------------------------------------------------------------------------
# 5. Assembler ordering — byte-identical stable prefix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assembler_prefix_byte_identical_across_calls() -> None:
    """The stable system/instruction prefix is byte-identical across calls.

    This is the vLLM APC hit-rate precondition: the prefix MUST be byte-
    identical for the same doc set so the cached KV is reused across follow-up
    queries (lld-long-context §3, R-00109/F4).
    """
    from iw_rag.modes.long_context.assembler import WholeDocAssembler

    assembler = WholeDocAssembler()
    doc_ids = [_DOC_A, _DOC_B]

    ctx1 = await assembler.assemble(doc_ids)
    ctx2 = await assembler.assemble(doc_ids)

    assert ctx1.system_prefix == ctx2.system_prefix, (
        "Stable system/instruction prefix must be byte-identical across calls "
        "for the same doc set (APC precondition)."
    )


@pytest.mark.asyncio
async def test_assembler_prefix_comes_first() -> None:
    """The stable system/instruction prefix appears at the START of full_text.

    Prefix-cache-friendly ordering: stable prefix FIRST, then per-doc blocks,
    then query last (lld-long-context §3, R-00109/F4).
    """
    from iw_rag.modes.long_context.assembler import WholeDocAssembler

    assembler = WholeDocAssembler()
    doc_ids = [_DOC_A]

    ctx = await assembler.assemble(doc_ids)

    assert ctx.full_text.startswith(ctx.system_prefix), (
        "full_text must start with the stable system/instruction prefix "
        "(prefix-cache-friendly ordering)."
    )
    # Prefix must be non-empty
    assert len(ctx.system_prefix) > 0


@pytest.mark.asyncio
async def test_assembler_yarn_never_enabled() -> None:
    """WholeDocAssembler never references enable_yarn=True.

    The assembler is bound to the native 32K window — YaRN is structurally
    excluded (lld-long-context §1 invariant, R-00109/F3).
    """
    from iw_rag.modes.long_context.assembler import WholeDocAssembler

    assembler = WholeDocAssembler()

    # The assembler should not have a public enable_yarn attribute set to True
    # (a False default or no attribute at all is acceptable)
    yarn_val = getattr(assembler, "enable_yarn", False)
    assert yarn_val is False, "Assembler must never enable YaRN (native 32K only)."


# ---------------------------------------------------------------------------
# 6. Contract test — build_context returns valid RetrievalResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_context_returns_valid_retrieval_result() -> None:
    """build_context() returns a structurally valid RetrievalResult.

    The result must:
    - Be a RetrievalResult instance.
    - Have retriever=="long_context".
    - Have acl_applied=False (the single sanctioned carve-out, CONVENTIONS §6).
    - Have non-empty chunks (document-level RetrievedChunks).
    - Have diagnostics with route=="long_context".
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A],
        estimated_input_tokens=1000,
    )

    result = await route.build_context("Summarise this doc", decision, _PUBLIC_CTX)

    assert isinstance(result, RetrievalResult)
    assert result.retriever == "long_context"
    assert result.acl_applied is False  # the sanctioned carve-out
    assert len(result.chunks) > 0
    assert result.diagnostics.get("route") == "long_context"


@pytest.mark.asyncio
async def test_build_context_raises_on_over_hard_ceiling() -> None:
    """build_context() raises LongContextOverBudget when assembled tokens exceed hard_ceiling.

    This ensures we never silently enable YaRN or overflow the native window
    (lld-long-context §5 error table).
    """
    from iw_rag.modes.long_context.route import LongContextOverBudget, LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000, hard_ceiling=500)
    # Assembler will return more tokens than the hard ceiling
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A],
        estimated_input_tokens=400,
    )

    with pytest.raises(LongContextOverBudget, match="hard_ceiling"):
        await route.build_context("Summarise", decision, _PUBLIC_CTX)


@pytest.mark.asyncio
async def test_build_context_document_level_citations() -> None:
    """build_context() produces document-level chunks (no per-claim attribution).

    Each RetrievedChunk in the result should have a document_id in metadata.
    This is the second structural exclusion: no per-claim citation, only
    document-level attribution (CONVENTIONS §6, T4).
    """
    from iw_rag.modes.long_context.route import LongContextRoute
    from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

    cfg = LongContextConfig(lc_max_input=24000)
    assembler = _make_assembler(token_estimate=1000)
    route = LongContextRoute(assembler=assembler, cfg=cfg)

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A],
        estimated_input_tokens=1000,
    )

    result = await route.build_context("Summarise", decision, _PUBLIC_CTX)

    # Each chunk should carry document_id in metadata (document-level attribution)
    assert len(result.chunks) >= 1, "Expected at least one document-level chunk"
    for chunk in result.chunks:
        assert "document_id" in chunk.metadata, (
            f"Chunk {chunk.chunk_id} missing document_id in metadata — "
            "long-context must emit document-level citations only."
        )
        # Pin the exact document_id value recorded in metadata
        assert chunk.metadata["document_id"] == str(_DOC_A), (
            f"Chunk metadata document_id must match the candidate doc id, got "
            f"{chunk.metadata['document_id']!r}"
        )


# ---------------------------------------------------------------------------
# 7. LongContextConfig defaults
# ---------------------------------------------------------------------------


def test_long_context_config_defaults() -> None:
    """LongContextConfig defaults match the spec (lld-long-context §6)."""
    from iw_rag.modes.long_context.types import LongContextConfig

    cfg = LongContextConfig()
    assert cfg.lc_max_input == 24_000
    assert cfg.hard_ceiling == 30_000
    assert cfg.enable_yarn is False
    assert cfg.enabled is False
    assert cfg.require_document_level_attribution_only is True


# ---------------------------------------------------------------------------
# 8. EntryDecision dataclass is frozen
# ---------------------------------------------------------------------------


def test_entry_decision_is_frozen() -> None:
    """EntryDecision is immutable (frozen dataclass)."""
    from iw_rag.modes.long_context.types import EntryDecision

    decision = EntryDecision(
        eligible=True,
        reason="eligible",
        candidate_doc_ids=[_DOC_A],
        estimated_input_tokens=1000,
    )

    with pytest.raises((AttributeError, TypeError), match="cannot assign"):
        decision.eligible = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. Assembler estimate_tokens with real WholeDocAssembler
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_assembler_estimate_tokens_returns_int() -> None:
    """WholeDocAssembler.estimate_tokens returns a non-negative int."""
    from iw_rag.modes.long_context.assembler import WholeDocAssembler

    assembler = WholeDocAssembler()
    result = await assembler.estimate_tokens([_DOC_A, _DOC_B])
    assert isinstance(result, int)
    assert result >= 0
