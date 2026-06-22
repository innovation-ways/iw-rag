"""Unit tests for the long-context router branch (WI-032 / Task 4c steps 6–8).

Tests cover:
- route_after_router dispatches "long_context" → "long_context" node when
  modes.long_context.enabled=True AND the node is wired.
- route_after_router collapses "long_context" → "retrieve" when
  modes.long_context.enabled=False (default — preserves the spine path).
- make_long_context_node returns correct chunks on eligible entry.
- make_long_context_node falls back to spine when entry is ineligible.
- make_long_context_node falls back to spine on LongContextOverBudget.
- make_long_context_node falls back to spine on any unexpected error.
- The node writes BOTH chunks AND graded (bypasses grade_documents).
- The fallback verdict has value="fallback" when spine is invoked.
- Long_context node → generate (bypasses sanitize and grade_documents wiring).

References:
- task-4c-long-context-route §6 (router wiring)
- lld-long-context.md §4 (diagram — LongContext → GradeGen directly)
- CONVENTIONS.md §6 (acl_applied=False carve-out)
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk

_PUBLIC_CTX = AccessContext(
    principal_id="u1",
    principal_groups=(),
    is_operator=False,
    request_id="req-lc-agent-test",
)

_DOC_A = uuid.uuid4()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str = "doc-1") -> RetrievedChunk:
    """Build a document-level RetrievedChunk.

    Args:
        chunk_id: The chunk identifier.

    Returns:
        A RetrievedChunk with acl_applied=False (long-context carve-out).
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=chunk_id,
        content="Document content for long-context.",
        score=1.0,
        source_uri="file:///doc.md",
        metadata={"document_id": chunk_id},
        retriever="long_context",
        acl_applied=False,
    )


def _make_spine_chunk() -> RetrievedChunk:
    """Build a spine-fallback RetrievedChunk.

    Returns:
        A RetrievedChunk with acl_applied=True (spine path).
    """
    return RetrievedChunk(
        chunk_id="spine-c1",
        document_id="doc-spine-c1",
        content="Spine fallback content.",
        score=0.7,
        source_uri="file:///spine.md",
        metadata={},
        retriever="spine",
        acl_applied=True,
    )


def _make_fake_long_context_route(
    *,
    eligible: bool = True,
    raise_over_budget: bool = False,
    raise_exc: type[Exception] | None = None,
) -> Any:
    """Build a fake LongContextRoute.

    Args:
        eligible: Whether evaluate_entry returns eligible=True.
        raise_over_budget: When True, build_context raises LongContextOverBudget.
        raise_exc: When set, evaluate_entry raises this exception type.

    Returns:
        A MagicMock LongContextRoute with async evaluate_entry and build_context.
    """
    from iw_rag.modes.long_context.route import LongContextOverBudget
    from iw_rag.modes.long_context.types import EntryDecision

    route = MagicMock()

    async def _evaluate_entry(query: str, access: Any, **kwargs: Any) -> EntryDecision:
        if raise_exc is not None:
            raise raise_exc("simulated error")
        return EntryDecision(
            eligible=eligible,
            reason="eligible" if eligible else "not_wholedoc",
            candidate_doc_ids=[_DOC_A] if eligible else [],
            estimated_input_tokens=5000 if eligible else 0,
        )

    async def _build_context(query: str, decision: Any, access: Any) -> RetrievalResult:
        if raise_over_budget:
            raise LongContextOverBudget("over hard ceiling")
        return RetrievalResult(
            query=query,
            chunks=[_make_chunk()],
            retriever="long_context",
            acl_applied=False,
            diagnostics={"route": "long_context", "input_tokens": 5000},
        )

    route.evaluate_entry = AsyncMock(side_effect=_evaluate_entry)
    route.build_context = AsyncMock(side_effect=_build_context)
    return route


def _make_fake_spine(chunks: list[RetrievedChunk] | None = None) -> Any:
    """Build a fake spine retriever.

    Args:
        chunks: Chunks to return on success (defaults to one spine chunk).

    Returns:
        A MagicMock spine with async retrieve method.
    """
    spine = MagicMock()
    effective_chunks = chunks if chunks is not None else [_make_spine_chunk()]

    async def _retrieve(query: str, ctx: Any, **kwargs: Any) -> RetrievalResult:
        return RetrievalResult(
            chunks=effective_chunks,
            retriever="spine",
            acl_applied=True,
            query=query,
        )

    spine.retrieve = AsyncMock(side_effect=_retrieve)
    return spine


# ---------------------------------------------------------------------------
# route_after_router tests
# ---------------------------------------------------------------------------


def test_route_after_router_long_context_disabled_collapses_to_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """route_after_router collapses 'long_context' → 'retrieve' when mode disabled.

    Default: modes.long_context.enabled=False → spine fallback.
    """
    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    monkeypatch.setenv("IW_RAG__MODES__LONG_CONTEXT__ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    try:
        state: GraphState = {  # type: ignore[typeddict-item]
            "route": "long_context",
            "question": "Summarise",
            "original_question": "Summarise",
            "verdicts": [],
            "retries": 0,
            "max_retries": 2,
        }
        result = route_after_router(state)
        assert result == "retrieve"
    finally:
        settings_mod.get_settings.cache_clear()


def test_route_after_router_long_context_enabled_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """route_after_router dispatches 'long_context' → 'long_context' when enabled."""
    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    monkeypatch.setenv("IW_RAG__MODES__LONG_CONTEXT__ENABLED", "true")
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
        settings_mod.get_settings.cache_clear()


def test_route_after_router_retrieve_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default retrieve path is byte-for-byte unchanged when long-context is off."""
    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    monkeypatch.setenv("IW_RAG__MODES__LONG_CONTEXT__ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    try:
        state: GraphState = {  # type: ignore[typeddict-item]
            "route": "retrieve",
            "question": "What is the policy?",
            "original_question": "What is the policy?",
            "verdicts": [],
            "retries": 0,
            "max_retries": 2,
        }
        result = route_after_router(state)
        assert result == "retrieve"
    finally:
        settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# make_long_context_node tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_long_context_node_eligible_returns_doc_level_chunks() -> None:
    """On eligible entry, node returns assembled doc-level chunks.

    Both chunks AND graded are set so generate can bypass grade_documents.
    """
    from iw_rag.agent.nodes import make_long_context_node
    from iw_rag.agent.state import GraphState

    lc_route = _make_fake_long_context_route(eligible=True)
    spine = _make_fake_spine()
    node = make_long_context_node(lc_route, spine, assembler=None)

    state: GraphState = {  # type: ignore[typeddict-item]
        "question": "Summarise this document",
        "original_question": "Summarise this document",
        "access_context": _PUBLIC_CTX,
        "query_type": "wholedoc",
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "diagnostics": {},
    }

    result = await node(state)

    assert "chunks" in result
    assert "graded" in result
    assert result["chunks"] == result["graded"], "chunks and graded must be identical on LC path"
    assert len(result["chunks"]) > 0
    assert len(result["verdicts"]) == 1
    verdict = result["verdicts"][0]
    assert verdict.kind == "long_context_retrieve"
    assert verdict.value == "long_context"


@pytest.mark.asyncio
async def test_long_context_node_ineligible_falls_back_to_spine() -> None:
    """On ineligible entry, node falls back to spine and records fallback verdict."""
    from iw_rag.agent.nodes import make_long_context_node
    from iw_rag.agent.state import GraphState

    lc_route = _make_fake_long_context_route(eligible=False)
    spine = _make_fake_spine()
    node = make_long_context_node(lc_route, spine, assembler=None)

    state: GraphState = {  # type: ignore[typeddict-item]
        "question": "What is the specific revenue in Q3?",
        "original_question": "What is the specific revenue in Q3?",
        "access_context": _PUBLIC_CTX,
        "query_type": "factual",
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "diagnostics": {},
    }

    result = await node(state)

    assert "chunks" in result
    assert "graded" in result
    assert len(result["chunks"]) > 0  # spine returned something
    assert len(result["verdicts"]) == 1
    verdict = result["verdicts"][0]
    assert verdict.kind == "long_context_retrieve"
    assert verdict.value == "fallback"


@pytest.mark.asyncio
async def test_long_context_node_over_budget_falls_back_to_spine() -> None:
    """On LongContextOverBudget, node falls back to spine with fallback verdict."""
    from iw_rag.agent.nodes import make_long_context_node
    from iw_rag.agent.state import GraphState

    lc_route = _make_fake_long_context_route(eligible=True, raise_over_budget=True)
    spine = _make_fake_spine()
    node = make_long_context_node(lc_route, spine, assembler=None)

    state: GraphState = {  # type: ignore[typeddict-item]
        "question": "Summarise this huge document",
        "original_question": "Summarise this huge document",
        "access_context": _PUBLIC_CTX,
        "query_type": "wholedoc",
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "diagnostics": {"candidate_doc_ids": [_DOC_A]},
    }

    result = await node(state)

    verdict = result["verdicts"][0]
    assert verdict.value == "fallback"
    assert "LongContextOverBudget" in verdict.reason


@pytest.mark.asyncio
async def test_long_context_node_spine_none_returns_empty_chunks() -> None:
    """On entry failure with spine=None, node returns empty chunks."""
    from iw_rag.agent.nodes import make_long_context_node
    from iw_rag.agent.state import GraphState

    lc_route = _make_fake_long_context_route(eligible=False)
    node = make_long_context_node(lc_route, spine=None, assembler=None)

    state: GraphState = {  # type: ignore[typeddict-item]
        "question": "Not a wholedoc question",
        "original_question": "Not a wholedoc question",
        "access_context": _PUBLIC_CTX,
        "query_type": "factual",
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "diagnostics": {},
    }

    result = await node(state)

    assert result["chunks"] == []
    assert result["graded"] == []
    assert result["verdicts"][0].value == "fallback"


@pytest.mark.asyncio
async def test_long_context_node_writes_both_chunks_and_graded() -> None:
    """Node writes identical chunks and graded on the long-context path.

    On the eligible long-context path both fields must be set to the same
    non-empty list so generate can consume graded without going through
    grade_documents (lld-long-context.md §4).
    """
    from iw_rag.agent.nodes import make_long_context_node
    from iw_rag.agent.state import GraphState

    lc_route = _make_fake_long_context_route(eligible=True)
    spine = _make_fake_spine()
    node = make_long_context_node(lc_route, spine, assembler=None)

    state: GraphState = {  # type: ignore[typeddict-item]
        "question": "Summarise",
        "original_question": "Summarise",
        "access_context": _PUBLIC_CTX,
        "query_type": "wholedoc",
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "diagnostics": {"candidate_doc_ids": [_DOC_A]},
    }

    result = await node(state)

    # Both must be set so generate can consume graded without grade_documents
    assert "chunks" in result
    assert "graded" in result
    # Exact value check: chunks and graded must be equal on the LC path
    assert result["chunks"] == result["graded"], (
        "On the long-context path chunks and graded must be identical lists"
    )
    # Both must be non-empty (the fake LC route returns one chunk)
    assert len(result["chunks"]) >= 1, (
        f"Expected at least one chunk on eligible long-context path, got {result['chunks']!r}"
    )
