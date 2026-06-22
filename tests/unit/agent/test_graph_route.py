"""Unit tests for the graph router branch (WI-029 / Task 4a steps 10–11).

Tests cover:
- route_after_router dispatches "graph" → "graph" node when modes.graphrag.enabled=True
  AND a graph retriever is wired.
- route_after_router collapses "graph" → "retrieve" when modes.graphrag.enabled=False
  (default, preserve the fallback behaviour).
- make_graph_retrieve_node returns correct chunks on success.
- make_graph_retrieve_node falls back to spine on GraphRouteTimeout.
- make_graph_retrieve_node falls back to spine on GraphStoreUnavailable.
- make_graph_retrieve_node falls back to spine on any unexpected error.
- The fallback verdict has value="fallback" when spine is invoked.

References:
- task-4a-graphrag-route §10–§11 (router branch wiring)
- lld-agentic-loop §3.1 (router graph branch)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], retriever)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.modes.graphrag.store import GraphRouteTimeout, GraphStoreUnavailable

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-graph-route",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str = "c1", content: str = "Test content") -> RetrievedChunk:
    """Build a minimal RetrievedChunk for test use.

    Args:
        chunk_id: Chunk identifier.
        content: Chunk text.

    Returns:
        A RetrievedChunk with retriever="graphrag", acl_applied=True.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        score=0.8,
        source_uri="file:///doc.md",
        retriever="graphrag",
        acl_applied=True,
    )


def _make_fake_graph_retriever(
    chunks: list[RetrievedChunk] | None = None,
    *,
    raise_exc: type[Exception] | None = None,
) -> Any:
    """Build a fake GraphRetriever.

    Args:
        chunks: Chunks to return on success (defaults to one test chunk).
        raise_exc: When set, retrieve() raises this exception type.

    Returns:
        A MagicMock GraphRetriever with async retrieve method.
    """
    retriever = MagicMock()
    effective_chunks = chunks if chunks is not None else [_make_chunk()]

    async def _retrieve(query: str, ctx: Any, **kwargs: Any) -> RetrievalResult:
        if raise_exc is not None:
            raise raise_exc("simulated error")
        return RetrievalResult(
            chunks=effective_chunks,
            retriever="graphrag",
            acl_applied=True,
            query=query,
            candidate_count=len(effective_chunks),
        )

    retriever.retrieve = AsyncMock(side_effect=_retrieve)
    return retriever


def _make_fake_spine(chunks: list[RetrievedChunk] | None = None) -> Any:
    """Build a fake spine retriever.

    Args:
        chunks: Chunks to return on success (defaults to one fallback chunk).

    Returns:
        A MagicMock spine with async retrieve method.
    """
    spine = MagicMock()
    effective_chunks = (
        chunks
        if chunks is not None
        else [
            RetrievedChunk(
                chunk_id="spine-c1",
                document_id="doc-spine-c1",
                content="Spine content",
                score=0.5,
                source_uri="file:///spine_doc.md",
                retriever="spine",
                acl_applied=True,
            )
        ]
    )

    async def _retrieve(query: str, ctx: Any) -> RetrievalResult:
        return RetrievalResult(
            chunks=effective_chunks,
            retriever="spine",
            acl_applied=True,
            query=query,
        )

    spine.retrieve = AsyncMock(side_effect=_retrieve)
    return spine


def _make_state(**kwargs: Any) -> Any:
    """Build a minimal GraphState dict.

    Args:
        **kwargs: Override default state fields.

    Returns:
        A dict suitable as GraphState.
    """
    state: dict[str, Any] = {
        "question": "Multi-hop test query?",
        "original_question": "Multi-hop test query?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "route": "graph",
    }
    state.update(kwargs)
    return state


# ---------------------------------------------------------------------------
# Tests: route_after_router — graph branch dispatch
# ---------------------------------------------------------------------------


def test_route_after_router_graph_disabled_collapses_to_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When modes.graphrag.enabled=False (default), "graph" route collapses to "retrieve"."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__MODES__GRAPHRAG__ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    try:
        from iw_rag.agent.edges import route_after_router

        state = _make_state(route="graph")
        result = route_after_router(state)
        assert result == "retrieve", (
            f"Disabled graphrag should collapse 'graph' → 'retrieve', got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__GRAPHRAG__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


def test_route_after_router_graph_enabled_dispatches_to_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When modes.graphrag.enabled=True, "graph" route dispatches to "graph" node."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__MODES__GRAPHRAG__ENABLED", "true")
    settings_mod.get_settings.cache_clear()
    try:
        from iw_rag.agent.edges import route_after_router

        state = _make_state(route="graph")
        result = route_after_router(state)
        assert result == "graph", (
            f"Enabled graphrag should dispatch 'graph' → 'graph', got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__GRAPHRAG__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


def test_route_after_router_non_graph_routes_unchanged() -> None:
    """Non-graph routes (retrieve, skip, visual) are not affected by graphrag changes."""
    from iw_rag.agent.edges import route_after_router

    # "retrieve" → "retrieve" (unchanged)
    state = _make_state(route="retrieve")
    assert route_after_router(state) == "retrieve"

    # "skip" → "answer_direct"
    state = _make_state(route="skip")
    assert route_after_router(state) == "answer_direct"


# ---------------------------------------------------------------------------
# Tests: make_graph_retrieve_node — node behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_retrieve_node_success_returns_chunks() -> None:
    """Graph retrieve node returns graph retriever chunks on success."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_chunks = [_make_chunk("g1", "Graph chunk")]
    graph_retriever = _make_fake_graph_retriever(chunks=graph_chunks)
    spine = _make_fake_spine()

    node = make_graph_retrieve_node(graph_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert "chunks" in result
    chunk_ids = [c.chunk_id for c in result["chunks"]]
    assert "g1" in chunk_ids

    # Verdict should be "graph" on success
    assert len(result["verdicts"]) == 1
    assert result["verdicts"][0].value == "graph"


@pytest.mark.asyncio
async def test_graph_retrieve_node_timeout_falls_back_to_spine() -> None:
    """GraphRouteTimeout → falls back to spine retriever; verdict value="fallback"."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_retriever = _make_fake_graph_retriever(raise_exc=GraphRouteTimeout)
    spine_chunks = [
        RetrievedChunk(
            chunk_id="spine-fallback",
            document_id="doc-spine-fallback",
            content="Spine fallback",
            score=0.5,
            source_uri="u1",
            retriever="spine",
            acl_applied=True,
        )
    ]
    spine = _make_fake_spine(chunks=spine_chunks)

    node = make_graph_retrieve_node(graph_retriever, spine)
    state = _make_state()

    result = await node(state)

    # Should have fallen back to spine chunks
    chunk_ids = [c.chunk_id for c in result["chunks"]]
    assert "spine-fallback" in chunk_ids

    # Verdict should indicate fallback
    assert result["verdicts"][0].value == "fallback"


@pytest.mark.asyncio
async def test_graph_retrieve_node_unavailable_falls_back_to_spine() -> None:
    """GraphStoreUnavailable → falls back to spine retriever; verdict value="fallback"."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_retriever = _make_fake_graph_retriever(raise_exc=GraphStoreUnavailable)
    spine = _make_fake_spine()

    node = make_graph_retrieve_node(graph_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert result["verdicts"][0].value == "fallback"
    assert len(result["chunks"]) > 0  # spine provided chunks


@pytest.mark.asyncio
async def test_graph_retrieve_node_unexpected_error_falls_back_to_spine() -> None:
    """Any unexpected error from graph retriever → falls back to spine (degrade, not 500)."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_retriever = _make_fake_graph_retriever(raise_exc=RuntimeError)
    spine = _make_fake_spine()

    node = make_graph_retrieve_node(graph_retriever, spine)
    state = _make_state()

    result = await node(state)

    # Must not raise; should return spine chunks
    assert "chunks" in result
    assert result["verdicts"][0].value == "fallback"


@pytest.mark.asyncio
async def test_graph_retrieve_node_spine_also_fails_returns_empty() -> None:
    """If both graph retriever and spine fail → return empty chunks (no 500)."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_retriever = _make_fake_graph_retriever(raise_exc=GraphRouteTimeout)

    # Spine also broken
    spine = MagicMock()

    async def _broken_spine(q: Any, c: Any) -> Any:
        raise RuntimeError("spine also broken")

    spine.retrieve = AsyncMock(side_effect=_broken_spine)

    node = make_graph_retrieve_node(graph_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert result["chunks"] == []
    assert result["verdicts"][0].value == "fallback"


@pytest.mark.asyncio
async def test_graph_retrieve_node_spine_none_returns_empty() -> None:
    """If spine is None → return empty chunks on graph retriever error (graceful)."""
    from iw_rag.agent.nodes import make_graph_retrieve_node

    graph_retriever = _make_fake_graph_retriever(raise_exc=GraphRouteTimeout)

    node = make_graph_retrieve_node(graph_retriever, None)
    state = _make_state()

    result = await node(state)

    assert result["chunks"] == []
    assert "verdicts" in result
