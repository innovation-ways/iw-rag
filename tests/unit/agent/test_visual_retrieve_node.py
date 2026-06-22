"""Unit tests for nodes.make_visual_retrieve_node and edges.route_after_router
for the visual branch (Phase-4 WI-031).

Tests cover:
- route_after_router returns "visual" when modes.visual.enabled=True and
  state["route"]="visual".
- route_after_router collapses "visual" to "retrieve" when
  modes.visual.enabled=False (default).
- Non-visual routes are not affected by the visual mode switch.
- make_visual_retrieve_node returns chunks from the visual retriever on success.
- The success verdict has kind="visual_retrieve" and value="visual".
- make_visual_retrieve_node falls back to spine on any retriever exception.
- The fallback verdict has value="fallback".
- make_visual_retrieve_node falls back to spine on VisualEncodeUnavailable.
- make_visual_retrieve_node returns empty chunks when both retriever and spine fail.
- make_visual_retrieve_node returns empty chunks when spine is None and
  retriever fails.
- Exactly one Verdict is appended per node invocation.

References:
- task-4b-visual-rag-route §10 (visual router branch wiring)
- lld-multimodal §5 (degrade, never block)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], retriever)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-visual-test",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_visual_chunk(chunk_id: str = "v1") -> RetrievedChunk:
    """Build a visual-origin RetrievedChunk.

    Args:
        chunk_id: Chunk identifier.

    Returns:
        A RetrievedChunk with retriever="visual".
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content="Visual page content",
        score=0.85,
        source_uri=f"file:///doc-{chunk_id}.pdf",
        retriever="visual",
        acl_applied=True,
    )


def _make_spine_chunk(chunk_id: str = "spine-1") -> RetrievedChunk:
    """Build a spine-fallback RetrievedChunk.

    Args:
        chunk_id: Chunk identifier.

    Returns:
        A RetrievedChunk with retriever="spine".
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content="Spine fallback content",
        score=0.6,
        source_uri=f"file:///spine-{chunk_id}.md",
        retriever="spine",
        acl_applied=True,
    )


def _make_fake_visual_retriever(
    chunks: list[RetrievedChunk] | None = None,
    *,
    raise_exc: type[Exception] | None = None,
) -> Any:
    """Build a fake VisualRetriever.

    Args:
        chunks: Chunks to return on success (defaults to one visual chunk).
        raise_exc: When set, retrieve() raises this exception type.

    Returns:
        A MagicMock VisualRetriever with async retrieve method.
    """
    retriever = MagicMock()
    effective_chunks = chunks if chunks is not None else [_make_visual_chunk()]

    async def _retrieve(query: str, ctx: Any) -> RetrievalResult:
        if raise_exc is not None:
            raise raise_exc("simulated visual error")
        return RetrievalResult(
            chunks=effective_chunks,
            retriever="visual",
            acl_applied=True,
            candidate_count=len(effective_chunks),
        )

    retriever.retrieve = AsyncMock(side_effect=_retrieve)
    return retriever


def _make_fake_spine(
    chunks: list[RetrievedChunk] | None = None,
    *,
    raise_exc: type[Exception] | None = None,
) -> Any:
    """Build a fake spine retriever.

    Args:
        chunks: Chunks to return on success (defaults to one spine chunk).
        raise_exc: When set, retrieve() raises this exception type.

    Returns:
        A MagicMock spine with async retrieve method.
    """
    spine = MagicMock()
    effective_chunks = chunks if chunks is not None else [_make_spine_chunk()]

    async def _retrieve(query: str, ctx: Any) -> RetrievalResult:
        if raise_exc is not None:
            raise raise_exc("simulated spine error")
        return RetrievalResult(
            chunks=effective_chunks,
            retriever="spine",
            acl_applied=True,
        )

    spine.retrieve = AsyncMock(side_effect=_retrieve)
    return spine


def _make_state(**kwargs: Any) -> dict[str, Any]:
    """Build a minimal GraphState dict for visual node tests.

    Args:
        **kwargs: Override default state fields.

    Returns:
        A dict suitable as GraphState.
    """
    state: dict[str, Any] = {
        "question": "What does the chart show?",
        "original_question": "What does the chart show?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "route": "visual",
    }
    state.update(kwargs)
    return state


# ---------------------------------------------------------------------------
# Tests: route_after_router — visual branch dispatch
# ---------------------------------------------------------------------------


def test_route_after_router_visual_disabled_collapses_to_retrieve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When modes.visual.enabled=False (default), 'visual' route collapses to 'retrieve'."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__MODES__VISUAL__ENABLED", "false")
    settings_mod.get_settings.cache_clear()
    try:
        from iw_rag.agent.edges import route_after_router

        state = _make_state(route="visual")
        result = route_after_router(state)
        assert result == "retrieve", (
            f"Disabled visual mode must collapse 'visual' → 'retrieve', got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__VISUAL__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


def test_route_after_router_visual_enabled_dispatches_to_visual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When modes.visual.enabled=True, 'visual' route dispatches to 'visual' node."""
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__MODES__VISUAL__ENABLED", "true")
    settings_mod.get_settings.cache_clear()
    try:
        from iw_rag.agent.edges import route_after_router

        state = _make_state(route="visual")
        result = route_after_router(state)
        assert result == "visual", (
            f"Enabled visual mode must dispatch 'visual' → 'visual', got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__VISUAL__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


def test_route_after_router_retrieve_unaffected_by_visual_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'retrieve' route is unchanged regardless of modes.visual.enabled."""
    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router

    monkeypatch.setenv("IW_RAG__MODES__VISUAL__ENABLED", "true")
    settings_mod.get_settings.cache_clear()
    try:
        state = _make_state(route="retrieve")
        result = route_after_router(state)
        assert result == "retrieve", (
            f"'retrieve' route must stay 'retrieve' regardless of visual mode, got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__VISUAL__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


def test_route_after_router_skip_route_unaffected_by_visual_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 'skip' route is unchanged regardless of modes.visual.enabled."""
    import iw_rag.settings as settings_mod
    from iw_rag.agent.edges import route_after_router

    monkeypatch.setenv("IW_RAG__MODES__VISUAL__ENABLED", "true")
    settings_mod.get_settings.cache_clear()
    try:
        state = _make_state(route="skip")
        result = route_after_router(state)
        assert result == "answer_direct", (
            f"'skip' route must dispatch to 'answer_direct', got {result!r}"
        )
    finally:
        monkeypatch.delenv("IW_RAG__MODES__VISUAL__ENABLED", raising=False)
        settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Tests: make_visual_retrieve_node — node behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_visual_retrieve_node_success_returns_visual_chunks() -> None:
    """On success, node returns visual retriever chunks with verdict value='visual'."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_chunks = [_make_visual_chunk("v1"), _make_visual_chunk("v2")]
    visual_retriever = _make_fake_visual_retriever(chunks=visual_chunks)
    spine = _make_fake_spine()

    node = make_visual_retrieve_node(visual_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert "chunks" in result, "Result must contain 'chunks' key"
    chunk_ids = [c.chunk_id for c in result["chunks"]]
    assert chunk_ids == ["v1", "v2"], f"Chunk IDs mismatch: {chunk_ids!r}"


@pytest.mark.asyncio
async def test_visual_retrieve_node_success_verdict_kind_and_value() -> None:
    """Success path appends exactly one Verdict with kind='visual_retrieve', value='visual'."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever()
    spine = _make_fake_spine()
    node = make_visual_retrieve_node(visual_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert len(result["verdicts"]) == 1, (
        f"Expected exactly one verdict, got {len(result['verdicts'])}"
    )
    v = result["verdicts"][0]
    assert v.kind == "visual_retrieve", f"Verdict kind={v.kind!r}"
    assert v.value == "visual", f"Verdict value={v.value!r}"


@pytest.mark.asyncio
async def test_visual_retrieve_node_encoder_error_falls_back_to_spine() -> None:
    """VisualEncodeUnavailable → falls back to spine; verdict value='fallback'."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever(raise_exc=RuntimeError)
    spine_chunks = [_make_spine_chunk("spine-fb")]
    spine = _make_fake_spine(chunks=spine_chunks)

    node = make_visual_retrieve_node(visual_retriever, spine)
    state = _make_state()

    result = await node(state)

    chunk_ids = [c.chunk_id for c in result["chunks"]]
    assert chunk_ids == ["spine-fb"], f"Fallback must use spine chunks: {chunk_ids!r}"
    assert result["verdicts"][0].value == "fallback", (
        f"Fallback verdict value={result['verdicts'][0].value!r}"
    )


@pytest.mark.asyncio
async def test_visual_retrieve_node_fallback_verdict_kind_correct() -> None:
    """Fallback verdict kind is still 'visual_retrieve' (not 'spine')."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever(raise_exc=ValueError)
    spine = _make_fake_spine()
    node = make_visual_retrieve_node(visual_retriever, spine)
    state = _make_state()

    result = await node(state)

    v = result["verdicts"][0]
    assert v.kind == "visual_retrieve", (
        f"Fallback verdict kind must be 'visual_retrieve', got {v.kind!r}"
    )
    assert v.value == "fallback", f"Fallback verdict value={v.value!r}"


@pytest.mark.asyncio
async def test_visual_retrieve_node_both_fail_returns_empty_chunks() -> None:
    """If both visual retriever and spine fail → return empty chunks (no 500)."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever(raise_exc=RuntimeError)
    spine = _make_fake_spine(raise_exc=RuntimeError)

    node = make_visual_retrieve_node(visual_retriever, spine)
    state = _make_state()

    result = await node(state)

    assert result["chunks"] == [], (
        f"Double-failure must return empty chunks, got {result['chunks']!r}"
    )
    assert result["verdicts"][0].value == "fallback", (
        f"Double-failure verdict must be 'fallback', got {result['verdicts'][0].value!r}"
    )


@pytest.mark.asyncio
async def test_visual_retrieve_node_spine_none_returns_empty_on_failure() -> None:
    """When spine=None and visual retriever fails → return empty chunks (graceful)."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever(raise_exc=RuntimeError)
    node = make_visual_retrieve_node(visual_retriever, spine=None)
    state = _make_state()

    result = await node(state)

    assert result["chunks"] == [], (
        f"spine=None failure must return empty chunks, got {result['chunks']!r}"
    )
    assert result["verdicts"][0].value == "fallback", (
        f"spine=None failure verdict must be 'fallback', got {result['verdicts'][0].value!r}"
    )


@pytest.mark.asyncio
async def test_visual_retrieve_node_appends_exactly_one_verdict() -> None:
    """The node always appends exactly one Verdict regardless of success or failure."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    for raise_exc in (None, RuntimeError):
        visual_retriever = _make_fake_visual_retriever(raise_exc=raise_exc)
        spine = _make_fake_spine()
        node = make_visual_retrieve_node(visual_retriever, spine)
        state = _make_state()

        result = await node(state)

        assert len(result["verdicts"]) == 1, (
            f"Expected exactly one verdict (raise_exc={raise_exc}), got {len(result['verdicts'])}"
        )


@pytest.mark.asyncio
async def test_visual_retrieve_node_writes_chunks_key() -> None:
    """The node always writes the 'chunks' key (even when empty)."""
    from iw_rag.agent.nodes import make_visual_retrieve_node

    visual_retriever = _make_fake_visual_retriever(raise_exc=RuntimeError)
    node = make_visual_retrieve_node(visual_retriever, spine=None)
    state = _make_state()

    result = await node(state)

    assert "chunks" in result, "Result must always contain 'chunks' key"
    assert result["chunks"] == [], f"Empty-fallback chunks must be [], got {result['chunks']!r}"
