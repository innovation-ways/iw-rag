"""Unit tests for GraphRetriever — TDD RED-first (WI-029 / Task 4a steps 7–11).

Tests cover:
- RetrievalResult adapter: graph chunk → RetrievedChunk shape, ordering by score,
  citation mapping (chunk-grounded; doc-level fallback when unresolved).
- ACL post-filter predicate: a chunk whose acl_tags don't overlap principal_groups
  is dropped; empty acl_tags → public (always allow); empty principal_groups +
  restricted tags → deny (fail-closed).
- Default mode = MIX: no explicit mode kwarg uses GraphQueryMode.MIX.
- Error handling: GraphRouteTimeout → degrade (empty RetrievalResult, never 500).
- Error handling: GraphStoreUnavailable → degrade (empty RetrievalResult, never 500).
- Error handling: empty entity_ids (graph not indexed) → empty RetrievalResult.
- Contract: retrieve() returns structurally valid RetrievalResult with
  retriever=="graphrag" and acl_applied=True.

All tests use fakes — no live models, no real Postgres, no AGE required.

References:
- task-4a-graphrag-route §7–§9 (retriever build)
- lld-graphrag §3 (GraphRetriever interface)
- lld-retrieval-spine §4 (RetrievalResult contract)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], retriever, acl_applied)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.modes.graphrag.store import GraphRouteTimeout, GraphStoreUnavailable
from iw_rag.modes.graphrag.types import GraphQueryMode, GraphRetrieverConfig, RawGraphContext

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

_CTX_PUBLIC = AccessContext(
    principal_id="user-public",
    principal_groups=("public",),
    is_operator=False,
    request_id="req-pub",
)

_CTX_FINANCE = AccessContext(
    principal_id="user-finance",
    principal_groups=("finance", "public"),
    is_operator=False,
    request_id="req-fin",
)

_CTX_HR = AccessContext(
    principal_id="user-hr",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-hr",
)


def _make_raw_graph_context(entity_ids: list[str]) -> RawGraphContext:
    """Build a fake RawGraphContext with the given entity id list.

    Args:
        entity_ids: Entity names to include in the context.

    Returns:
        A RawGraphContext with the given entity_ids (entities/relations empty).
    """
    return RawGraphContext(entity_ids=entity_ids)


def _make_fake_store(
    *,
    entity_ids_to_chunk_ids: dict[str, list[str]] | None = None,
    raise_on_query: type[Exception] | None = None,
    raise_on_resolve: type[Exception] | None = None,
) -> Any:
    """Build a fake AgeGraphStore with controllable responses.

    The fake ``query_subgraph`` returns entity_ids that are the KEYS of
    ``entity_ids_to_chunk_ids`` (not the seeds from the query text) so that
    ``resolve_to_chunks`` can look them up deterministically regardless of what
    the seed-extraction step produces from the query string.

    Args:
        entity_ids_to_chunk_ids: Mapping from entity name to list of chunk_id
            strings.  When an entity name is looked up, its chunk_ids are returned.
            The keys of this mapping are also returned as entity_ids by
            query_subgraph so tests can control the full traversal output.
        raise_on_query: When set, query_subgraph raises this exception type.
        raise_on_resolve: When set, resolve_to_chunks raises this exception type.

    Returns:
        A MagicMock AgeGraphStore with async query_subgraph / resolve_to_chunks.
    """
    store = MagicMock()
    mapping = entity_ids_to_chunk_ids or {}

    async def _query_subgraph(seeds: list[str], hops: int = 2) -> RawGraphContext:
        if raise_on_query is not None:
            raise raise_on_query("simulated error")
        # Return the keys of the mapping as entity_ids (not the seeds), so
        # resolve_to_chunks can find them.  This allows tests to control the
        # full retrieval regardless of what seeds the query produces.
        entity_ids = list(mapping.keys()) if mapping else []
        return _make_raw_graph_context(entity_ids)

    async def _resolve_to_chunks(entity_ids: list[str]) -> list[str]:
        if raise_on_resolve is not None:
            raise raise_on_resolve("simulated error")
        result: list[str] = []
        for eid in entity_ids:
            result.extend(mapping.get(eid, []))
        # Deduplicate preserving order
        seen: set[str] = set()
        deduped = []
        for cid in result:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        return deduped

    store.query_subgraph = AsyncMock(side_effect=_query_subgraph)
    store.resolve_to_chunks = AsyncMock(side_effect=_resolve_to_chunks)
    return store


def _make_fake_chunk_repo(chunk_rows: list[dict[str, Any]]) -> Any:
    """Build a fake ChunkRepository that returns pre-defined chunk rows.

    Args:
        chunk_rows: List of dicts with keys: chunk_id, content, source_uri,
            acl_tags, metadata (optional).

    Returns:
        A MagicMock ChunkRepository with async get_by_str_ids.
    """
    by_id = {row["chunk_id"]: row for row in chunk_rows}

    async def _get(chunk_ids: list[str]) -> list[dict[str, Any]]:
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    repo = MagicMock()
    repo.get_by_str_ids = AsyncMock(side_effect=_get)
    return repo


def _make_retriever(
    store: Any,
    chunk_repo: Any,
    cfg: GraphRetrieverConfig | None = None,
) -> Any:
    """Import and construct a GraphRetriever with fakes.

    Args:
        store: Fake AgeGraphStore.
        chunk_repo: Fake ChunkRepository (with get_by_str_ids).
        cfg: Optional retriever config.

    Returns:
        A GraphRetriever instance.
    """
    from iw_rag.modes.graphrag.retriever import GraphRetriever

    return GraphRetriever(store=store, chunk_repo=chunk_repo, cfg=cfg or GraphRetrieverConfig())


# ---------------------------------------------------------------------------
# Tests: ACL post-filter predicate
# ---------------------------------------------------------------------------


def test_acl_postfilter_empty_tags_is_public_allow() -> None:
    """Empty acl_tags means publicly readable — always allow (any principal)."""
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    assert _acl_postfilter([], ["public"]) is True
    assert _acl_postfilter([], []) is True  # even empty principal → public chunk allowed


def test_acl_postfilter_overlap_allows() -> None:
    """Chunk tags ∩ principal_groups non-empty → allow."""
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    assert _acl_postfilter(["finance", "hr"], ["finance"]) is True
    assert _acl_postfilter(["hr"], ["hr", "finance"]) is True


def test_acl_postfilter_no_overlap_denies() -> None:
    """Chunk tags ∩ principal_groups empty + tags non-empty → deny (fail-closed)."""
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    assert _acl_postfilter(["hr"], ["finance"]) is False
    assert _acl_postfilter(["confidential"], ["public"]) is False


def test_acl_postfilter_empty_principal_restricted_chunk_denies() -> None:
    """Empty principal_groups with restricted acl_tags → deny (fail-closed)."""
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    assert _acl_postfilter(["finance"], []) is False


# ---------------------------------------------------------------------------
# Tests: RetrievalResult adapter / _adapt_graph_chunk_to_retrieved_chunk
# ---------------------------------------------------------------------------


def test_adapt_chunk_shape() -> None:
    """Adapted chunk has correct chunk_id, content, score, source_uri, retriever, acl_applied."""
    from iw_rag.modes.graphrag.retriever import _adapt_graph_chunk

    row = {
        "chunk_id": "c-abc",
        "content": "Hello world",
        "source_uri": "file:///doc.md",
        "acl_tags": [],
        "metadata": {"section": "intro"},
    }
    chunk = _adapt_graph_chunk(row, score=0.75)
    assert isinstance(chunk, RetrievedChunk)
    assert chunk.chunk_id == "c-abc"
    assert chunk.content == "Hello world"
    assert chunk.score == pytest.approx(0.75)
    assert chunk.source_uri == "file:///doc.md"
    assert chunk.retriever == "graphrag"
    assert chunk.acl_applied is True


def test_adapt_chunk_doc_level_fallback_flagged_in_metadata() -> None:
    """When chunk_id is missing / empty → doc-level citation flagged in metadata."""
    from iw_rag.modes.graphrag.retriever import _adapt_graph_chunk

    row = {
        "chunk_id": "",  # unresolved → doc-level fallback
        "content": "Entity context",
        "source_uri": "file:///doc.md",
        "acl_tags": [],
        "metadata": {},
    }
    chunk = _adapt_graph_chunk(row, score=0.5)
    # Doc-level fallback should be flagged
    assert chunk.metadata.get("doc_level_citation") is True


def test_adapt_chunk_ordering_by_score() -> None:
    """Multiple adapted chunks ordered by score descending matches rank order."""
    from iw_rag.modes.graphrag.retriever import _adapt_graph_chunk

    rows = [
        {"chunk_id": "c1", "content": "A", "source_uri": "u1", "acl_tags": [], "metadata": {}},
        {"chunk_id": "c2", "content": "B", "source_uri": "u2", "acl_tags": [], "metadata": {}},
        {"chunk_id": "c3", "content": "C", "source_uri": "u3", "acl_tags": [], "metadata": {}},
    ]
    scores = [0.3, 0.9, 0.6]
    adapted = [_adapt_graph_chunk(r, score=s) for r, s in zip(rows, scores, strict=True)]
    sorted_chunks = sorted(adapted, key=lambda c: c.score, reverse=True)
    assert sorted_chunks[0].chunk_id == "c2"
    assert sorted_chunks[1].chunk_id == "c3"
    assert sorted_chunks[2].chunk_id == "c1"


# ---------------------------------------------------------------------------
# Tests: GraphRetriever.retrieve — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_returns_valid_retrieval_result_contract() -> None:
    """retrieve() returns a structurally valid RetrievalResult (contract test).

    Checks: type, retriever=="graphrag", acl_applied=True, chunks is a list.
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c1"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c1",
                "content": "Alpha text",
                "source_uri": "file:///doc.md",
                "acl_tags": [],
                "metadata": {},
            }
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("Who is EntityA?", _CTX_PUBLIC)

    assert isinstance(result, RetrievalResult)
    assert result.retriever == "graphrag"
    assert result.acl_applied is True
    assert isinstance(result.chunks, list)


@pytest.mark.asyncio
async def test_retrieve_default_mode_is_mix() -> None:
    """Default mode is MIX (GraphQueryMode.MIX) when none is provided."""
    from iw_rag.modes.graphrag.retriever import GraphRetriever

    store = _make_fake_store()
    chunk_repo = _make_fake_chunk_repo([])

    cfg = GraphRetrieverConfig(default_mode=GraphQueryMode.MIX)
    retriever = GraphRetriever(store=store, chunk_repo=chunk_repo, cfg=cfg)

    # Call retrieve with no mode kwarg
    result = await retriever.retrieve("test query", _CTX_PUBLIC)

    # The diagnostics should record the graph_mode used
    assert result.diagnostics.get("graph_mode") == "mix"


@pytest.mark.asyncio
async def test_retrieve_acl_drops_non_overlapping_chunks() -> None:
    """ACL post-filter drops chunks whose acl_tags don't overlap principal_groups."""
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c-public"], "EntityB": ["c-hr"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c-public",
                "content": "Public text",
                "source_uri": "u1",
                "acl_tags": [],
                "metadata": {},
            },
            {
                "chunk_id": "c-hr",
                "content": "HR text",
                "source_uri": "u2",
                "acl_tags": ["hr"],
                "metadata": {},
            },
        ]
    )
    # Access context: finance user (not in hr)
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("multi-hop query", _CTX_FINANCE)

    # Only the public chunk should survive (c-hr is HR-restricted)
    chunk_ids = {c.chunk_id for c in result.chunks}
    assert "c-public" in chunk_ids
    assert "c-hr" not in chunk_ids


@pytest.mark.asyncio
async def test_retrieve_acl_drops_all_returns_empty() -> None:
    """If ACL drops all chunks, return empty RetrievalResult (no error)."""
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c-restricted"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c-restricted",
                "content": "Restricted",
                "source_uri": "u1",
                "acl_tags": ["confidential"],
                "metadata": {},
            },
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_PUBLIC)

    assert result.chunks == []
    assert result.acl_applied is True
    assert result.retriever == "graphrag"


# ---------------------------------------------------------------------------
# Tests: GraphRetriever.retrieve — error / degrade paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_timeout_returns_empty_result() -> None:
    """GraphRouteTimeout during subgraph query → empty RetrievalResult (degrade, not 500)."""
    store = _make_fake_store(raise_on_query=GraphRouteTimeout)  # type: ignore[type-abstract]
    chunk_repo = _make_fake_chunk_repo([])
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_PUBLIC)

    assert isinstance(result, RetrievalResult)
    assert result.chunks == []
    assert result.retriever == "graphrag"
    assert result.acl_applied is True


@pytest.mark.asyncio
async def test_retrieve_graph_store_unavailable_returns_empty_result() -> None:
    """GraphStoreUnavailable → empty RetrievalResult (degrade, not 500)."""
    store = _make_fake_store(raise_on_query=GraphStoreUnavailable)  # type: ignore[type-abstract]
    chunk_repo = _make_fake_chunk_repo([])
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_PUBLIC)

    assert isinstance(result, RetrievalResult)
    assert result.chunks == []
    assert result.retriever == "graphrag"
    assert result.acl_applied is True


@pytest.mark.asyncio
async def test_retrieve_empty_entity_ids_returns_empty_result() -> None:
    """Graph not yet indexed (entity_ids empty from subgraph) → empty RetrievalResult."""
    store = _make_fake_store(entity_ids_to_chunk_ids={})  # no entities → empty subgraph
    chunk_repo = _make_fake_chunk_repo([])
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("multi-hop query", _CTX_PUBLIC)

    assert result.chunks == []
    assert result.retriever == "graphrag"
    assert result.acl_applied is True


@pytest.mark.asyncio
async def test_retrieve_resolve_unavailable_returns_empty_result() -> None:
    """GraphStoreUnavailable from resolve_to_chunks → empty RetrievalResult (degrade)."""
    store = _make_fake_store(raise_on_resolve=GraphStoreUnavailable)  # type: ignore[type-abstract]
    chunk_repo = _make_fake_chunk_repo([])
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_PUBLIC)

    assert isinstance(result, RetrievalResult)
    assert result.chunks == []


# ---------------------------------------------------------------------------
# Tests: candidate_count field
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_candidate_count_reflects_resolved_before_acl() -> None:
    """candidate_count reflects the number of resolved chunks before ACL filtering."""
    store = _make_fake_store(entity_ids_to_chunk_ids={"E": ["c1", "c2", "c3"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {"chunk_id": "c1", "content": "A", "source_uri": "u1", "acl_tags": [], "metadata": {}},
            {
                "chunk_id": "c2",
                "content": "B",
                "source_uri": "u2",
                "acl_tags": ["hr"],
                "metadata": {},
            },
            {"chunk_id": "c3", "content": "C", "source_uri": "u3", "acl_tags": [], "metadata": {}},
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_FINANCE)  # no hr

    # candidate_count = 3 (before ACL); chunks = 2 (c2 dropped by ACL)
    assert result.candidate_count == 3
    assert len(result.chunks) == 2


# ---------------------------------------------------------------------------
# Contract test: retrieve output is accepted by the generator/eval pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_contract_retrieve_result_fields_match_retrieval_result_spec() -> None:
    """Contract test: GraphRetriever.retrieve produces a RetrievalResult identical
    in structure to the spine output — generator + grade_generation accept it
    unchanged (CONVENTIONS.md §1, lld-retrieval-spine §4).
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"Entity": ["cid-42"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "cid-42",
                "content": "The component handles the request.",
                "source_uri": "file:///component.md",
                "acl_tags": [],
                "metadata": {"section": "core"},
            }
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("How does the component work?", _CTX_PUBLIC)

    # Structural checks against the canonical RetrievalResult contract
    assert isinstance(result, RetrievalResult), "Must return RetrievalResult"
    assert result.retriever == "graphrag", "retriever field must be 'graphrag'"
    assert result.acl_applied is True, "acl_applied must be True"
    assert isinstance(result.chunks, list), "chunks must be a list"
    assert result.query == "How does the component work?", "query must be preserved"

    if result.chunks:
        chunk = result.chunks[0]
        assert isinstance(chunk, RetrievedChunk), "each item must be RetrievedChunk"
        assert chunk.retriever == "graphrag", "chunk.retriever must be 'graphrag'"
        assert chunk.acl_applied is True, "chunk.acl_applied must be True"
        assert isinstance(chunk.chunk_id, str), "chunk_id must be str"
        assert isinstance(chunk.content, str), "content must be str"
        assert isinstance(chunk.score, float), "score must be float"
        assert isinstance(chunk.source_uri, str), "source_uri must be str"
