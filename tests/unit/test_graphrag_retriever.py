"""Unit tests for GraphRetriever — TDD RED-first (WI-029 / Task 4a steps 7–11).

Tests cover:
- RetrievalResult adapter: graph chunk → RetrievedChunk shape, ordering by score,
  citation mapping (chunk-grounded; doc-level fallback when unresolved).
- ACL pre-filter wiring: ``GraphRetriever.retrieve`` threads ``principal_groups``
  into ``chunk_repo.get_by_str_ids`` so the fetch SQL filters via
  ``acl_tags && :principal_groups`` (F-00097, Option A).  An empty
  ``principal_groups`` is fail-closed (the fetch returns ``[]``); a chunk with
  ``acl_tags = []`` is invisible to every principal (matches the spine's
  ``&&`` semantics, matches ``enrich.py`` ingestion contract).
- ACL post-filter predicate (defense-in-depth): a chunk whose ``acl_tags``
  do not overlap ``principal_groups`` is dropped; both empty
  ``chunk_acl_tags`` and empty ``principal_groups`` are fail-closed
  (matches the spine's pre-filter); on correctly pre-filtered data the
  post-filter is a no-op (drops zero items) and a pre-filter defect emits a
  ``logging.WARNING``.
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
- F-00097 (Graph/Visual ACL pre-filter — pre-filter wiring + post-filter
  defense-in-depth)
"""

from __future__ import annotations

import logging
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

    The fake mirrors the production ``_StrChunkRepo.get_by_str_ids`` contract
    (F-00097): it accepts an optional ``principal_groups`` keyword argument
    and **applies the same SQL pre-filter** at the seam, so restricted chunks
    are never returned to Python.  When ``principal_groups`` is omitted, the
    fake degrades to returning all rows (back-compat for tests that don't
    care about ACL).

    Args:
        chunk_rows: List of dicts with keys: chunk_id, content, source_uri,
            acl_tags, metadata (optional).

    Returns:
        A MagicMock ChunkRepository with async get_by_str_ids.
    """
    by_id = {row["chunk_id"]: row for row in chunk_rows}

    async def _get(
        chunk_ids: list[str],
        principal_groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        # Simulate the production adapter: when principal_groups is provided,
        # filter rows by `acl_tags && set(principal_groups)` (pre-filter at
        # the SQL boundary).  Empty principal_groups → fail-closed: [].
        if principal_groups is not None:
            pg_set = set(principal_groups)
            visible_rows = [
                by_id[cid]
                for cid in chunk_ids
                if cid in by_id and bool(set(by_id[cid].get("acl_tags") or []) & pg_set)
            ]
            return visible_rows
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
# Tests: ACL post-filter predicate (defense-in-depth, matches spine semantics)
# ---------------------------------------------------------------------------


def test_acl_postfilter_blocks_when_acl_tags_empty() -> None:
    """Empty acl_tags ⇒ INVISIBLE to every principal (fail-closed).

    After S03 the post-filter matches the spine's
    ``acl_tags && :principal_groups`` semantics and the ingestion contract
    in :mod:`iw_rag.ingestion.enrich` ("an empty resolver result yields
    ``acl_tags = []`` → the document is invisible to every principal; fail-
    closed, never world-readable").  The previous behavior — treat ``[]``
    as publicly readable — was a security defect that this step closes
    (F-00097, Boundary Behavior table; Invariant 5).  Broadly-readable
    documents carry an explicit shared tag (e.g. ``"public"``) that
    principals also hold.
    """
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    principal_groups = ["some-group"]
    chunk_acl_tags: list[str] = []
    assert _acl_postfilter(chunk_acl_tags, principal_groups) is False, (
        "Empty acl_tags must be invisible (deny) — matches spine semantics "
        "and enrich.py ingestion contract (fail-closed, never world-readable)."
    )


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


def test_acl_postfilter_blocks_when_both_empty() -> None:
    """Empty principal_groups AND empty chunk tags ⇒ deny (fail-closed).

    Both sides of the intersection empty: nothing matches; deny.  Matches
    the spine's ``[] && [] == false`` (F-00097, Invariant 5).
    """
    from iw_rag.modes.graphrag.retriever import _acl_postfilter

    assert _acl_postfilter([], []) is False


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
                "acl_tags": ["public"],
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
    """ACL pre-filter drops chunks whose acl_tags don't overlap principal_groups.

    After S03, the fetch SQL ``acl_tags && :principal_groups`` returns no
    restricted rows for the principal; the post-filter is a no-op.  This
    test exercises the contract end-to-end.
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c-public"], "EntityB": ["c-hr"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c-public",
                "content": "Public text",
                "source_uri": "u1",
                "acl_tags": ["public"],
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
async def test_retrieve_candidate_count_reflects_post_prefilter_rows() -> None:
    """candidate_count reflects the number of rows the pre-filter returned.

    After S03 the fetch SQL ``acl_tags && :principal_groups`` filters before
    the rows ever reach Python.  ``candidate_count`` therefore counts the
    **readable** rows the fetch returned (the c2 hr-restricted row is never
    fetched, so it is not counted).  The post-filter is a no-op on correctly
    pre-filtered data.
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"E": ["c1", "c2", "c3"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c1",
                "content": "A",
                "source_uri": "u1",
                "acl_tags": ["public"],
                "metadata": {},
            },
            {
                "chunk_id": "c2",
                "content": "B",
                "source_uri": "u2",
                "acl_tags": ["hr"],
                "metadata": {},
            },
            {
                "chunk_id": "c3",
                "content": "C",
                "source_uri": "u3",
                "acl_tags": ["public"],
                "metadata": {},
            },
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    result = await retriever.retrieve("query", _CTX_FINANCE)  # no hr

    # Pre-filter returns 2 rows (c1, c3 — public-readable; c2 is hr-restricted
    # and is never fetched).  candidate_count = 2, len(chunks) = 2.
    assert result.candidate_count == 2
    assert len(result.chunks) == 2
    chunk_ids = {c.chunk_id for c in result.chunks}
    assert chunk_ids == {"c1", "c3"}


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
                "acl_tags": ["public"],
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


# ---------------------------------------------------------------------------
# F-00097 — GraphRAG ACL pre-filter wiring (Option A: SQL-boundary pre-filter)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_passes_principal_groups_to_get_by_str_ids() -> None:
    """GraphRetriever MUST thread ``principal_groups`` into ``get_by_str_ids``.

    The graph pre-filter is enforced at the chunk-fetch SQL boundary via
    ``acl_tags && :principal_groups`` (matches the spine, GIN-accelerated by
    ``chunks_acl_gin``).  If the retriever does not thread
    ``principal_groups`` from the authenticated ``AccessContext`` into the
    ``chunk_repo.get_by_str_ids`` call, the fetch SQL runs unfiltered and the
    pre-filter invariant is violated (F-00097, Invariant 2; AC2).

    The ``principal_groups`` MUST come from the ``AccessContext`` — never
    from the query text or any request field (lld-guardrails §3.1).
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c1"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c1",
                "content": "A",
                "source_uri": "u1",
                "acl_tags": ["public"],
                "metadata": {},
            }
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    ctx = AccessContext(principal_id="user-pub", principal_groups=("public",))
    await retriever.retrieve("test", ctx)

    # The retriever must invoke the fetch with principal_groups so the SQL
    # pre-filter (`acl_tags && :principal_groups`) can run.
    chunk_repo.get_by_str_ids.assert_called_once()
    call = chunk_repo.get_by_str_ids.call_args
    assert "principal_groups" in call.kwargs, (
        "get_by_str_ids must receive principal_groups (from AccessContext) so "
        "the fetch SQL pre-filter runs. "
        f"Got kwargs: {list(call.kwargs.keys())}"
    )
    assert list(call.kwargs["principal_groups"]) == ["public"], (
        "principal_groups must come from AccessContext, not from query text. "
        f"Got: {call.kwargs['principal_groups']!r}"
    )


@pytest.mark.asyncio
async def test_retrieve_prefilter_excludes_restricted_chunks() -> None:
    """Restricted chunks must never appear in the result (pre-filter is the seam).

    The fake simulates the production ``_StrChunkRepo.get_by_str_ids``: when
    ``principal_groups`` is provided, the fake applies the same
    ``acl_tags && :principal_groups`` pre-filter, so the hr-restricted row
    is never returned to Python.  The retriever's post-filter is a no-op on
    the surviving rows (the pre-filter already excluded the restricted row).
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c-pub", "c-hr"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c-pub",
                "content": "Public text",
                "source_uri": "u1",
                "acl_tags": ["public"],
                "metadata": {},
            },
            {
                "chunk_id": "c-hr",
                "content": "HR-only text",
                "source_uri": "u2",
                "acl_tags": ["hr"],
                "metadata": {},
            },
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    # Principal holds only "public" — c-hr is filtered out by the SQL pre-filter.
    ctx = AccessContext(principal_id="u", principal_groups=("public",))
    result = await retriever.retrieve("test", ctx)

    chunk_ids = {c.chunk_id for c in result.chunks}
    assert "c-pub" in chunk_ids, "Public chunk must be present"
    assert "c-hr" not in chunk_ids, (
        "Hr-restricted chunk must never be returned — the fetch SQL's "
        "`acl_tags && :principal_groups` is the enforcement seam (AC2)."
    )


@pytest.mark.asyncio
async def test_retrieve_empty_principal_groups_fails_closed() -> None:
    """Empty principal_groups ⇒ no fetch happens ⇒ empty result (fail-closed).

    Mirrors :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`
    semantics.  A principal with no groups must not see any chunk, even if
    chunks exist in the store (F-00097, Boundary Behavior table).  The
    retriever short-circuits to an empty result without invoking
    ``get_by_str_ids`` (no unfiltered query is ever issued).
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"EntityA": ["c1", "c2"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c1",
                "content": "A",
                "source_uri": "u1",
                "acl_tags": ["public"],
                "metadata": {},
            },
            {
                "chunk_id": "c2",
                "content": "B",
                "source_uri": "u2",
                "acl_tags": ["public"],
                "metadata": {},
            },
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    ctx = AccessContext(principal_id="u", principal_groups=())
    result = await retriever.retrieve("test", ctx)

    assert result.chunks == [], (
        f"Empty principal_groups must yield empty chunks; got {len(result.chunks)}"
    )
    assert result.candidate_count == 0
    assert result.acl_applied is True

    # The retriever must not have issued an unfiltered fetch (fail-closed).
    (
        chunk_repo.get_by_str_ids.assert_not_called(),
        (
            "Empty principal_groups must short-circuit before get_by_str_ids — "
            "no unfiltered query should ever be issued (mirrors AclPredicate.is_empty)."
        ),
    )


@pytest.mark.asyncio
async def test_retrieve_logs_warning_when_postfilter_drops_chunk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Post-filter dropping a chunk ⇒ logging.WARNING (pre-filter defect signal).

    Simulates a pre-filter bug: the chunk_repo returns a row that the
    post-filter would drop (e.g. ``acl_tags=["hr"]`` for a public principal).
    The post-filter must drop it AND emit a ``logging.WARNING`` identifying
    the chunk id — this is the AC3 / Invariant 6 defense-in-depth contract.
    """
    # Force the post-filter to actually fire by giving the fake a row that
    # bypasses its pre-filter check.  We do this by making the fake return
    # all rows regardless of principal_groups (i.e. we drop the fake's
    # pre-filter simulation just for this test).
    by_id: dict[str, dict[str, Any]] = {
        "leak-chunk-id": {
            "chunk_id": "leak-chunk-id",
            "content": "should not have arrived here",
            "source_uri": "u1",
            "acl_tags": ["hr"],
            "metadata": {},
        }
    }

    async def _get_no_prefilter(
        chunk_ids: list[str],
        principal_groups: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    chunk_repo = MagicMock()
    chunk_repo.get_by_str_ids = AsyncMock(side_effect=_get_no_prefilter)

    store = _make_fake_store(entity_ids_to_chunk_ids={"E": ["leak-chunk-id"]})
    retriever = _make_retriever(store, chunk_repo)

    ctx = AccessContext(principal_id="u", principal_groups=("public",))

    with caplog.at_level(logging.WARNING, logger="iw_rag.modes.graphrag.retriever"):
        result = await retriever.retrieve("test", ctx)

    # The post-filter must have dropped the chunk (no leak).
    assert all(c.chunk_id != "leak-chunk-id" for c in result.chunks), (
        "Post-filter must drop the leaked chunk — no cross-tenant leak."
    )
    # And emitted a WARNING identifying the chunk id.
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_records, (
        "Post-filter dropping a chunk must emit a WARNING (pre-filter defect signal)."
    )
    assert any("leak-chunk-id" in r.getMessage() for r in warning_records), (
        f"WARNING must identify the dropped chunk id; got: "
        f"{[r.getMessage() for r in warning_records]}"
    )


@pytest.mark.asyncio
async def test_retrieve_postfilter_is_noop_on_prefiltered_data(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """On correctly pre-filtered data, the post-filter is a no-op (drops zero).

    AC3 / Invariant 6: the post-filter is a defense-in-depth secondary check
    that runs after the pre-filter.  When the pre-filter has already
    excluded unreadable rows, the post-filter should drop nothing and emit
    no WARNING.
    """
    store = _make_fake_store(entity_ids_to_chunk_ids={"E": ["c-pub"]})
    chunk_repo = _make_fake_chunk_repo(
        [
            {
                "chunk_id": "c-pub",
                "content": "Public text",
                "source_uri": "u1",
                "acl_tags": ["public"],
                "metadata": {},
            }
        ]
    )
    retriever = _make_retriever(store, chunk_repo)

    ctx = AccessContext(principal_id="u", principal_groups=("public",))

    with caplog.at_level(logging.WARNING, logger="iw_rag.modes.graphrag.retriever"):
        result = await retriever.retrieve("test", ctx)

    # The pre-filter passes c-pub; the post-filter must be a no-op.
    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_id == "c-pub"

    # No WARNING records on the post-filter path (pre-filter did its job).
    warning_records = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert not warning_records, (
        "Post-filter must be a no-op on correctly pre-filtered data; got "
        f"WARNING records: {[r.getMessage() for r in warning_records]}"
    )
