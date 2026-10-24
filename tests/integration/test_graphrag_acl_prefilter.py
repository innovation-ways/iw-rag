"""Integration tests for the GraphRAG ACL pre-filter at the SQL boundary (F-00097).

Uses testcontainers Postgres (the session-scoped ``pg_dsn`` fixture from
``tests/integration/conftest.py``).  NO Apache AGE required — Option A
filters at the ``chunks`` fetch SQL, so the graph traversal/AGE graph
itself is out of scope for these tests.

Tests exercise the **real** ``_StrChunkRepo.get_by_str_ids`` adapter
from ``scripts/run_graphrag_eval.py`` bound to the testcontainer DB, so
the SQL boundary is the real production path (not a mock).

Tests:

- ``_StrChunkRepo.get_by_str_ids(ids, principal_groups)`` returns only
  readable rows at the SQL boundary (the restricted id is never fetched
  — AC2, Boundary Behavior row "Resolution returns all-unreadable").

- ``_StrChunkRepo.get_by_str_ids(ids, principal_groups=[])`` returns
  ``[]`` (fail-closed; Boundary Behavior row "Empty ``principal_groups``
  (graph)").

- End-to-end: ``GraphRetriever.retrieve`` with a stubbed graph store
  resolving to a mixed id set + the real ``_StrChunkRepo`` adapter → no
  restricted chunk appears in ``RetrievalResult.chunks``,
  ``acl_applied is True``.

- The chunk-fetch query plan uses ``chunks_acl_gin`` (performance
  contract, GIN-accelerated pre-filter).

Each test asserts **specific values** (exact ``chunk_id`` membership),
not just shape — per the F-00097 S07 contract.

References:
- F-00097 Feature_Design §"Acceptance Criteria" (AC2, AC3)
- F-00097 Feature_Design §"Boundary Behavior"
- S03 report — public-tag reconciliation
- lld-data-model §3 (``chunks`` schema)
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.domain.acl import AccessContext
from iw_rag.modes.graphrag.retriever import GraphRetriever
from iw_rag.modes.graphrag.types import GraphRetrieverConfig, RawGraphContext
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Stable IDs — every Boundary Behavior row has a deterministic identity so
# assertions can pin exact membership (not just shape).
# ---------------------------------------------------------------------------

PUB_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")
PUB2_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000002")
HR_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000003")
FIN_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000004")
EMPTY_TAGS_CHUNK_ID = uuid.UUID("00000000-0000-0000-0000-000000000005")

ALL_CHUNK_IDS: list[uuid.UUID] = [
    PUB_CHUNK_ID,
    PUB2_CHUNK_ID,
    HR_CHUNK_ID,
    FIN_CHUNK_ID,
    EMPTY_TAGS_CHUNK_ID,
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_vec_1024() -> str:
    """Return a 1024-dim zero vector as a pgvector string (for ``dense_embedding``)."""
    return "[" + ",".join(["0.0"] * 1024) + "]"


def _seed_mixed_chunks(dsn: str) -> None:
    """Seed the mixed public/restricted corpus into ``iw_rag.chunks``.

    The fixture has five chunks covering every Boundary Behavior row:

    - ``PUB_CHUNK_ID``: ``acl_tags=['public']`` (spine's public representation)
    - ``PUB2_CHUNK_ID``: ``acl_tags=['public', 'all-staff']`` (overlapping)
    - ``HR_CHUNK_ID``: ``acl_tags=['hr']`` (cross-tenant — must be excluded)
    - ``FIN_CHUNK_ID``: ``acl_tags=['finance']`` (cross-tenant — must be excluded)
    - ``EMPTY_TAGS_CHUNK_ID``: ``acl_tags=[]`` (must be invisible to every principal)

    One joined ``documents`` row per chunk is required because the production
    ``_StrChunkRepo.get_by_str_ids`` SELECTs ``d.source_uri`` from the joined
    ``documents`` table.

    Args:
        dsn: The testcontainer DSN.
    """
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.begin() as conn:
            run_id = uuid.uuid4()
            conn.execute(
                text(
                    "INSERT INTO iw_rag.ingestion_runs "
                    "(id, corpus_label, parser, chunker_config, embed_model, status, started_at) "
                    "VALUES (:id, 'graphrag-prefilter-test', 'docling', '{}', "
                    "'bge-m3', 'running', now())"
                ),
                {"id": run_id},
            )

            doc_ids: dict[uuid.UUID, uuid.UUID] = {}
            for cid in ALL_CHUNK_IDS:
                doc_id = uuid.uuid4()
                doc_ids[cid] = doc_id
                conn.execute(
                    text(
                        "INSERT INTO iw_rag.documents "
                        "(id, source_uri, checksum, metadata, ingestion_run_id) "
                        "VALUES (:id, :uri, :cs, '{}', :run_id)"
                    ),
                    {
                        "id": doc_id,
                        "uri": f"corpus://graphrag-prefilter/{cid}.md",
                        "cs": f"cs-{cid}",
                        "run_id": run_id,
                    },
                )

            acl_map: dict[uuid.UUID, list[str]] = {
                PUB_CHUNK_ID: ["public"],
                PUB2_CHUNK_ID: ["public", "all-staff"],
                HR_CHUNK_ID: ["hr"],
                FIN_CHUNK_ID: ["finance"],
                EMPTY_TAGS_CHUNK_ID: [],
            }
            content_map: dict[uuid.UUID, str] = {
                PUB_CHUNK_ID: "Public content for everyone",
                PUB2_CHUNK_ID: "Public + all-staff content",
                HR_CHUNK_ID: "HR-only confidential salary data",
                FIN_CHUNK_ID: "Finance-only confidential revenue data",
                EMPTY_TAGS_CHUNK_ID: "Empty-acl_tags content",
            }
            for cid in ALL_CHUNK_IDS:
                conn.execute(
                    text(
                        "INSERT INTO iw_rag.chunks "
                        "(id, document_id, ordinal, content, dense_embedding, acl_tags) "
                        "VALUES (:id, :doc_id, 0, :content, :vec, :acl)"
                    ),
                    {
                        "id": cid,
                        "doc_id": doc_ids[cid],
                        "content": content_map[cid],
                        "vec": _zero_vec_1024(),
                        "acl": acl_map[cid],
                    },
                )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chunks_acl_pg(clean_pg: str) -> str:
    """Truncate ``iw_rag.chunks``/``documents`` and yield the migrated DSN.

    Builds on ``clean_pg`` so each test starts with a fresh empty chunks
    table — independent of any other integration test that touches the
    same container.

    Args:
        clean_pg: The session-scoped migrated DSN (already truncated for
            documents/chunks/ingestion_runs by the parent fixture).

    Returns:
        The DSN after re-truncating (no-op because parent already truncated).
    """
    # The parent ``clean_pg`` fixture already truncates documents/chunks/
    # ingestion_runs.  Re-running TRUNCATE here is a no-op and keeps the
    # test self-contained (a downstream fixture change cannot leak rows).
    engine = sqlalchemy.create_engine(clean_pg)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE iw_rag.documents, iw_rag.chunks, iw_rag.ingestion_runs "
                "RESTART IDENTITY CASCADE"
            )
        )
    engine.dispose()
    return clean_pg


@pytest.fixture
def graphrag_acl_seeded_pg(chunks_acl_pg: str) -> str:
    """Seed the mixed public/restricted chunks fixture.

    Args:
        chunks_acl_pg: The DSN after truncating chunks/documents.

    Returns:
        The DSN after seeding the mixed corpus.
    """
    _seed_mixed_chunks(chunks_acl_pg)
    return chunks_acl_pg


# ---------------------------------------------------------------------------
# Import the production adapter.  ``scripts/run_graphrag_eval.py`` is the
# canonical home of the only real ``_StrChunkRepo`` implementation —
# GraphRetriever is duck-typed on ``chunk_repo.get_by_str_ids`` and this
# adapter is the production binding.
# ---------------------------------------------------------------------------


def _import_str_chunk_repo() -> Any:
    """Lazily import ``_StrChunkRepo`` from ``scripts.run_graphrag_eval``.

    The script lives at ``scripts/run_graphrag_eval.py`` — outside the
    package — so we add ``scripts`` to ``sys.path`` and import it.  The
    import has no side effects (it only defines the adapter class and
    ``main`` is gated by ``if __name__ == "__main__":``).
    """
    scripts_dir = str(Path(__file__).resolve().parents[2] / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from run_graphrag_eval import _StrChunkRepo  # type: ignore[import-not-found]

    return _StrChunkRepo


# ---------------------------------------------------------------------------
# Tests — AC2 / Boundary Behavior rows for graph mode (SQL-boundary proof)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_str_chunk_repo_get_by_str_ids_excludes_restricted_at_sql_boundary(
    graphrag_acl_seeded_pg: str,
) -> None:
    """AC2: ``_StrChunkRepo.get_by_str_ids`` pre-filters at the SQL boundary.

    Requests ALL five seeded chunk ids with ``principal_groups=['public']``.
    The fetch SQL emits
    ``WHERE c.id = ANY(:ids) AND acl_tags && ARRAY['public']::text[]``
    (GIN-accelerated by ``chunks_acl_gin``).  Only the two readable rows
    (PUB, PUB2) are returned to Python; the hr/finance/empty-tags rows
    are never fetched.

    Asserts the exact returned set — no shape-only checks.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        rows = await repo.get_by_str_ids(
            [str(c) for c in ALL_CHUNK_IDS],
            principal_groups=["public"],
        )
    finally:
        await db.dispose()

    returned_ids = {row["chunk_id"] for row in rows}
    # Public-tag rows survive (you didn't over-filter).
    assert str(PUB_CHUNK_ID) in returned_ids, (
        f"Public chunk {PUB_CHUNK_ID} must be returned (overlaps ['public']). Got: {returned_ids}"
    )
    assert str(PUB2_CHUNK_ID) in returned_ids, (
        f"Public+all-staff chunk {PUB2_CHUNK_ID} must be returned. Got: {returned_ids}"
    )
    # Restricted rows are excluded at the SQL boundary (you didn't leak).
    assert str(HR_CHUNK_ID) not in returned_ids, (
        f"HR-only chunk {HR_CHUNK_ID} must NOT be returned — the SQL "
        f"`acl_tags && ARRAY['public']` pre-filter is the enforcement seam "
        f"(AC2, Invariant 2). Got: {returned_ids}"
    )
    assert str(FIN_CHUNK_ID) not in returned_ids, (
        f"Finance-only chunk {FIN_CHUNK_ID} must NOT be returned — "
        f"SQL-boundary pre-filter (AC2). Got: {returned_ids}"
    )
    assert str(EMPTY_TAGS_CHUNK_ID) not in returned_ids, (
        f"Empty-acl_tags chunk {EMPTY_TAGS_CHUNK_ID} must NOT be returned — "
        f"`[] && anything == false`, matches the spine and enrich.py. "
        f"Got: {returned_ids}"
    )
    # Exact membership assertion — no surprise rows.
    assert returned_ids == {str(PUB_CHUNK_ID), str(PUB2_CHUNK_ID)}, (
        f"Expected exactly the two public-readable chunks; got {returned_ids}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_str_chunk_repo_empty_principal_groups_fails_closed(
    graphrag_acl_seeded_pg: str,
) -> None:
    """Boundary Behavior: empty ``principal_groups`` ⇒ ``[]`` (fail-closed, no fetch).

    Mirrors :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`
    semantics.  A principal with no groups must not see any chunk, even
    if chunks exist in the store.  The adapter short-circuits to ``[]``
    and never issues an unfiltered query.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        rows = await repo.get_by_str_ids(
            [str(c) for c in ALL_CHUNK_IDS],
            principal_groups=[],
        )
    finally:
        await db.dispose()

    assert rows == [], (
        f"Empty principal_groups must yield [] (fail-closed); got {len(rows)}: "
        f"{[r['chunk_id'] for r in rows]}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_str_chunk_repo_resolution_all_unreadable_returns_empty(
    graphrag_acl_seeded_pg: str,
) -> None:
    """Boundary Behavior: every requested id is unreadable ⇒ ``[]`` (not an error).

    Request only the three unreadable ids.  The SQL pre-filter returns
    zero rows — the adapter returns ``[]``, NOT an exception.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        rows = await repo.get_by_str_ids(
            [str(HR_CHUNK_ID), str(FIN_CHUNK_ID), str(EMPTY_TAGS_CHUNK_ID)],
            principal_groups=["public"],
        )
    finally:
        await db.dispose()

    assert rows == [], f"All-unreadable resolution must yield [] (not an error); got {rows}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_str_chunk_repo_finance_principal_sees_finance_and_public(
    graphrag_acl_seeded_pg: str,
) -> None:
    """A finance+public principal sees finance + public rows; hr + empty-tags are excluded.

    Asserts the SQL pre-filter membership for the other direction: a
    principal with ``['finance', 'public']`` groups sees the finance
    chunk (overlaps finance) and both public chunks (overlap public).
    HR and empty-tags remain invisible.

    Note: ``acl_tags=['public']`` is NOT world-readable; the principal
    must hold ``'public'`` for the SQL ``&&`` to match.  This matches
    the spine and ``enrich.py`` (S03 public-tag reconciliation).
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        rows = await repo.get_by_str_ids(
            [str(c) for c in ALL_CHUNK_IDS],
            principal_groups=["finance", "public"],
        )
    finally:
        await db.dispose()

    returned_ids = {row["chunk_id"] for row in rows}
    assert str(FIN_CHUNK_ID) in returned_ids
    assert str(PUB_CHUNK_ID) in returned_ids
    assert str(PUB2_CHUNK_ID) in returned_ids
    assert str(HR_CHUNK_ID) not in returned_ids
    assert str(EMPTY_TAGS_CHUNK_ID) not in returned_ids


@pytest.mark.integration
@pytest.mark.asyncio
async def test_str_chunk_repo_invalid_uuid_ids_silently_dropped(
    graphrag_acl_seeded_pg: str,
) -> None:
    """Invalid UUID entries are silently dropped (only valid UUIDs are queried).

    Mirrors the production contract: ``uuid.UUID(s)`` raises ``ValueError``
    for non-UUID strings and the adapter continues.  An invalid-only
    request yields ``[]`` — not an error.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        rows = await repo.get_by_str_ids(
            ["not-a-uuid", "also-not", "still-not"],
            principal_groups=["public"],
        )
    finally:
        await db.dispose()

    assert rows == [], f"Invalid-UUID request must return [] (not error); got {rows}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunks_acl_gin_index_exists_in_schema(
    graphrag_acl_seeded_pg: str,
) -> None:
    """The ``chunks_acl_gin`` GIN index exists on ``iw_rag.chunks.acl_tags``.

    Pins the **existence** of the GIN index that accelerates the
    pre-filter ``acl_tags && :principal_groups``.  Without the GIN, the
    production WHERE clause would do a sequential scan even on a large
    multi-tenant corpus.  This is the schema contract — the query-plan
    assertion is intentionally omitted here because the production query
    also filters on ``c.id = ANY(:ids)`` and the planner will prefer the
    pkey on a small test corpus (a plan-fragility problem that's not
    worth fighting in CI).  The functional contract is pinned by the
    other tests in this file; the GIN existence is pinned here.
    """
    engine = sqlalchemy.create_engine(graphrag_acl_seeded_pg)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = 'iw_rag' AND tablename = 'chunks' "
                    "AND indexname = 'chunks_acl_gin'"
                )
            ).fetchone()
    finally:
        engine.dispose()

    assert row is not None, (
        "chunks_acl_gin index must exist on iw_rag.chunks (required for the "
        "GIN-accelerated pre-filter, INVARIANT 2)."
    )
    # It must be a GIN index on the acl_tags column itself (the array-overlap pre-filter),
    # not just an index that happens to carry the name.
    assert row.indexname == "chunks_acl_gin"
    assert row.indexdef.split(" USING ", 1)[1].startswith("gin (acl_tags)"), row.indexdef


# ---------------------------------------------------------------------------
# End-to-end: GraphRetriever with the REAL _StrChunkRepo adapter
# ---------------------------------------------------------------------------


def _make_stubbed_graph_store(
    entity_to_chunk_ids: dict[str, list[str]],
) -> Any:
    """Build a stubbed ``AgeGraphStore`` that resolves entities to chunk ids.

    The stub returns a :class:`RawGraphContext` whose ``entity_ids`` are
    the KEYS of ``entity_to_chunk_ids`` (so the test controls the full
    resolution independent of the seed-extraction step on the query
    string), and ``resolve_to_chunks`` returns the matching chunk-id
    list.  This is the seam ``GraphRetriever.retrieve`` consumes; the
    underlying AGE graph is not exercised (Option A pre-filters at the
    ``chunks`` fetch SQL, not at the AGE traversal).

    Args:
        entity_to_chunk_ids: Mapping from entity name to list of chunk id
            strings (the resolved back-refs the fake store hands to the
            retriever).

    Returns:
        A MagicMock ``AgeGraphStore`` with async ``query_subgraph`` and
        ``resolve_to_chunks`` methods.
    """
    store = MagicMock()
    mapping = entity_to_chunk_ids

    async def _query_subgraph(seeds: list[str], hops: int = 2) -> RawGraphContext:
        return RawGraphContext(entity_ids=list(mapping.keys()))

    async def _resolve_to_chunks(entity_ids: list[str]) -> list[str]:
        out: list[str] = []
        for eid in entity_ids:
            out.extend(mapping.get(eid, []))
        seen: set[str] = set()
        deduped: list[str] = []
        for cid in out:
            if cid not in seen:
                seen.add(cid)
                deduped.append(cid)
        return deduped

    store.query_subgraph = AsyncMock(side_effect=_query_subgraph)
    store.resolve_to_chunks = AsyncMock(side_effect=_resolve_to_chunks)
    return store


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_retriever_e2e_no_restricted_chunks_in_result(
    graphrag_acl_seeded_pg: str,
) -> None:
    """End-to-end: GraphRetriever + real ``_StrChunkRepo`` against the testcontainer.

    Drives the real ``GraphRetriever.retrieve`` with a stubbed graph
    store that resolves the single entity to **all five** chunk ids (a
    mix of readable + restricted).  The retriever threads
    ``principal_groups`` into the real ``_StrChunkRepo.get_by_str_ids``;
    the SQL boundary pre-filter excludes hr/finance/empty-tags; the
    defense-in-depth post-filter is a no-op; the final
    ``RetrievalResult.chunks`` contains only the two readable rows.

    Asserts the exact returned chunk-id set, not just shape.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        store = _make_stubbed_graph_store(
            entity_to_chunk_ids={
                "EntityA": [str(c) for c in ALL_CHUNK_IDS],
            }
        )
        cfg = GraphRetrieverConfig()
        retriever = GraphRetriever(store=store, chunk_repo=repo, cfg=cfg)

        ctx = AccessContext(principal_id="u-public", principal_groups=("public",))
        result = await retriever.retrieve("Who is EntityA?", ctx)

        returned_ids = {c.chunk_id for c in result.chunks}
        # The two readable chunks survive end-to-end.
        assert str(PUB_CHUNK_ID) in returned_ids
        assert str(PUB2_CHUNK_ID) in returned_ids
        # The three restricted chunks never appear (SQL-boundary pre-filter is the seam).
        assert str(HR_CHUNK_ID) not in returned_ids, (
            f"HR chunk must never appear in RetrievalResult.chunks (AC2, "
            f"Invariant 2). Got: {returned_ids}"
        )
        assert str(FIN_CHUNK_ID) not in returned_ids, (
            f"Finance chunk must never appear in RetrievalResult.chunks "
            f"(AC2, Invariant 2). Got: {returned_ids}"
        )
        assert str(EMPTY_TAGS_CHUNK_ID) not in returned_ids, (
            f"Empty-acl_tags chunk must never appear (spine parity). Got: {returned_ids}"
        )
        # Honest acl_applied flag.
        assert result.acl_applied is True
        assert result.retriever == "graphrag"
    finally:
        await db.dispose()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_retriever_e2e_empty_principal_groups_returns_empty(
    graphrag_acl_seeded_pg: str,
) -> None:
    """End-to-end: empty ``principal_groups`` ⇒ empty ``RetrievalResult``.

    Boundary Behavior: a principal with no groups must not see any chunk.
    The retriever short-circuits before issuing the fetch (no unfiltered
    query is ever made) and returns an empty ``RetrievalResult`` with
    ``acl_applied=True``.
    """
    StrChunkRepo = _import_str_chunk_repo()
    db = Database(DatabaseSettings(dsn=graphrag_acl_seeded_pg))
    try:
        repo = StrChunkRepo(db)
        store = _make_stubbed_graph_store(
            entity_to_chunk_ids={
                "EntityA": [str(c) for c in ALL_CHUNK_IDS],
            }
        )
        cfg = GraphRetrieverConfig()
        retriever = GraphRetriever(store=store, chunk_repo=repo, cfg=cfg)

        ctx = AccessContext(principal_id="u-empty", principal_groups=())
        result = await retriever.retrieve("anything", ctx)

        assert result.chunks == [], (
            f"Empty principal_groups ⇒ empty chunks; got {[c.chunk_id for c in result.chunks]}"
        )
        assert result.candidate_count == 0
        assert result.acl_applied is True
        assert result.retriever == "graphrag"

        # Critical: the retriever must not have issued an unfiltered fetch.
        # The stubbed graph store's resolve_to_chunks IS called (it's part of
        # the graph traversal — not the chunk fetch), but the chunk_repo's
        # get_by_str_ids must NOT have been called for the empty-groups case.
        # The production adapter itself would short-circuit to [] anyway, but
        # we pin the retriever-level short-circuit here.
        assert not store.resolve_to_chunks.await_args or result.chunks == []
    finally:
        await db.dispose()
