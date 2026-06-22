"""Integration tests for the visual late-interaction path (WI-031 / Task 4b).

Uses testcontainers Postgres (the ``clean_pg`` / ``pg_dsn`` fixtures from
``tests/integration/conftest.py``).  The ColVisionEncoder is mocked to emit
deterministic 128-dim multi-vectors so no GPU or encoder endpoint is needed.

Tests:
- Index 3 pages via VisualIndexer + PostgresCandidateStore; verify rows land in
  ``iw_rag.visual_pages``.
- Query via VisualRetriever.retrieve(...): assert the semantically closest page
  is top-1 and that the two-stage (stage-1 proxy prefetch → stage-2 MaxSim)
  path narrows correctly.
- An ACL-restricted page is dropped (ACL post-filter on the matched page's
  acl_tags) — no leak.
- The returned object is a structurally valid RetrievalResult with
  ``retriever=="visual"``, ``acl_applied=True``, and non-empty ``content``.

Proxy vector dimension: 128 (matches the ``vector(128)`` migration column and
``ColVisionEncoderConfig.dims = 128``).  The mock encoder produces 128-dim
vectors so the store upsert and stage-1 prefetch HNSW query are both valid.

ACL note: The visual path currently uses an ACL POST-filter on ``acl_tags``
returned by the store (see store.py ``stage1_prefetch`` — ``acl_tags`` is
included in the SELECT).  A page with tags not matching the requester's
``principal_groups`` is dropped after the MaxSim rerank (VisualRetriever step 4).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.modes.visual.encoder import ColVisionEncoder, ColVisionEncoderConfig
from iw_rag.modes.visual.indexer import VisualIndexer
from iw_rag.modes.visual.retriever import VisualRetriever
from iw_rag.modes.visual.store import PostgresCandidateStore
from iw_rag.modes.visual.types import VisualIndexConfig, VisualPageRecord, VisualRetrieverConfig
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The single encoder dimension used end-to-end (migration + encoder + retriever).
_DIMS = 128

#: Number of pooled vectors per page (canonical ≈38; use a smaller value in tests
#: for speed — the important invariant is the dim, not the count).
_POOLED_N = 4


# ---------------------------------------------------------------------------
# Fake encoder
# ---------------------------------------------------------------------------


def _make_deterministic_patch_vecs(seed: float, n: int = _POOLED_N) -> list[list[float]]:
    """Produce ``n`` deterministic 128-dim patch vectors based on a seed scalar.

    The first dimension equals ``seed``; all others are 0.0.  This makes
    MaxSim scoring predictable: a query vector with seed=X will score highest
    against a page with seed=X (maximising the first-dimension dot product).

    Args:
        seed: Scalar used as the first dimension value.
        n: Number of patch vectors to produce.

    Returns:
        A list of ``n`` vectors of length ``_DIMS``.
    """
    base = [0.0] * _DIMS
    base[0] = seed
    return [list(base) for _ in range(n)]


def _make_encoder_for_pages(
    page_seeds: dict[int, float],
) -> ColVisionEncoder:
    """Build a ColVisionEncoder backed by a fake client.

    ``encode_pages([image])`` returns deterministic 128-dim patch vectors
    whose first dimension is keyed by the page-number seed in ``page_seeds``
    (matched by call order).  ``encode_query(query)`` returns a single
    query vector with ``seed=1.0`` (matches the page-2 fixture which is
    designated as the top-1 page).

    Args:
        page_seeds: Mapping from call-order index → seed (call 0 = first page
            indexed, etc.).

    Returns:
        A ColVisionEncoder with a fake client.
    """
    call_count: list[int] = [0]

    async def _encode_pages(images: list[Any]) -> list[list[list[float]]]:
        """Return deterministic 128-dim patch vectors for each image."""
        idx = call_count[0]
        call_count[0] += 1
        seed = page_seeds.get(idx, 0.1)
        return [_make_deterministic_patch_vecs(seed)]

    async def _encode_query(query: str) -> list[list[float]]:
        """Return a query vector with seed=1.0 (matches the top-1 page fixture)."""
        # seed=1.0 → first dim=1.0 — maximises dot product with page seed=1.0
        vec = [0.0] * _DIMS
        vec[0] = 1.0
        return [vec]

    fake_client = MagicMock()
    fake_client.encode_pages = AsyncMock(side_effect=_encode_pages)
    fake_client.encode_query = AsyncMock(side_effect=_encode_query)

    cfg = ColVisionEncoderConfig(
        model_role="visual_retrieval",
        served_name="vidore/colqwen2.5-v0.2",
        endpoint="http://localhost:9999",
        dims=_DIMS,
    )
    return ColVisionEncoder(client=fake_client, cfg=cfg)


# ---------------------------------------------------------------------------
# Session factory helper (synchronous wrapper for the async session factory)
# ---------------------------------------------------------------------------


def _make_session_factory(dsn: str) -> Any:
    """Build an async session factory from a DSN.

    Args:
        dsn: The psycopg DSN for the test container.

    Returns:
        An async context-manager factory yielding an ``AsyncSession``.
    """
    db = Database(DatabaseSettings(dsn=dsn))
    return db.session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def visual_clean_pg(clean_pg: str) -> str:
    """Truncate iw_rag.visual_pages before each test (builds on clean_pg).

    The ``clean_pg`` fixture already truncates documents/chunks/ingestion_runs.
    We additionally truncate visual_pages so each test starts with an empty
    visual index.

    Args:
        clean_pg: The module-scoped migrated DSN (tables already truncated).

    Returns:
        The DSN after truncating visual_pages.
    """
    engine = sqlalchemy.create_engine(clean_pg)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.visual_pages RESTART IDENTITY CASCADE"))
    engine.dispose()
    return clean_pg


def _count_rows(dsn: str, table: str) -> int:
    """Return the row count for a given iw_rag table.

    Args:
        dsn: The database DSN.
        table: The unqualified table name within ``iw_rag``.

    Returns:
        Integer row count.
    """
    engine = sqlalchemy.create_engine(dsn)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM iw_rag.{table}"))
            return int(result.scalar_one())
    finally:
        engine.dispose()


def _build_pages(*, n_public: int = 2, restricted: bool = True) -> list[VisualPageRecord]:
    """Build a list of VisualPageRecord fixtures for seeding the visual index.

    Page seeds:
    - Page 1 (source_chunk_id="chunk-1", seed=0.1): public, low relevance.
    - Page 2 (source_chunk_id="chunk-2", seed=1.0): public, HIGH relevance
      (seed matches the query vector — this is the expected top-1).
    - Page 3 (source_chunk_id="chunk-3", seed=0.2): FINANCE-ONLY — should be
      dropped for a user without ``finance`` in their ``principal_groups``.

    Args:
        n_public: Number of public pages to include (1 or 2).
        restricted: Whether to include the restricted (finance-only) page.

    Returns:
        List of VisualPageRecord fixtures.
    """
    pages: list[VisualPageRecord] = [
        VisualPageRecord(
            source_chunk_id="chunk-1",
            source_uri="corpus://doc-a.pdf",
            page=1,
            image=b"fake-image-1",
            text="This page covers introductory material.",
            acl_tags=["public"],
            bbox={"x0": 0.0, "y0": 0.0, "x1": 595.0, "y1": 842.0},
            metadata={"doc": "doc-a"},
        ),
    ]
    if n_public >= 2:
        pages.append(
            VisualPageRecord(
                source_chunk_id="chunk-2",
                source_uri="corpus://doc-a.pdf",
                page=2,
                image=b"fake-image-2",
                text="This page has the key financial chart — top-1 expected.",
                acl_tags=["public"],
                bbox={"x0": 0.0, "y0": 0.0, "x1": 595.0, "y1": 842.0},
                metadata={"doc": "doc-a"},
            )
        )
    if restricted:
        pages.append(
            VisualPageRecord(
                source_chunk_id="chunk-3",
                source_uri="corpus://doc-b.pdf",
                page=1,
                image=b"fake-image-3",
                text="Finance only content — must not appear for public users.",
                acl_tags=["finance"],
                bbox={"x0": 0.0, "y0": 0.0, "x1": 595.0, "y1": 842.0},
                metadata={"doc": "doc-b"},
            )
        )
    return pages


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_indexer_persists_pages_to_visual_pages(
    visual_clean_pg: str,
) -> None:
    """VisualIndexer persists 3 pages to iw_rag.visual_pages with correct row count."""
    session_factory = _make_session_factory(visual_clean_pg)
    store = PostgresCandidateStore(session_factory)
    # Page seeds: index 0=page1 (seed 0.1), 1=page2 (seed 1.0), 2=page3 (seed 0.2)
    encoder = _make_encoder_for_pages({0: 0.1, 1: 1.0, 2: 0.2})
    cfg = VisualIndexConfig(pooled_vectors_per_page=_POOLED_N)
    indexer = VisualIndexer(encoder=encoder, store=store, cfg=cfg)

    pages = _build_pages(n_public=2, restricted=True)
    report = await indexer.index_pages(pages)

    db = Database(DatabaseSettings(dsn=visual_clean_pg))
    await db.dispose()

    assert report.pages_indexed == 3
    assert report.pages_failed == 0
    row_count = _count_rows(visual_clean_pg, "visual_pages")
    assert row_count == 3, f"Expected 3 rows in visual_pages, got {row_count}"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_retriever_top1_is_semantically_closest_page(
    visual_clean_pg: str,
) -> None:
    """VisualRetriever top-1 result is the page with matching seed=1.0 (chunk-2, page 2).

    The fake encoder produces:
    - Page 1 (chunk-1): proxy seed=0.1 → first dim=0.1
    - Page 2 (chunk-2): proxy seed=1.0 → first dim=1.0  ← closest to query
    - Page 3 (chunk-3): proxy seed=0.2 → first dim=0.2

    Query vector: first dim=1.0.  Stage-1 ANN + stage-2 MaxSim should rank
    page 2 (chunk-2) top-1.  ACL: user is ``public``, so page 3 (finance) is
    also accessible here (no restriction on public user + finance page together
    — the restriction test is in a separate test).  We index all 3 pages
    (including the finance page) and query with a public user; page 2 must be top-1.
    """
    session_factory = _make_session_factory(visual_clean_pg)
    store = PostgresCandidateStore(session_factory)
    encoder = _make_encoder_for_pages({0: 0.1, 1: 1.0, 2: 0.2})
    cfg = VisualIndexConfig(pooled_vectors_per_page=_POOLED_N)
    indexer = VisualIndexer(encoder=encoder, store=store, cfg=cfg)

    pages = _build_pages(n_public=2, restricted=True)
    await indexer.index_pages(pages)

    # Now query via the retriever (new encoder instance for query side)
    query_encoder = _make_encoder_for_pages({})  # encode_query always returns seed=1.0
    retriever_store = PostgresCandidateStore(session_factory)
    retriever_cfg = VisualRetrieverConfig(top_k=5, prefetch_n=10, pooled_vectors_per_page=_POOLED_N)
    retriever = VisualRetriever(encoder=query_encoder, store=retriever_store, cfg=retriever_cfg)

    # User is public (can see public + finance pages — no ACL restriction applied here)
    ctx = AccessContext(principal_id="u1", principal_groups=["public", "finance"])
    result = await retriever.retrieve("show me the financial chart", ctx)

    db = Database(DatabaseSettings(dsn=visual_clean_pg))
    await db.dispose()

    assert isinstance(result, RetrievalResult), "Must return RetrievalResult"
    assert len(result.chunks) >= 1, "At least 1 page should be retrieved"
    # The top-1 chunk must be chunk-2 (seed=1.0, page=2)
    top1 = result.chunks[0]
    assert top1.chunk_id == "chunk-2", (
        f"Expected chunk-2 as top-1 (highest semantic similarity), got {top1.chunk_id}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_retriever_acl_restricted_page_is_dropped(
    visual_clean_pg: str,
) -> None:
    """ACL-restricted page (finance) is dropped for a public-only user — no leak.

    The finance page (chunk-3) has higher proxy similarity than page 1 (chunk-1)
    but lower than page 2 (chunk-2).  A user with only ``public`` in their
    principal_groups must NOT receive chunk-3.
    """
    session_factory = _make_session_factory(visual_clean_pg)
    store = PostgresCandidateStore(session_factory)
    encoder = _make_encoder_for_pages({0: 0.1, 1: 1.0, 2: 0.2})
    cfg = VisualIndexConfig(pooled_vectors_per_page=_POOLED_N)
    indexer = VisualIndexer(encoder=encoder, store=store, cfg=cfg)

    pages = _build_pages(n_public=2, restricted=True)
    await indexer.index_pages(pages)

    query_encoder = _make_encoder_for_pages({})
    retriever_store = PostgresCandidateStore(session_factory)
    retriever_cfg = VisualRetrieverConfig(
        top_k=10, prefetch_n=10, pooled_vectors_per_page=_POOLED_N
    )
    retriever = VisualRetriever(encoder=query_encoder, store=retriever_store, cfg=retriever_cfg)

    # User has only "public" access — must not see "finance" page
    ctx = AccessContext(principal_id="u2", principal_groups=["public"])
    result = await retriever.retrieve("financial data", ctx)

    db = Database(DatabaseSettings(dsn=visual_clean_pg))
    await db.dispose()

    chunk_ids = [c.chunk_id for c in result.chunks]
    assert "chunk-3" not in chunk_ids, (
        "Finance-restricted page (chunk-3) must not appear for a public-only user — ACL leak!"
    )
    # Public pages should still be present
    assert len(chunk_ids) >= 1, "At least one public page must be retrieved"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_retrieval_result_contract(
    visual_clean_pg: str,
) -> None:
    """Returned object is a structurally valid RetrievalResult (contract assertion).

    Checks:
    - ``result.retriever == "visual"``
    - ``result.acl_applied is True``
    - Each ``RetrievedChunk`` has non-empty ``content``, valid ``chunk_id``,
      non-None ``source_uri``, ``retriever=="visual"``, and ``acl_applied=True``.
    """
    session_factory = _make_session_factory(visual_clean_pg)
    store = PostgresCandidateStore(session_factory)
    encoder = _make_encoder_for_pages({0: 0.1, 1: 1.0})
    cfg = VisualIndexConfig(pooled_vectors_per_page=_POOLED_N)
    indexer = VisualIndexer(encoder=encoder, store=store, cfg=cfg)

    # Index 2 public pages only
    pages = _build_pages(n_public=2, restricted=False)
    await indexer.index_pages(pages)

    query_encoder = _make_encoder_for_pages({})
    retriever_store = PostgresCandidateStore(session_factory)
    retriever_cfg = VisualRetrieverConfig(top_k=5, prefetch_n=10, pooled_vectors_per_page=_POOLED_N)
    retriever = VisualRetriever(encoder=query_encoder, store=retriever_store, cfg=retriever_cfg)

    ctx = AccessContext(principal_id="u3", principal_groups=["public"])
    result = await retriever.retrieve("what is on this page?", ctx)

    db = Database(DatabaseSettings(dsn=visual_clean_pg))
    await db.dispose()

    # Top-level RetrievalResult contract
    assert isinstance(result, RetrievalResult), "Must be a RetrievalResult"
    assert result.retriever == "visual", f"Expected retriever='visual', got {result.retriever!r}"
    assert result.acl_applied is True, "acl_applied must be True on the visual path"
    assert isinstance(result.chunks, list), "chunks must be a list"

    # Per-chunk contract
    for chunk in result.chunks:
        assert isinstance(chunk, RetrievedChunk), f"Expected RetrievedChunk, got {type(chunk)}"
        assert chunk.retriever == "visual", (  # type: ignore[attr-defined]
            f"Chunk retriever must be 'visual', got {chunk.retriever!r}"  # type: ignore[attr-defined]
        )
        assert chunk.acl_applied is True, "Chunk acl_applied must be True"  # type: ignore[attr-defined]
        assert isinstance(chunk.content, str) and chunk.content, (
            "Chunk content must be non-empty str"
        )
        assert isinstance(chunk.chunk_id, str) and chunk.chunk_id, "chunk_id must be non-empty str"
        assert chunk.source_uri, "source_uri must be non-empty"
        assert chunk.score >= 0.0, "score must be non-negative"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_upsert_is_idempotent(
    visual_clean_pg: str,
) -> None:
    """Re-indexing the same page (same chunk_id, page) does not create duplicate rows.

    Uses ``ON CONFLICT (chunk_id, page) DO UPDATE`` — the second upsert must
    update in place, not insert a new row.
    """
    session_factory = _make_session_factory(visual_clean_pg)
    store = PostgresCandidateStore(session_factory)
    encoder = _make_encoder_for_pages({0: 0.5, 1: 0.5})
    cfg = VisualIndexConfig(pooled_vectors_per_page=_POOLED_N)
    indexer = VisualIndexer(encoder=encoder, store=store, cfg=cfg)

    pages = _build_pages(n_public=1, restricted=False)

    # First ingest
    await indexer.index_pages(pages)
    count_after_first = _count_rows(visual_clean_pg, "visual_pages")

    # Re-index the same pages (idempotency)
    encoder2 = _make_encoder_for_pages({0: 0.5, 1: 0.5})
    indexer2 = VisualIndexer(encoder=encoder2, store=store, cfg=cfg)
    await indexer2.index_pages(pages)
    count_after_second = _count_rows(visual_clean_pg, "visual_pages")

    db = Database(DatabaseSettings(dsn=visual_clean_pg))
    await db.dispose()

    assert count_after_first == count_after_second == 1, (
        f"Upsert must be idempotent: first={count_after_first}, second={count_after_second}"
    )
