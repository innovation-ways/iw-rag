"""Integration tests for the Visual ACL pre-filter at the SQL boundary (F-00097).

Uses testcontainers Postgres (the session-scoped ``pg_dsn`` fixture from
``tests/integration/conftest.py``).  No GPU, no encoder, no AGE required —
the test exercises the stage-1 SQL pre-filter directly through the real
``PostgresCandidateStore`` against a seeded ``iw_rag.visual_pages`` table.

Tests:

- The stage-1 SQL pre-filter (``WHERE acl_tags && :principal_groups``) is
  the enforcement seam: finance-only pages never appear in the candidate
  set for a ``public`` principal (AC1, Boundary Behavior row
  "Pre-filter excludes them in SQL — they are never returned to Python").

- An empty ``principal_groups`` fails closed: zero candidates (Boundary
  Behavior row "Empty ``principal_groups`` (visual)").

- A page with the explicit shared tag ``"public"`` is visible to a
  ``public`` principal (public-tag reconciliation finding from S01).

- A page with ``acl_tags = []`` is invisible to every principal (matches
  the spine and the ingestion contract — Boundary Behavior row
  "Empty-``acl_tags`` chunk/page").

- The defense-in-depth post-filter is a no-op on correctly pre-filtered
  data (AC3 / Invariant 6): when the stage-1 SQL pre-filter excludes
  everything, the post-filter drops zero additional pages.

Each test asserts **specific values** (the exact set of returned
``chunk_id``s), not just shape — per the F-00097 S07 contract.

References:
- F-00097 Feature_Design §"Acceptance Criteria" (AC1, AC3)
- F-00097 Feature_Design §"Boundary Behavior"
- S01 report — public-tag reconciliation
- lld-data-model §8 (visual_pages schema)
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.domain.acl import AccessContext
from iw_rag.modes.visual.retriever import VisualRetriever, VisualRetrieverConfig
from iw_rag.modes.visual.store import PostgresCandidateStore
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants — stable across the suite so any inadvertent change to the
# fixture data triggers an explicit failure here, not a cascade.
# ---------------------------------------------------------------------------

#: Pooled-vector count per seeded page (small for speed; the dim is what matters).
_POOLED_N = 2
#: Proxy-vector dimension (matches the ``vector(128)`` column in 0004).
_DIMS = 128

#: Stable, easy-to-grep page/chunk IDs for semantic assertions.
PUB_CHUNK_ID = "chunk-public"
PUB2_CHUNK_ID = "chunk-public-2"
FIN_CHUNK_ID = "chunk-finance"
HR_CHUNK_ID = "chunk-hr"
EMPTY_TAGS_CHUNK_ID = "chunk-empty-tags"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session_factory(dsn: str) -> Any:
    """Build an async session factory bound to the testcontainer DSN.

    Args:
        dsn: The psycopg DSN for the test container.

    Returns:
        An async context-manager factory yielding an ``AsyncSession``
        (this is what ``PostgresCandidateStore`` consumes).
    """
    db = Database(DatabaseSettings(dsn=dsn))
    return db.session


def _seed_visual_page(
    engine: sqlalchemy.engine.base.Engine,  # type: ignore[type-arg]
    *,
    chunk_id: str,
    source_uri: str,
    page: int,
    content: str,
    acl_tags: list[str],
    proxy_first_dim: float,
) -> None:
    """Seed a single row into ``iw_rag.visual_pages``.

    Uses a deterministic 128-d proxy vector whose first dimension equals
    ``proxy_first_dim`` so ordering is predictable across tests.  The
    pooled/quantized columns are small JSONB blobs (the pre-filter ignores
    them; they exist only to satisfy the NOT-NULL column constraints).

    Args:
        engine: The synchronous SQLAlchemy engine.
        chunk_id: The page's ``chunk_id`` (back-ref to ``chunks.id``).
        source_uri: The document URI this page belongs to.
        page: 1-based page number.
        content: The page text (becomes ``RetrievedChunk.content``).
        acl_tags: The page's ACL entitlement tags.
        proxy_first_dim: The scalar to seed the first dim of ``proxy_vector``.
    """
    proxy = [0.0] * _DIMS
    proxy[0] = proxy_first_dim
    proxy_str = "[" + ",".join(str(v) for v in proxy) + "]"
    pooled = [[0.0] * _DIMS for _ in range(_POOLED_N)]
    quantized = [[0] for _ in range(_POOLED_N)]
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO iw_rag.visual_pages "
                "(chunk_id, source_uri, page, content, acl_tags, "
                " pooled_vectors, quantized_vectors, proxy_vector, bbox, "
                " page_image_ref, metadata) "
                "VALUES "
                "(:chunk_id, :source_uri, :page, :content, :acl_tags, "
                " CAST(:pooled AS jsonb), CAST(:quantized AS jsonb), "
                " CAST(:proxy AS vector), CAST(:bbox AS jsonb), "
                " :image_ref, CAST(:metadata AS jsonb))"
            ),
            {
                "chunk_id": chunk_id,
                "source_uri": source_uri,
                "page": page,
                "content": content,
                "acl_tags": acl_tags,
                "pooled": json.dumps(pooled),
                "quantized": json.dumps(quantized),
                "proxy": proxy_str,
                "bbox": json.dumps({"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}),
                "image_ref": f"{source_uri}#page={page}",
                "metadata": json.dumps({"seeded_for": "F-00097"}),
            },
        )


def _seed_visual_pages_mixed(dsn: str) -> None:
    """Seed a mixed public/restricted corpus into ``iw_rag.visual_pages``.

    The fixture has five pages covering every Boundary Behavior row:

    - ``chunk-public``: ``acl_tags=["public"]`` (spine's public representation)
    - ``chunk-public-2``: ``acl_tags=["public", "all-staff"]`` (overlapping)
    - ``chunk-finance``: ``acl_tags=["finance"]`` (cross-tenant — must be excluded)
    - ``chunk-hr``: ``acl_tags=["hr"]`` (cross-tenant — must be excluded)
    - ``chunk-empty-tags``: ``acl_tags=[]`` (must be invisible to every principal)

    Args:
        dsn: The testcontainer DSN.
    """
    engine = sqlalchemy.create_engine(dsn)
    try:
        _seed_visual_page(
            engine,
            chunk_id=PUB_CHUNK_ID,
            source_uri="corpus://pub-doc.pdf",
            page=1,
            content="Public page 1",
            acl_tags=["public"],
            proxy_first_dim=0.9,
        )
        _seed_visual_page(
            engine,
            chunk_id=PUB2_CHUNK_ID,
            source_uri="corpus://pub-doc.pdf",
            page=2,
            content="Public page 2 (also all-staff)",
            acl_tags=["public", "all-staff"],
            proxy_first_dim=0.85,
        )
        _seed_visual_page(
            engine,
            chunk_id=FIN_CHUNK_ID,
            source_uri="corpus://finance-doc.pdf",
            page=1,
            content="Finance-only page — must be excluded for public",
            acl_tags=["finance"],
            proxy_first_dim=0.95,
        )
        _seed_visual_page(
            engine,
            chunk_id=HR_CHUNK_ID,
            source_uri="corpus://hr-doc.pdf",
            page=1,
            content="HR-only page — must be excluded for public",
            acl_tags=["hr"],
            proxy_first_dim=0.8,
        )
        _seed_visual_page(
            engine,
            chunk_id=EMPTY_TAGS_CHUNK_ID,
            source_uri="corpus://empty-tags.pdf",
            page=1,
            content="Page with empty acl_tags — must be invisible to every principal",
            acl_tags=[],
            proxy_first_dim=0.99,
        )
    finally:
        engine.dispose()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def visual_acl_pg(clean_pg: str) -> str:
    """Truncate ``iw_rag.visual_pages`` and yield the migrated DSN.

    Builds on ``clean_pg`` so each test starts with a fresh, empty visual
    index — independent of any other integration test that touches the
    same container.

    Args:
        clean_pg: The session-scoped migrated DSN (already truncated for
            documents/chunks/ingestion_runs by the parent fixture).

    Returns:
        The DSN after truncating visual_pages.
    """
    engine = sqlalchemy.create_engine(clean_pg)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.visual_pages RESTART IDENTITY CASCADE"))
    engine.dispose()
    return clean_pg


@pytest.fixture
def visual_acl_seeded_pg(visual_acl_pg: str) -> str:
    """Seed the mixed public/restricted visual_pages fixture.

    Builds on ``visual_acl_pg`` so each test starts with a known mixed
    corpus.  Five rows covering every Boundary Behavior row (public,
    public-all-staff, finance, hr, empty-tags).

    Args:
        visual_acl_pg: The DSN after truncating visual_pages.

    Returns:
        The DSN after seeding the mixed corpus.
    """
    _seed_visual_pages_mixed(visual_acl_pg)
    return visual_acl_pg


# ---------------------------------------------------------------------------
# Tests — AC1 / Boundary Behavior rows for visual mode
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_stage1_prefetch_excludes_finance_for_public(
    visual_acl_seeded_pg: str,
) -> None:
    """AC1: stage1_prefetch pre-filters at SQL — finance-only pages are not returned.

    Seeded: ``chunk-finance`` and ``chunk-hr`` are finance/hr-only.  For a
    principal holding only ``["public"]``, the stage-1 SQL pre-filter
    ``WHERE acl_tags && ARRAY['public']`` must exclude them at the SQL
    boundary — they are never returned to Python.

    Asserts the specific returned set: only ``chunk-public`` and
    ``chunk-public-2`` survive.  No shape-only checks.
    """
    session_factory = _make_session_factory(visual_acl_seeded_pg)
    store = PostgresCandidateStore(session_factory)

    # Query proxy: arbitrary 128-dim vector; ranking is irrelevant to this
    # test — we only assert the ACL pre-filter membership.
    query_proxy = [0.0] * _DIMS
    query_proxy[0] = 0.9

    candidates = await store.stage1_prefetch(
        query_proxy,
        prefetch_n=200,
        principal_groups=["public"],
    )

    returned_ids = {c["chunk_id"] for c in candidates}
    # Public-tag pages are present (you didn't over-filter).
    assert PUB_CHUNK_ID in returned_ids, (
        f"Public page {PUB_CHUNK_ID!r} must be visible to a public principal — "
        f"the pre-filter must not over-filter explicit shared tags. Got: {returned_ids}"
    )
    assert PUB2_CHUNK_ID in returned_ids, (
        f"Public+all-staff page {PUB2_CHUNK_ID!r} must be visible (intersects "
        f"public). Got: {returned_ids}"
    )
    # Restricted pages are absent (you didn't leak).
    assert FIN_CHUNK_ID not in returned_ids, (
        f"Finance-only page {FIN_CHUNK_ID!r} must be excluded at the SQL "
        f"boundary by `acl_tags && ARRAY['public']`. Got: {returned_ids}"
    )
    assert HR_CHUNK_ID not in returned_ids, (
        f"HR-only page {HR_CHUNK_ID!r} must be excluded at the SQL boundary. Got: {returned_ids}"
    )
    # Empty-acl_tags is invisible to every principal (spine parity).
    assert EMPTY_TAGS_CHUNK_ID not in returned_ids, (
        f"Page with acl_tags=[] must be invisible to every principal "
        f"(matches spine `acl_tags && anything == false` and enrich.py "
        f"'empty resolver result ⇒ acl_tags=[] ⇒ invisible to every "
        f"principal; fail-closed, never world-readable'). Got: {returned_ids}"
    )
    # Exact membership assertion — no surprise rows.
    assert returned_ids == {PUB_CHUNK_ID, PUB2_CHUNK_ID}, (
        f"Expected exactly the two public-readable pages; got {returned_ids}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_stage1_prefetch_empty_principal_groups_fails_closed(
    visual_acl_seeded_pg: str,
) -> None:
    """Boundary Behavior: empty ``principal_groups`` returns zero candidates.

    Mirrors :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`
    semantics.  A principal with no groups must not see any page, even if
    pages exist in the store.  The store short-circuits to ``[]`` without
    issuing an unfiltered query.
    """
    session_factory = _make_session_factory(visual_acl_seeded_pg)
    store = PostgresCandidateStore(session_factory)

    query_proxy = [0.0] * _DIMS
    query_proxy[0] = 0.9

    candidates = await store.stage1_prefetch(
        query_proxy,
        prefetch_n=200,
        principal_groups=[],
    )

    assert candidates == [], (
        f"Empty principal_groups must yield zero candidates (fail-closed); "
        f"got {len(candidates)}: {[c['chunk_id'] for c in candidates]}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_stage1_prefetch_principal_in_finance_sees_finance_only(
    visual_acl_seeded_pg: str,
) -> None:
    """A finance+public principal sees finance + public pages; hr + empty-tags are excluded.

    Asserts the SQL pre-filter membership for the other direction: a
    principal with ``['finance', 'public']`` groups sees the finance page
    (overlaps finance) and both public pages (overlap public).  HR and
    empty-tags remain invisible.

    Note: ``acl_tags=['public']`` is NOT world-readable; the principal
    must hold ``'public'`` for the SQL ``&&`` to match.  This matches the
    spine and ``enrich.py`` (S01 public-tag reconciliation).
    """
    session_factory = _make_session_factory(visual_acl_seeded_pg)
    store = PostgresCandidateStore(session_factory)

    query_proxy = [0.0] * _DIMS
    query_proxy[0] = 0.9

    candidates = await store.stage1_prefetch(
        query_proxy,
        prefetch_n=200,
        principal_groups=["finance", "public"],
    )

    returned_ids = {c["chunk_id"] for c in candidates}
    assert FIN_CHUNK_ID in returned_ids, (
        f"Finance-only page {FIN_CHUNK_ID!r} must be visible to a "
        f"finance+public principal (acl_tags=['finance'] ∩ "
        f"['finance','public'] non-empty). Got: {returned_ids}"
    )
    assert PUB_CHUNK_ID in returned_ids, (
        f"Public page {PUB_CHUNK_ID!r} must be visible to a finance+public "
        f"principal (overlap public). Got: {returned_ids}"
    )
    assert PUB2_CHUNK_ID in returned_ids, (
        f"Public+all-staff page {PUB2_CHUNK_ID!r} must be visible to a "
        f"finance+public principal (overlap public). Got: {returned_ids}"
    )
    assert HR_CHUNK_ID not in returned_ids, (
        f"HR-only page {HR_CHUNK_ID!r} must be excluded for a finance+public "
        f"principal (no overlap). Got: {returned_ids}"
    )
    assert EMPTY_TAGS_CHUNK_ID not in returned_ids, (
        f"Empty-tags page must remain invisible (spine parity). Got: {returned_ids}"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_stage1_prefetch_uses_acl_gin_index(
    visual_acl_seeded_pg: str,
) -> None:
    """The stage-1 pre-filter query plan uses ``visual_pages_acl_gin`` (performance contract).

    F-00097 INVARIANT 1 + S01 recommendation #4: pin the performance
    contract, not just the security contract.  ``EXPLAIN`` the
    ``stage1_prefetch`` query shape and assert the GIN index appears in
    the plan.  This guards against a regression where the pre-filter is
    silently demoted to a sequential scan.

    Implementation note: the EXPLAIN runs with ``enable_seqscan=OFF`` so
    the cost-based planner picks an index plan even on the small seeded
    corpus.  This isolates the planner from row-count noise while still
    proving the GIN index is the chosen access path for the
    ``acl_tags && :principal_groups`` filter (no other useful index
    exists on ``visual_pages.acl_tags``).
    """
    engine = sqlalchemy.create_engine(visual_acl_seeded_pg)
    try:
        with engine.connect() as conn:
            # Force the planner to use an index (disable Seq Scan) so the GIN
            # is exercised on the small test corpus — without this, the
            # planner prefers Seq Scan when the row count is tiny.
            conn.execute(text("SET LOCAL enable_seqscan = OFF"))
            # Mimic the production SQL shape (same clause, same parameter).
            explain_rows = conn.execute(
                text(
                    "EXPLAIN "
                    "SELECT chunk_id FROM iw_rag.visual_pages "
                    "WHERE acl_tags && ARRAY['public']::text[] "
                    "ORDER BY proxy_vector <=> CAST('[0.9,0,0]' AS vector) "
                    "LIMIT 200"
                )
            ).all()
        plan_text = "\n".join(str(r[0]) for r in explain_rows)
    finally:
        engine.dispose()

    lines = [ln.strip() for ln in plan_text.splitlines()]
    gin_scans = [
        ln for ln in lines if ln.startswith("->  Bitmap Index Scan on visual_pages_acl_gin")
    ]
    assert len(gin_scans) == 1, (
        f"EXPLAIN must show exactly one scan of visual_pages_acl_gin (GIN-accelerated "
        f"pre-filter, INVARIANT 1).  Plan:\n{plan_text}"
    )
    # ...and that scan must be driven by the ACL overlap predicate itself.
    assert lines[lines.index(gin_scans[0]) + 1].startswith("Index Cond: (acl_tags &&"), plan_text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_retriever_full_path_excludes_restricted_pages(
    visual_acl_seeded_pg: str,
) -> None:
    """End-to-end ``VisualRetriever.retrieve`` returns only readable pages.

    Drives the real retriever through the real store against the seeded
    testcontainer corpus, with a stubbed encoder (no GPU/network).  The
    stage-1 SQL pre-filter excludes finance/hr/empty-tags at the SQL
    boundary; the defense-in-depth post-filter is a no-op on the
    pre-filtered set; ``RetrievalResult.chunks`` contains only the two
    public pages.
    """
    from unittest.mock import AsyncMock, MagicMock

    session_factory = _make_session_factory(visual_acl_seeded_pg)
    store = PostgresCandidateStore(session_factory)

    # Stubbed encoder: encode_query returns a single 128-dim vector.
    fake_encoder = MagicMock()
    fake_encoder.encode_query = AsyncMock(return_value=[[1.0] + [0.0] * (_DIMS - 1)])

    cfg = VisualRetrieverConfig(top_k=10, prefetch_n=200, pooled_vectors_per_page=_POOLED_N)
    retriever = VisualRetriever(encoder=fake_encoder, store=store, cfg=cfg)

    ctx = AccessContext(principal_id="u-public", principal_groups=("public",))
    result = await retriever.retrieve("show me the chart", ctx)

    returned_ids = {c.chunk_id for c in result.chunks}
    # The two public pages survive end-to-end.
    assert PUB_CHUNK_ID in returned_ids
    assert PUB2_CHUNK_ID in returned_ids
    # The finance page is excluded (the SQL pre-filter is the seam).
    assert FIN_CHUNK_ID not in returned_ids, (
        f"Finance-only page must never appear in RetrievalResult.chunks "
        f"(AC1, Invariant 1). Got: {returned_ids}"
    )
    assert HR_CHUNK_ID not in returned_ids
    assert EMPTY_TAGS_CHUNK_ID not in returned_ids
    # Honest acl_applied flag.
    assert result.acl_applied is True
    assert result.retriever == "visual"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_visual_retriever_empty_principal_groups_returns_empty_result(
    visual_acl_seeded_pg: str,
) -> None:
    """End-to-end: empty ``principal_groups`` ⇒ empty ``RetrievalResult``.

    Boundary Behavior: a principal with no groups must not see any page.
    The store's stage-1 SQL pre-filter short-circuits to ``[]`` (mirrors
    :class:`~iw_rag.retrieval.acl_filter.AclPredicate.is_empty`), and the
    retriever returns an empty ``RetrievalResult`` with
    ``acl_applied=True``.
    """
    from unittest.mock import AsyncMock, MagicMock

    session_factory = _make_session_factory(visual_acl_seeded_pg)
    store = PostgresCandidateStore(session_factory)

    fake_encoder = MagicMock()
    fake_encoder.encode_query = AsyncMock(return_value=[[1.0] + [0.0] * (_DIMS - 1)])

    cfg = VisualRetrieverConfig(top_k=10, prefetch_n=200, pooled_vectors_per_page=_POOLED_N)
    retriever = VisualRetriever(encoder=fake_encoder, store=store, cfg=cfg)

    ctx = AccessContext(principal_id="u-empty", principal_groups=())
    result = await retriever.retrieve("anything", ctx)

    assert result.chunks == [], (
        f"Empty principal_groups ⇒ empty chunks; got {[c.chunk_id for c in result.chunks]}"
    )
    assert result.candidate_count == 0
    assert result.acl_applied is True
    assert result.retriever == "visual"
