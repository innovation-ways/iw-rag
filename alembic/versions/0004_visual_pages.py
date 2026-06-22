"""iw_rag.visual_pages — ColVision multi-vector store for visual late-interaction (WI-031).

Creates ``iw_rag.visual_pages``, the storage table for pooled+binary multi-vectors
produced by the offline VisualIndexer (ColQwen2.5 / ColNomic-3B).

Table design (lld-data-model §8, ADR-010):
- ``chunk_id`` + ``page`` form a logical composite key; ON CONFLICT target in
  ``PostgresCandidateStore.upsert``.  A surrogate uuid pk is added for
  operational convenience (FK from future tables).
- ``proxy_vector vector(128)``: mean-pooled single-vector proxy for stage-1 HNSW
  ANN prefetch (cosine distance, ``<=>``).  Dimension fixed at 128 so Postgres can
  build the HNSW index — ColQwen2.5 and ColNomic-3B both produce 128-d patch
  vectors (``ColVisionEncoderConfig.dims = 128``).
- ``pooled_vectors jsonb``: ~38 pooled float vectors per page (ADR-010 §3).
- ``quantized_vectors jsonb``: binary-quantized pooled vectors (sign bits packed
  as byte ints).
- ``acl_tags text[]``: copied from source manifest; enables a future stage-1
  GIN pre-filter once the open item (task-4b Risks §2) is resolved.

Indexes:
- HNSW on ``proxy_vector`` (cosine) — mirrors ``chunks_dense_hnsw`` in 0001.
- GIN on ``acl_tags`` — required for future ``acl_tags && :groups`` pre-filter
  promotion (currently ACL is a post-filter; GIN is added now so promoting it
  to stage-1 is a query change only, not a schema change).
- B-tree on ``chunk_id`` — supports back-reference lookups from chunks.

Mode-enabled note:
The table is always created by this migration regardless of
``modes.visual.enabled``.  The table existing is harmless when the visual mode
is off — it will remain empty until the offline VisualIndexer runs.  This avoids
making the migration conditional, which would complicate the migration graph.

Fully reversible: ``downgrade()`` drops indexes (explicitly, then) the table
(``CASCADE`` drops any remaining indexes).

References:
- task-4b-visual-rag-route §4–§5 (VisualIndexer, store abstraction)
- lld-data-model §8 (visual_pages table spec)
- ADR-007 (Postgres-first store)
- ADR-010 (pooled+binary, no raw multi-vectors, 128-d patch vectors)
- store.py (``PostgresCandidateStore`` — authoritative SQL contract)

Revision ID: 0004_visual_pages
Revises: 0003_audit_log
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0004_visual_pages"
down_revision: str | None = "0003_audit_log"  # chains from the audit-log migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.visual_pages with HNSW proxy, pooled+binary JSONB columns, indexes.

    Creates:
    - ``iw_rag.visual_pages`` table with uuid PK, all columns matching
      ``PostgresCandidateStore.upsert`` and ``stage1_prefetch`` SQL exactly.
    - UNIQUE constraint on ``(chunk_id, page)`` — the ``ON CONFLICT`` target.
    - HNSW index on ``proxy_vector vector_cosine_ops`` for stage-1 ANN.
    - GIN index on ``acl_tags`` for future stage-1 pre-filter promotion.
    - B-tree index on ``chunk_id`` for back-reference lookups.
    """
    op.execute(
        """
        CREATE TABLE iw_rag.visual_pages (
            id                uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
            chunk_id          text        NOT NULL,
            source_uri        text        NOT NULL,
            page              int         NOT NULL,
            content           text        NOT NULL,
            acl_tags          text[]      NOT NULL DEFAULT '{}',
            pooled_vectors    jsonb       NOT NULL,
            quantized_vectors jsonb       NOT NULL,
            proxy_vector      vector(128) NOT NULL,
            bbox              jsonb       NOT NULL DEFAULT '{}'::jsonb,
            page_image_ref    text        NOT NULL,
            metadata          jsonb       NOT NULL DEFAULT '{}'::jsonb,
            UNIQUE (chunk_id, page)
        )
        """
    )

    # Stage-1 ANN index: cosine over the mean-pooled proxy vector.
    # Mirrors chunks_dense_hnsw in 0001 (m=16, ef_construction=64).
    op.execute(
        "CREATE INDEX visual_pages_proxy_hnsw ON iw_rag.visual_pages "
        "USING hnsw (proxy_vector vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    # GIN index for future stage-1 ACL pre-filter (task-4b Risks §2 open item).
    # Currently ACL is a post-filter in VisualRetriever; the GIN is added now
    # so promoting to pre-filter requires only a query rewrite, not a migration.
    op.execute("CREATE INDEX visual_pages_acl_gin ON iw_rag.visual_pages USING gin (acl_tags)")

    # B-tree index on chunk_id for back-reference lookups from chunks.
    op.execute("CREATE INDEX visual_pages_chunk_id ON iw_rag.visual_pages (chunk_id)")


def downgrade() -> None:
    """Drop the visual_pages indexes and table (full reversal).

    Drops in reverse creation order:
    1. The three explicit indexes (HNSW + GIN + B-tree).
    2. The table (``CASCADE`` drops the UNIQUE index + any remaining indexes).
    """
    op.execute("DROP INDEX IF EXISTS iw_rag.visual_pages_proxy_hnsw")
    op.execute("DROP INDEX IF EXISTS iw_rag.visual_pages_acl_gin")
    op.execute("DROP INDEX IF EXISTS iw_rag.visual_pages_chunk_id")
    op.execute("DROP TABLE IF EXISTS iw_rag.visual_pages CASCADE")
