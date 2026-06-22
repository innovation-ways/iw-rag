"""initial iw_rag schema — documents, chunks, ACL, ingestion_runs, eval_*.

Creates the single auditable store (lld-data-model §3/§4): extensions, the
``iw_rag`` schema, all core tables, and the six ``chunks`` indexes (dense HNSW,
BM25, two GIN, parent, document). Extensions are asserted here (belt-and-braces)
so a fresh testcontainer with no ``docker/postgres/init`` still gets them.

Revision ID: 0001
Revises: (base)
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None  # base migration — initial schema
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create extensions, the iw_rag schema, all tables, and chunk indexes."""
    # --- extensions + schema (belt-and-braces; also bootstrapped in init SQL) ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_search")
    op.execute("CREATE SCHEMA IF NOT EXISTS iw_rag")

    # --- ingestion_runs (referenced by documents) ---
    op.execute(
        """
        CREATE TABLE iw_rag.ingestion_runs (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            corpus_label  text,
            parser        text,
            chunker_config text,
            embed_model   text,
            status        text NOT NULL DEFAULT 'running',
            started_at    timestamptz NOT NULL DEFAULT now(),
            finished_at   timestamptz
        )
        """
    )

    # --- documents (UNIQUE(source_uri, checksum) → idempotent re-ingest) ---
    op.execute(
        """
        CREATE TABLE iw_rag.documents (
            id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            source_uri       text NOT NULL,
            title            text,
            content_type     text NOT NULL DEFAULT 'document',
            checksum         text NOT NULL,
            metadata         jsonb NOT NULL DEFAULT '{}'::jsonb,
            ingestion_run_id uuid REFERENCES iw_rag.ingestion_runs(id),
            created_at       timestamptz NOT NULL DEFAULT now(),
            UNIQUE (source_uri, checksum)
        )
        """
    )

    # --- chunks (dense vector + learned-sparse + ACL tags + parent link) ---
    op.execute(
        """
        CREATE TABLE iw_rag.chunks (
            id              uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id     uuid NOT NULL REFERENCES iw_rag.documents(id) ON DELETE CASCADE,
            parent_id       uuid REFERENCES iw_rag.chunks(id) ON DELETE CASCADE,
            ordinal         int  NOT NULL,
            content         text NOT NULL,
            context_prefix  text,
            dense_embedding vector(1024) NOT NULL,
            sparse_weights  jsonb,
            content_type    text NOT NULL DEFAULT 'text',
            section_path    text,
            acl_tags        text[] NOT NULL DEFAULT '{}',
            metadata        jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at      timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    # --- document_acl (principal→permission source of truth; chunks.acl_tags denorm) ---
    op.execute(
        """
        CREATE TABLE iw_rag.document_acl (
            id          uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id uuid NOT NULL REFERENCES iw_rag.documents(id) ON DELETE CASCADE,
            principal   text NOT NULL,
            permission  text NOT NULL DEFAULT 'read'
        )
        """
    )

    # --- eval tables (append-only regression history) ---
    op.execute(
        """
        CREATE TABLE iw_rag.eval_datasets (
            id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            name       text NOT NULL,
            version    text NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE iw_rag.eval_queries (
            id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id         uuid NOT NULL REFERENCES iw_rag.eval_datasets(id) ON DELETE CASCADE,
            question           text NOT NULL,
            reference_answer   text,
            relevant_chunk_ids uuid[] NOT NULL DEFAULT '{}',
            query_type         text
        )
        """
    )
    op.execute(
        """
        CREATE TABLE iw_rag.eval_runs (
            id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            dataset_id uuid REFERENCES iw_rag.eval_datasets(id),
            experiment text NOT NULL,
            git_sha    text,
            config     jsonb NOT NULL DEFAULT '{}'::jsonb,
            created_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE iw_rag.eval_results (
            id                 uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id             uuid NOT NULL REFERENCES iw_rag.eval_runs(id) ON DELETE CASCADE,
            query_id           uuid REFERENCES iw_rag.eval_queries(id),
            retrieval_metrics  jsonb,
            generation_metrics jsonb,
            trace_ref          jsonb
        )
        """
    )

    # --- chunk indexes (lld-data-model §4) ---
    # Dense ANN: pgvector HNSW, cosine for normalized BGE-M3 embeddings.
    op.execute(
        "CREATE INDEX chunks_dense_hnsw ON iw_rag.chunks "
        "USING hnsw (dense_embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )
    # Lexical/BM25: ParadeDB pg_search. DDL pinned to the installed tag (Q-AGPL);
    # verified against paradedb/paradedb:latest by the index-existence test.
    op.execute(
        "CREATE INDEX chunks_bm25 ON iw_rag.chunks "
        "USING bm25 (id, content, content_type, section_path) WITH (key_field='id')"
    )
    # ACL + metadata pre-filtering (the enforcement point, R-00108).
    op.execute("CREATE INDEX chunks_acl_gin ON iw_rag.chunks USING gin (acl_tags)")
    op.execute("CREATE INDEX chunks_metadata_gin ON iw_rag.chunks USING gin (metadata)")
    op.execute("CREATE INDEX chunks_parent ON iw_rag.chunks (parent_id)")
    op.execute("CREATE INDEX chunks_document ON iw_rag.chunks (document_id)")


def downgrade() -> None:
    """Drop the entire iw_rag schema (cascades all tables + indexes)."""
    op.execute("DROP SCHEMA IF EXISTS iw_rag CASCADE")
