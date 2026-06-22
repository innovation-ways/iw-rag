"""Integration tests for DB schema, migrations, and repositories.

Uses testcontainers with ``paradedb/paradedb:latest`` so pgvector + pg_search
are available. Tests verify:

- Alembic upgrade head creates the iw_rag schema + all required tables/indexes.
- downgrade base reverses cleanly; re-upgrade is idempotent.
- Both extensions (vector, pg_search) are present after migration.
- All 6 chunk indexes exist on iw_rag.chunks.
- Repository round-trip: insert document + chunk with 1024-d vector; read back;
  assert vector dim=1024 and acl_tags preserved.
- UNIQUE (source_uri, checksum) constraint rejects a duplicate.

RED-first note: these tests were written before the implementation existed.
The paradedb/paradedb:latest image is required (plain postgres image lacks
pg_search / BM25 extension).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import pytest
import sqlalchemy
from sqlalchemy import text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = "paradedb/paradedb:latest"


@pytest.fixture(scope="module")
def db_container() -> Generator[str, None, None]:
    """Spin up a fresh paradedb/paradedb container and yield its DSN.

    Returns:
        Generator yielding a psycopg-driver DSN for the container.
    """
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image=PARADEDB_IMAGE,
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as pg:
        # testcontainers hands back psycopg2 URL; swap driver to psycopg (v3)
        raw_url = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        yield dsn


@pytest.fixture(scope="module")
def migrated_engine(db_container: str) -> Generator[sqlalchemy.Engine, None, None]:
    """Run Alembic upgrade head against the container and yield the engine.

    Sets the IW_RAG__* env directly (module-scoped, so it cannot use the
    function-scoped ``monkeypatch``) and cleans it up on teardown.

    Args:
        db_container: DSN for the testcontainer.

    Yields:
        Synchronous SQLAlchemy engine connected to the migrated DB.
    """
    import iw_rag.settings as settings_mod
    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    # Point Settings at the container
    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    with rebased_alembic_config(db_container) as rebased:
        command.upgrade(rebased.cfg, "head")

    engine = sqlalchemy.create_engine(db_container)
    yield engine

    engine.dispose()
    # Clean up env
    for key in [
        "IW_RAG__DATABASE__DSN",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
        "IW_RAG__MODELS__RERANKER__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
    ]:
        os.environ.pop(key, None)
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Migration tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_upgrade_creates_schema(migrated_engine: sqlalchemy.Engine) -> None:
    """Verifies Alembic upgrade head creates the iw_rag schema."""
    with migrated_engine.connect() as conn:
        result = conn.execute(
            text("SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'iw_rag'")
        )
        row = result.fetchone()
    assert row is not None, "Schema iw_rag was not created"
    assert row[0] == "iw_rag", f"Expected schema_name 'iw_rag'; got {row[0]!r}"


@pytest.mark.integration
def test_both_extensions_present(migrated_engine: sqlalchemy.Engine) -> None:
    """Verifies vector and pg_search extensions are present after migration."""
    with migrated_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT extname FROM pg_extension "
                "WHERE extname IN ('vector', 'pg_search') "
                "ORDER BY extname"
            )
        )
        names = {row[0] for row in result}
    assert "vector" in names, "pgvector extension missing"
    assert "pg_search" in names, "pg_search extension missing"
    assert names == {"vector", "pg_search"}, (
        f"Expected exactly {{'vector', 'pg_search'}}; got {names!r}"
    )


@pytest.mark.integration
def test_all_six_chunk_indexes_exist(migrated_engine: sqlalchemy.Engine) -> None:
    """Verifies all 6 required indexes exist on iw_rag.chunks."""
    expected_indexes = {
        "chunks_dense_hnsw",
        "chunks_bm25",
        "chunks_acl_gin",
        "chunks_metadata_gin",
        "chunks_parent",
        "chunks_document",
    }
    with migrated_engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'iw_rag' AND tablename = 'chunks'"
            )
        )
        actual_indexes = {row[0] for row in result}
    missing = expected_indexes - actual_indexes
    assert not missing, f"Missing chunk indexes: {missing}"


@pytest.mark.integration
def test_downgrade_and_reupgrade_idempotent(db_container: str) -> None:
    """Verifies downgrade base reverses schema; re-upgrade head is idempotent."""
    import iw_rag.settings as settings_mod
    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    with rebased_alembic_config(db_container) as rebased:
        alembic_cfg = rebased.cfg

        # Downgrade to base — iw_rag schema should be gone
        command.downgrade(alembic_cfg, "base")

        engine = sqlalchemy.create_engine(db_container)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name = 'iw_rag'"
                )
            )
            row = result.fetchone()
        assert row is None, "Schema iw_rag should have been dropped by downgrade"

        # Re-upgrade — idempotency check
        command.upgrade(alembic_cfg, "head")

        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT schema_name FROM information_schema.schemata "
                    "WHERE schema_name = 'iw_rag'"
                )
            )
            row = result.fetchone()
        assert row is not None, "Schema iw_rag not recreated after re-upgrade"
        engine.dispose()


# ---------------------------------------------------------------------------
# Repository round-trip tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_document_and_chunk_round_trip(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """Round-trip a document + chunk THROUGH the repositories (1024-d vector + acl_tags).

    Exercises ``DocumentRepository.upsert_document`` and
    ``ChunkRepository.bulk_upsert_chunks``/``get_parents`` against real ParadeDB,
    then reads back the vector dim and acl_tags to confirm shape preservation.
    """
    from iw_rag.settings import DatabaseSettings
    from iw_rag.storage.db import Database
    from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

    db = Database(DatabaseSettings(dsn=db_container))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            doc_id = await docs.upsert_document(
                source_uri="file:///roundtrip/doc.pdf",
                checksum="rt-checksum-1",
                title="Round-trip Doc",
            )
            # Idempotent: a second upsert of the same (uri, checksum) returns the same id.
            doc_id_again = await docs.upsert_document(
                source_uri="file:///roundtrip/doc.pdf", checksum="rt-checksum-1"
            )
            assert doc_id_again == doc_id, "upsert_document not idempotent on (uri, checksum)"

            chunks_repo = ChunkRepository(session)
            parent = NewChunk(
                document_id=doc_id,
                ordinal=0,
                content="parent content",
                dense_embedding=[0.0] * 1024,
                acl_tags=["finance", "all-staff"],
            )
            (parent_id,) = await chunks_repo.bulk_upsert_chunks([parent])
            leaf = NewChunk(
                document_id=doc_id,
                ordinal=1,
                content="leaf content",
                dense_embedding=[0.1] * 1024,
                parent_id=parent_id,
                acl_tags=["finance"],
            )
            (leaf_id,) = await chunks_repo.bulk_upsert_chunks([leaf])
            await session.commit()

            # get_parents resolves the parent for small-to-big expansion.
            parents = await chunks_repo.get_parents([parent_id])
            assert len(parents) == 1
            assert parents[0]["content"] == "parent content"

            # Read back vector dim + acl_tags for the leaf.
            row = (
                await session.execute(
                    text(
                        "SELECT vector_dims(dense_embedding) AS dim, acl_tags "
                        "FROM iw_rag.chunks WHERE id = :id"
                    ),
                    {"id": leaf_id},
                )
            ).first()
    finally:
        await db.dispose()

    assert row is not None, "Chunk was not found after insert"
    assert row[0] == 1024, f"Expected vector dim=1024, got {row[0]}"
    assert "finance" in row[1], "acl_tags not preserved"


@pytest.mark.integration
def test_unique_source_uri_checksum_constraint(migrated_engine: sqlalchemy.Engine) -> None:
    """UNIQUE (source_uri, checksum) rejects a duplicate document row."""
    import sqlalchemy.exc

    with migrated_engine.connect() as conn:
        run_id = uuid.uuid4()
        conn.execute(
            text(
                "INSERT INTO iw_rag.ingestion_runs "
                "(id, corpus_label, parser, chunker_config, embed_model, status, started_at) "
                "VALUES (:id, 'test', 'docling', '{}', 'bge-m3', 'running', now())"
            ),
            {"id": run_id},
        )

        doc_id1 = uuid.uuid4()
        conn.execute(
            text(
                "INSERT INTO iw_rag.documents "
                "(id, source_uri, checksum, metadata, ingestion_run_id) "
                "VALUES (:id, 'file:///unique-test.pdf', 'sha256abc', '{}', :run_id)"
            ),
            {"id": doc_id1, "run_id": run_id},
        )
        conn.commit()

        doc_id2 = uuid.uuid4()
        with pytest.raises(
            sqlalchemy.exc.IntegrityError,
            match="documents_source_uri_checksum_key",
        ):
            conn.execute(
                text(
                    "INSERT INTO iw_rag.documents "
                    "(id, source_uri, checksum, metadata, ingestion_run_id) "
                    "VALUES (:id, 'file:///unique-test.pdf', 'sha256abc', '{}', :run_id)"
                ),
                {"id": doc_id2, "run_id": run_id},
            )
