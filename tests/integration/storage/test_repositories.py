"""Integration tests for ``iw_rag.storage.repositories`` — untested methods.

Uses a real testcontainers ParadeDB Postgres (the project convention; the
shared ``pg_dsn`` / ``clean_pg`` fixtures from ``tests/integration/conftest.py``
are re-used).  The DB is NEVER mocked.

Coverage:
- ``DocumentRepository.purge_by_id``: seed doc + chunks (+ graph_chunk_backrefs +
  visual_pages rows), call purge, assert ALL related rows are gone.
- ``DocumentRepository.list_visible``: owner-only, common, attachment exclusion,
  include_attachments, chunk_count populated.
- ``DocumentRepository.ids_by_session``: returns only docs whose metadata carries
  the requested session_id.
- ``DocumentRepository.delete_by_source_uri``: deletes all versions + cascades;
  count returned.
- ``ChunkRepository.get_by_ids``: id-order preservation; missing ids skipped.
- ``ChunkRepository.get_retrievable``: chunk→document join; ``document_id`` and
  ``source_uri`` populated; id-order preserved.
- ``IngestionRunRepository.create`` → ``finish``: status transition +
  ``finished_at`` set.
- ``EphemeralSessionRepository.expired``: returns only sessions whose
  ``expires_at`` is in the past; fresh sessions are excluded.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EphemeralSessionRepository,
    IngestionRunRepository,
    NewChunk,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro: Any) -> Any:
    """Run an async coroutine in a blocking context (for sync test bodies)."""
    return asyncio.run(coro)


async def _upsert_doc(
    db: Database,
    *,
    source_uri: str,
    checksum: str,
    owner_id: str | None = None,
    visibility: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> uuid.UUID:
    """Insert a document row and return its id.

    Args:
        db: Open database handle.
        source_uri: The document source URI.
        checksum: The document content checksum.
        owner_id: Uploading principal id, if any.
        visibility: ``common`` | ``specific`` | ``None``.
        metadata: Optional document metadata.

    Returns:
        The inserted document's UUID.
    """
    async with db.session() as session:
        repo = DocumentRepository(session)
        doc_id = await repo.upsert_document(
            source_uri=source_uri,
            checksum=checksum,
            title="Test Doc",
            metadata=metadata or {},
        )
        if owner_id is not None or visibility is not None:
            await repo.set_attributes(doc_id, owner_id=owner_id, visibility=visibility)
        await session.commit()
    return doc_id


async def _insert_chunk(
    db: Database,
    doc_id: uuid.UUID,
    *,
    ordinal: int = 0,
    acl_tags: list[str] | None = None,
) -> uuid.UUID:
    """Insert a single chunk for ``doc_id`` and return its id.

    Args:
        db: Open database handle.
        doc_id: The owning document UUID.
        ordinal: Position within the document.
        acl_tags: ACL tag list for this chunk.

    Returns:
        The inserted chunk's UUID.
    """
    async with db.session() as session:
        repo = ChunkRepository(session)
        ids = await repo.bulk_upsert_chunks(
            [
                NewChunk(
                    document_id=doc_id,
                    ordinal=ordinal,
                    content=f"chunk content ordinal={ordinal}",
                    dense_embedding=[0.0] * 1024,
                    acl_tags=acl_tags or ["public"],
                )
            ]
        )
        await session.commit()
    return ids[0]


# ---------------------------------------------------------------------------
# DocumentRepository.purge_by_id
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_purge_by_id_removes_doc_and_all_chunks(clean_pg: str) -> None:
    """``purge_by_id`` deletes the document row and all its chunks via CASCADE."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(
                db, source_uri="file:///purge/doc.pdf", checksum="purge-cs-1"
            )
            await _insert_chunk(db, doc_id, ordinal=0)
            await _insert_chunk(db, doc_id, ordinal=1)

            async with db.session() as session:
                repo = DocumentRepository(session)
                source_uri = await repo.purge_by_id(doc_id)
                await session.commit()

            assert source_uri == "file:///purge/doc.pdf"

            # The document row is gone.
            async with db.session() as session:
                repo = DocumentRepository(session)
                row = await repo.get_by_id(doc_id)
            assert row is None, "document row should have been deleted"

            # All chunks are gone (CASCADE).
            engine = sqlalchemy.create_engine(clean_pg)
            with engine.connect() as conn:
                count = conn.execute(
                    text("SELECT count(*) FROM iw_rag.chunks WHERE document_id = :doc_id"),
                    {"doc_id": doc_id},
                ).scalar()
            engine.dispose()
            assert count == 0, f"expected 0 chunks after purge, got {count}"
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_purge_by_id_removes_graph_chunk_backrefs(clean_pg: str) -> None:
    """``purge_by_id`` deletes ``graph_chunk_backrefs`` rows referencing the doc's chunks."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(
                db, source_uri="file:///purge/graph.pdf", checksum="purge-cs-graph"
            )
            chunk_id = await _insert_chunk(db, doc_id, ordinal=0)

            # Seed a graph_chunk_backrefs row referencing this chunk (text id).
            engine = sqlalchemy.create_engine(clean_pg)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO iw_rag.graph_chunk_backrefs "
                        "(entity_name, chunk_id, graph_name) "
                        "VALUES ('TestEntity', :cid, 'iw_rag_graph')"
                    ),
                    {"cid": str(chunk_id)},
                )
            engine.dispose()

            async with db.session() as session:
                repo = DocumentRepository(session)
                await repo.purge_by_id(doc_id)
                await session.commit()

            # graph_chunk_backrefs row must be gone.
            engine2 = sqlalchemy.create_engine(clean_pg)
            with engine2.connect() as conn:
                count = conn.execute(
                    text("SELECT count(*) FROM iw_rag.graph_chunk_backrefs WHERE chunk_id = :cid"),
                    {"cid": str(chunk_id)},
                ).scalar()
            engine2.dispose()
            assert count == 0, f"expected 0 backrefs after purge, got {count}"
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_purge_by_id_removes_visual_pages(clean_pg: str) -> None:
    """``purge_by_id`` deletes ``visual_pages`` rows keyed on the document's source_uri."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            source_uri = "file:///purge/visual.pdf"
            doc_id = await _upsert_doc(db, source_uri=source_uri, checksum="purge-cs-visual")
            chunk_id = await _insert_chunk(db, doc_id, ordinal=0)

            # Seed a visual_pages row for this source_uri.
            # Build the vector literal inline to avoid the :param::vector
            # notation that psycopg3 misparses (it reads ":" as a bind-param
            # prefix even inside a cast expression).
            proxy_vector_literal = "[" + ",".join(["0.1"] * 128) + "]"
            engine = sqlalchemy.create_engine(clean_pg)
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO iw_rag.visual_pages "
                        "(chunk_id, source_uri, page, content, acl_tags, "
                        "pooled_vectors, quantized_vectors, proxy_vector, "
                        "bbox, page_image_ref, metadata) "
                        f"VALUES (:cid, :uri, 1, 'page text', '{{}}', "
                        f"'[]'::jsonb, '[]'::jsonb, '{proxy_vector_literal}'::vector, "
                        f"'{{}}' ::jsonb, 'ref.png', '{{}}' ::jsonb)"
                    ),
                    {"cid": str(chunk_id), "uri": source_uri},
                )
            engine.dispose()

            async with db.session() as session:
                repo = DocumentRepository(session)
                await repo.purge_by_id(doc_id)
                await session.commit()

            # visual_pages row must be gone.
            engine2 = sqlalchemy.create_engine(clean_pg)
            with engine2.connect() as conn:
                count = conn.execute(
                    text("SELECT count(*) FROM iw_rag.visual_pages WHERE source_uri = :uri"),
                    {"uri": source_uri},
                ).scalar()
            engine2.dispose()
            assert count == 0, f"expected 0 visual_pages after purge, got {count}"
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_purge_by_id_returns_none_for_absent_document(clean_pg: str) -> None:
    """``purge_by_id`` returns ``None`` when the document does not exist."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = DocumentRepository(session)
                result = await repo.purge_by_id(uuid.uuid4())
                await session.commit()
            assert result is None
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# DocumentRepository.list_visible
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_visible_returns_owned_and_common_docs(clean_pg: str) -> None:
    """``list_visible`` returns docs owned by the principal OR with ``visibility='common'``."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            # owner doc
            await _upsert_doc(
                db,
                source_uri="file:///list/owned.pdf",
                checksum="lv-cs-1",
                owner_id="user:alice",
                visibility="specific",
            )
            # common doc
            await _upsert_doc(
                db,
                source_uri="file:///list/common.pdf",
                checksum="lv-cs-2",
                owner_id="user:bob",
                visibility="common",
            )
            # another user's specific doc — must NOT appear
            await _upsert_doc(
                db,
                source_uri="file:///list/bob-private.pdf",
                checksum="lv-cs-3",
                owner_id="user:bob",
                visibility="specific",
            )

            async with db.session() as session:
                repo = DocumentRepository(session)
                rows = await repo.list_visible("user:alice")

            uris = {r["source_uri"] for r in rows}
            assert "file:///list/owned.pdf" in uris
            assert "file:///list/common.pdf" in uris
            assert "file:///list/bob-private.pdf" not in uris
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_list_visible_excludes_attachments_by_default(clean_pg: str) -> None:
    """``list_visible`` hides attachment docs unless ``include_attachments=True``."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            await _upsert_doc(
                db,
                source_uri="file:///attach/file.pdf",
                checksum="att-cs-1",
                owner_id="user:alice",
                visibility="specific",
                metadata={"kind": "attachment"},
            )
            await _upsert_doc(
                db,
                source_uri="file:///attach/regular.pdf",
                checksum="att-cs-2",
                owner_id="user:alice",
                visibility="specific",
                metadata={"kind": "document"},
            )

            async with db.session() as session:
                repo = DocumentRepository(session)
                without_attach = await repo.list_visible("user:alice")
                with_attach = await repo.list_visible("user:alice", include_attachments=True)

            uris_without = {r["source_uri"] for r in without_attach}
            uris_with = {r["source_uri"] for r in with_attach}

            assert "file:///attach/file.pdf" not in uris_without
            assert "file:///attach/regular.pdf" in uris_without
            assert "file:///attach/file.pdf" in uris_with
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_list_visible_includes_chunk_count(clean_pg: str) -> None:
    """``list_visible`` returns the ``chunk_count`` subquery value per document."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(
                db,
                source_uri="file:///list/counted.pdf",
                checksum="lv-cs-count",
                owner_id="user:alice",
                visibility="specific",
            )
            await _insert_chunk(db, doc_id, ordinal=0)
            await _insert_chunk(db, doc_id, ordinal=1)
            await _insert_chunk(db, doc_id, ordinal=2)

            async with db.session() as session:
                repo = DocumentRepository(session)
                rows = await repo.list_visible("user:alice")

            matching = [r for r in rows if r["source_uri"] == "file:///list/counted.pdf"]
            assert len(matching) == 1
            assert matching[0]["chunk_count"] == 3
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# DocumentRepository.ids_by_session
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ids_by_session_returns_matching_doc_ids(clean_pg: str) -> None:
    """``ids_by_session`` returns document ids whose metadata carries the session_id."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            session_id = "sess-abc-123"
            other_session_id = "sess-xyz-999"

            doc_id_a = await _upsert_doc(
                db,
                source_uri="file:///sess/a.pdf",
                checksum="sess-cs-a",
                metadata={"session_id": session_id},
            )
            doc_id_b = await _upsert_doc(
                db,
                source_uri="file:///sess/b.pdf",
                checksum="sess-cs-b",
                metadata={"session_id": session_id},
            )
            # Different session — must NOT appear.
            await _upsert_doc(
                db,
                source_uri="file:///sess/c.pdf",
                checksum="sess-cs-c",
                metadata={"session_id": other_session_id},
            )

            async with db.session() as session:
                repo = DocumentRepository(session)
                ids = await repo.ids_by_session(session_id)

            assert set(ids) == {doc_id_a, doc_id_b}
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_ids_by_session_returns_empty_for_unknown_session(clean_pg: str) -> None:
    """``ids_by_session`` returns an empty list when no docs carry the session id."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = DocumentRepository(session)
                ids = await repo.ids_by_session("nonexistent-session")
            assert ids == []
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# DocumentRepository.delete_by_source_uri
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_delete_by_source_uri_removes_all_versions(clean_pg: str) -> None:
    """``delete_by_source_uri`` removes every document with the given source_uri."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            uri = "file:///multi/versioned.pdf"
            doc_id_v1 = await _upsert_doc(db, source_uri=uri, checksum="multi-cs-v1")
            doc_id_v2 = await _upsert_doc(db, source_uri=uri, checksum="multi-cs-v2")
            # Different uri — must survive.
            other_id = await _upsert_doc(
                db, source_uri="file:///multi/other.pdf", checksum="multi-cs-other"
            )

            async with db.session() as session:
                repo = DocumentRepository(session)
                count = await repo.delete_by_source_uri(uri)
                await session.commit()

            assert count == 2

            async with db.session() as session:
                repo = DocumentRepository(session)
                assert await repo.get_by_id(doc_id_v1) is None
                assert await repo.get_by_id(doc_id_v2) is None
                # Other doc survives.
                assert await repo.get_by_id(other_id) is not None
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_delete_by_source_uri_returns_zero_for_unknown_uri(clean_pg: str) -> None:
    """``delete_by_source_uri`` returns 0 when the source_uri is not present."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = DocumentRepository(session)
                count = await repo.delete_by_source_uri("file:///no-such/doc.pdf")
                await session.commit()
            assert count == 0
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# ChunkRepository.get_by_ids — id-order preservation; missing ids skipped
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_by_ids_preserves_input_order(clean_pg: str) -> None:
    """``get_by_ids`` returns chunks in the order of the supplied id sequence."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(db, source_uri="file:///order/doc.pdf", checksum="ord-cs-1")
            cid0 = await _insert_chunk(db, doc_id, ordinal=0)
            cid1 = await _insert_chunk(db, doc_id, ordinal=1)
            cid2 = await _insert_chunk(db, doc_id, ordinal=2)

            # Request in reverse order.
            requested = [cid2, cid0, cid1]
            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_by_ids(requested)

            returned_ids = [r["id"] for r in rows]
            assert returned_ids == requested, (
                f"id order not preserved; expected {requested}, got {returned_ids}"
            )
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_get_by_ids_skips_missing_ids(clean_pg: str) -> None:
    """``get_by_ids`` silently skips ids that have no matching chunk row."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(db, source_uri="file:///skip/doc.pdf", checksum="skip-cs-1")
            real_id = await _insert_chunk(db, doc_id, ordinal=0)
            ghost_id = uuid.uuid4()  # not in the DB

            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_by_ids([real_id, ghost_id])

            assert len(rows) == 1
            assert rows[0]["id"] == real_id
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_get_by_ids_returns_empty_for_empty_input(clean_pg: str) -> None:
    """``get_by_ids`` returns an empty list when called with an empty sequence."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_by_ids([])
            assert rows == []
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# ChunkRepository.get_retrievable — join + document_id + source_uri threading
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_retrievable_threads_document_id_and_source_uri(clean_pg: str) -> None:
    """``get_retrievable`` joins chunk→document and populates ``document_id`` and ``source_uri``."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            source_uri = "file:///retrieve/doc.pdf"
            doc_id = await _upsert_doc(db, source_uri=source_uri, checksum="ret-cs-1")
            cid = await _insert_chunk(db, doc_id, ordinal=0)

            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_retrievable([cid])

            assert len(rows) == 1
            row = rows[0]
            assert row["id"] == cid
            assert str(row["document_id"]) == str(doc_id)
            assert row["source_uri"] == source_uri
            assert "content" in row
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_get_retrievable_preserves_id_order(clean_pg: str) -> None:
    """``get_retrievable`` returns rows in the input id order (rerank survivors order)."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            doc_id = await _upsert_doc(
                db, source_uri="file:///retrieve/order.pdf", checksum="ret-cs-ord"
            )
            cid0 = await _insert_chunk(db, doc_id, ordinal=0)
            cid1 = await _insert_chunk(db, doc_id, ordinal=1)
            cid2 = await _insert_chunk(db, doc_id, ordinal=2)

            requested = [cid2, cid0, cid1]
            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_retrievable(requested)

            returned_ids = [r["id"] for r in rows]
            assert returned_ids == requested
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_get_retrievable_returns_empty_for_empty_input(clean_pg: str) -> None:
    """``get_retrievable`` returns an empty list when called with an empty sequence."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = ChunkRepository(session)
                rows = await repo.get_retrievable([])
            assert rows == []
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# IngestionRunRepository.create → finish: status transition + finished_at
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ingestion_run_create_and_finish(clean_pg: str) -> None:
    """``create`` inserts a ``running`` row; ``finish`` sets status + ``finished_at``."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = IngestionRunRepository(session)
                run_id = await repo.create(
                    corpus_label="test-corpus",
                    parser="docling",
                    embed_model="bge-m3",
                )
                await session.commit()

            # Verify initial ``running`` status.
            engine = sqlalchemy.create_engine(clean_pg)
            with engine.connect() as conn:
                row = conn.execute(
                    text("SELECT status, finished_at FROM iw_rag.ingestion_runs WHERE id = :id"),
                    {"id": run_id},
                ).first()
            engine.dispose()
            assert row is not None
            assert row[0] == "running", f"expected 'running', got {row[0]!r}"
            assert row[1] is None, "finished_at should be NULL initially"

            # Finish the run.
            async with db.session() as session:
                repo = IngestionRunRepository(session)
                await repo.finish(run_id, "success")
                await session.commit()

            # Verify terminal status + finished_at set.
            engine2 = sqlalchemy.create_engine(clean_pg)
            with engine2.connect() as conn:
                row2 = conn.execute(
                    text("SELECT status, finished_at FROM iw_rag.ingestion_runs WHERE id = :id"),
                    {"id": run_id},
                ).first()
            engine2.dispose()
            assert row2 is not None
            assert row2[0] == "success", f"expected 'success', got {row2[0]!r}"
            assert row2[1] is not None, "finished_at should be set after finish()"
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_ingestion_run_finish_can_set_failed_status(clean_pg: str) -> None:
    """``finish`` accepts any terminal status string (e.g. ``'failed'``)."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg))
        try:
            async with db.session() as session:
                repo = IngestionRunRepository(session)
                run_id = await repo.create(
                    corpus_label="test-corpus",
                    parser="docling",
                    embed_model="bge-m3",
                )
                await session.commit()

            async with db.session() as session:
                repo = IngestionRunRepository(session)
                await repo.finish(run_id, "failed")
                await session.commit()

            engine = sqlalchemy.create_engine(clean_pg)
            with engine.connect() as conn:
                status = conn.execute(
                    text("SELECT status FROM iw_rag.ingestion_runs WHERE id = :id"),
                    {"id": run_id},
                ).scalar()
            engine.dispose()
            assert status == "failed"
        finally:
            await db.dispose()

    _run(_run_test())


# ---------------------------------------------------------------------------
# EphemeralSessionRepository.expired — TTL purge query
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_pg_with_sessions(clean_pg: str) -> str:
    """Truncate ephemeral_sessions before the test; return the DSN.

    The shared ``clean_pg`` fixture truncates documents/chunks/ingestion_runs
    but not ephemeral_sessions.  We add that here.

    Args:
        clean_pg: DSN of the pre-truncated container from the parent fixture.

    Returns:
        The same DSN with ephemeral_sessions also truncated.
    """
    engine = sqlalchemy.create_engine(clean_pg)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.ephemeral_sessions"))
    engine.dispose()
    return clean_pg


@pytest.mark.integration
def test_expired_returns_only_expired_sessions(clean_pg_with_sessions: str) -> None:
    """``expired(now)`` returns only sessions whose ``expires_at`` is before ``now``."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg_with_sessions))
        try:
            now = datetime.now(UTC)
            past = now - timedelta(hours=1)
            future = now + timedelta(hours=1)

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                await repo.ensure(
                    "sess-expired",
                    principal_id="user:alice",
                    expires_at=past,
                )
                await repo.ensure(
                    "sess-fresh",
                    principal_id="user:bob",
                    expires_at=future,
                )
                await session.commit()

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                expired_ids = await repo.expired(now)

            assert expired_ids == ["sess-expired"], (
                f"expected only the expired session; got {expired_ids}"
            )
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_expired_returns_empty_when_no_sessions_expired(clean_pg_with_sessions: str) -> None:
    """``expired(now)`` returns an empty list when all sessions are still fresh."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg_with_sessions))
        try:
            now = datetime.now(UTC)
            future = now + timedelta(hours=2)

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                await repo.ensure(
                    "sess-still-fresh",
                    principal_id="user:alice",
                    expires_at=future,
                )
                await session.commit()

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                expired_ids = await repo.expired(now)

            assert expired_ids == []
        finally:
            await db.dispose()

    _run(_run_test())


@pytest.mark.integration
def test_expired_returns_multiple_expired_sessions(clean_pg_with_sessions: str) -> None:
    """``expired(now)`` returns all expired session ids when several are past TTL."""

    async def _run_test() -> None:
        db = Database(DatabaseSettings(dsn=clean_pg_with_sessions))
        try:
            now = datetime.now(UTC)
            past = now - timedelta(hours=1)

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                await repo.ensure(
                    "sess-old-1",
                    principal_id="user:a",
                    expires_at=past,
                )
                await repo.ensure(
                    "sess-old-2",
                    principal_id="user:b",
                    expires_at=past,
                )
                await session.commit()

            async with db.session() as session:
                repo = EphemeralSessionRepository(session)
                expired_ids = await repo.expired(now)

            assert set(expired_ids) == {"sess-old-1", "sess-old-2"}
        finally:
            await db.dispose()

    _run(_run_test())
