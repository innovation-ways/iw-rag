"""Integration tests for DocumentService write-path authorisation negatives (Wave 1).

These tests cover the owner-only / cross-user-session security boundaries on the
write path of :class:`~iw_rag.api.documents.DocumentService`:

* ``delete`` — raises :class:`~iw_rag.domain.errors.NotFoundError` when the
  document is not visible to the caller (existence is never leaked), and raises
  :class:`~iw_rag.domain.errors.AuthorizationError` when the document is visible
  (``common``) but owned by someone else and the caller is not an operator.
* ``attach`` — raises :class:`~iw_rag.domain.errors.AuthorizationError` when the
  chat session is already owned by a different principal.
* ``purge_session`` — raises :class:`~iw_rag.domain.errors.AuthorizationError`
  when the session is owned by a different principal.

All tests run against a real ParadeDB testcontainers instance (``clean_pg``
fixture) with inert fakes for the embedder and chunker; the DB layer is never
mocked (tests/CLAUDE.md critical rule).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import text

from iw_rag.api.auth import PUBLIC_GROUP, user_group
from iw_rag.api.documents import CharTokenizer, DocumentService
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, NotFoundError
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EphemeralSessionRepository,
    NewChunk,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fakes — bypass embedder/chunker/parser; tests seed rows directly
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """A no-op embedder. Tests seed chunks with a deterministic 1024-d vector."""

    async def encode(self, text: str) -> list[float]:  # noqa: ARG002
        """Return a zero vector (never called in write-authz tests)."""
        return [0.0] * EMBEDDING_DIMS

    async def health(self) -> bool:
        """Report healthy (not exercised in write-authz tests)."""
        return True


def _unit_vec(seed: int) -> list[float]:
    """A 1024-d unit vector with 1.0 at position ``seed % 1024``.

    Args:
        seed: Integer seed; determines which dimension is set to 1.0.

    Returns:
        A 1024-element list with one 1.0 at index ``seed % 1024``.
    """
    vec = [0.0] * EMBEDDING_DIMS
    vec[seed % EMBEDDING_DIMS] = 1.0
    return vec


def _make_service(clean_pg: str, tmp_path: Path) -> DocumentService:
    """Build a DocumentService with inert fakes for embed/chunker/pipeline.

    The pipeline is built but never invoked; tests insert chunks directly so
    the embed and chunker paths are not exercised here.  A real
    :class:`~iw_rag.storage.db.Database` is used so the repository layer runs
    against a real Postgres.

    Args:
        clean_pg: DSN of the clean testcontainer Postgres instance.
        tmp_path: A temporary directory used as the upload staging area.

    Returns:
        A :class:`~iw_rag.api.documents.DocumentService` backed by real Postgres
        and inert embed/chunker fakes.
    """
    return DocumentService(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64, parent_max_tokens=256), CharTokenizer()),
        upload_dir=str(tmp_path),
    )


async def _seed_doc_with_staged(
    clean_pg: str,
    *,
    upload_dir: Path,
    acl_tags: list[str],
    owner_id: str | None = None,
    visibility: str | None = None,
) -> tuple[str, Path]:
    """Seed a document + one chunk + actual staged bytes; return ``(doc_id, path)``.

    Writes ``hello.txt`` into ``upload_dir/<doc_uuid>/`` and stamps the path
    into the document's ``metadata.stored_path`` (the field that
    :meth:`~iw_rag.api.documents.DocumentService.delete` reads when cleaning up
    staged bytes after a successful deletion).

    Args:
        clean_pg: DSN of the test container.
        upload_dir: Directory where the staged bytes are written.
        acl_tags: Tags stamped onto the chunk (the document's effective ACL).
        owner_id: Optional uploading principal (``None`` for corpus docs).
        visibility: Optional visibility (``common`` / ``specific`` / ``None``).

    Returns:
        ``(document_id, stored_path)`` — the document id as a string and the
        absolute path to the staged bytes file.
    """
    doc_uuid = uuid.uuid4()
    stored_dir = upload_dir / doc_uuid.hex
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / "hello.txt"
    stored_path.write_bytes(b"hello world\n")
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://test/{doc_uuid}/hello.txt",
                checksum=f"cs-{doc_uuid}",
            )
            await docs.set_attributes(
                doc_id,
                owner_id=owner_id,
                visibility=visibility,
                metadata={
                    "filename": "hello.txt",
                    "mime": "text/plain",
                    "size_bytes": 12,
                    "kind": "upload",
                    "stored_path": str(stored_path),
                },
            )
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=doc_id,
                        ordinal=0,
                        content="hello world",
                        dense_embedding=_unit_vec(0),
                        acl_tags=acl_tags,
                    ),
                ]
            )
            await session.commit()
    finally:
        await db.dispose()
    return str(doc_id), stored_path


async def _seed_ephemeral_session(
    clean_pg: str,
    *,
    session_id: str,
    principal_id: str,
    ttl_hours: int = 1,
) -> None:
    """Insert an ephemeral-session ownership row into the DB.

    Args:
        clean_pg: DSN of the test container.
        session_id: The chat session id to register.
        principal_id: The principal that owns the session.
        ttl_hours: Hours from now until the session expires (default 1).
    """
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            repo = EphemeralSessionRepository(session)
            await repo.ensure(
                session_id, principal_id, datetime.now(UTC) + timedelta(hours=ttl_hours)
            )
            await session.commit()
    finally:
        await db.dispose()


def _truncate_ephemeral_sessions(pg_dsn: str) -> None:
    """Truncate the ephemeral_sessions table for inter-test isolation.

    ``clean_pg`` does not truncate ``ephemeral_sessions`` (it only truncates
    documents, chunks, and ingestion_runs).  Tests that seed session rows
    call this helper in their setup to start from a known-clean state.

    Args:
        pg_dsn: DSN of the test container Postgres.
    """
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.ephemeral_sessions RESTART IDENTITY CASCADE"))
    engine.dispose()


# ---------------------------------------------------------------------------
# delete — NotFoundError when document is not visible
# ---------------------------------------------------------------------------


async def test_delete_not_visible_doc_raises_not_found(clean_pg: str, tmp_path: Path) -> None:
    """Delete of a ``specific`` (user:alice) doc by bob raises NotFoundError.

    Bob cannot see alice's ``specific`` document, so ``_visible_to`` returns
    ``False`` and the service raises ``NotFoundError("document not found")`` —
    existence is never leaked (the same message shape as a missing document).
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
    )
    with pytest.raises(NotFoundError, match="document not found"):
        await svc.delete(bob_ctx, doc_id)


async def test_delete_non_existent_doc_raises_not_found(clean_pg: str, tmp_path: Path) -> None:
    """Delete of an unknown document id raises NotFoundError.

    The unknown id resolves to a real UUID format but has no row in the DB.
    The service raises ``NotFoundError("document not found")`` without leaking
    whether any row exists.
    """
    svc = _make_service(clean_pg, tmp_path)
    ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    with pytest.raises(NotFoundError, match="document not found"):
        await svc.delete(ctx, str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# delete — AuthorizationError when visible but non-owner non-operator
# ---------------------------------------------------------------------------


async def test_delete_common_doc_by_non_owner_raises_authorization_error(
    clean_pg: str, tmp_path: Path
) -> None:
    """Delete of a ``common`` doc by a non-owner non-operator raises AuthorizationError.

    Bob can SEE alice's ``common`` document (``_visible_to`` returns ``True``
    because ``visibility == "common"``), but he is not the owner and not an
    operator, so the service raises
    ``AuthorizationError("only the owner may delete this document")``.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[PUBLIC_GROUP],
        owner_id="alice",
        visibility="common",
    )
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
        is_operator=False,
    )
    with pytest.raises(AuthorizationError, match="only the owner may delete this document"):
        await svc.delete(bob_ctx, doc_id)

    # Post-condition: the row must still exist after a denied delete.
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            row = await DocumentRepository(session).get_by_id(uuid.UUID(doc_id))
    finally:
        await db.dispose()
    assert row is not None, "document must still exist after a denied delete"
    assert row["owner_id"] == "alice", "row owner must remain 'alice' — delete was denied"


# ---------------------------------------------------------------------------
# delete — positive control: owner CAN delete (proves the negatives are not vacuous)
# ---------------------------------------------------------------------------


async def test_delete_owner_succeeds(clean_pg: str, tmp_path: Path) -> None:
    """The owner of a ``specific`` document can delete it successfully.

    This positive control proves the negative tests above are not vacuous: the
    same code path that raises for non-owners succeeds for the actual owner.
    After deletion the document row is gone from the DB.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    alice_ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    # Must not raise.
    await svc.delete(alice_ctx, doc_id)

    # Post-condition: the row must be gone.
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            row = await DocumentRepository(session).get_by_id(uuid.UUID(doc_id))
    finally:
        await db.dispose()
    assert row is None, "document row must be absent after a successful delete"


# ---------------------------------------------------------------------------
# attach — AuthorizationError when session belongs to another user
# ---------------------------------------------------------------------------


async def test_attach_to_another_users_session_raises_authorization_error(
    clean_pg: str, tmp_path: Path
) -> None:
    """Attaching to a chat session owned by another principal raises AuthorizationError.

    Alice's session is pre-registered in the ephemeral-sessions table.  When
    bob calls :meth:`~iw_rag.api.documents.DocumentService.attach` with the
    same session_id, the service detects the principal mismatch and raises
    ``AuthorizationError("this chat session belongs to another user")``.
    """
    _truncate_ephemeral_sessions(clean_pg)
    session_id = f"chat-authz-{uuid.uuid4().hex[:8]}"
    # Seed alice's ownership row before bob tries to attach.
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    svc = _make_service(clean_pg, tmp_path)
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
    )
    with pytest.raises(AuthorizationError, match="another user"):
        await svc.attach(
            bob_ctx,
            session_id=session_id,
            filename="test.txt",
            data=b"hello",
        )


# ---------------------------------------------------------------------------
# purge_session — AuthorizationError when session belongs to another user
# ---------------------------------------------------------------------------


async def test_purge_session_of_another_user_raises_authorization_error(
    clean_pg: str, tmp_path: Path
) -> None:
    """Purging a chat session owned by another principal raises AuthorizationError.

    Alice owns the session.  When bob calls
    :meth:`~iw_rag.api.documents.DocumentService.purge_session` with that
    session_id, the service detects the mismatch and raises
    ``AuthorizationError("cannot purge another user's chat session")``.
    """
    _truncate_ephemeral_sessions(clean_pg)
    session_id = f"chat-purge-authz-{uuid.uuid4().hex[:8]}"
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    svc = _make_service(clean_pg, tmp_path)
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
    )
    with pytest.raises(AuthorizationError, match="another user"):
        await svc.purge_session(bob_ctx, session_id)

    # Post-condition: alice's session row must still be present.
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            repo = EphemeralSessionRepository(session)
            row = await repo.get(session_id)
    finally:
        await db.dispose()
    assert row is not None, "session row must survive a denied purge"
    assert row["principal_id"] == "alice"


# ---------------------------------------------------------------------------
# purge_session — positive control: owner CAN purge (not vacuous)
# ---------------------------------------------------------------------------


async def test_purge_session_owner_succeeds(clean_pg: str, tmp_path: Path) -> None:
    """The session owner can purge their own chat session without error.

    This positive control proves the negative test above is not vacuous:
    the same code path that raises for non-owners succeeds for the actual owner.
    After purging, the session row is gone from the DB.
    """
    _truncate_ephemeral_sessions(clean_pg)
    session_id = f"chat-purge-ok-{uuid.uuid4().hex[:8]}"
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    svc = _make_service(clean_pg, tmp_path)
    alice_ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    removed = await svc.purge_session(alice_ctx, session_id)
    # No attachment docs were seeded, so 0 docs are removed.
    assert removed == 0

    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            row = await EphemeralSessionRepository(session).get(session_id)
    finally:
        await db.dispose()
    assert row is None, "session row must be gone after a successful purge"
