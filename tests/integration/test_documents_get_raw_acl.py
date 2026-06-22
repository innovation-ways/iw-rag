"""Integration tests for ``DocumentService.get_raw`` ACL-overlap authorization
(CR-00098 / S03).

The download endpoint ``GET /documents/{id}/raw`` must mirror the *retrieval*
ACL (CR-00098 §"Desired Behavior" / AC4): a caller is authorized iff their
``principal_groups`` overlap the document's ``chunks.acl_tags``. The previous
narrow gate (``_visible_to`` — owner-or-``common``) refuses corpus docs and
session attachments even when the caller could legitimately retrieve them.

These tests exercise the full service path against a real ParadeDB test
container, seeding documents and chunks directly via the repositories and
asserting the authorization outcome of :meth:`DocumentService.get_raw`. The
fixture's ``tmp_path`` is used as the upload directory so the service can
actually read the staged bytes.

Coverage matrix (AC4 grant-only invariant — every old-gate grant must remain):

* **Owner-uploaded** ``specific`` doc, caller is the owner → ``RawDocument``.
* **Common** doc, any authenticated caller → ``RawDocument`` (``public`` overlap).
* **Group-tagged corpus** doc, caller carries the group → ``RawDocument``.
* **Session attachment** (``session:<id>``), caller owns the session →
  ``RawDocument`` (session group granted by the service after re-validation).
* **Foreign user** with no overlap → ``NotFoundError`` (existence never leaked).
* **Missing staged bytes** → ``NotFoundError("...unavailable")`` (preserved
  error message shape).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from iw_rag.api.auth import PUBLIC_GROUP, user_group
from iw_rag.api.documents import CharTokenizer, DocumentService, RawDocument
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import NotFoundError
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fakes — bypass embedder, chunker, and parser; the test seeds the rows itself
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """A no-op embedder. Tests seed chunks with a deterministic 1024-d vector."""

    async def encode(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * EMBEDDING_DIMS

    async def health(self) -> bool:
        return True


def _unit_vec(seed: int) -> list[float]:
    """A 1024-d unit vector with 1.0 at position ``seed % 1024``."""
    vec = [0.0] * EMBEDDING_DIMS
    vec[seed % EMBEDDING_DIMS] = 1.0
    return vec


def _make_service(clean_pg: str, tmp_path: Path) -> DocumentService:
    """Build a DocumentService whose embed/chunker/pipeline are inert fakes.

    The pipeline is built but never invoked; tests insert chunks directly so
    embed/chunker are not exercised here. A real ``Database`` is used so the
    repo layer (and the new ``acl_tags_for_document`` query) runs against a
    real Postgres.
    """
    return DocumentService(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64, parent_max_tokens=256), CharTokenizer()),
        upload_dir=str(tmp_path),
    )


async def _seed_doc_with_chunks(
    clean_pg: str,
    *,
    acl_tags: list[str],
    owner_id: str | None = None,
    visibility: str | None = None,
    stored_filename: str = "hello.txt",
    staged_bytes: bytes = b"hello world\n",
) -> tuple[str, str, list[str]]:
    """Seed a document + one chunk; write staged bytes; return ids + stored_path.

    Args:
        clean_pg: DSN of the test container.
        acl_tags: Tags stamped onto the chunk (and therefore the document's
            effective ACL — retrieval runs on this denormalized column).
        owner_id: Optional uploading principal (None for corpus docs).
        visibility: Optional visibility (``common``/``specific``/None).
        stored_filename: Filename to write to the upload dir.
        staged_bytes: Content to write.

    Returns:
        ``(document_id, stored_path, acl_tags)`` — the document id as a
        string, the absolute staged-bytes path, and the echo of the tags.
    """
    doc_uuid = uuid.uuid4()
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://test/{doc_uuid}/{stored_filename}",
                checksum=f"cs-{doc_uuid}",
            )
            await docs.set_attributes(doc_id, owner_id=owner_id, visibility=visibility, metadata={})
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
    # Resolve stored_path from the documents row's metadata (set by the service
    # during a real upload). The repo's `set_attributes` here doesn't write a
    # stored_path, so the test files are responsible for the upload dir.
    return str(doc_id), stored_filename, list(acl_tags)


async def _seed_doc_with_staged(
    clean_pg: str,
    *,
    upload_dir: Path,
    acl_tags: list[str],
    owner_id: str | None = None,
    visibility: str | None = None,
) -> tuple[str, Path]:
    """Seed a document + chunk + actual staged bytes (so ``get_raw`` can read them).

    Writes ``hello.txt`` into ``upload_dir/<doc_uuid>/`` and stamps the path
    into the document's ``metadata.stored_path`` (the location ``get_raw``
    reads from).

    Returns:
        ``(document_id, stored_path)`` — the document id as a string and the
        absolute path to the staged bytes.
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


async def _seed_ephemeral_session(clean_pg: str, *, session_id: str, principal_id: str) -> None:
    """Insert an ephemeral-session ownership row (so ``session_group_for`` grants it)."""
    from datetime import UTC, datetime, timedelta

    from iw_rag.storage.repositories import EphemeralSessionRepository

    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            repo = EphemeralSessionRepository(session)
            await repo.ensure(session_id, principal_id, datetime.now(UTC) + timedelta(hours=1))
            await session.commit()
    finally:
        await db.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_get_raw_owner_succeeds(clean_pg: str, tmp_path: Path) -> None:
    """An owner-uploaded doc is downloadable by its owner (``user:<id>`` overlap)."""
    svc = _make_service(clean_pg, tmp_path)
    doc_id, stored_path = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["user:alice"],
        owner_id="alice",
        visibility="specific",
    )
    ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    raw = await svc.get_raw(ctx, doc_id)
    assert isinstance(raw, RawDocument)
    assert raw.path == stored_path
    assert raw.mime == "text/plain"
    assert raw.filename == "hello.txt"


async def test_get_raw_common_succeeds_for_any_authenticated_caller(
    clean_pg: str, tmp_path: Path
) -> None:
    """A ``common`` (``public``-tagged) doc is downloadable by every authenticated caller.

    This pins the grant-only invariant: the old ``_visible_to`` gate
    authorized ``common`` docs and we MUST preserve that — but now via the
    ``public`` group that ``augment_groups`` always mints.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[PUBLIC_GROUP],
        owner_id="alice",  # owner is alice, but bob should still download
        visibility="common",
    )
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
    )
    raw = await svc.get_raw(bob_ctx, doc_id)
    assert isinstance(raw, RawDocument)
    assert raw.filename == "hello.txt"


async def test_get_raw_group_tagged_corpus_doc_succeeds_for_member(
    clean_pg: str, tmp_path: Path
) -> None:
    """An offline-corpus doc with group tags is downloadable by a member of that group.

    This is the headline grant: under the old ``_visible_to`` gate, a corpus
    doc with ``owner_id == NULL`` and ``visibility == NULL`` (no
    owner-or-``common``) was REFUSED — even when the caller's
    ``principal_groups`` overlapped the chunk's ``acl_tags``. The new
    ACL-overlap gate MUST grant.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,  # corpus doc — no owner
        visibility=None,  # corpus doc — not "common"
    )
    finance_ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, "finance", user_group("alice")),
    )
    raw = await svc.get_raw(finance_ctx, doc_id)
    assert isinstance(raw, RawDocument)
    assert raw.filename == "hello.txt"


async def test_get_raw_session_attachment_succeeds_for_session_owner(
    clean_pg: str, tmp_path: Path
) -> None:
    """A ``session:<id>``-tagged attachment is downloadable after the service
    grants the synthetic ``session:<id>`` group.

    Exercises the full S04-ish flow: the caller asks for the session group,
    the service re-validates ownership and adds it to the access context,
    and the ACL overlap then matches.
    """
    svc = _make_service(clean_pg, tmp_path)
    session_id = "chat-1"
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}"],
        owner_id="alice",
        visibility="specific",
    )
    # The service grants the session group only after re-validating ownership —
    # seed the ownership row so the check passes.
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")
    base_ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    # The service grants the session group only after re-validating ownership.
    session_group_str = await svc.session_group_for(base_ctx, session_id)
    assert session_group_str == f"session:{session_id}"
    widened_ctx = AccessContext(
        principal_id=base_ctx.principal_id,
        principal_groups=(*base_ctx.principal_groups, session_group_str),
    )
    raw = await svc.get_raw(widened_ctx, doc_id)
    assert isinstance(raw, RawDocument)
    assert raw.filename == "hello.txt"


async def test_get_raw_denies_non_overlap(clean_pg: str, tmp_path: Path) -> None:
    """A caller without overlapping ``principal_groups`` is refused (NotFoundError).

    Existence is never leaked: the same exception/message shape as a missing
    document. This is the AC4 / keystone invariant.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
    )
    # ``bob`` carries only the synthetic groups; no overlap with {finance}.
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=(PUBLIC_GROUP, user_group("bob")),
    )
    with pytest.raises(NotFoundError, match="document not found") as exc_info:
        await svc.get_raw(bob_ctx, doc_id)
    assert str(exc_info.value) == "document not found"


async def test_get_raw_denies_missing_document_id(clean_pg: str, tmp_path: Path) -> None:
    """An unknown document id maps to NotFoundError (never 500)."""
    svc = _make_service(clean_pg, tmp_path)
    ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    with pytest.raises(NotFoundError, match="document not found"):
        await svc.get_raw(ctx, str(uuid.uuid4()))


async def test_get_raw_denies_when_staged_bytes_missing(clean_pg: str, tmp_path: Path) -> None:
    """When the ACL overlap authorises the caller but the staged bytes are gone,
    ``get_raw`` raises NotFoundError with the existing ``content is unavailable``
    message — the historical contract is preserved.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, stored_path = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["user:alice"],
        owner_id="alice",
        visibility="specific",
    )
    # Wipe the staged bytes — the metadata.stored_path still resolves to a
    # path, but the file is gone.
    stored_path.unlink()

    ctx = AccessContext(
        principal_id="alice",
        principal_groups=(PUBLIC_GROUP, user_group("alice")),
    )
    with pytest.raises(NotFoundError, match="document content is unavailable") as exc_info:
        await svc.get_raw(ctx, doc_id)
    assert str(exc_info.value) == "document content is unavailable"


# ---------------------------------------------------------------------------
# Repository: acl_tags_for_document
# ---------------------------------------------------------------------------


async def test_acl_tags_for_document_returns_distinct_union(clean_pg: str) -> None:
    """``acl_tags_for_document`` returns the distinct union of every chunk's
    ``acl_tags`` for the given document, in a single query.
    """
    doc_uuid = uuid.uuid4()
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://test/{doc_uuid}/multi.txt",
                checksum=f"cs-{doc_uuid}",
            )
            # Two chunks with overlapping tag sets — the distinct union
            # should be {finance, public} (and 'finance' should appear once,
            # not twice).
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=doc_id,
                        ordinal=0,
                        content="a",
                        dense_embedding=_unit_vec(0),
                        acl_tags=["finance", PUBLIC_GROUP],
                    ),
                    NewChunk(
                        document_id=doc_id,
                        ordinal=1,
                        content="b",
                        dense_embedding=_unit_vec(1),
                        acl_tags=["finance", "public"],
                    ),
                ]
            )
            await session.commit()

            tags = await DocumentRepository(session).acl_tags_for_document(doc_id)
    finally:
        await db.dispose()

    assert set(tags) == {"finance", "public"}
    # 'finance' must appear exactly once in the distinct set.
    assert tags.count("finance") == 1


async def test_acl_tags_for_document_empty_when_no_chunks(clean_pg: str) -> None:
    """A document with zero chunks yields an empty list (not an error)."""
    doc_uuid = uuid.uuid4()
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://test/{doc_uuid}/empty.txt",
                checksum=f"cs-{doc_uuid}",
            )
            await session.commit()

            tags = await DocumentRepository(session).acl_tags_for_document(doc_id)
    finally:
        await db.dispose()
    assert tags == []
