"""Document-management service + online upload chunker seam (WI-DOCMGMT / I-00131).

This module owns two closely-related concerns for user-uploaded documents:

1. **Chunker seam** (``online_chunk_config`` / ``make_online_chunker``) — the
   online counterpart of the offline ``IngestionPipeline``
   (``iw_rag.ingestion.pipeline``).  It builds a ``HybridChunker`` pre-configured
   for the runtime constraints of user uploads.

   The online path uses a ``CharTokenizer`` (1 token == 1 char) — the chunker
   does not share a BGE-M3 tokenizer with the embedder the way the offline path
   does — and the chunker output is consumed by small-to-big parent expansion, so
   a parent that exceeds ``RetrievalSettings.parent_expansion_max_chars`` would
   silently overflow the generator's ``max_model_len`` and trip a 503 (I-00131
   design §"Root Cause Analysis" cause 1).  We therefore:

     a. Cap the leaf size via ``IngestionSettings.online_chunk_leaf_tokens``;
     b. Cap the accumulated parent size at the **same** char budget the runtime
        gate enforces (``RetrievalSettings.parent_expansion_max_chars``) so that
        bounded parents always fit the expansion budget.  This is the single
        source of truth — there is no second literal to drift.

   The offline ``IngestionPipeline`` is intentionally left untouched
   (``ChunkConfig.parent_max_tokens=None`` by default): offline chunking was
   never the bug and is out of scope.

2. **DocumentService** — the user-facing document layer behind ``/documents``.  It
   runs *inside the API container*, so it ingests with the dependency-light online
   path: parse with :func:`~iw_rag.ingestion.light_parse.select_light_parser`
   (text/markdown native, PDF via ``pypdf``, images via OCR), chunk with
   :func:`make_online_chunker` (parents bounded — see above), embed with the pooled
   :class:`~iw_rag.ingestion.http_embed.InfinityDocumentEmbedder` (same BGE-M3 space
   as the corpus), and persist through the existing ``IngestionPipeline``.

   ACL is the keystone: a document's chunk ``acl_tags`` are derived solely from the
   upload's *visibility* + the authenticated owner — never from content.
   ``common`` → ``public`` (every principal carries it); ``specific`` →
   ``user:<owner_id>`` (only the owner carries it).  Ephemeral chat attachments use
   a server-issued ``session:<id>`` tag.

   Delete purges the document and every derived artifact (chunks + embeddings, plus
   ``visual_pages`` and ``graph_chunk_backrefs``), so a removed document is
   unqueryable through every retrieval route.
"""

from __future__ import annotations

import logging
import mimetypes
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, NotFoundError, ValidationError
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EmbeddingClient
from iw_rag.ingestion.enrich import IngestContext
from iw_rag.ingestion.light_parse import OcrEngine, select_light_parser
from iw_rag.ingestion.parse import ParseConfig, ParseHints, Parser, SourceRef
from iw_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
from iw_rag.settings import Settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import DocumentRepository, EphemeralSessionRepository

if TYPE_CHECKING:
    from iw_rag.ingestion.caption import CaptionEngine

logger = logging.getLogger(__name__)

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


# --------------------------------------------------------------------------- seam
class CharTokenizer:
    """Online-path tokenizer: 1 token == 1 char (deterministic, model-free)."""

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of char ordinals."""
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        """Decode char ordinals back to text."""
        return "".join(chr(i) for i in ids)


def online_chunk_config(settings: Settings) -> ChunkConfig:
    """Build the ``ChunkConfig`` used by the online upload path (I-00131).

    The parent-size cap is sourced from
    ``RetrievalSettings.parent_expansion_max_chars`` — the single source of
    truth shared with the runtime small-to-big expansion gate.  The leaf size
    comes from ``IngestionSettings.online_chunk_leaf_tokens`` (default 1600,
    larger than the offline default because the online path uses
    ``CharTokenizer`` and the Late-Chunking embedder pools at the doc level).

    Args:
        settings: The validated ``Settings`` instance.

    Returns:
        A ``ChunkConfig`` ready to be paired with ``CharTokenizer``.
    """
    return ChunkConfig(
        leaf_tokens=settings.ingestion.online_chunk_leaf_tokens,
        parent_max_tokens=settings.retrieval.parent_expansion_max_chars,
    )


def make_online_chunker(settings: Settings) -> HybridChunker:
    """Build the ``HybridChunker`` for the online upload path (I-00131).

    Wires the bounded ``ChunkConfig`` (see :func:`online_chunk_config`) with
    the online ``CharTokenizer``.  Pair with a parser via
    ``iw_rag.ingestion.parse.select_parser``.

    Args:
        settings: The validated ``Settings`` instance.

    Returns:
        A ``HybridChunker`` whose parents are guaranteed to fit the runtime
        expansion budget.
    """
    return HybridChunker(online_chunk_config(settings), CharTokenizer())


# ----------------------------------------------------------------------- ACL/util
def session_group(session_id: str) -> str:
    """The synthetic ACL group that entitles a chat session's ephemeral attachments."""
    return f"session:{session_id}"


def safe_filename(name: str) -> str:
    """Reduce an uploaded filename to a safe basename (no path traversal)."""
    base = Path(name).name or "upload"
    cleaned = _UNSAFE.sub("_", base).strip("._") or "upload"
    return cleaned[:200]


def acl_tags_for(*, visibility: str, owner_id: str, session_id: str | None) -> list[str]:
    """Compute chunk ACL tags from visibility + owner (the only ACL inputs).

    Args:
        visibility: ``common`` or ``specific``.
        owner_id: The uploading principal id.
        session_id: When set, the document is a chat-scoped ephemeral attachment;
            its only tag is ``session:<session_id>`` (invisible outside that chat).

    Returns:
        The entitlement tags written to every chunk of the document.
    """
    if session_id is not None:
        return [f"session:{session_id}"]
    if visibility == "common":
        return ["public"]
    return [f"user:{owner_id}"]


class _FixedAclResolver:
    """An :class:`~iw_rag.ingestion.enrich.AclResolver` returning fixed tags."""

    def __init__(self, tags: list[str]) -> None:
        self._tags = list(tags)

    def resolve(self, source: SourceRef) -> list[str]:
        """Return the fixed entitlement tags for this upload."""
        return list(self._tags)


@dataclass(frozen=True, slots=True)
class DocumentInfo:
    """A document row as surfaced to the management UI / API."""

    id: str
    filename: str
    title: str | None
    mime: str
    visibility: str | None
    owner_id: str | None
    size_bytes: int
    chunk_count: int
    created_at: str
    is_owner: bool
    kind: str

    @classmethod
    def from_row(cls, row: dict[str, Any], principal_id: str) -> DocumentInfo:
        """Build from a ``documents`` row (with ``chunk_count``) for a principal."""
        meta = row.get("metadata") or {}
        created = row.get("created_at")
        return cls(
            id=str(row["id"]),
            filename=str(meta.get("filename") or row.get("title") or row["source_uri"]),
            title=row.get("title"),
            mime=str(meta.get("mime") or "application/octet-stream"),
            visibility=row.get("visibility"),
            owner_id=row.get("owner_id"),
            size_bytes=int(meta.get("size_bytes") or 0),
            chunk_count=int(row.get("chunk_count") or 0),
            created_at=created.isoformat() if created is not None else "",
            is_owner=row.get("owner_id") == principal_id,
            kind=str(meta.get("kind") or "upload"),
        )


@dataclass(frozen=True, slots=True)
class RawDocument:
    """A staged upload's bytes location for view/download."""

    path: Path
    mime: str
    filename: str


class DocumentService:
    """Upload / list / delete / view documents under the caller's ACL (WI-DOCMGMT)."""

    def __init__(
        self,
        db: Database,
        embedder: EmbeddingClient,
        chunker: HybridChunker,
        *,
        upload_dir: str,
        ocr: OcrEngine | None = None,
        caption_engine: CaptionEngine | None = None,
        max_upload_bytes: int = 50 * 1024 * 1024,
        embed_model_label: str = "bge-m3-pooled",
        ephemeral_ttl_minutes: int = 240,
    ) -> None:
        """Wire the service collaborators.

        Args:
            db: The database handle (owns sessions/transactions).
            embedder: The pooled document embedder (Infinity-backed).
            chunker: The bounded online chunker (build via :func:`make_online_chunker`
                so uploaded-document parents fit the runtime expansion budget — I-00131).
            upload_dir: Directory where uploaded bytes are staged.
            ocr: OCR engine for image/scanned inputs (default Tesseract).
            caption_engine: Optional VLM caption fallback for images OCR can't
                read (visual mode); None disables it.
            max_upload_bytes: Reject uploads larger than this.
            embed_model_label: Recorded on ingestion-run rows.
            ephemeral_ttl_minutes: Chat-attachment session lifetime before the
                TTL sweep purges it.
        """
        self._db = db
        self._embedder = embedder
        self._upload_dir = Path(upload_dir)
        self._ocr = ocr
        self._caption_engine = caption_engine
        self._max_upload_bytes = max_upload_bytes
        self._ephemeral_ttl = timedelta(minutes=ephemeral_ttl_minutes)
        self._parse_cfg = ParseConfig()
        self._chunker = chunker
        self._pipeline = IngestionPipeline(
            parser_factory=self._parser_factory,
            chunker=chunker,
            embedder=embedder,
            db=db,
            cfg=IngestionConfig(concurrency=1, embed_model_label=embed_model_label),
        )

    def _parser_factory(self, source: SourceRef) -> Parser:
        """Resolve a parser for an uploaded source (dependency-light path)."""
        return select_light_parser(source, self._parse_cfg, self._ocr, self._caption_engine)

    @staticmethod
    def _visible_to(row: dict[str, Any], ctx: AccessContext) -> bool:
        """Whether ``ctx`` may see this document in the management UI."""
        return row.get("owner_id") == ctx.principal_id or row.get("visibility") == "common"

    def _guess_mime(self, filename: str, declared: str | None) -> str:
        """Pick a MIME type from the declared content-type or the filename."""
        if declared and declared != "application/octet-stream":
            return declared
        guessed, _ = mimetypes.guess_type(filename)
        return guessed or "application/octet-stream"

    async def upload(
        self,
        ctx: AccessContext,
        *,
        filename: str,
        data: bytes,
        visibility: str,
        declared_mime: str | None = None,
        kind: str = "upload",
        session_id: str | None = None,
    ) -> DocumentInfo:
        """Stage, parse, chunk, embed, and persist one uploaded document.

        Args:
            ctx: The authenticated uploader (its ``principal_id`` is the owner).
            filename: The original filename.
            data: The raw file bytes.
            visibility: ``common`` (→ public) or ``specific`` (→ owner-only).
            declared_mime: The browser-declared content-type, if any.
            kind: ``upload`` (managed doc) or ``attachment`` (ephemeral).
            session_id: Chat session id for ``attachment`` kind (ACL scoping).

        Returns:
            The persisted document's :class:`DocumentInfo`.

        Raises:
            ValidationError: Empty/oversized file, unsupported type, or no content
                could be extracted (parse/embed failure).
        """
        if not data:
            raise ValidationError("uploaded file is empty")
        if len(data) > self._max_upload_bytes:
            raise ValidationError(
                f"file too large ({len(data)} bytes > {self._max_upload_bytes} limit)"
            )
        if visibility not in ("common", "specific"):
            raise ValidationError("visibility must be 'common' or 'specific'")

        doc_uuid = uuid.uuid4().hex
        fname = safe_filename(filename)
        mime = self._guess_mime(fname, declared_mime)
        stored = self._upload_dir / doc_uuid / fname
        await anyio.to_thread.run_sync(_stage_bytes, stored, data)

        scheme = "attachment" if kind == "attachment" else "upload"
        source = SourceRef(
            uri=f"{scheme}://{ctx.principal_id}/{doc_uuid}/{fname}",
            mime=mime,
            hints=ParseHints(),
            path=stored,
        )
        tags = acl_tags_for(visibility=visibility, owner_id=ctx.principal_id, session_id=session_id)
        ictx = IngestContext(
            acl_resolver=_FixedAclResolver(tags),
            corpus_label=f"{scheme}:{ctx.principal_id}",
            ingestion_run_id=None,
        )

        result = await self._pipeline.ingest_document(source, ictx)
        if result.status == "failed" or result.document_id is None:
            await anyio.to_thread.run_sync(_cleanup_dir, stored.parent)
            raise ValidationError(
                f"could not ingest '{fname}': {result.error or 'no extractable content'}"
            )

        metadata = {
            "filename": fname,
            "size_bytes": len(data),
            "mime": mime,
            "kind": kind,
            "stored_path": str(stored),
            "ocr": mime.startswith("image/"),
            **({"session_id": session_id} if session_id else {}),
        }
        async with self._db.session() as session:
            repo = DocumentRepository(session)
            await repo.set_attributes(
                result.document_id,
                owner_id=ctx.principal_id,
                visibility=visibility,
                metadata=metadata,
            )
            row = await repo.get_by_id(result.document_id)
            await session.commit()

        assert row is not None
        row["chunk_count"] = result.chunk_count
        logger.info(
            "document uploaded: id=%s owner=%s visibility=%s chunks=%s kind=%s",
            result.document_id,
            ctx.principal_id,
            visibility,
            result.chunk_count,
            kind,
        )
        return DocumentInfo.from_row(row, ctx.principal_id)

    async def list_documents(self, ctx: AccessContext) -> list[DocumentInfo]:
        """List documents the caller manages (owned + common), newest first."""
        async with self._db.session() as session:
            rows = await DocumentRepository(session).list_visible(ctx.principal_id)
        return [DocumentInfo.from_row(r, ctx.principal_id) for r in rows]

    async def delete(self, ctx: AccessContext, document_id: str) -> None:
        """Delete a document the caller owns; purge all derived artifacts.

        Raises:
            NotFoundError: If the document does not exist or is not visible to the
                caller (existence is never leaked).
            AuthorizationError: If the document is visible (common) but owned by
                someone else and the caller is not an operator.
        """
        doc_uuid = _parse_uuid(document_id)
        async with self._db.session() as session:
            repo = DocumentRepository(session)
            row = await repo.get_by_id(doc_uuid)
            if row is None or not self._visible_to(row, ctx):
                raise NotFoundError("document not found")
            if row.get("owner_id") != ctx.principal_id and not ctx.is_operator:
                raise AuthorizationError("only the owner may delete this document")
            await repo.purge_by_id(doc_uuid)
            await session.commit()
        # Best-effort removal of the staged bytes.
        meta = row.get("metadata") or {}
        stored = meta.get("stored_path")
        if stored:
            await anyio.to_thread.run_sync(_cleanup_dir, Path(str(stored)).parent)
        logger.info("document deleted: id=%s by=%s", document_id, ctx.principal_id)

    async def get_raw(self, ctx: AccessContext, document_id: str) -> RawDocument:
        """Return the staged bytes location for a document the caller may see.

        Authorization mirrors the **retrieval** ACL: a caller is allowed to
        download iff ``set(chunks.acl_tags) ∩ set(ctx.principal_groups)`` is
        non-empty.  This broadens the prior owner-or-``common`` check
        (``_visible_to``) to also grant corpus docs (no ``owner_id`` /
        ``visibility``) and session attachments (``session:<id>``) that the
        caller could legitimately retrieve, while still refusing callers
        with no overlap (existence is never leaked — the
        ``NotFoundError("document not found")`` message is the same as for
        a missing document).  See CR-00098 §"Desired Behavior" / AC4.

        The caller's ``principal_groups`` are populated by the bearer auth
        path (``augment_groups``) and, on the signed-token path (S04), by
        the authenticator's ``resolve_principal`` (re-resolved server-side
        from the token's ``principal_id`` — never from the token payload).
        The grant-only invariant holds: the tags stamped at ingest
        (``acl_tags_for`` in :mod:`iw_rag.api.documents` —
        ``user:<owner>`` / ``public`` / ``session:<id>`` / corpus group)
        are exactly the tags ``augment_groups`` + the session-group grant
        mint for the corresponding caller, so every owner and every
        ``common`` caller remains authorized.

        Raises:
            NotFoundError: If absent, ACL overlap is empty, or the staged
                bytes are unavailable (preserved message shape — see
                ``acl_tags_for`` / :mod:`iw_rag.domain.errors`).
        """
        doc_uuid = _parse_uuid(document_id)
        async with self._db.session() as session:
            row = await DocumentRepository(session).get_by_id(doc_uuid)
            if row is None:
                raise NotFoundError("document not found")
            doc_tags = await DocumentRepository(session).acl_tags_for_document(doc_uuid)
        if not doc_tags or not (set(doc_tags) & set(ctx.principal_groups)):
            raise NotFoundError("document not found")
        meta = row.get("metadata") or {}
        stored = meta.get("stored_path")
        exists = bool(stored) and await anyio.to_thread.run_sync(_is_file, Path(str(stored)))
        if not exists:
            raise NotFoundError("document content is unavailable")
        return RawDocument(
            path=Path(str(stored)),
            mime=str(meta.get("mime") or "application/octet-stream"),
            filename=str(meta.get("filename") or "document"),
        )

    # ----------------------------------------------------------------- ephemeral
    async def attach(
        self,
        ctx: AccessContext,
        *,
        session_id: str,
        filename: str,
        data: bytes,
        declared_mime: str | None = None,
    ) -> DocumentInfo:
        """Attach a file to a chat session (queryable in-chat, purged on chat end).

        Records (idempotently) that ``session_id`` is owned by the caller, then
        ingests the file with the single ACL tag ``session:<session_id>`` so it is
        visible *only* to a request that presents that session (granted
        server-side after re-validating ownership — see :meth:`session_group_for`).

        Args:
            ctx: The authenticated caller (owns the session + attachment).
            session_id: The chat session id (from the frontend).
            filename: The original filename.
            data: The raw file bytes.
            declared_mime: Browser-declared content-type, if any.

        Returns:
            The persisted attachment's :class:`DocumentInfo`.

        Raises:
            AuthorizationError: If ``session_id`` is already owned by someone else.
            ValidationError: On an empty/oversized/unparseable file.
        """
        await self.sweep_expired()
        expires_at = datetime.now(UTC) + self._ephemeral_ttl
        async with self._db.session() as session:
            repo = EphemeralSessionRepository(session)
            row = await repo.ensure(session_id, ctx.principal_id, expires_at)
            if row["principal_id"] != ctx.principal_id:
                raise AuthorizationError("this chat session belongs to another user")
            await session.commit()
        return await self.upload(
            ctx,
            filename=filename,
            data=data,
            visibility="specific",
            declared_mime=declared_mime,
            kind="attachment",
            session_id=session_id,
        )

    async def session_group_for(self, ctx: AccessContext, session_id: str) -> str | None:
        """Return ``session:<id>`` iff the caller owns a live session, else None.

        This is the *only* place a request gains an ephemeral session group — the
        server grants it after confirming the session exists, is owned by the
        authenticated caller, and has not expired.  Never trusts request input.
        """
        if not session_id:
            return None
        async with self._db.session() as session:
            row = await EphemeralSessionRepository(session).get(session_id)
        if row is None or row["principal_id"] != ctx.principal_id:
            return None
        expires = row.get("expires_at")
        if expires is not None and expires < datetime.now(UTC):
            return None
        return session_group(session_id)

    async def purge_session(self, ctx: AccessContext, session_id: str) -> int:
        """Purge a chat session's attachments + ownership row (owner-only).

        Returns the number of attachment documents removed.  Idempotent: an
        unknown/already-purged session removes nothing and returns 0.

        Raises:
            AuthorizationError: If the session is owned by another principal.
        """
        async with self._db.session() as session:
            ephem = EphemeralSessionRepository(session)
            row = await ephem.get(session_id)
            if row is not None and row["principal_id"] != ctx.principal_id:
                raise AuthorizationError("cannot purge another user's chat session")
            docs = DocumentRepository(session)
            doc_ids = await docs.ids_by_session(session_id)
            stored_dirs: list[Path] = []
            for doc_id in doc_ids:
                doc = await docs.get_by_id(doc_id)
                if doc is not None:
                    stored = (doc.get("metadata") or {}).get("stored_path")
                    if stored:
                        stored_dirs.append(Path(str(stored)).parent)
                await docs.purge_by_id(doc_id)
            await ephem.delete(session_id)
            await session.commit()
        for directory in stored_dirs:
            await anyio.to_thread.run_sync(_cleanup_dir, directory)
        logger.info(
            "ephemeral session purged: session=%s docs=%s by=%s",
            session_id,
            len(doc_ids),
            ctx.principal_id,
        )
        return len(doc_ids)

    async def sweep_expired(self) -> int:
        """Purge all expired ephemeral sessions (TTL backstop). Returns docs removed.

        Safe to call opportunistically (it is invoked on each :meth:`attach`) and
        from a periodic task.  Never raises for a single bad session.
        """
        now = datetime.now(UTC)
        async with self._db.session() as session:
            expired = await EphemeralSessionRepository(session).expired(now)
        total = 0
        for session_id in expired:
            try:
                total += await self._purge_session_unchecked(session_id)
            except Exception:  # noqa: BLE001 - a bad session must not abort the sweep
                logger.warning("sweep failed for session %s", session_id, exc_info=True)
        if expired:
            logger.info("ephemeral TTL sweep purged %s sessions", len(expired))
        return total

    async def _purge_session_unchecked(self, session_id: str) -> int:
        """Purge a session's docs + row without an ownership check (TTL sweep only)."""
        async with self._db.session() as session:
            ephem = EphemeralSessionRepository(session)
            docs = DocumentRepository(session)
            doc_ids = await docs.ids_by_session(session_id)
            dirs: list[Path] = []
            for doc_id in doc_ids:
                doc = await docs.get_by_id(doc_id)
                if doc is not None:
                    stored = (doc.get("metadata") or {}).get("stored_path")
                    if stored:
                        dirs.append(Path(str(stored)).parent)
                await docs.purge_by_id(doc_id)
            await ephem.delete(session_id)
            await session.commit()
        for directory in dirs:
            await anyio.to_thread.run_sync(_cleanup_dir, directory)
        return len(doc_ids)


def _stage_bytes(path: Path, data: bytes) -> None:
    """Write uploaded bytes to ``path`` (creating parent dirs). Blocking — off-loop."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _is_file(path: Path) -> bool:
    """Whether ``path`` is a regular file. Blocking — off-loop."""
    return path.is_file()


def _cleanup_dir(directory: Path) -> None:
    """Remove a staged upload directory (best-effort). Blocking — off-loop."""
    try:
        for child in directory.glob("*"):
            child.unlink(missing_ok=True)
        directory.rmdir()
    except OSError:
        logger.warning("failed to clean staged dir %s", directory, exc_info=True)


def _parse_uuid(value: str) -> uuid.UUID:
    """Parse a document-id string, mapping a bad id to NotFound (never 500)."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError) as exc:
        raise NotFoundError("document not found") from exc


__all__ = [
    "CharTokenizer",
    "DocumentInfo",
    "DocumentService",
    "RawDocument",
    "acl_tags_for",
    "make_online_chunker",
    "online_chunk_config",
    "safe_filename",
    "session_group",
]
