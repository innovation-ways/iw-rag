"""Repositories — the typed query surface over the ``iw_rag`` tables.

Repositories own all CRUD the higher layers use; raw BM25/RRF SQL is *not* here
(it is encapsulated in ``retrieval/sparse.py`` + ``retrieval/fusion.py`` per the
repo-structure §2 exception). Re-ingest is idempotent on
``(source_uri, checksum)`` (lld-data-model §9), so :meth:`DocumentRepository.
upsert_document` is a no-op insert on conflict.

Task 3.4 adds :class:`AuditRepository` — an **insert-only** surface for the
append-only decision trail.  No UPDATE/DELETE method exists; enforcement is also
at the DB layer (trigger + REVOKE, ``0003_audit_log.py``).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any, cast

from sqlalchemy import delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.storage import schema


@dataclass(slots=True)
class NewChunk:
    """An insertable chunk row.

    Attributes:
        document_id: Owning document id.
        ordinal: Position within the parent/document.
        content: Chunk text.
        dense_embedding: 1024-d BGE-M3 dense vector.
        parent_id: Parent chunk id (small-to-big), or None for a leaf root.
        context_prefix: Optional contextual-retrieval prefix.
        sparse_weights: Optional BGE-M3 learned-sparse term→weight map.
        content_type: ``text|table|figure_caption``.
        section_path: Section breadcrumb.
        acl_tags: Group entitlements (denormalised from DOCUMENT_ACL at ingest).
        metadata: Free-form per-chunk metadata.
    """

    document_id: uuid.UUID
    ordinal: int
    content: str
    dense_embedding: list[float]
    parent_id: uuid.UUID | None = None
    context_prefix: str | None = None
    sparse_weights: dict[str, float] | None = None
    content_type: str = "text"
    section_path: str | None = None
    acl_tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentRepository:
    """CRUD for ``iw_rag.documents`` keyed by ``(source_uri, checksum)``."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open session.

        Args:
            session: The active async session (transaction scope owned by caller).
        """
        self._session = session

    async def get_by_source_checksum(self, source_uri: str, checksum: str) -> uuid.UUID | None:
        """Return the document id for a ``(source_uri, checksum)`` pair, if present.

        Args:
            source_uri: The document's source URI.
            checksum: The document's content checksum.

        Returns:
            The document id, or ``None`` if no matching row exists.
        """
        stmt = select(schema.documents.c.id).where(
            schema.documents.c.source_uri == source_uri,
            schema.documents.c.checksum == checksum,
        )
        result = await self._session.execute(stmt)
        row = result.first()
        return None if row is None else cast(uuid.UUID, row[0])

    async def upsert_document(
        self,
        *,
        source_uri: str,
        checksum: str,
        title: str | None = None,
        content_type: str = "document",
        metadata: dict[str, Any] | None = None,
        ingestion_run_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Insert a document, or return the existing id on ``(uri, checksum)`` conflict.

        Idempotent on the unique key (lld-data-model §9) — re-ingesting an
        unchanged document does not create a duplicate row.

        Args:
            source_uri: The document's source URI.
            checksum: The document's content checksum.
            title: Optional title.
            content_type: Whole-document content type (default ``'document'``).
            metadata: Optional document metadata.
            ingestion_run_id: Owning ingestion run, if any.

        Returns:
            The id of the inserted or pre-existing document.
        """
        stmt = (
            pg_insert(schema.documents)
            .values(
                source_uri=source_uri,
                checksum=checksum,
                title=title,
                content_type=content_type,
                metadata=metadata or {},
                ingestion_run_id=ingestion_run_id,
            )
            .on_conflict_do_nothing(index_elements=["source_uri", "checksum"])
            .returning(schema.documents.c.id)
        )
        result = await self._session.execute(stmt)
        row = result.first()
        if row is not None:
            return cast(uuid.UUID, row[0])
        # Conflict → row already existed; fetch its id.
        existing = await self.get_by_source_checksum(source_uri, checksum)
        assert existing is not None  # the conflict guarantees a row exists
        return existing

    async def delete_by_source_uri(self, source_uri: str) -> int:
        """Delete all document rows for a source uri (cascades to chunks).

        Used by ``supersede=true`` re-ingest to drop prior versions.

        Args:
            source_uri: The source uri whose document versions to remove.

        Returns:
            The number of document rows deleted.
        """
        result = await self._session.execute(
            delete(schema.documents).where(schema.documents.c.source_uri == source_uri)
        )
        return int(cast(Any, result).rowcount or 0)

    async def set_attributes(
        self,
        document_id: uuid.UUID,
        *,
        owner_id: str | None,
        visibility: str | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Set the management attributes (owner/visibility/metadata) on a document.

        Called after ingestion to record who uploaded the document and how it is
        shared, plus the original filename/size/mime in ``metadata`` for the UI.

        Args:
            document_id: The document to update.
            owner_id: Uploading principal id (None for corpus docs).
            visibility: ``common`` | ``specific`` (None for corpus docs).
            metadata: Optional metadata to store (merged-replace, not deep-merge).
        """
        values: dict[str, Any] = {"owner_id": owner_id, "visibility": visibility}
        if metadata is not None:
            values["metadata"] = metadata
        await self._session.execute(
            schema.documents.update().where(schema.documents.c.id == document_id).values(**values)
        )

    async def get_by_id(self, document_id: uuid.UUID) -> dict[str, Any] | None:
        """Fetch a single document row by id (None if absent)."""
        stmt = select(schema.documents).where(schema.documents.c.id == document_id)
        result = await self._session.execute(stmt)
        row = result.mappings().first()
        return None if row is None else dict(row)

    async def list_visible(
        self, principal_id: str, *, include_attachments: bool = False
    ) -> list[dict[str, Any]]:
        """List documents the principal manages: owned + common (with chunk counts).

        Returns rows for documents where ``owner_id = principal_id`` OR
        ``visibility = 'common'`` — never another user's ``specific`` documents.
        Ephemeral chat attachments (``metadata->>'kind' = 'attachment'``) are
        excluded unless ``include_attachments`` is True.

        Args:
            principal_id: The requesting principal id.
            include_attachments: Whether to include ephemeral attachment rows.

        Returns:
            Document rows (id, source_uri, title, owner_id, visibility, metadata,
            created_at, chunk_count), newest first.
        """
        d, c = schema.documents, schema.chunks
        chunk_count = (
            select(func.count())
            .select_from(c)
            .where(c.c.document_id == d.c.id)
            .scalar_subquery()
            .label("chunk_count")
        )
        conditions: list[Any] = [
            (d.c.owner_id == principal_id) | (d.c.visibility == "common"),
        ]
        if not include_attachments:
            conditions.append(func.coalesce(d.c.metadata["kind"].astext, "") != "attachment")
        stmt = (
            select(
                d.c.id,
                d.c.source_uri,
                d.c.title,
                d.c.content_type,
                d.c.owner_id,
                d.c.visibility,
                d.c.metadata,
                d.c.created_at,
                chunk_count,
            )
            .where(*conditions)
            .order_by(d.c.created_at.desc())
        )
        result = await self._session.execute(stmt)
        return [dict(row) for row in result.mappings()]

    async def purge_by_id(self, document_id: uuid.UUID) -> str | None:
        """Delete a document and ALL derived artifacts; return its source_uri.

        Removes (in one transaction scope owned by the caller):
        - ``graph_chunk_backrefs`` rows referencing this document's chunk ids
          (no FK cascade exists — they key on ``chunk_id`` text);
        - ``visual_pages`` rows for this document's ``source_uri`` (no FK cascade);
        - the ``documents`` row, which cascades to ``chunks`` (and their
          embeddings) and ``document_acl`` via ON DELETE CASCADE.

        After this the document is unqueryable through every retrieval path
        (dense, sparse, visual, graph).

        Args:
            document_id: The document to purge.

        Returns:
            The deleted document's ``source_uri``, or ``None`` if it did not exist.
        """
        row = await self.get_by_id(document_id)
        if row is None:
            return None
        source_uri = str(row["source_uri"])

        # Graph back-refs key on chunk_id (text) — delete before the cascade drops
        # the chunks, so the subquery can still see them.
        await self._session.execute(
            text(
                "DELETE FROM iw_rag.graph_chunk_backrefs WHERE chunk_id IN "
                "(SELECT id::text FROM iw_rag.chunks WHERE document_id = :doc_id)"
            ),
            {"doc_id": document_id},
        )
        # Visual pages key on source_uri (no FK to chunks).
        await self._session.execute(
            text("DELETE FROM iw_rag.visual_pages WHERE source_uri = :uri"),
            {"uri": source_uri},
        )
        await self._session.execute(
            delete(schema.documents).where(schema.documents.c.id == document_id)
        )
        return source_uri

    async def ids_by_session(self, session_id: str) -> list[uuid.UUID]:
        """Return document ids for an ephemeral chat session (metadata.session_id)."""
        stmt = select(schema.documents.c.id).where(
            schema.documents.c.metadata["session_id"].astext == session_id
        )
        result = await self._session.execute(stmt)
        return [cast(uuid.UUID, row[0]) for row in result]

    async def acl_tags_for_document(self, document_id: uuid.UUID) -> list[str]:
        """Return the distinct union of every chunk's ``acl_tags`` for a document.

        The retrieval ACL pre-filter operates on the denormalized
        ``chunks.acl_tags`` array (the canonical source of truth for who may
        see a document — see :mod:`iw_rag.retrieval.acl_filter`).  This method
        surfaces that set for the download authorization gate, which must
        mirror retrieval: a caller is allowed to download iff
        ``set(doc_acl_tags) & set(ctx.principal_groups)`` is non-empty
        (CR-00098 S03 / AC4).  Without this method the download gate would
        fall back to the narrower owner-or-``common`` check and refuse
        corpus docs and session attachments the caller could legitimately
        retrieve.

        The query unnests every chunk's ``acl_tags`` array and projects
        ``DISTINCT`` so duplicates across chunks collapse.  A document with
        no chunks (or an unknown id) yields an empty list — never an
        exception.

        Args:
            document_id: The document whose effective ACL tags to return.

        Returns:
            The distinct ``acl_tags`` for the document (empty list if the
            document has no chunks or does not exist).
        """
        stmt = select(func.distinct(func.unnest(schema.chunks.c.acl_tags)).label("tag")).where(
            schema.chunks.c.document_id == document_id
        )
        result = await self._session.execute(stmt)
        return [str(row[0]) for row in result]


class EphemeralSessionRepository:
    """Ownership records for chat-scoped ephemeral attachment sessions (WI-DOCMGMT).

    Binds a ``session_id`` to the ``principal_id`` that created it so the API can
    verify, server-side, that a request may be granted the synthetic
    ``session:<id>`` acl group (keeping the ACL keystone honest).  Rows carry an
    ``expires_at`` for the TTL purge.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open session."""
        self._session = session

    async def get(self, session_id: str) -> dict[str, Any] | None:
        """Fetch an ephemeral-session row by id (None if absent)."""
        stmt = select(schema.ephemeral_sessions).where(
            schema.ephemeral_sessions.c.session_id == session_id
        )
        row = (await self._session.execute(stmt)).mappings().first()
        return None if row is None else dict(row)

    async def ensure(self, session_id: str, principal_id: str, expires_at: Any) -> dict[str, Any]:
        """Create the session row if absent; return the (existing or new) row.

        Idempotent on ``session_id``.  The caller is responsible for rejecting a
        principal mismatch (the row carries the owning ``principal_id``).
        """
        stmt = (
            pg_insert(schema.ephemeral_sessions)
            .values(
                session_id=session_id,
                principal_id=principal_id,
                expires_at=expires_at,
            )
            .on_conflict_do_nothing(index_elements=["session_id"])
        )
        await self._session.execute(stmt)
        existing = await self.get(session_id)
        assert existing is not None
        return existing

    async def delete(self, session_id: str) -> None:
        """Delete one ephemeral-session row."""
        await self._session.execute(
            delete(schema.ephemeral_sessions).where(
                schema.ephemeral_sessions.c.session_id == session_id
            )
        )

    async def expired(self, now: Any) -> list[str]:
        """Return session ids whose ``expires_at`` is in the past."""
        stmt = select(schema.ephemeral_sessions.c.session_id).where(
            schema.ephemeral_sessions.c.expires_at < now
        )
        result = await self._session.execute(stmt)
        return [str(row[0]) for row in result]


class ConversationRepository:
    """CRUD for multi-turn chat memory (``conversations`` + ``conversation_messages``).

    A conversation is owned by exactly one ``principal_id`` (sourced from the
    authenticated AccessContext, never request input).  Callers MUST verify
    ownership via :meth:`get` before reading history or deleting a thread — the
    repository itself is ACL-agnostic, mirroring the other repositories.

    History reads return turns oldest-first so the agent replays them in order;
    :meth:`recent_messages` caps the window (token-budget guard, I-CTXOVERFLOW).
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open session."""
        self._session = session

    async def create(self, principal_id: str, *, title: str | None = None) -> uuid.UUID:
        """Insert a conversation owned by ``principal_id``; return its id."""
        stmt = (
            insert(schema.conversations)
            .values(principal_id=principal_id, title=title)
            .returning(schema.conversations.c.id)
        )
        result = await self._session.execute(stmt)
        return cast(uuid.UUID, result.scalar_one())

    async def get(self, conversation_id: uuid.UUID) -> dict[str, Any] | None:
        """Fetch a conversation row by id (None if absent) — for ownership checks."""
        stmt = select(schema.conversations).where(schema.conversations.c.id == conversation_id)
        row = (await self._session.execute(stmt)).mappings().first()
        return None if row is None else dict(row)

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        disposition: str | None = None,
    ) -> uuid.UUID:
        """Append one turn and bump the conversation's ``last_active_at``.

        ``role`` is ``"user"`` or ``"assistant"`` (DB CHECK).  Assistant turns
        carry their resolved ``citations`` + ``disposition``; user turns leave
        both at their defaults.
        """
        stmt = (
            insert(schema.conversation_messages)
            .values(
                conversation_id=conversation_id,
                role=role,
                content=content,
                citations=citations or [],
                disposition=disposition,
            )
            .returning(schema.conversation_messages.c.id)
        )
        result = await self._session.execute(stmt)
        message_id = cast(uuid.UUID, result.scalar_one())
        await self._session.execute(
            schema.conversations.update()
            .where(schema.conversations.c.id == conversation_id)
            .values(last_active_at=func.now())
        )
        return message_id

    async def recent_messages(
        self, conversation_id: uuid.UUID, *, limit: int
    ) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` turns, oldest-first (replay order).

        Selects the newest ``limit`` rows (so a long thread does not blow the
        context window) then re-sorts ascending so the agent and the UI see them
        chronologically.
        """
        if limit <= 0:
            return []
        stmt = (
            select(schema.conversation_messages)
            .where(schema.conversation_messages.c.conversation_id == conversation_id)
            .order_by(schema.conversation_messages.c.created_at.desc())
            .limit(limit)
        )
        rows = [dict(r) for r in (await self._session.execute(stmt)).mappings()]
        rows.reverse()
        return rows

    async def delete(self, conversation_id: uuid.UUID) -> None:
        """Delete a conversation; its messages cascade (the "Clear" button)."""
        await self._session.execute(
            delete(schema.conversations).where(schema.conversations.c.id == conversation_id)
        )


class IngestionRunRepository:
    """Create/finish ``iw_rag.ingestion_runs`` rows for a corpus run."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open session."""
        self._session = session

    async def create(self, *, corpus_label: str, parser: str, embed_model: str) -> uuid.UUID:
        """Insert a ``running`` ingestion-run row and return its id."""
        stmt = (
            insert(schema.ingestion_runs)
            .values(
                corpus_label=corpus_label,
                parser=parser,
                embed_model=embed_model,
                status="running",
            )
            .returning(schema.ingestion_runs.c.id)
        )
        result = await self._session.execute(stmt)
        return cast(uuid.UUID, result.scalar_one())

    async def finish(self, run_id: uuid.UUID, status: str) -> None:
        """Mark a run finished with the given terminal status."""
        await self._session.execute(
            schema.ingestion_runs.update()
            .where(schema.ingestion_runs.c.id == run_id)
            .values(status=status, finished_at=func.now())
        )


class ChunkRepository:
    """CRUD for ``iw_rag.chunks`` (bulk upsert + parent expansion entry points)."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open session.

        Args:
            session: The active async session (transaction scope owned by caller).
        """
        self._session = session

    async def bulk_upsert_chunks(self, chunks: Sequence[NewChunk]) -> list[uuid.UUID]:
        """Insert chunk rows in one statement and return their ids.

        Args:
            chunks: The chunk rows to insert.

        Returns:
            The ids of the inserted chunks (empty list if ``chunks`` is empty).
        """
        if not chunks:
            return []
        values = [
            {
                "document_id": c.document_id,
                "parent_id": c.parent_id,
                "ordinal": c.ordinal,
                "content": c.content,
                "context_prefix": c.context_prefix,
                "dense_embedding": c.dense_embedding,
                "sparse_weights": c.sparse_weights,
                "content_type": c.content_type,
                "section_path": c.section_path,
                "acl_tags": c.acl_tags,
                "metadata": c.metadata,
            }
            for c in chunks
        ]
        stmt = pg_insert(schema.chunks).returning(schema.chunks.c.id)
        result = await self._session.execute(stmt, values)
        return [cast(uuid.UUID, row[0]) for row in result]

    async def get_by_ids(self, chunk_ids: Sequence[uuid.UUID]) -> list[dict[str, Any]]:
        """Fetch chunk rows by id, preserving the input id order.

        Args:
            chunk_ids: Chunk ids to fetch (e.g. reranked survivors).

        Returns:
            The matching chunk rows as dicts, ordered to match ``chunk_ids``
            (ids with no row are skipped).
        """
        if not chunk_ids:
            return []
        stmt = select(schema.chunks).where(schema.chunks.c.id.in_(list(chunk_ids)))
        result = await self._session.execute(stmt)
        by_id = {row["id"]: dict(row) for row in result.mappings()}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def get_retrievable(self, chunk_ids: Sequence[uuid.UUID]) -> list[dict[str, Any]]:
        """Fetch chunks joined to their document's ``source_uri`` (id-order preserved).

        Retrieval results need ``source_uri`` (which lives on ``documents``), so this
        joins chunk → document. Returns dicts with ``id``, ``content``,
        ``source_uri``, ``metadata``, ``parent_id``, and ``document_id`` (the owning
        ``documents.id`` PK as a string, threaded onto :class:`RetrievedChunk` so
        downstream citation download links can address
        ``GET /documents/{document_id}/raw`` — CR-00098). The UUID embedded in
        ``source_uri`` is a *storage* uuid, not the PK; only ``document_id`` is
        the addressable identifier for the download flow.

        Args:
            chunk_ids: Chunk ids to materialize.

        Returns:
            Rows ordered to match ``chunk_ids`` (missing ids skipped).
        """
        if not chunk_ids:
            return []
        c, d = schema.chunks, schema.documents
        stmt = (
            select(
                c.c.id,
                c.c.content,
                c.c.metadata,
                c.c.parent_id,
                c.c.document_id,
                d.c.source_uri,
            )
            .select_from(c.join(d, d.c.id == c.c.document_id))
            .where(c.c.id.in_(list(chunk_ids)))
        )
        result = await self._session.execute(stmt)
        by_id = {row["id"]: dict(row) for row in result.mappings()}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    async def get_section_ids(self, chunk_ids: Sequence[uuid.UUID]) -> frozenset[str]:
        """Return which of ``chunk_ids`` are sections, that is, have at least one child chunk.

        A section (parent) row is searchable like any chunk, but it is not a leaf: it needs
        its own handling in the spine.  A row is a section when another row points at it, which
        the ``chunks_parent`` index answers without a schema change.

        Args:
            chunk_ids: Candidate chunk ids.

        Returns:
            The ids (as strings) among ``chunk_ids`` that are parents of other chunks.
        """
        if not chunk_ids:
            return frozenset()
        c = schema.chunks
        stmt = select(c.c.parent_id).where(c.c.parent_id.in_(list(chunk_ids))).distinct()
        result = await self._session.execute(stmt)
        return frozenset(str(row[0]) for row in result)

    async def get_parents(self, parent_ids: Sequence[uuid.UUID]) -> list[dict[str, Any]]:
        """Fetch parent chunk rows for small-to-big expansion.

        Args:
            parent_ids: Parent chunk ids to fetch.

        Returns:
            The matching chunk rows as dicts (empty list if ``parent_ids`` empty).
        """
        if not parent_ids:
            return []
        stmt = select(schema.chunks).where(schema.chunks.c.id.in_(list(parent_ids)))
        result = await self._session.execute(stmt)
        return [dict(row) for row in result.mappings()]


# ---------------------------------------------------------------------------
# AuditRepository — Task 3.4 append-only audit log
# ---------------------------------------------------------------------------


class AuditRepository:
    """Insert-only repository for ``iw_rag.audit_log`` (Task 3.4).

    Exposes **only** :meth:`append` — no UPDATE or DELETE method exists.
    DB-layer enforcement (trigger + REVOKE in ``0003_audit_log.py``) provides
    defense-in-depth: even if this class were mistakenly extended with a mutating
    method, the trigger would catch it.

    The session's transaction is owned by the caller; ``AuditRepository``
    does NOT commit.  The caller must commit (or let the session context
    manager do so) after calling ``append``.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open async session.

        Args:
            session: The active async session; transaction scope owned by caller.
        """
        self._session = session

    async def append(self, record: Any) -> None:
        """Insert one audit record — the ONLY write operation on this repository.

        Serialises ``guard_verdicts`` and ``redactions`` to JSON-ready lists
        before inserting.  The ``created_at`` column defaults to ``now()`` in
        Postgres so no timestamp is sent from the application layer.

        Args:
            record: An :class:`~iw_rag.guardrails.audit.AuditRecord` instance.

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On any DB error (propagated to
                caller; ``audit_node`` converts this to a fail-closed refusal).
        """
        from datetime import datetime

        from iw_rag.guardrails.audit import _serialize_guard_verdicts

        guard_verdicts_json = _serialize_guard_verdicts(record.guard_verdicts)

        # Parse the ISO-8601 timestamp string to a timezone-aware datetime so
        # SQLAlchemy/psycopg can pass a proper timestamptz to Postgres.
        try:
            ts = datetime.fromisoformat(record.timestamp)
        except (ValueError, TypeError):
            ts = datetime.now(UTC)

        stmt = insert(schema.audit_log).values(
            request_id=record.request_id,
            principal=record.principal,
            query=record.query,
            acl_filter=record.acl_filter,
            retrieved_chunk_ids=record.retrieved_chunk_ids,
            guard_verdicts=guard_verdicts_json,
            redactions=record.redactions,
            disposition=record.disposition,
            created_at=ts,
        )
        await self._session.execute(stmt)


# ---------------------------------------------------------------------------
# ConfigChangeRepository — F-00096 S03 (Operator Console live-config audit)
# ---------------------------------------------------------------------------


class ConfigChangeRepository:
    """Append-only repository for ``iw_rag.config_change_log`` (F-00096 S03).

    Mirrors the :class:`AuditRepository` immutability pattern (Task 3.4
    / ``alembic/versions/0003_audit_log.py``): the **only** mutation
    surface is :meth:`append`.  No UPDATE or DELETE method exists in
    code; DB-layer defense in depth (trigger + REVOKE) lives in
    ``alembic/versions/0009_config_change_log.py`` (S01).

    The session's transaction is owned by the caller —
    ``ConfigChangeRepository`` does NOT commit.

    Pending implementation per F-00096 S03 — TDD RED scaffold.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open async session.

        Args:
            session: The active async session; transaction scope owned by caller.
        """
        self._session = session

    async def append(
        self,
        *,
        principal_id: str,
        config_key: str,
        old_value: str | None,
        new_value: str,
        applied_mode: str,
        trace_id: str | None = None,
    ) -> uuid.UUID:
        """Insert one config-change row — the ONLY write operation on this repository.

        Issues ``insert(schema.config_change_log).values(...).returning(id)``
        and returns the freshly generated uuid.  The session's
        transaction is owned by the caller — this method does NOT
        commit (mirrors :class:`AuditRepository.append`).

        Args:
            principal_id: The authenticated principal id from the
                :class:`AccessContext` (Invariant 3 — never from a
                request body).
            config_key: Dotted config key that was edited.
            old_value: Stringified prior value (``None`` for the first
                ever overlay write for a given key).
            new_value: Stringified new value.
            applied_mode: ``"hot"`` or ``"pending_restart"`` (CHECK
                enforced in the migration DDL).
            trace_id: Optional request correlation id.

        Returns:
            The new row's id (uuid).

        Raises:
            sqlalchemy.exc.SQLAlchemyError: On any DB error (propagated
                to caller; the service layer converts to a typed error).
        """
        stmt = (
            insert(schema.config_change_log)
            .values(
                principal_id=principal_id,
                config_key=config_key,
                old_value=old_value,
                new_value=new_value,
                applied_mode=applied_mode,
                trace_id=trace_id,
            )
            .returning(schema.config_change_log.c.id)
        )
        result = await self._session.execute(stmt)
        return cast(uuid.UUID, result.scalar_one())

    async def list_changes(
        self,
        *,
        config_key: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read the most recent change-log rows, newest first.

        Reads via ``.mappings()`` so the caller receives plain
        ``dict[str, Any]`` rows (JSONB columns round-trip as native
        Python dicts; the columns here are all scalars, but using
        ``.mappings()`` keeps the contract uniform across repositories
        — see :class:`BenchmarkReportRepository.get_report`).

        Args:
            config_key: Optional dotted-key filter; ``None`` returns
                rows for every key.
            limit: Maximum number of rows to return (default 100).

        Returns:
            A list of row dicts (newest first).
        """
        stmt = select(schema.config_change_log).order_by(schema.config_change_log.c.ts.desc())
        if config_key is not None:
            stmt = stmt.where(schema.config_change_log.c.config_key == config_key)
        stmt = stmt.limit(limit)
        result = await self._session.execute(stmt)
        return [dict(row) for row in result.mappings()]
