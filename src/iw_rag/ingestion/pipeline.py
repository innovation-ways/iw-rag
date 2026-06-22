"""Orchestration — parse → chunk → enrich → embed → transactional upsert.

The offline ingestion entry point (lld-ingestion §2.5). Idempotent on
``(source_uri, checksum)``: an unchanged file is ``skipped_unchanged``; a changed
file is a new document (or replaces the prior version when ``supersede=true``).
Embedding happens before any write, so an embed failure persists zero rows
(atomic). Parents are inserted first to obtain ids, then leaves reference them —
all in one committed session.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.domain.errors import EmbedError, IngestionError, ParseError
from iw_rag.ingestion.chunk import ChunkNode, ChunkTree, HybridChunker
from iw_rag.ingestion.embed import EmbeddingClient, _l2_normalize, _mean
from iw_rag.ingestion.enrich import IngestContext, enrich
from iw_rag.ingestion.parse import Parser, SourceRef
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    IngestionRunRepository,
    NewChunk,
)

IngestStatus = Literal["ingested", "skipped_unchanged", "failed"]


@dataclass(frozen=True, slots=True)
class IngestResult:
    """Outcome of ingesting one document.

    Attributes:
        document_id: The document id (None when skipped/failed before insert).
        status: ``ingested|skipped_unchanged|failed``.
        chunk_count: Number of leaf chunks persisted (0 for skip/failure).
        error: Error message when ``failed``.
    """

    document_id: UUID | None
    status: IngestStatus
    chunk_count: int
    error: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusIngestResult:
    """Aggregate outcome of a corpus run.

    Attributes:
        ingestion_run_id: The run row id.
        results: Per-document results.
    """

    ingestion_run_id: UUID
    results: list[IngestResult]


@dataclass(frozen=True, slots=True)
class IngestionConfig:
    """Pipeline-level knobs (subset of IngestionSettings).

    Attributes:
        concurrency: Max documents processed in parallel (offline; respects GPU).
        supersede: Replace prior versions on a checksum change.
        embed_model_label: Recorded on the ingestion_run row.
    """

    concurrency: int = 2
    supersede: bool = False
    embed_model_label: str = "bge-m3"


def _checksum(source: SourceRef) -> str:
    """Content checksum (sha256). Falls back to hashing the uri if unreadable."""
    try:
        return hashlib.sha256(source.local_path().read_bytes()).hexdigest()
    except (OSError, ParseError):
        return hashlib.sha256(source.uri.encode("utf-8")).hexdigest()


class IngestionPipeline:
    """Wraps the transform stages + transactional upsert (idempotent, offline)."""

    def __init__(
        self,
        parser_factory: Callable[[SourceRef], Parser],
        chunker: HybridChunker,
        embedder: EmbeddingClient,
        db: Database,
        cfg: IngestionConfig | None = None,
    ) -> None:
        """Wire the pipeline collaborators.

        Args:
            parser_factory: Maps a source to its parser (e.g. ``select_parser`` bound to config).
            chunker: The tokenizer-aware chunker (tokenizer matches the embedder).
            embedder: The Late-Chunking embedding client.
            db: The database handle (owns sessions/transactions).
            cfg: Pipeline knobs.
        """
        self._parser_factory = parser_factory
        self._chunker = chunker
        self._embedder = embedder
        self._db = db
        self._cfg = cfg or IngestionConfig()

    async def ingest_document(self, source: SourceRef, ctx: IngestContext) -> IngestResult:
        """Ingest one document end-to-end.

        Args:
            source: The source to ingest.
            ctx: The ingestion context (ACL resolver + run id).

        Returns:
            An :class:`IngestResult` describing the outcome (never raises for
            per-document parse/embed failures — they map to ``failed``).
        """
        checksum = _checksum(source)
        try:
            async with self._db.session() as session:
                existing = await DocumentRepository(session).get_by_source_checksum(
                    source.uri, checksum
                )
            if existing is not None:
                return IngestResult(existing, "skipped_unchanged", 0, None)

            # Transform stages (no DB writes yet → embed failure ⇒ zero rows).
            doc = await self._parser_factory(source).parse(source)
            tree = enrich(self._chunker.chunk(doc), source, ctx)
            embeddings = await self._embedder.embed_document(tree.doc_token_ids, tree.leaves)
            dense_by_local = {e.local_id: e.dense for e in embeddings}

            async with self._db.session() as session:
                if self._cfg.supersede:
                    await DocumentRepository(session).delete_by_source_uri(source.uri)
                doc_id = await DocumentRepository(session).upsert_document(
                    source_uri=source.uri,
                    checksum=checksum,
                    title=doc.title,
                    ingestion_run_id=ctx.ingestion_run_id,
                )
                count = await self._persist_chunks(session, doc_id, tree, dense_by_local)
                await session.commit()
            return IngestResult(doc_id, "ingested", count, None)
        except (ParseError, EmbedError, IngestionError) as exc:
            return IngestResult(None, "failed", 0, str(exc))

    async def _persist_chunks(
        self,
        session: AsyncSession,
        doc_id: UUID,
        tree: ChunkTree,
        dense_by_local: dict[str, list[float]],
    ) -> int:
        """Insert parents (for ids), then leaves referencing them. Returns leaf count."""
        chunk_repo = ChunkRepository(session)

        # Parent dense = normalized mean of its children's leaf vectors.
        children: dict[str, list[str]] = {p.local_id: [] for p in tree.parents}
        for leaf in tree.leaves:
            if leaf.parent_local_id in children:
                children[leaf.parent_local_id].append(leaf.local_id)

        parent_rows: list[NewChunk] = []
        for parent in tree.parents:
            kid_vecs = [
                dense_by_local[lid] for lid in children[parent.local_id] if lid in dense_by_local
            ]
            parent_dense = _l2_normalize(_mean(kid_vecs)) if kid_vecs else [0.0] * 1024
            parent_rows.append(_to_new_chunk(parent, doc_id, None, parent_dense))
        parent_ids = await chunk_repo.bulk_upsert_chunks(parent_rows)
        local_to_db = {p.local_id: pid for p, pid in zip(tree.parents, parent_ids, strict=True)}

        leaf_rows = [
            _to_new_chunk(
                leaf,
                doc_id,
                local_to_db.get(leaf.parent_local_id) if leaf.parent_local_id else None,
                dense_by_local[leaf.local_id],
            )
            for leaf in tree.leaves
        ]
        await chunk_repo.bulk_upsert_chunks(leaf_rows)
        return len(leaf_rows)

    async def ingest_corpus(
        self, sources: Iterable[SourceRef], ctx: IngestContext
    ) -> CorpusIngestResult:
        """Ingest many sources with bounded concurrency, recording a run.

        Args:
            sources: The sources to ingest.
            ctx: The ingestion context (run id is assigned here).

        Returns:
            A :class:`CorpusIngestResult` with the run id and per-document results.
        """
        async with self._db.session() as session:
            run_id = await IngestionRunRepository(session).create(
                corpus_label=ctx.corpus_label,
                parser="docling/text",
                embed_model=self._cfg.embed_model_label,
            )
            await session.commit()

        run_ctx = replace(ctx, ingestion_run_id=run_id)
        sem = asyncio.Semaphore(max(1, self._cfg.concurrency))

        async def _one(src: SourceRef) -> IngestResult:
            async with sem:
                return await self.ingest_document(src, run_ctx)

        results = await asyncio.gather(*[_one(s) for s in sources])
        status = "failed" if any(r.status == "failed" for r in results) else "succeeded"
        async with self._db.session() as session:
            await IngestionRunRepository(session).finish(run_id, status)
            await session.commit()
        return CorpusIngestResult(ingestion_run_id=run_id, results=list(results))


def _to_new_chunk(
    node: ChunkNode, doc_id: UUID, parent_id: UUID | None, dense: list[float]
) -> NewChunk:
    """Map a ChunkNode + its dense vector onto a persistable NewChunk."""
    return NewChunk(
        document_id=doc_id,
        ordinal=node.ordinal,
        content=node.content,
        dense_embedding=dense,
        parent_id=parent_id,
        content_type=node.content_type,
        section_path=node.section_path,
        acl_tags=list(node.acl_tags),
        metadata=node.payload,
    )
