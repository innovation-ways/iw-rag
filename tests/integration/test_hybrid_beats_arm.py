"""Hybrid > either arm — BM25 surfaces an exact-match chunk dense-only drops.

Seeds filler chunks whose vectors sit on the query vector, plus one "exact" chunk
(an error code) whose vector is orthogonal to the query (so dense, at a small
candidate_k, drops it) but whose text contains the code (so BM25 finds it). The
fused spine result must include the exact chunk; a dense-only pass must not.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.acl_filter import acl_predicate
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

DIMS = 1024


def _unit(idx: int) -> list[float]:
    vec = [0.0] * DIMS
    vec[idx] = 1.0
    return vec


class FakeEncoder:
    async def encode(self, query: str) -> list[float]:  # noqa: ARG002
        return _unit(0)  # query vector lies on the filler chunks, far from the exact chunk


class IdentityReranker:
    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        return [
            ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
            for i, c in enumerate(candidates[:top_n])
        ]

    async def health(self) -> bool:
        return True


async def _seed(dsn: str) -> uuid.UUID:
    """Seed fillers (on the query vector) + one exact-code chunk (orthogonal). Returns exact id."""
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc = await docs.upsert_document(source_uri="corpus://finance/kb.md", checksum="k1")
            issues = await docs.upsert_document(
                source_uri="corpus://finance/issues.md", checksum="i1"
            )
            filler = [
                NewChunk(
                    document_id=doc,
                    ordinal=i,
                    content=text,
                    dense_embedding=_unit(0),
                    acl_tags=["finance"],
                )
                for i, text in enumerate(
                    [
                        "general invoice information",
                        "quarterly totals summary",
                        "payment terms net 30",
                    ]
                )
            ]
            await chunks.bulk_upsert_chunks(filler)
            (exact_id,) = await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=issues,
                        ordinal=0,
                        content="Known issue: error code E4421 occurs on export",
                        dense_embedding=_unit(1),  # orthogonal to the query vector
                        acl_tags=["finance"],
                    )
                ]
            )
            await session.commit()
            return exact_id
    finally:
        await db.dispose()


@pytest.mark.integration
async def test_dense_only_drops_exact_but_hybrid_keeps_it(clean_pg: str) -> None:
    exact_id = await _seed(clean_pg)
    ctx = AccessContext(principal_id="u", principal_groups=("finance",))

    # Dense-only, small candidate budget → the orthogonal exact chunk is NOT returned.
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            dense_hits = await DenseRetriever().retrieve(session, _unit(0), acl_predicate(ctx), k=2)
    finally:
        await db.dispose()
    assert str(exact_id) not in {h.chunk_id for h in dense_hits}, "dense unexpectedly surfaced it"

    # Full hybrid spine (same candidate_k=2) → BM25 surfaces it, fused keeps it.
    spine = RetrievalSpine(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        encoder=FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=2, fused_k=50, top_k=10, expansion_mode="leaf"),
    )
    result = await spine.retrieve("E4421 export error", ctx)
    assert str(exact_id) in {c.chunk_id for c in result.chunks}, (
        "hybrid failed to keep the exact match"
    )
