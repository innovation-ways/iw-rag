"""HEADLINE security test — the ACL pre-filter never leaks across groups.

Seeds {hr} and {finance} chunks; a {finance}-only principal must NEVER receive an
{hr} chunk at any candidate_k (proves pre-filter, not post-filter). Also asserts
the core RetrievalResult invariants. Encoder + reranker are faked (deterministic);
dense (pgvector) + sparse (BM25) run against real ParadeDB.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

DIMS = 1024


def _unit(idx: int) -> list[float]:
    """A 1024-d unit vector with 1.0 at position ``idx``."""
    vec = [0.0] * DIMS
    vec[idx] = 1.0
    return vec


class FakeEncoder:
    """Returns a fixed query vector aligned to the 'finance' seed chunk."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    async def encode(self, query: str) -> list[float]:  # noqa: ARG002
        return self._vec


class IdentityReranker:
    """Keeps fused order, scoring 1/(i+1); takes top_n (no model)."""

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        return [
            ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
            for i, c in enumerate(candidates[:top_n])
        ]

    async def health(self) -> bool:
        return True


async def _seed(dsn: str) -> None:
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            fin_doc = await docs.upsert_document(
                source_uri="corpus://finance/inv.md", checksum="f1"
            )
            hr_doc = await docs.upsert_document(source_uri="corpus://hr/policy.md", checksum="h1")
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=fin_doc,
                        ordinal=0,
                        content="Invoice total for Q3 is 15000 USD",
                        dense_embedding=_unit(0),
                        acl_tags=["finance"],
                    ),
                    NewChunk(
                        document_id=hr_doc,
                        ordinal=0,
                        content="Invoice total leave policy salary 15000",
                        dense_embedding=_unit(0),  # same vector → only ACL keeps it out
                        acl_tags=["hr"],
                    ),
                ]
            )
            await session.commit()
    finally:
        await db.dispose()


def _spine(dsn: str, candidate_k: int) -> RetrievalSpine:
    return RetrievalSpine(
        db=Database(DatabaseSettings(dsn=dsn)),
        encoder=FakeEncoder(_unit(0)),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=candidate_k, fused_k=50, top_k=10, expansion_mode="leaf"),
    )


@pytest.mark.integration
@pytest.mark.parametrize("candidate_k", [1, 5, 50])
async def test_finance_principal_never_sees_hr_chunk(clean_pg: str, candidate_k: int) -> None:
    await _seed(clean_pg)
    spine = _spine(clean_pg, candidate_k)
    result = await spine.retrieve(
        "invoice total", AccessContext(principal_id="u", principal_groups=("finance",))
    )
    assert result.acl_applied is True
    assert result.chunks, "finance principal should retrieve its own chunk"
    for chunk in result.chunks:
        assert "/hr/" not in chunk.source_uri, f"ACL LEAK: hr chunk surfaced ({chunk.source_uri})"
        assert chunk.source_uri  # non-empty invariant


@pytest.mark.integration
async def test_result_invariants(clean_pg: str) -> None:
    await _seed(clean_pg)
    spine = _spine(clean_pg, candidate_k=50)
    result = await spine.retrieve(
        "invoice total", AccessContext(principal_id="u", principal_groups=("finance",))
    )
    assert len(result.chunks) <= 10  # top_k
    scores = [c.score for c in result.chunks]
    assert scores == sorted(scores, reverse=True)
    assert all(c.source_uri for c in result.chunks)
    assert all(uuid.UUID(c.chunk_id) for c in result.chunks)  # valid uuids


class FailingReranker:
    """A reranker that always errors — to exercise degrade-not-fail."""

    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        from iw_rag.domain.errors import RerankError

        raise RerankError("simulated reranker outage")

    async def health(self) -> bool:
        return False


@pytest.mark.integration
async def test_reranker_degrade_returns_fused_order_with_flag(clean_pg: str) -> None:
    await _seed(clean_pg)
    spine = RetrievalSpine(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        encoder=FakeEncoder(_unit(0)),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=FailingReranker(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=10, expansion_mode="leaf"),
    )
    result = await spine.retrieve(
        "invoice total", AccessContext(principal_id="u", principal_groups=("finance",))
    )
    assert result.diagnostics.get("rerank") == "degraded"
    assert result.chunks, "degrade must still return fused-order results"
    assert all("/hr/" not in c.source_uri for c in result.chunks)  # ACL still holds


@pytest.mark.integration
async def test_no_groups_returns_empty(clean_pg: str) -> None:
    await _seed(clean_pg)
    spine = _spine(clean_pg, candidate_k=50)
    result = await spine.retrieve(
        "invoice total", AccessContext(principal_id="u", principal_groups=())
    )
    assert result.chunks == []
    assert result.acl_applied is True
