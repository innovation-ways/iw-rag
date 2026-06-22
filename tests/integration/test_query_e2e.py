"""End-to-end /query — real app + spine (testcontainers), faked model clients.

Ties the API to the spine's ACL guarantee: a {finance} principal gets a cited
answer drawn only from {finance}/{public} chunks; an {hr} doc never appears. An
unanswerable query refuses at HTTP 200. Deep health reports the database.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.api.service import QueryService
from iw_rag.generation.answer import AnswerAssembler
from iw_rag.generation.llm import GenerationParams, LlmResponse, Message, ToolSpec
from iw_rag.generation.prompts import REFUSAL_TEXT, CitationPrompt
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
        return _unit(0)


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


class QueryAwareLlm:
    """Refuses when the user asks the 'address' question; otherwise cites [1]."""

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        user = str(messages[-1]["content"]).lower()
        if "address" in user:
            return LlmResponse(text=REFUSAL_TEXT)
        return LlmResponse(text="The total is grounded in the source [1].")

    async def complete_tools(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec], params: GenerationParams
    ) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")

    async def complete_structured(
        self, messages: Sequence[Message], json_schema: dict, params: GenerationParams
    ) -> LlmResponse:  # pragma: no cover
        return LlmResponse(text="")

    async def health(self) -> bool:
        return True


async def _seed(dsn: str) -> None:
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            fin = await docs.upsert_document(source_uri="corpus://finance/inv.md", checksum="f1")
            hr = await docs.upsert_document(source_uri="corpus://hr/policy.md", checksum="h1")
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=fin,
                        ordinal=0,
                        content="The Q3 invoice total is 15000 USD.",
                        dense_embedding=_unit(0),
                        acl_tags=["finance"],
                    ),
                    NewChunk(
                        document_id=hr,
                        ordinal=0,
                        content="Salary and leave policy for staff; total compensation.",
                        dense_embedding=_unit(0),
                        acl_tags=["hr"],
                    ),
                ]
            )
            await session.commit()
    finally:
        await db.dispose()


def _client(dsn: str) -> TestClient:
    db = Database(DatabaseSettings(dsn=dsn))
    spine = RetrievalSpine(
        db=db,
        encoder=FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=5, expansion_mode="leaf"),
    )
    assembler = AnswerAssembler(QueryAwareLlm(), CitationPrompt(), GenerationParams(model="m"))
    app = create_app(
        authenticator=StaticTokenAuthenticator.from_config({"fin": ["finance"]}, []),
        query_service=QueryService(spine, assembler),
        database=db,
        llm=QueryAwareLlm(),
    )
    return TestClient(app)


@pytest.mark.integration
def test_query_grounded_finance_only(clean_pg: str) -> None:
    asyncio.run(_seed(clean_pg))
    resp = _client(clean_pg).post(
        "/query",
        json={"query": "what was the invoice total?"},
        headers={"authorization": "Bearer fin"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    assert body["citations"], "expected a resolved citation"
    for cite in body["citations"]:
        assert "/hr/" not in cite["source_uri"], "ACL LEAK via API"
        assert "/finance/" in cite["source_uri"] or "/public/" in cite["source_uri"]
    assert body["retriever"] == "spine"
    assert body["trace_id"]


@pytest.mark.integration
def test_query_unanswerable_refuses_at_200(clean_pg: str) -> None:
    asyncio.run(_seed(clean_pg))
    resp = _client(clean_pg).post(
        "/query",
        json={"query": "what is the CEO home address?"},
        headers={"authorization": "Bearer fin"},
    )
    assert resp.status_code == 200
    assert resp.json()["grounded"] is False


@pytest.mark.integration
def test_deep_health_reports_database(clean_pg: str) -> None:
    resp = _client(clean_pg).get("/health", params={"deep": "true"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["checks"]["database"]["ok"] is True
    assert body["status"] == "ok"
