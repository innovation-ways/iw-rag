"""Phase-0 exit-gate smoke — the spine MVP answers the manual query set correctly.

Aggregates the Phase-0 acceptance through the real app + spine (testcontainers,
faked model clients): each manual query yields the expected disposition
(grounded / fail-closed refusal), and ACL isolation holds on every answer (a
{finance} principal never receives an {hr} source). Model clients are faked; the
retrieval + ACL + citation machinery is real.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml
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

MANUAL = Path(__file__).resolve().parents[1] / "fixtures" / "manual_queries.yaml"
DIMS = 1024


def _unit(idx: int) -> list[float]:
    vec = [0.0] * DIMS
    vec[idx] = 1.0
    return vec


class _FakeEncoder:
    async def encode(self, query: str) -> list[float]:  # noqa: ARG002
        return _unit(0)


class _Identity:
    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        return [ScoredChunk(c.chunk_id, 1.0 / (i + 1)) for i, c in enumerate(candidates[:top_n])]

    async def health(self) -> bool:
        return True


class _Llm:
    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
        if "address" in str(messages[-1]["content"]).lower():
            return LlmResponse(text=REFUSAL_TEXT)
        return LlmResponse(text="Grounded in the source [1].")

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
                        content="Q3 invoice total is 15000 USD with two line items.",
                        dense_embedding=_unit(0),
                        acl_tags=["finance"],
                    ),
                    NewChunk(
                        document_id=hr,
                        ordinal=0,
                        content="Leave policy and salary details for staff.",
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
        encoder=_FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=_Identity(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=5, expansion_mode="leaf"),
    )
    assembler = AnswerAssembler(_Llm(), CitationPrompt(), GenerationParams(model="m"))
    app = create_app(
        authenticator=StaticTokenAuthenticator.from_config({"fin": ["finance"]}, []),
        query_service=QueryService(spine, assembler),
        database=db,
    )
    return TestClient(app)


@pytest.mark.integration
def test_phase0_manual_query_set(clean_pg: str) -> None:
    asyncio.run(_seed(clean_pg))
    client = _client(clean_pg)
    cases: dict[str, Any] = yaml.safe_load(MANUAL.read_text(encoding="utf-8"))

    for case in cases["queries"]:
        resp = client.post(
            "/query", json={"query": case["query"]}, headers={"authorization": "Bearer fin"}
        )
        assert resp.status_code == 200, case["query"]
        body = resp.json()
        if case["expect"] == "grounded":
            assert body["grounded"] is True, f"expected grounded: {case['query']}"
            assert body["citations"], case["query"]
        else:
            assert body["grounded"] is False, f"expected refusal: {case['query']}"
        # ACL isolation holds on every answer (no hr source ever surfaces).
        for cite in body["citations"]:
            assert "/hr/" not in cite["source_uri"], f"ACL LEAK on: {case['query']}"
