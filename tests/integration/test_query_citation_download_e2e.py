"""Integration tests for the ``/query`` (and SSE ``final`` event) citation
download flow end-to-end (CR-00098 / S07).

The S05 unit tests in :mod:`tests.unit.api.test_citation_download_url`
cover the response shape: ``CitationOut`` carries the additive
``document_id`` + ``download_url``, both for the non-streaming
``/query`` and the SSE ``final`` event.  S07 is responsible for the
**end-to-end mint→navigate** flow:

* A grounded ``/query`` response surfaces a non-empty
  ``download_url`` for every citation.
* Each ``download_url`` is a real URL whose token verifies for the
  citation's document + principal.
* Navigating to the ``download_url`` (with no ``Authorization`` header)
  returns the original file bytes — same filename, same MIME.
* The same contract holds for the SSE ``final`` event.

Every test runs against a real ParadeDB test container (``clean_pg``
fixture) and a real :class:`QueryService` / :class:`DocumentService` so
the citation's ``document_id`` (from the documents.id PK) and the
served bytes line up.  The TestClient is the same shape as the S04
file; the focus here is the citation→download seam.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator, StaticTokenAuthenticator
from iw_rag.api.documents import CharTokenizer, DocumentService
from iw_rag.api.download_tokens import verify
from iw_rag.generation.answer import AnswerAssembler
from iw_rag.generation.llm import GenerationParams, LlmResponse, Message, ToolSpec
from iw_rag.generation.prompts import CitationPrompt
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.retrieval.dense import DenseRetriever
from iw_rag.retrieval.rerank import RerankCandidate, ScoredChunk
from iw_rag.retrieval.sparse import SparseRetriever
from iw_rag.retrieval.spine import RetrievalConfig, RetrievalSpine
from iw_rag.settings import ApiSettings, DatabaseSettings, Settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DIMS: Final = 1024
_SECRET: Final = "test-secret-must-be-long-enough-for-hmac-sha256"
_FILE_BODY: Final = b"invoice total $15000 USD\n"
_FILE_NAME: Final = "invoice.txt"
_FILE_MIME: Final = "text/plain"


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _unit(idx: int) -> list[float]:
    vec = [0.0] * _DIMS
    vec[idx % _DIMS] = 1.0
    return vec


class _FakeEncoder:
    async def encode(self, query: str) -> list[float]:  # noqa: ARG002
        return _unit(0)


class _IdentityReranker:
    async def rerank(
        self, query: str, candidates: Sequence[RerankCandidate], top_n: int
    ) -> list[ScoredChunk]:  # noqa: ARG002
        return [
            ScoredChunk(chunk_id=c.chunk_id, score=1.0 / (i + 1))
            for i, c in enumerate(candidates[:top_n])
        ]

    async def health(self) -> bool:
        return True


class _CitingLlm:
    """A canned LLM that returns a single citation marker ``[1]``."""

    async def complete(self, messages: Sequence[Message], params: GenerationParams) -> LlmResponse:
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


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute-style)."""

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


class _FakeEmbedder:
    """A no-op embedder used by the DocumentService (the S07 path doesn't ingest)."""

    async def encode(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * EMBEDDING_DIMS

    async def health(self) -> bool:
        return True


def _make_service(clean_pg: str, tmp_path: Path) -> DocumentService:
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
    content: str = "The Q3 invoice total is $15000 USD.",
    stored_filename: str = _FILE_NAME,
    stored_bytes: bytes = _FILE_BODY,
) -> tuple[str, Path]:
    """Seed a document + chunk + actual staged bytes; returns ``(id, path)``."""
    doc_uuid = uuid.uuid4()
    stored_dir = upload_dir / doc_uuid.hex
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / stored_filename
    stored_path.write_bytes(stored_bytes)
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://finance/{doc_uuid}/{stored_filename}",
                checksum=f"cs-{doc_uuid}",
            )
            await docs.set_attributes(
                doc_id,
                owner_id=owner_id,
                visibility=visibility,
                metadata={
                    "filename": stored_filename,
                    "mime": _FILE_MIME,
                    "size_bytes": len(stored_bytes),
                    "kind": "upload",
                    "stored_path": str(stored_path),
                },
            )
            await chunks.bulk_upsert_chunks(
                [
                    NewChunk(
                        document_id=doc_id,
                        ordinal=0,
                        content=content,
                        dense_embedding=_unit(0),
                        acl_tags=acl_tags,
                    )
                ]
            )
            await session.commit()
    finally:
        await db.dispose()
    return str(doc_id), stored_path


def _build_app(
    clean_pg: str,
    tmp_path: Path,
    *,
    auth_mode: str = "demo",
) -> tuple[TestClient, Settings]:
    """Build a full TestClient + Settings, wired to a real QueryService + DocumentService.

    The QueryService runs against the test container's documents
    (seeded by ``_seed_doc_with_staged``) so a grounded query
    surfaces a real ``Citation`` with the document's ``document_id``
    (the documents.id PK).  The DocumentService is the S03 path the
    S04 download route delegates to.
    """
    db = Database(DatabaseSettings(dsn=clean_pg))
    spine = RetrievalSpine(
        db=db,
        encoder=_FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=_IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=5, expansion_mode="leaf"),
    )
    assembler = AnswerAssembler(_CitingLlm(), CitationPrompt(), GenerationParams(model="m"))
    from iw_rag.api.service import QueryService

    query_svc = QueryService(spine, assembler)
    doc_svc = _make_service(clean_pg, tmp_path)

    if auth_mode == "demo":
        auth = DemoUserAuthenticator.from_demo_users(
            {
                "alice": _FakeDemoCfg(
                    password="pw-alice",
                    token="alice-token",
                    groups=["finance"],
                    operator=False,
                ),
            }
        )
    else:
        auth = StaticTokenAuthenticator.from_config(
            {"alice-token": ["finance"]}, operator_tokens=[]
        )

    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={},
            demo_users={},
            download_token_secret=_SECRET,
            download_token_ttl_seconds=300,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=query_svc,
        database=db,
        document_service=doc_svc,
        llm=_CitingLlm(),
    )
    return TestClient(app), settings


# ---------------------------------------------------------------------------
# Helpers — SSE parsing
# ---------------------------------------------------------------------------


def _parse_sse(raw: str) -> list[dict[str, str]]:
    """Parse raw SSE text into a list of event dicts."""
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# 1. /query non-streaming — citation download_url is non-empty + works
# ---------------------------------------------------------------------------


async def test_query_citation_download_url_is_nonempty_and_works_end_to_end(
    clean_pg: str, tmp_path: Path
) -> None:
    """A grounded /query response surfaces a non-empty ``download_url`` per
    citation, and navigating to that URL (no Authorization header)
    downloads the original file.

    This is the **headline end-to-end** — a citation's
    ``download_url`` must (a) be non-empty, (b) be a verifiable
    signed token for the citation's document + principal, AND
    (c) actually serve the file when navigated.

    The setup seeds a real document + chunk + staged bytes in the
    test container; the QueryService is real (spine + assembler
    + fake LLM that emits a single ``[1]``); the TestClient is real.
    The full request flow: ``POST /query`` → ``GET
    /documents/{id}/raw?token=…`` (no Authorization).
    """
    client, settings = _build_app(clean_pg, tmp_path, auth_mode="demo")
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,  # corpus doc — group-tagged only
        visibility=None,
        content="The Q3 invoice total is $15000 USD.",
    )

    resp = client.post(
        "/query",
        json={"query": "what was the invoice total?"},
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grounded"] is True
    assert len(body["citations"]) == 1, body
    cit = body["citations"][0]

    # The citation carries the document_id of the cited chunk (the PK).
    assert cit["document_id"] == doc_id
    # The download_url is non-empty and has the documented form.
    assert cit["download_url"], "download_url must be non-empty for a grounded citation"
    assert cit["download_url"].startswith(f"/documents/{doc_id}/raw?token=")
    # The token verifies for the citation's document + principal.
    token = cit["download_url"].rsplit("token=", 1)[-1]
    verified = verify(token, secret=settings.api.download_token_secret)
    assert verified is not None
    assert verified.document_id == doc_id
    assert verified.principal_id == "alice"

    # End-to-end: navigating to the download_url (no Authorization
    # header) returns the original file.
    download = client.get(cit["download_url"])
    assert download.status_code == 200, download.text
    assert download.content == _FILE_BODY
    assert f'filename="{_FILE_NAME}"' in download.headers["content-disposition"]


async def test_query_multiple_citations_carry_distinct_download_urls(
    clean_pg: str, tmp_path: Path
) -> None:
    """A multi-citation answer carries a distinct download_url per citation.

    The S05 mint path issues one token per citation (each bound to its
    own document + principal).  The cross-cutting pin: distinct
    citations → distinct tokens → distinct URLs that each download
    the right file.

    To exercise the multi-citation path, the LLM in this test emits
    ``[1] and [2]`` and the seed sets up two corpus documents with
    overlapping tags.
    """
    db = Database(DatabaseSettings(dsn=clean_pg))
    db_engine = db

    # Seed TWO documents (both with ``finance`` group tags so alice sees them).
    doc_a, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
        content="Invoice A: $100.",
        stored_filename="a.txt",
        stored_bytes=b"invoice-a contents\n",
    )
    doc_b, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
        content="Invoice B: $200.",
        stored_filename="b.txt",
        stored_bytes=b"invoice-b contents\n",
    )

    # Build a TestClient with a 2-citation LLM.
    spine = RetrievalSpine(
        db=db_engine,
        encoder=_FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=_IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=5, expansion_mode="leaf"),
    )

    class _TwoCitationLlm:
        async def complete(
            self, messages: Sequence[Message], params: GenerationParams
        ) -> LlmResponse:
            return LlmResponse(text="Citing both [1] and [2].")

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

    assembler = AnswerAssembler(_TwoCitationLlm(), CitationPrompt(), GenerationParams(model="m"))
    from iw_rag.api.service import QueryService

    query_svc = QueryService(spine, assembler)
    doc_svc = _make_service(clean_pg, tmp_path)
    auth = DemoUserAuthenticator.from_demo_users(
        {
            "alice": _FakeDemoCfg(
                password="pw-alice", token="alice-token", groups=["finance"], operator=False
            ),
        }
    )
    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={},
            demo_users={},
            download_token_secret=_SECRET,
            download_token_ttl_seconds=300,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=query_svc,
        database=db_engine,
        document_service=doc_svc,
        llm=_TwoCitationLlm(),
    )
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what are the invoices?"},
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["grounded"] is True
    # At least 2 citations, with distinct document_ids and download_urls.
    cits = body["citations"]
    assert len(cits) >= 2, cits
    doc_ids = [c["document_id"] for c in cits]
    download_urls = [c["download_url"] for c in cits]
    assert len(set(doc_ids)) == len(doc_ids), f"document_ids must be distinct, got {doc_ids!r}"
    assert len(set(download_urls)) == len(download_urls), (
        f"download_urls must be distinct, got {download_urls!r}"
    )

    # Each download_url actually serves the right file.
    for cit in cits:
        # Pick the matching stored bytes by document_id.
        expected = (
            b"invoice-a contents\n" if cit["document_id"] == doc_a else b"invoice-b contents\n"
        )
        download = client.get(cit["download_url"])
        assert download.status_code == 200, download.text
        assert download.content == expected


# ---------------------------------------------------------------------------
# 2. /query/stream SSE final event — citation download_url is non-empty + works
# ---------------------------------------------------------------------------


async def test_stream_final_event_citation_download_url_is_nonempty_and_works(
    clean_pg: str, tmp_path: Path
) -> None:
    """The SSE ``final`` event's citation carries a non-empty
    ``download_url`` that actually downloads the file.
    """
    client, settings = _build_app(clean_pg, tmp_path, auth_mode="demo")
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
        content="The Q3 invoice total is $15000 USD.",
    )

    resp = client.post(
        "/query/stream",
        json={"query": "what was the invoice total?"},
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    events = _parse_sse(resp.text)
    final_events = [e for e in events if e.get("event") == "final"]
    assert final_events, f"Expected a 'final' event, got: {[e.get('event') for e in events]}"
    payload = json.loads(final_events[0]["data"])
    citations = payload.get("citations") or []
    assert len(citations) == 1, payload
    cit = citations[0]

    # Same contract as the non-streaming path.
    assert cit["document_id"] == doc_id
    assert cit["download_url"].startswith(f"/documents/{doc_id}/raw?token=")
    token = cit["download_url"].rsplit("token=", 1)[-1]
    verified = verify(token, secret=settings.api.download_token_secret)
    assert verified is not None
    assert verified.document_id == doc_id
    assert verified.principal_id == "alice"

    # End-to-end: the URL downloads the file.
    download = client.get(cit["download_url"])
    assert download.status_code == 200, download.text
    assert download.content == _FILE_BODY


# ---------------------------------------------------------------------------
# 3. Fail-soft path — when the secret is unset, download_url is empty
#    and the response still carries document_id (the field is additive)
# ---------------------------------------------------------------------------


async def test_query_citation_carries_document_id_but_empty_download_url_when_secret_unset(
    clean_pg: str, tmp_path: Path
) -> None:
    """When the env-only download secret is unset, ``download_url`` is empty
    (no 500), but ``document_id`` is still populated.

    The fail-soft contract: the field is additive — a misconfigured
    deployment never breaks the query response, the frontend falls
    back to plain-text rendering of the citation.
    """
    # Build a TestClient with download_token_secret="" (the fail-safe).
    db = Database(DatabaseSettings(dsn=clean_pg))
    spine = RetrievalSpine(
        db=db,
        encoder=_FakeEncoder(),
        dense=DenseRetriever(),
        sparse=SparseRetriever(),
        reranker=_IdentityReranker(),
        cfg=RetrievalConfig(candidate_k=50, fused_k=50, top_k=5, expansion_mode="leaf"),
    )
    assembler = AnswerAssembler(_CitingLlm(), CitationPrompt(), GenerationParams(model="m"))
    from iw_rag.api.service import QueryService

    query_svc = QueryService(spine, assembler)
    doc_svc = _make_service(clean_pg, tmp_path)
    auth = DemoUserAuthenticator.from_demo_users(
        {
            "alice": _FakeDemoCfg(
                password="pw-alice", token="alice-token", groups=["finance"], operator=False
            ),
        }
    )
    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={},
            demo_users={},
            download_token_secret="",  # fail-safe: feature off
            download_token_ttl_seconds=300,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=query_svc,
        database=db,
        document_service=doc_svc,
        llm=_CitingLlm(),
    )
    client = TestClient(app)

    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
        content="The Q3 invoice total is $15000 USD.",
    )

    resp = client.post(
        "/query",
        json={"query": "what was the invoice total?"},
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 200, resp.text
    cit = resp.json()["citations"][0]
    # document_id is still populated (additive field).
    assert cit["document_id"] == doc_id
    # download_url is the empty string (no 500).
    assert cit["download_url"] == ""
