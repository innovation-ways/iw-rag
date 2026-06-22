"""Unit tests for the chat-memory HTTP surface.

Covers the ``/query`` memory orchestration (resolve → replay → record → echo the
``conversation_id``) and the ``/conversations`` CRUD endpoints, using an
injected fake :class:`ConversationService` + a recording query service.  No DB,
no network — FastAPI ``TestClient`` only (same pattern as
``tests/unit/api/test_citation_download_url.py``).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator
from iw_rag.api.conversation import StoredMessage
from iw_rag.api.service import QueryOutcome
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import GeneratedAnswer, RetrievedChunk


class _FakeDemoCfg:
    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


_MEM_CHUNK = RetrievedChunk(
    chunk_id="conv:1",
    document_id="",
    content='Earlier in this conversation, the user said: "My name is Sergio"',
    score=1.0,
    source_uri="conversation://conv-1",
    metadata={"kind": "conversation", "role": "user"},
)


class _RecordingQueryService:
    """Records the kwargs it was called with and returns a canned grounded answer."""

    def __init__(self) -> None:
        self.last_kwargs: dict[str, Any] = {}

    async def answer(
        self,
        query: str,
        ctx: AccessContext,
        *,
        loop: bool = True,
        allow_ungrounded: bool = False,
        history_chunks: list[RetrievedChunk] | None = None,
    ) -> QueryOutcome:
        # Mirror AgentQueryService.answer's explicit signature so the route's
        # inspect-based kwarg gating actually forwards history_chunks.
        self.last_kwargs = {
            "loop": loop,
            "allow_ungrounded": allow_ungrounded,
            "history_chunks": history_chunks,
        }
        ans = GeneratedAnswer(answer="Your name is Sergio [1].", grounded=True, citations=[])
        return QueryOutcome(
            answer=ans, retriever="agent", acl_applied=True, diagnostics={"disposition": "answered"}
        )


class _FakeConversationService:
    """In-memory conversation service tracking the calls the route makes."""

    def __init__(self) -> None:
        self.recorded: list[dict[str, Any]] = []
        self.cleared: list[str] = []
        self.history: list[RetrievedChunk] = [_MEM_CHUNK]

    async def create(self, ctx: AccessContext) -> str:
        return "conv-new"

    async def resolve(self, ctx: AccessContext, conversation_id: str | None) -> str:
        return conversation_id or "conv-1"

    async def history_chunks(
        self, ctx: AccessContext, conversation_id: str
    ) -> list[RetrievedChunk]:
        return self.history

    async def record_turn(self, conversation_id: str, **kwargs: Any) -> None:
        self.recorded.append({"conversation_id": conversation_id, **kwargs})

    async def get_history(self, ctx: AccessContext, conversation_id: str) -> list[StoredMessage]:
        return [
            StoredMessage(
                role="user",
                content="My name is Sergio",
                citations=[],
                disposition=None,
                created_at="t0",
            ),
            StoredMessage(
                role="assistant",
                content="Your name is Sergio [1].",
                citations=[],
                disposition="answered",
                created_at="t1",
            ),
        ]

    async def clear(self, ctx: AccessContext, conversation_id: str) -> bool:
        self.cleared.append(conversation_id)
        return True


@pytest.fixture
def demo_auth() -> DemoUserAuthenticator:
    return DemoUserAuthenticator.from_demo_users(
        {"alice": _FakeDemoCfg(password="pw", token="tok", groups=["hr"], operator=False)}
    )


_AUTH = {"authorization": "Bearer tok"}


def _client(demo_auth: DemoUserAuthenticator, **overrides: Any) -> tuple[TestClient, Any]:
    qs = _RecordingQueryService()
    cs = _FakeConversationService()
    app = create_app(
        authenticator=demo_auth, query_service=qs, conversation_service=cs, **overrides
    )
    return TestClient(app), (qs, cs)


# ---------------------------------------------------------------------------
# /query — memory orchestration
# ---------------------------------------------------------------------------


def test_query_replays_history_and_echoes_conversation_id(
    demo_auth: DemoUserAuthenticator,
) -> None:
    client, (qs, cs) = _client(demo_auth)
    resp = client.post("/query", json={"query": "what is my name?"}, headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()

    # The resolved conversation id is echoed back for the client to persist.
    assert body["conversation_id"] == "conv-1"
    # History was replayed into the agent loop.
    assert qs.last_kwargs.get("history_chunks") == [_MEM_CHUNK]
    # The completed turn was persisted (user question + assistant answer).
    assert len(cs.recorded) == 1
    rec = cs.recorded[0]
    assert rec["user_text"] == "what is my name?"
    assert rec["answer_text"] == "Your name is Sergio [1]."
    assert rec["disposition"] == "answered"


def test_query_passes_client_conversation_id(demo_auth: DemoUserAuthenticator) -> None:
    client, (_qs, cs) = _client(demo_auth)
    resp = client.post("/query", json={"query": "hi", "conversation_id": "conv-xyz"}, headers=_AUTH)
    assert resp.json()["conversation_id"] == "conv-xyz"
    assert cs.recorded[0]["conversation_id"] == "conv-xyz"


def test_query_without_memory_service_degrades_gracefully(
    demo_auth: DemoUserAuthenticator,
) -> None:
    """No conversation service wired → query still answers, conversation_id empty."""
    qs = _RecordingQueryService()
    app = create_app(authenticator=demo_auth, query_service=qs)  # no conversation_service
    client = TestClient(app)
    resp = client.post("/query", json={"query": "hi"}, headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == ""
    assert qs.last_kwargs.get("history_chunks") is None


# ---------------------------------------------------------------------------
# /conversations CRUD
# ---------------------------------------------------------------------------


def test_create_conversation(demo_auth: DemoUserAuthenticator) -> None:
    client, _ = _client(demo_auth)
    resp = client.post("/conversations", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json()["conversation_id"] == "conv-new"


def test_get_conversation_history(demo_auth: DemoUserAuthenticator) -> None:
    client, _ = _client(demo_auth)
    resp = client.get("/conversations/conv-1", headers=_AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"] == "conv-1"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assert body["messages"][0]["content"] == "My name is Sergio"


def test_delete_conversation(demo_auth: DemoUserAuthenticator) -> None:
    client, (_qs, cs) = _client(demo_auth)
    resp = client.request("DELETE", "/conversations/conv-1", headers=_AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"cleared": True}
    assert cs.cleared == ["conv-1"]


def test_conversation_endpoints_require_auth(demo_auth: DemoUserAuthenticator) -> None:
    client, _ = _client(demo_auth)
    assert client.post("/conversations").status_code == 401
    assert client.get("/conversations/conv-1").status_code == 401
    assert client.request("DELETE", "/conversations/conv-1").status_code == 401
