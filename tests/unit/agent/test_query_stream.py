"""Unit tests for the /query/stream SSE endpoint.

Verifies:
- POST /query/stream streams text/event-stream with a final or refusal event.
- 401 without a valid bearer token.
- Events are properly formatted SSE (event:\\ndata:\\n\\n).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.api.service import QueryOutcome
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer


class _StreamingFakeService:
    """Fake query service that supports both answer() and stream_answer()."""

    async def answer(self, query: str, ctx: AccessContext, *, loop: bool = True) -> QueryOutcome:
        """Return a canned QueryOutcome."""
        grounded = "unanswerable" not in query
        ans = GeneratedAnswer(
            answer="Streamed answer [1]." if grounded else "I don't have enough information.",
            grounded=grounded,
            citations=[
                Citation(
                    marker=1,
                    chunk_id="c0",
                    document_id="doc-0",
                    source_uri="uri0",
                    quote="ev",
                )
            ]
            if grounded
            else [],
            used_chunk_ids=["c0"] if grounded else [],
        )
        return QueryOutcome(answer=ans, retriever="spine", acl_applied=True, diagnostics={})


@pytest.fixture
def streaming_client() -> TestClient:
    """TestClient wired with a fake streaming-capable service."""
    svc = _StreamingFakeService()
    auth = StaticTokenAuthenticator.from_config({"tok": ["finance"], "op": ["all-staff"]}, ["op"])
    app = create_app(authenticator=auth, query_service=svc)
    return TestClient(app)


def _parse_sse(raw: str) -> list[dict]:
    """Parse raw SSE text into a list of event dicts.

    Args:
        raw: The raw SSE response text.

    Returns:
        List of dicts with 'event' and 'data' keys.
    """
    events = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
        elif line.startswith(":"):
            pass  # heartbeat / comment
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


def test_stream_no_token_is_401(streaming_client: TestClient) -> None:
    """Verifies /query/stream returns 401 without a bearer token."""
    resp = streaming_client.post("/query/stream", json={"query": "hello"})
    assert resp.status_code == 401


def test_stream_invalid_token_is_401(streaming_client: TestClient) -> None:
    """Verifies /query/stream returns 401 with an unknown token."""
    resp = streaming_client.post(
        "/query/stream",
        json={"query": "hello"},
        headers={"authorization": "Bearer bad-token"},
    )
    assert resp.status_code == 401


def test_stream_returns_event_stream_content_type(streaming_client: TestClient) -> None:
    """Verifies /query/stream returns text/event-stream content-type."""
    resp = streaming_client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")


def test_stream_ends_with_final_event_for_grounded(streaming_client: TestClient) -> None:
    """Verifies a grounded query stream ends with a 'final' event."""
    resp = streaming_client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    event_types = [e.get("event") for e in events]
    assert "final" in event_types, f"Expected 'final' event, got: {event_types}"


def test_stream_final_event_has_answer(streaming_client: TestClient) -> None:
    """Verifies the 'final' SSE event carries an 'answer' field."""
    resp = streaming_client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final_events = [e for e in events if e.get("event") == "final"]
    assert final_events, "No final event found"
    payload = json.loads(final_events[0]["data"])
    assert "answer" in payload
    assert "citations" in payload


def test_stream_ends_with_refusal_event_for_unanswerable(streaming_client: TestClient) -> None:
    """Verifies an unanswerable query stream ends with a 'refusal' event."""
    resp = streaming_client.post(
        "/query/stream",
        json={"query": "unanswerable question please"},
        headers={"authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    event_types = [e.get("event") for e in events]
    assert "refusal" in event_types, f"Expected 'refusal' event, got: {event_types}"


class _FailingService:
    """A query service whose answer() raises, to exercise the SSE error path."""

    async def answer(self, query: str, ctx: AccessContext, *, loop: bool = True) -> QueryOutcome:
        raise RuntimeError("boom: secret internal detail that must NOT leak to the client")


def test_stream_emits_canonical_error_event_on_failure() -> None:
    """A mid-stream failure ends with the canonical Q-SSE-SCHEMA 'error' event.

    The error frame must carry a generic ``message`` + ``trace_ref`` and MUST NOT
    leak the exception text (lld-api §error mapping).
    """
    auth = StaticTokenAuthenticator.from_config({"tok": ["finance"]}, ["tok"])
    app = create_app(authenticator=auth, query_service=_FailingService())
    client = TestClient(app)
    resp = client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer tok"},
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error_events = [e for e in events if e.get("event") == "error"]
    assert error_events, f"Expected an 'error' event, got: {[e.get('event') for e in events]}"
    payload = json.loads(error_events[0]["data"])
    assert "message" in payload
    assert "trace_ref" in payload
    assert "boom" not in resp.text  # exception detail never leaks to the client
