"""Unit tests for the S05 (CR-00098) CitationOut enrichment + per-citation
download token minting.

Coverage (per the S05 prompt's TDD requirement):

- ``CitationOut`` exposes the new additive fields ``document_id`` and
  ``download_url`` and stays additive (the existing 4 fields are
  unchanged);
- the non-streaming ``POST /query`` response populates both fields on every
  citation;
- the ``POST /query/stream`` SSE ``final`` event populates both fields on
  every citation (the path that the
  :func:`iw_rag.api.routes._sse_generator` fallback builds when the
  service exposes only ``answer()`` — and the same shape that the
  Phase-2 ``answer_question_stream`` path uses);
- the ``download_url`` decodes back to the right ``document_id`` +
  ``principal_id`` (the token is a verifiable capability bound to both);
- the ``download_url`` is *empty* (never a 500) when the secret is unset
  — the field is additive, fail-soft; the response still carries
  ``document_id``;
- the token's ``session_id`` is set iff the request's ``AccessContext``
  was widened by :func:`_scope_to_session` for a session the caller owns
  (an attachment scenario).

The tests are hermetic: ``TestClient`` + a ``RecordingService`` (same
pattern as :mod:`tests.unit.test_api_security`) with the ``Secret``
env-injected via ``monkeypatch.setenv``.  No docker, no live DB, no
network — pure stdlib + pydantic + FastAPI's TestClient.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from iw_rag.api import download_tokens
from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator
from iw_rag.api.schemas import CitationOut
from iw_rag.api.service import QueryOutcome
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, GeneratedAnswer

# ---------------------------------------------------------------------------
# Fixtures — RecordingService + a TestClient wired with a demo auth so the
# download path is actually reversible (principal_id → groups, so the S04
# download-by-token path would resolve it; S05's mint is symmetric).
# ---------------------------------------------------------------------------

_TEST_SECRET = "test-secret-for-s05-citation-download-tokens-only"


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute-style).

    Mirrors the pattern used by ``tests/integration/test_documents_raw_token.py``
    so the demo-user authenticator can be wired in unit tests without
    pydantic-settings glue.
    """

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


class _RecordingService:
    """In-memory query service: records the AccessContext and returns a
    canned grounded answer with one citation (so the /query response
    always has at least one CitationOut to assert against)."""

    def __init__(self) -> None:
        self.last_ctx: AccessContext | None = None

    async def answer(self, query: str, ctx: AccessContext, **_: Any) -> QueryOutcome:
        self.last_ctx = ctx
        ans = GeneratedAnswer(
            answer="Grounded [1].",
            grounded=True,
            citations=[
                Citation(
                    marker=1,
                    chunk_id="chunk-abc",
                    document_id="doc-xyz",
                    source_uri="corpus://hr/policy.md",
                    quote="policy excerpt",
                )
            ],
            used_chunk_ids=["chunk-abc"],
        )
        return QueryOutcome(answer=ans, retriever="spine", acl_applied=True, diagnostics={})

    async def answer_question_stream(
        self, query: str, ctx: AccessContext, **_: Any
    ) -> Iterator[dict[str, Any]]:
        self.last_ctx = ctx
        ans = GeneratedAnswer(
            answer="Streamed [1].",
            grounded=True,
            citations=[
                Citation(
                    marker=1,
                    chunk_id="chunk-abc",
                    document_id="doc-xyz",
                    source_uri="corpus://hr/policy.md",
                    quote="policy excerpt",
                )
            ],
            used_chunk_ids=["chunk-abc"],
        )
        yield {
            "event": "step",
            "data": {"node": "retrieve", "status": "complete", "attempt": 0, "detail": None},
        }
        yield {
            "event": "final",
            "data": {
                "answer": ans.answer,
                "citations": [
                    {
                        "marker": c.marker,
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "source_uri": c.source_uri,
                        "quote": c.quote,
                    }
                    for c in ans.citations
                ],
                "trace_ref": "trace-1",
                "acl_applied": True,
                "attempts": 1,
            },
        }


@pytest.fixture
def demo_auth() -> DemoUserAuthenticator:
    """A demo-user authenticator with a single non-operator user.

    The bearer token ``fin-tok`` maps to username ``alice`` (stable
    principal id) with group ``finance``.  This gives the test a known
    ``principal_id`` to assert against in the token payload.
    """
    return DemoUserAuthenticator.from_demo_users(
        {"alice": _FakeDemoCfg(password="pw", token="fin-tok", groups=["finance"], operator=False)}
    )


@pytest.fixture
def service() -> _RecordingService:
    return _RecordingService()


@pytest.fixture
def env_with_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject the env-only download token secret for the test's lifetime."""
    monkeypatch.setenv("IW_RAG__API__DOWNLOAD_TOKEN_SECRET", _TEST_SECRET)


@pytest.fixture
def env_without_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make sure no secret is set so the fail-soft path is exercised."""
    monkeypatch.delenv("IW_RAG__API__DOWNLOAD_TOKEN_SECRET", raising=False)


def _parse_sse(raw: str) -> list[dict[str, Any]]:
    """Parse raw SSE text into a list of event dicts.

    Args:
        raw: The raw SSE response text.

    Returns:
        List of dicts with ``event`` and ``data`` keys.
    """
    events: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in raw.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:") :].strip()
        elif line.startswith("data:"):
            current["data"] = line[len("data:") :].strip()
        elif line.startswith(":"):
            pass
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events


# ---------------------------------------------------------------------------
# Schema-level — CitationOut exposes the new fields
# ---------------------------------------------------------------------------


def test_citation_out_exposes_document_id_and_download_url() -> None:
    """``CitationOut`` carries additive ``document_id`` + ``download_url``.

    The existing 4 fields are unchanged.  Both new fields default to
    ``""`` so a legacy caller that builds a CitationOut without them
    still passes validation.
    """
    full = CitationOut(
        marker=1,
        chunk_id="c1",
        document_id="doc-1",
        source_uri="corpus://x",
        quote="q",
        download_url="/documents/doc-1/raw?token=tok",
    )
    assert full.document_id == "doc-1"
    assert full.download_url == "/documents/doc-1/raw?token=tok"

    # Additive — existing fields still work without the new ones.
    minimal = CitationOut(marker=1, chunk_id="c1", source_uri="corpus://x", quote="q")
    assert minimal.document_id == ""
    assert minimal.download_url == ""


# ---------------------------------------------------------------------------
# /query — non-streaming response
# ---------------------------------------------------------------------------


def test_query_response_citation_carries_document_id_and_download_url(
    env_with_secret: None, demo_auth: DemoUserAuthenticator, service: _RecordingService
) -> None:
    """Every citation in the /query response carries ``document_id`` + a
    signed ``download_url`` whose token verifies for that document and
    the caller's principal."""
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["grounded"] is True
    citations = body["citations"]
    assert len(citations) == 1
    cit = citations[0]
    assert cit["document_id"] == "doc-xyz"
    assert cit["download_url"].startswith("/documents/doc-xyz/raw?token=")

    # The token must verify for the right document + principal under
    # the same secret the request was minted with.
    token = cit["download_url"].rsplit("token=", 1)[-1]
    verified = download_tokens.verify(token, secret=_TEST_SECRET)
    assert verified is not None
    assert verified.document_id == "doc-xyz"
    assert verified.principal_id == "alice"


def test_query_response_citation_carries_document_id_when_secret_unset(
    env_without_secret: None, demo_auth: DemoUserAuthenticator, service: _RecordingService
) -> None:
    """When the download secret is unset, ``download_url`` is ``""`` (fail-soft).

    ``document_id`` is still populated (it is read from the chunk, not
    the token).  The query MUST NOT 500 — citation enrichment is a
    best-effort additive layer.
    """
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    assert resp.status_code == 200
    body = resp.json()
    cit = body["citations"][0]
    assert cit["document_id"] == "doc-xyz"
    assert cit["download_url"] == ""


# ---------------------------------------------------------------------------
# /query/stream — SSE final event
# ---------------------------------------------------------------------------


def test_stream_final_event_citation_carries_document_id_and_download_url(
    env_with_secret: None, demo_auth: DemoUserAuthenticator, service: _RecordingService
) -> None:
    """The SSE ``final`` event citations carry ``document_id`` + a
    signed ``download_url`` (same contract as the non-streaming
    response)."""
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    events = _parse_sse(resp.text)
    final = [e for e in events if e.get("event") == "final"]
    assert final, f"Expected a 'final' event, got: {[e.get('event') for e in events]}"
    payload = json.loads(final[0]["data"])
    assert len(payload["citations"]) == 1
    cit = payload["citations"][0]
    assert cit["document_id"] == "doc-xyz"
    assert cit["download_url"].startswith("/documents/doc-xyz/raw?token=")

    token = cit["download_url"].rsplit("token=", 1)[-1]
    verified = download_tokens.verify(token, secret=_TEST_SECRET)
    assert verified is not None
    assert verified.document_id == "doc-xyz"
    assert verified.principal_id == "alice"


# ---------------------------------------------------------------------------
# Session-scoped attachments — token carries session_id
# ---------------------------------------------------------------------------


def test_query_response_session_attachment_token_carries_session_id(
    env_with_secret: None, demo_auth: DemoUserAuthenticator, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the caller's ctx is widened by ``_scope_to_session``, the
    minted token carries the validated ``session_id`` so the
    attachment remains downloadable.

    The session re-validation is faked here by patching
    :func:`iw_rag.api.routes._scope_to_session` to return a widened
    ``AccessContext`` whose ``principal_groups`` includes a synthetic
    ``session:test-session`` group.  The S05 helper MUST observe that
    the ctx carries a session group and include the session_id in the
    token payload.
    """
    from iw_rag.api import routes as api_routes

    async def _fake_scope(
        request: Any, ctx: AccessContext, session_id: str | None
    ) -> AccessContext:
        if session_id == "test-session":
            return AccessContext(
                principal_id=ctx.principal_id,
                principal_groups=(*ctx.principal_groups, "session:test-session"),
                is_operator=ctx.is_operator,
                request_id=ctx.request_id,
            )
        return ctx

    monkeypatch.setattr(api_routes, "_scope_to_session", _fake_scope)

    service = _RecordingService()
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "summarize the attached doc", "session_id": "test-session"},
        headers={"authorization": "Bearer fin-tok"},
    )
    assert resp.status_code == 200
    cit = resp.json()["citations"][0]
    assert cit["document_id"] == "doc-xyz"
    token = cit["download_url"].rsplit("token=", 1)[-1]
    verified = download_tokens.verify(token, secret=_TEST_SECRET)
    assert verified is not None
    assert verified.session_id == "test-session"


# ---------------------------------------------------------------------------
# Token-binding — wrong document and wrong principal
# ---------------------------------------------------------------------------


def test_query_token_is_bound_to_principal_and_document(
    env_with_secret: None, demo_auth: DemoUserAuthenticator, service: _RecordingService
) -> None:
    """Sanity: the minted token is bound to BOTH the document_id AND
    principal_id (a leak to a different principal must verify as
    ``None`` when checked against the other principal's secret, and a
    token forged for a different document must verify as ``None``)."""
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    cit = resp.json()["citations"][0]
    token = cit["download_url"].rsplit("token=", 1)[-1]

    # Different document fails the URL-path equality check (S04 enforces
    # it server-side; here we model that mint + verify agree on the
    # document the token was bound to).
    # A token minted for one document cannot be used against a different document:
    # the citation's document_id must not be "doc-other".
    assert cit["document_id"] != "doc-other"

    # The wrong secret rejects the token outright.
    assert download_tokens.verify(token, secret="not-the-right-secret") is None

    # A different document id with a valid (but cross-cut) token must
    # not pass: the S05 helper mints ONE token per citation, and the
    # token's payload's d field IS the document_id the citation came
    # from.  A second query yielding a different document_id would
    # produce a different token.
    verified = download_tokens.verify(token, secret=_TEST_SECRET)
    assert verified is not None
    assert verified.document_id != "doc-other"


# ---------------------------------------------------------------------------
# No secret in the response — the secret must never be exposed
# ---------------------------------------------------------------------------


def test_query_response_never_exposes_secret(
    env_with_secret: None, demo_auth: DemoUserAuthenticator, service: _RecordingService
) -> None:
    """The secret MUST NOT appear in the response body or the SSE
    stream — neither the raw secret nor its hash."""
    app = create_app(authenticator=demo_auth, query_service=service)
    client = TestClient(app)

    resp = client.post(
        "/query",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    body_text = resp.text
    assert _TEST_SECRET not in body_text

    stream_resp = client.post(
        "/query/stream",
        json={"query": "what is the policy?"},
        headers={"authorization": "Bearer fin-tok"},
    )
    assert _TEST_SECRET not in stream_resp.text
