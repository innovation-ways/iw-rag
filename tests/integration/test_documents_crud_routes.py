"""Integration tests for the document CRUD HTTP surface (Wave 3).

Exercises ``POST /documents``, ``DELETE /documents/{id}``, ``GET /documents``,
``POST /attachments``, ``DELETE /attachments/{session_id}``, upload validation
branches, and the ``_service``-unwired → 400 branch via the ``TestClient`` +
testcontainers Postgres.

Out of scope (covered by sibling tests):
- Service-level authz negatives (delete/attach/purge AuthorizationError) →
  ``test_documents_write_authz.py``.
- Raw-download ACL matrix → ``test_get_raw_acl_matrix.py``.
- Signed-token download path → ``test_documents_raw_token.py``.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Final

import pytest
import sqlalchemy
from fastapi.testclient import TestClient
from sqlalchemy import text

from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator
from iw_rag.api.documents import CharTokenizer, DocumentService
from iw_rag.ingestion.chunk import ChunkConfig, ChunkNode, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS, ChunkEmbedding
from iw_rag.settings import ApiSettings, DatabaseSettings, Settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import DocumentRepository

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_UPLOAD_BYTES: Final = 64 * 1024  # small ceiling for tests (64 KiB)
_PLAINTEXT_MIME: Final = "text/plain"
_FILE_BODY: Final = b"hello world\n"


# ---------------------------------------------------------------------------
# Fakes — bypass embedder/chunker/parser; allow the real pipeline for upload
# ---------------------------------------------------------------------------


class _FakeEmbedder:
    """Fake EmbeddingClient that returns a zero-vector for every leaf.

    Implements the ``EmbeddingClient`` protocol (``embed_document`` + ``health``).
    The pipeline calls ``embed_document(doc_token_ids, leaves)``; we return one
    zero-vector :class:`~iw_rag.ingestion.embed.ChunkEmbedding` per leaf so
    the ingestion can complete without a live Infinity endpoint.
    """

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Return a zero-vector embedding for every leaf (no Infinity call)."""
        return [
            ChunkEmbedding(local_id=leaf.local_id, dense=[0.0] * EMBEDDING_DIMS, sparse_weights={})
            for leaf in leaves
        ]

    async def health(self) -> bool:
        """Report healthy (no real endpoint required in tests)."""
        return True


class _BrokenEmbedder:
    """Fake EmbeddingClient that always raises EmbedError (triggers failed path)."""

    async def embed_document(
        self, doc_token_ids: list[int], leaves: Sequence[ChunkNode]
    ) -> list[ChunkEmbedding]:
        """Always raise EmbedError to simulate an embed failure."""
        from iw_rag.domain.errors import EmbedError

        raise EmbedError("embed endpoint unreachable (test-injected failure)")

    async def health(self) -> bool:
        """Report unhealthy."""
        return False


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute-style)."""

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        """Wire a demo user config object used by DemoUserAuthenticator."""
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


class _StubQueryService:
    """Minimal stub satisfying ``create_app``'s ``query_service`` seam."""

    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
        """Raise always — not exercised in CRUD route tests."""
        raise RuntimeError("not used in CRUD route tests")


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _make_service(
    clean_pg: str, tmp_path: Path, *, embedder: object | None = None
) -> DocumentService:
    """Build a DocumentService backed by real Postgres + an optional fake embedder.

    Args:
        clean_pg: DSN of the clean testcontainer Postgres instance.
        tmp_path: Temporary directory used as the upload staging area.
        embedder: Override the embedder (default: :class:`_FakeEmbedder`).

    Returns:
        A :class:`~iw_rag.api.documents.DocumentService` wired for tests.
    """
    return DocumentService(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        embedder=embedder or _FakeEmbedder(),  # type: ignore[arg-type]
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64, parent_max_tokens=256), CharTokenizer()),
        upload_dir=str(tmp_path),
        max_upload_bytes=_MAX_UPLOAD_BYTES,
    )


def _client(
    clean_pg: str,
    tmp_path: Path,
    *,
    embedder: object | None = None,
) -> TestClient:
    """Build a TestClient wired with demo-user auth + a real DocumentService.

    The demo authenticator exposes two principals: ``alice`` and ``bob``.
    Their tokens are ``alice-token`` and ``bob-token`` respectively.

    Args:
        clean_pg: DSN of the clean testcontainer Postgres instance.
        tmp_path: Temporary directory used as the upload staging area.
        embedder: Optional embedder override (default: :class:`_FakeEmbedder`).

    Returns:
        A :class:`~fastapi.testclient.TestClient` for the configured app.
    """
    svc = _make_service(clean_pg, tmp_path, embedder=embedder)
    auth = DemoUserAuthenticator.from_demo_users(
        {
            "alice": _FakeDemoCfg(
                password="pw-alice", token="alice-token", groups=[], operator=False
            ),
            "bob": _FakeDemoCfg(password="pw-bob", token="bob-token", groups=[], operator=False),
        }
    )
    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={},
            demo_users={},
            download_token_secret="",
            download_token_ttl_seconds=300,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=_StubQueryService(),
        database=Database(DatabaseSettings(dsn=clean_pg)),
        document_service=svc,
    )
    return TestClient(app)


def _client_no_service(clean_pg: str) -> TestClient:
    """Build a TestClient with NO document_service wired (tests the 400 branch).

    Args:
        clean_pg: DSN used for the database wiring only (service stays None).

    Returns:
        A :class:`~fastapi.testclient.TestClient` where ``document_service`` is
        ``None`` on ``app.state``.
    """
    auth = DemoUserAuthenticator.from_demo_users(
        {
            "alice": _FakeDemoCfg(
                password="pw-alice", token="alice-token", groups=[], operator=False
            ),
        }
    )
    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={},
            demo_users={},
            download_token_secret="",
            download_token_ttl_seconds=300,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=_StubQueryService(),
        database=Database(DatabaseSettings(dsn=clean_pg)),
        document_service=None,  # intentionally unwired
    )
    return TestClient(app)


def _upload_form(
    *,
    content: bytes = _FILE_BODY,
    filename: str = "test.txt",
    mime: str = _PLAINTEXT_MIME,
    visibility: str = "specific",
) -> dict:
    """Build a multipart form payload for ``POST /documents``.

    Args:
        content: Raw file bytes.
        filename: The filename to send in the multipart boundary.
        mime: The declared content-type for the file part.
        visibility: The ``visibility`` form field value.

    Returns:
        A ``files`` dict compatible with ``httpx`` / ``requests`` multipart.
    """
    return {
        "file": (filename, io.BytesIO(content), mime),
        "visibility": (None, visibility),
    }


async def _get_doc_row(clean_pg: str, doc_id: str) -> dict | None:
    """Fetch a raw document row from the DB for post-condition assertions.

    Args:
        clean_pg: DSN of the testcontainer Postgres instance.
        doc_id: The document id string to look up.

    Returns:
        The raw ``documents`` row dict, or ``None`` when absent.
    """
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            row = await DocumentRepository(session).get_by_id(uuid.UUID(doc_id))
    finally:
        await db.dispose()
    return row


def _truncate_ephemeral_sessions(pg_dsn: str) -> None:
    """Truncate the ephemeral_sessions table for inter-test isolation.

    Args:
        pg_dsn: DSN of the testcontainer Postgres.
    """
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE iw_rag.ephemeral_sessions RESTART IDENTITY CASCADE"))
    engine.dispose()


# ---------------------------------------------------------------------------
# POST /documents — 201 success + owner stamped from auth
# ---------------------------------------------------------------------------


async def test_upload_returns_201_with_doc_id(clean_pg: str, tmp_path: Path) -> None:
    """Successful upload returns HTTP 201 and a non-empty ``id`` field."""
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "id" in body
    assert body["id"]  # non-empty string


async def test_upload_stamps_owner_from_auth_not_body(clean_pg: str, tmp_path: Path) -> None:
    """Upload stamps ``owner_id`` from auth (alice's principal_id), not from the request body.

    Security invariant: the ``owner_id`` in the persisted document row must equal
    ``alice`` (the authenticated principal), even though the multipart form has no
    owner field — the route NEVER reads an owner from the body.
    """
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 201, resp.text
    doc_id = resp.json()["id"]

    # Verify the DB row: owner_id must come from auth (alice), never from the body.
    row = await _get_doc_row(clean_pg, doc_id)
    assert row is not None, "document row must exist after a successful upload"
    assert row["owner_id"] == "alice", (
        f"SECURITY BUG: owner_id is {row['owner_id']!r}, expected 'alice' (from auth)"
    )


async def test_upload_response_carries_doc_id(clean_pg: str, tmp_path: Path) -> None:
    """The 201 response carries the persisted document's id in the body."""
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(content=b"sample content for id test"),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "id" in body
    # The id must be a parseable UUID string.
    parsed = uuid.UUID(body["id"])  # raises ValueError if invalid
    assert str(parsed)


# ---------------------------------------------------------------------------
# Upload validation negatives
# ---------------------------------------------------------------------------


async def test_upload_empty_file_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """An empty-body upload is rejected with HTTP 400."""
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(content=b""),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "error" in body
    assert "ValidationError" in body["error"]


async def test_upload_oversized_file_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """A file exceeding max_upload_bytes is rejected with HTTP 400."""
    client = _client(clean_pg, tmp_path)
    oversized = b"x" * (_MAX_UPLOAD_BYTES + 1)
    resp = client.post(
        "/documents",
        files=_upload_form(content=oversized),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "error" in body
    assert "ValidationError" in body["error"]


async def test_upload_invalid_visibility_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """An unrecognised ``visibility`` value is rejected with HTTP 400."""
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(visibility="public"),  # not 'common' or 'specific'
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "error" in body
    assert "ValidationError" in body["error"]


async def test_upload_invalid_visibility_error_message(clean_pg: str, tmp_path: Path) -> None:
    """The 400 body for an invalid visibility carries a meaningful error type."""
    client = _client(clean_pg, tmp_path)
    resp = client.post(
        "/documents",
        files=_upload_form(visibility="all"),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    # The error field names the exception class (from the app's error handler).
    assert body.get("error") == "ValidationError"


# ---------------------------------------------------------------------------
# Parse/embed failure → cleanup branch (documents.py:343-347)
# ---------------------------------------------------------------------------


async def test_upload_embed_failure_cleans_up_partial_row(clean_pg: str, tmp_path: Path) -> None:
    """When embed fails the ingestion status is 'failed' and no document row is left.

    The pipeline never writes DB rows before embedding succeeds, so a failed
    embed means the document table has no partial row.  The upload method also
    calls ``_cleanup_dir`` to remove staged bytes (best-effort).
    """
    client = _client(clean_pg, tmp_path, embedder=_BrokenEmbedder())
    resp = client.post(
        "/documents",
        files=_upload_form(content=b"some content that will fail to embed"),
        headers={"authorization": "Bearer alice-token"},
    )
    # The route propagates ValidationError (from the failed ingest) → 400.
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "error" in body
    assert "ValidationError" in body["error"]

    # DB post-condition: no document row was left behind.
    engine = sqlalchemy.create_engine(clean_pg)
    with engine.begin() as conn:
        result = conn.execute(
            text("SELECT COUNT(*) FROM iw_rag.documents WHERE owner_id = 'alice'")
        )
        count = result.scalar()
    engine.dispose()
    assert count == 0, f"Expected no document rows after embed failure, found {count}"


# ---------------------------------------------------------------------------
# DELETE /documents/{id} — 204 by owner; 404 for unknown id
# ---------------------------------------------------------------------------


async def test_delete_by_owner_returns_204_and_row_gone(clean_pg: str, tmp_path: Path) -> None:
    """Owner can delete their document; HTTP 204 is returned and the row is absent."""
    client = _client(clean_pg, tmp_path)

    # First upload a document as alice.
    upload_resp = client.post(
        "/documents",
        files=_upload_form(content=b"delete me"),
        headers={"authorization": "Bearer alice-token"},
    )
    assert upload_resp.status_code == 201, upload_resp.text
    doc_id = upload_resp.json()["id"]

    # Now delete it.
    del_resp = client.delete(
        f"/documents/{doc_id}",
        headers={"authorization": "Bearer alice-token"},
    )
    assert del_resp.status_code == 204, del_resp.text

    # DB post-condition: the document row must be gone.
    row = await _get_doc_row(clean_pg, doc_id)
    assert row is None, f"Document row {doc_id} must be absent after deletion"


async def test_delete_unknown_id_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """Deleting an unknown document id returns HTTP 404 (existence never leaked)."""
    client = _client(clean_pg, tmp_path)
    unknown_id = str(uuid.uuid4())
    resp = client.delete(
        f"/documents/{unknown_id}",
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# GET /documents — list returns only the caller's visible docs
# ---------------------------------------------------------------------------


async def test_list_documents_returns_only_caller_visible(clean_pg: str, tmp_path: Path) -> None:
    """``GET /documents`` returns only the caller's own + common docs, not other users' specific.

    Seeding:
    - alice uploads a ``specific`` doc (visible only to alice).
    - alice uploads a ``common`` doc (visible to everyone).
    - bob uploads a ``specific`` doc (visible only to bob).

    When alice calls ``GET /documents`` she should see exactly her two docs.
    When bob calls ``GET /documents`` he should see exactly his specific doc
    plus alice's common doc (2 total), but NOT alice's specific doc.
    """
    client = _client(clean_pg, tmp_path)

    # Alice uploads a specific doc.
    alice_specific = client.post(
        "/documents",
        files=_upload_form(content=b"alice specific doc", visibility="specific"),
        headers={"authorization": "Bearer alice-token"},
    )
    assert alice_specific.status_code == 201, alice_specific.text
    alice_specific_id = alice_specific.json()["id"]

    # Alice uploads a common doc.
    alice_common = client.post(
        "/documents",
        files=_upload_form(content=b"alice common doc", visibility="common"),
        headers={"authorization": "Bearer alice-token"},
    )
    assert alice_common.status_code == 201, alice_common.text
    alice_common_id = alice_common.json()["id"]

    # Bob uploads a specific doc.
    bob_specific = client.post(
        "/documents",
        files=_upload_form(content=b"bob specific doc", visibility="specific"),
        headers={"authorization": "Bearer bob-token"},
    )
    assert bob_specific.status_code == 201, bob_specific.text
    bob_specific_id = bob_specific.json()["id"]

    # Alice lists — must see her own two docs.
    alice_list = client.get("/documents", headers={"authorization": "Bearer alice-token"})
    assert alice_list.status_code == 200, alice_list.text
    alice_ids = {d["id"] for d in alice_list.json()}
    assert alice_specific_id in alice_ids, "Alice must see her own specific doc"
    assert alice_common_id in alice_ids, "Alice must see her own common doc"
    assert bob_specific_id not in alice_ids, "Alice must NOT see bob's specific doc"

    # Bob lists — must see alice's common doc + his own specific doc.
    bob_list = client.get("/documents", headers={"authorization": "Bearer bob-token"})
    assert bob_list.status_code == 200, bob_list.text
    bob_ids = {d["id"] for d in bob_list.json()}
    assert bob_specific_id in bob_ids, "Bob must see his own specific doc"
    assert alice_common_id in bob_ids, "Bob must see alice's common doc"
    assert alice_specific_id not in bob_ids, "Bob must NOT see alice's specific doc"


# ---------------------------------------------------------------------------
# POST /attachments — 201 happy path
# ---------------------------------------------------------------------------


async def test_attach_file_returns_201_with_document_id(clean_pg: str, tmp_path: Path) -> None:
    """Attaching a file to a chat session returns HTTP 201 with a ``document_id``."""
    _truncate_ephemeral_sessions(clean_pg)
    client = _client(clean_pg, tmp_path)
    session_id = f"chat-{uuid.uuid4().hex[:8]}"

    resp = client.post(
        "/attachments",
        files={
            "file": ("note.txt", io.BytesIO(b"attachment content"), _PLAINTEXT_MIME),
            "session_id": (None, session_id),
        },
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert "document_id" in body
    assert body["document_id"]
    assert body["session_id"] == session_id


# ---------------------------------------------------------------------------
# DELETE /attachments/{session_id} — 204 happy path
# ---------------------------------------------------------------------------


async def test_purge_attachments_returns_204(clean_pg: str, tmp_path: Path) -> None:
    """Purging a chat session's attachments returns HTTP 204."""
    _truncate_ephemeral_sessions(clean_pg)
    client = _client(clean_pg, tmp_path)
    session_id = f"chat-purge-{uuid.uuid4().hex[:8]}"

    # First attach a file to register the session.
    attach_resp = client.post(
        "/attachments",
        files={
            "file": ("note.txt", io.BytesIO(b"ephemeral content"), _PLAINTEXT_MIME),
            "session_id": (None, session_id),
        },
        headers={"authorization": "Bearer alice-token"},
    )
    assert attach_resp.status_code == 201, attach_resp.text

    # Now purge the session.
    purge_resp = client.delete(
        f"/attachments/{session_id}",
        headers={"authorization": "Bearer alice-token"},
    )
    assert purge_resp.status_code == 204, purge_resp.text


# ---------------------------------------------------------------------------
# _service-unwired → 400 branch (documents_routes.py:42-47)
# ---------------------------------------------------------------------------


async def test_upload_without_service_wired_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """When ``document_service`` is not wired on ``app.state``, routes return 400.

    ``_service()`` raises :class:`ValidationError` which the app's error handler
    maps to HTTP 400 (not 503 as the docstring comment implies — ``ValidationError``
    is in the 400 bucket per ``app.py``'s ``_ERROR_STATUS`` table).
    """
    client = _client_no_service(clean_pg)
    resp = client.post(
        "/documents",
        files=_upload_form(),
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert "error" in body
    assert "ValidationError" in body["error"]


async def test_list_without_service_wired_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """``GET /documents`` with no service wired returns 400."""
    client = _client_no_service(clean_pg)
    resp = client.get("/documents", headers={"authorization": "Bearer alice-token"})
    assert resp.status_code == 400, resp.text


async def test_delete_without_service_wired_returns_400(clean_pg: str, tmp_path: Path) -> None:
    """``DELETE /documents/{id}`` with no service wired returns 400."""
    client = _client_no_service(clean_pg)
    resp = client.delete(
        f"/documents/{uuid.uuid4()}",
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 400, resp.text
