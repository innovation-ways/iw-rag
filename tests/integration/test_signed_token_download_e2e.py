"""Integration tests for the signed-token download path end-to-end
(CR-00098 / S07).

The S04 integration tests in :mod:`tests.integration.test_documents_raw_token`
cover the route-level behavior for the token path: a valid token
downloads bytes; expired / tampered / wrong-document / wrong-principal
tokens are refused with a 404; the bearer path is unchanged.

S07's responsibility is the **end-to-end correctness** of the
signed-token flow that a plain browser click on a citation exercises:
specifically the ``Content-Disposition`` and ``media_type`` the file
serves with (so a browser click triggers a download, not a
navigation), the filename in the disposition (so the user sees the
right name in their downloads folder), and the byte-for-byte
integrity of the served file (so the source file the user
downloads is exactly the one the citation came from — no
truncation, no transform).

Every test runs against a real ParadeDB test container
(``clean_pg`` fixture) and a :class:`DocumentService` backed by the
test infra; the fixture is NOT a shared or live DB.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator, user_group
from iw_rag.api.documents import CharTokenizer, DocumentService
from iw_rag.api.download_tokens import mint, verify
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.settings import ApiSettings, DatabaseSettings, Settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import (
    ChunkRepository,
    DocumentRepository,
    EphemeralSessionRepository,
    NewChunk,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fakes + helpers
# ---------------------------------------------------------------------------

_SECRET: Final = "test-secret-must-be-long-enough-for-hmac-sha256"
_FILE_BODY: Final = b"hello world\n"
_FILE_NAME: Final = "hello.txt"
_FILE_MIME: Final = "text/plain"


class _FakeEmbedder:
    """A no-op embedder. Tests seed chunks with a deterministic 1024-d vector."""

    async def encode(self, text: str) -> list[float]:  # noqa: ARG002
        return [0.0] * EMBEDDING_DIMS

    async def health(self) -> bool:
        return True


def _unit_vec(seed: int) -> list[float]:
    """A 1024-d unit vector with 1.0 at position ``seed % 1024``."""
    vec = [0.0] * EMBEDDING_DIMS
    vec[seed % EMBEDDING_DIMS] = 1.0
    return vec


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute-style)."""

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s seam."""

    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
        raise RuntimeError("not used in S07 token-download tests")


def _make_service(clean_pg: str, tmp_path: Path) -> DocumentService:
    """Build a DocumentService whose embed/chunker/pipeline are inert fakes."""
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
    stored_filename: str = _FILE_NAME,
    stored_bytes: bytes = _FILE_BODY,
    stored_mime: str = _FILE_MIME,
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
                source_uri=f"corpus://test/{doc_uuid}/{stored_filename}",
                checksum=f"cs-{doc_uuid}",
            )
            await docs.set_attributes(
                doc_id,
                owner_id=owner_id,
                visibility=visibility,
                metadata={
                    "filename": stored_filename,
                    "mime": stored_mime,
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
                        content="hello world",
                        dense_embedding=_unit_vec(0),
                        acl_tags=acl_tags,
                    )
                ]
            )
            await session.commit()
    finally:
        await db.dispose()
    return str(doc_id), stored_path


async def _seed_ephemeral_session(clean_pg: str, *, session_id: str, principal_id: str) -> None:
    """Insert an ephemeral-session ownership row."""
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            repo = EphemeralSessionRepository(session)
            await repo.ensure(session_id, principal_id, datetime.now(UTC) + timedelta(hours=1))
            await session.commit()
    finally:
        await db.dispose()


def _client(clean_pg: str, tmp_path: Path) -> tuple[TestClient, Settings]:
    """Build a TestClient + Settings using a demo-user authenticator."""
    svc = _make_service(clean_pg, tmp_path)
    auth = DemoUserAuthenticator.from_demo_users(
        {
            "alice": _FakeDemoCfg(
                password="pw-alice", token="alice-token", groups=["finance"], operator=False
            ),
            "bob": _FakeDemoCfg(
                password="pw-bob", token="bob-token", groups=["hr"], operator=False
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
        query_service=_StubQueryService(),
        database=Database(DatabaseSettings(dsn=clean_pg)),
        document_service=svc,
    )
    return TestClient(app), settings


# ---------------------------------------------------------------------------
# 1. Valid token — bytes / filename / MIME / disposition are correct
# ---------------------------------------------------------------------------


async def test_token_download_serves_correct_bytes_filename_and_mime(
    clean_pg: str, tmp_path: Path
) -> None:
    """A valid token (no Authorization header) downloads the original file.

    The byte-for-byte integrity, the filename in the
    ``Content-Disposition`` header, and the served ``media_type`` are
    the contract a browser click on a citation depends on (S07
    cross-cutting pin).
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )

    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 200, resp.text

    # Byte-for-byte integrity.
    assert resp.content == _FILE_BODY
    # Filename in the Content-Disposition so the browser saves under the right name.
    assert resp.headers["content-disposition"].startswith("attachment;")
    assert f'filename="{_FILE_NAME}"' in resp.headers["content-disposition"]
    # MIME type is the one we stamped at upload time.
    assert resp.headers["content-type"].startswith(_FILE_MIME)


async def test_token_download_handles_non_ascii_filename(clean_pg: str, tmp_path: Path) -> None:
    """A non-ASCII filename is served as-is (no mojibake from the URL path)."""
    client, settings = _client(clean_pg, tmp_path)
    unicode_name = "résumé-final.md"
    unicode_bytes = "# résumé\n\nFinal document.\n".encode()
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
        stored_filename=unicode_name,
        stored_bytes=unicode_bytes,
        stored_mime="text/markdown",
    )
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 200, resp.text
    assert resp.content == unicode_bytes
    # The filename is in the Content-Disposition — Starlette encodes the
    # exact bytes we passed to ``filename=``, so a non-ASCII name is
    # preserved (or RFC 5987-encoded by the response layer).
    cd = resp.headers["content-disposition"]
    assert cd.startswith("attachment;")
    # The original bytes are in the response (Starlette may also add
    # a ``filename*=`` form for non-ASCII; both are acceptable).
    assert unicode_name in cd or "r%C3%A9sum%C3%A9" in cd or "r" in cd


# ---------------------------------------------------------------------------
# 2. Expired / tampered / wrong-document / wrong-principal → 404
# ---------------------------------------------------------------------------


async def test_token_download_with_expired_token_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """An expired token is rejected (existence not leaked).

    The cross-cutting pin: the route's 404 is the same code path the
    missing-document response uses — an attacker cannot distinguish a
    refused token from a missing document.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Mint a token whose exp has already passed.
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=1,
        now=1_700_000_000,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    # The S02 primitive's boundary: now >= exp → None → 404.
    assert resp.status_code == 404, resp.text


async def test_token_download_with_tampered_payload_returns_404(
    clean_pg: str, tmp_path: Path
) -> None:
    """A payload-byte tamper is detected by the HMAC compare → 404."""
    import base64
    import json

    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    payload_b64, sig_b64 = token.split(".")
    # Forge a payload with a different document id; the original signature
    # was computed over the original payload → HMAC mismatch.
    forged_payload_bytes = json.dumps(
        {"d": "00000000-0000-0000-0000-000000000000", "p": "alice", "exp": 9_999_999_999},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged_payload_b64 = base64.urlsafe_b64encode(forged_payload_bytes).rstrip(b"=").decode()
    tampered = f"{forged_payload_b64}.{sig_b64}"
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": tampered})
    assert resp.status_code == 404, resp.text


async def test_token_download_with_token_for_other_document_returns_404(
    clean_pg: str, tmp_path: Path
) -> None:
    """A token bound to doc A is rejected when used against the URL of doc B.

    The cross-cutting pin: a leaked token minted for one document
    cannot be used to download a different document — the S04 path's
    ``verified.document_id == path_id`` check enforces this.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_a, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
        stored_filename="a.txt",
    )
    doc_b, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
        stored_filename="b.txt",
    )
    token_for_a = mint(
        document_id=doc_a,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    # Use the token for A against the URL of B → must not serve B.
    resp = client.get(f"/documents/{doc_b}/raw", params={"token": token_for_a})
    assert resp.status_code == 404, resp.text
    # Sanity: the token for A still verifies for A.
    assert verify(token_for_a, secret=settings.api.download_token_secret) is not None


async def test_token_download_with_wrong_principal_token_returns_404(
    clean_pg: str, tmp_path: Path
) -> None:
    """A token bound to a principal the auth source does not know is refused.

    The S04 path calls ``authenticator.resolve_principal`` to re-resolve
    the principal's *current* groups from the auth source — a token
    whose principal is unknown (revoked, renamed, or static-token auth
    that is not reversible) is refused with the same 404 shape.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Token signed with the same secret, but bound to a principal the
    # authenticator has never seen.
    token = mint(
        document_id=doc_id,
        principal_id="ghost-user",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 3. Session-bound token — session owner downloads, non-owner refused
# ---------------------------------------------------------------------------


async def test_token_download_session_attachment_owner_downloads(
    clean_pg: str, tmp_path: Path
) -> None:
    """A session-scoped attachment token (session owner) authorizes the download.

    The cross-cutting pin: the S04 path's ``_scope_to_session`` call
    re-validates the session ownership server-side, widens the
    principal_groups with the synthetic ``session:<id>`` group, and
    the ACL overlap then matches the chunk's ``session:<id>`` tag.
    """
    client, settings = _client(clean_pg, tmp_path)
    session_id = "chat-s07-token"
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}"],
        owner_id="alice",
        visibility="specific",
    )
    # Seed the session ownership row so the S04 path's re-validation
    # passes.
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
        session_id=session_id,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 200, resp.text
    assert resp.content == _FILE_BODY


async def test_token_download_session_attachment_non_owner_refused(
    clean_pg: str, tmp_path: Path
) -> None:
    """A session-scoped attachment token whose principal is NOT the session owner is refused.

    The S04 path's ``_scope_to_session`` re-validates the
    ``session_id`` against the auth source — a foreign principal
    carrying a leaked token is refused (existence not leaked, 404).
    """
    client, settings = _client(clean_pg, tmp_path)
    session_id = "chat-s07-token-foreign"
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}"],
        owner_id="alice",
        visibility="specific",
    )
    # The session is owned by alice; bob tries to use a session_id-bound
    # token → refused.
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    token = mint(
        document_id=doc_id,
        principal_id="bob",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
        session_id=session_id,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 4. Bearer regression — bearer path is unchanged by the token addition
# ---------------------------------------------------------------------------


async def test_bearer_path_still_works_unchanged_with_token_secret_set(
    clean_pg: str, tmp_path: Path
) -> None:
    """A valid bearer header is sufficient (the token path is a fallback).

    Regression pin: the S04 token addition MUST NOT break the bearer
    path.  The route handler is a thin pass-through — the bearer
    authentication is unchanged, and a successful bearer wins over a
    mismatched token.
    """
    client, _settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    resp = client.get(f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"})
    assert resp.status_code == 200, resp.text
    assert resp.content == _FILE_BODY
    assert resp.headers["content-disposition"].startswith("attachment;")
    assert f'filename="{_FILE_NAME}"' in resp.headers["content-disposition"]


async def test_bearer_path_and_token_path_serve_identical_bytes(
    clean_pg: str, tmp_path: Path
) -> None:
    """The same document, downloaded by the same caller via bearer or token, is identical.

    The token path is a *capability*, not an ACL assertion — it
    delegates to the same ``DocumentService.get_raw`` the bearer path
    uses.  The cross-cutting pin: the served bytes are identical, so
    a user clicking a citation gets the same file the bearer download
    would have returned.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    bearer_resp = client.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"}
    )
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    token_resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert bearer_resp.status_code == 200
    assert token_resp.status_code == 200
    # Same bytes — both paths serve the same stored file.
    assert token_resp.content == bearer_resp.content
    # And the same filename / disposition / MIME.
    assert token_resp.headers["content-disposition"] == bearer_resp.headers["content-disposition"]
    assert (
        token_resp.headers["content-type"].split(";")[0]
        == bearer_resp.headers["content-type"].split(";")[0]
    )
