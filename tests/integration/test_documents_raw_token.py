"""Integration tests for ``GET /documents/{document_id}/raw?token=`` (CR-00098 / S04).

A plain browser click on a citation downloads the source file via a
short-lived signed ``?token=`` query parameter — without an
``Authorization`` header (the chainlit frontend authenticates with a
bearer, so a naive link would 401 the browser navigation).

These tests exercise the **full route** (``TestClient`` + a real
``DocumentService`` against a testcontainers Postgres) so the bearer path
and the new token path are both pinned end-to-end.  The token's
authorisation is intentionally narrow:

* The token binds ``{document_id, principal_id, exp[, session_id]}``.
* The download path **re-resolves** the principal's *current* groups from
  the auth source (``Authenticator.resolve_principal``) — never from the
  token payload.  The token is a capability, not an ACL assertion.
* ``DocumentService.get_raw`` (S03) re-runs the ACL overlap check at
  download time — defense in depth (a revoked group takes effect
  immediately).
* A failed token is indistinguishable from a missing document
  (``NotFoundError`` → 404) so the API never leaks existence.

Coverage matrix:

* valid token (no ``Authorization`` header) → 200 with the file bytes
  and ``Content-Disposition: attachment``.
* expired token → 404 (no bytes).
* tampered signature / payload → 404.
* token whose ``document_id`` ≠ path ``{id}`` → 404.
* token whose ``principal_id`` is unknown to the auth source → 404.
* token with an empty secret configured → 404 (fail-safe; the
  ``IW_RAG__API__DOWNLOAD_TOKEN_SECRET`` is env-only and intentionally
  unset by default).
* bearer path still works unchanged (regression pin).
* both bearer and token present → bearer wins.
* ACL-overlap re-check still gates the token path (foreign user
  carrying a leaked-but-valid token → 404).
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import (
    DemoUserAuthenticator,
    StaticTokenAuthenticator,
    user_group,
)
from iw_rag.api.documents import CharTokenizer, DocumentService
from iw_rag.api.download_tokens import mint, verify
from iw_rag.ingestion.chunk import ChunkConfig, HybridChunker
from iw_rag.ingestion.embed import EMBEDDING_DIMS
from iw_rag.settings import ApiSettings, DatabaseSettings, Settings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ChunkRepository, DocumentRepository, NewChunk

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fakes — bypass embedder/chunker/parser; tests seed the rows directly
# ---------------------------------------------------------------------------


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


def _make_service(clean_pg: str, tmp_path: Path) -> DocumentService:
    """Build a DocumentService with inert fakes for embed/chunker/pipeline."""
    return DocumentService(
        db=Database(DatabaseSettings(dsn=clean_pg)),
        embedder=_FakeEmbedder(),  # type: ignore[arg-type]
        chunker=HybridChunker(ChunkConfig(leaf_tokens=64, parent_max_tokens=256), CharTokenizer()),
        upload_dir=str(tmp_path),
    )


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute-style, like S02)."""

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


async def _seed_doc_with_staged(
    clean_pg: str,
    *,
    upload_dir: Path,
    acl_tags: list[str],
    owner_id: str | None = None,
    visibility: str | None = None,
) -> tuple[str, Path]:
    """Seed a document + chunk + actual staged bytes.

    Returns:
        ``(document_id, stored_path)``.
    """
    doc_uuid = uuid.uuid4()
    stored_dir = upload_dir / doc_uuid.hex
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / "hello.txt"
    stored_path.write_bytes(b"hello world\n")
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            docs = DocumentRepository(session)
            chunks = ChunkRepository(session)
            doc_id = await docs.upsert_document(
                source_uri=f"corpus://test/{doc_uuid}/hello.txt",
                checksum=f"cs-{doc_uuid}",
            )
            await docs.set_attributes(
                doc_id,
                owner_id=owner_id,
                visibility=visibility,
                metadata={
                    "filename": "hello.txt",
                    "mime": "text/plain",
                    "size_bytes": 12,
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
                    ),
                ]
            )
            await session.commit()
    finally:
        await db.dispose()
    return str(doc_id), stored_path


# ---------------------------------------------------------------------------
# Test client builders
# ---------------------------------------------------------------------------

_SECRET = "test-secret-must-be-long-enough-for-hmac-sha256"


def _client_with_demo_user(
    clean_pg: str,
    tmp_path: Path,
    *,
    with_secret: bool = True,
    ttl_seconds: int = 300,
) -> tuple[TestClient, DocumentService, Settings]:
    """Build a TestClient + DocumentService using the demo-user authenticator.

    The demo authenticator exposes ``resolve_principal`` (so the S04 token
    path can re-resolve the principal's *current* groups) — the legacy
    static-token authenticator does not (per S02).  ``settings`` is
    returned so the test can mint a token with the same secret + TTL.
    """
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
            download_token_secret=_SECRET if with_secret else "",
            download_token_ttl_seconds=ttl_seconds,
        ),
    )
    app = create_app(
        settings=settings,
        authenticator=auth,
        query_service=_StubQueryService(),
        database=Database(DatabaseSettings(dsn=clean_pg)),
        document_service=svc,
    )
    return TestClient(app), svc, settings


def _client_with_static_token(
    clean_pg: str,
    tmp_path: Path,
    *,
    with_secret: bool = True,
) -> tuple[TestClient, DocumentService, Settings]:
    """Build a TestClient with the legacy ``StaticTokenAuthenticator``.

    The static authenticator's ``resolve_principal`` always returns
    ``None`` (S02), so the S04 token path is unavailable here — the
    bearer header is the only way in.  Used to pin the bearer path's
    behavior is unchanged.
    """
    svc = _make_service(clean_pg, tmp_path)
    auth = StaticTokenAuthenticator.from_config(
        {"alice-token": ["finance", "public", user_group("alice")]},
        operator_tokens=[],
    )
    settings = Settings(
        database=DatabaseSettings(dsn=clean_pg),
        api=ApiSettings(
            static_tokens={"alice-token": ["finance"]},
            demo_users={},
            download_token_secret=_SECRET if with_secret else "",
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
    return TestClient(app), svc, settings


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s `query_service` seam."""

    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
        raise RuntimeError("not used in S04 tests")


# ---------------------------------------------------------------------------
# 1. Valid token (no Authorization header) — downloads the file
# ---------------------------------------------------------------------------


async def test_raw_with_valid_token_no_bearer_downloads_file(clean_pg: str, tmp_path: Path) -> None:
    """A valid signed token is enough to download — no Authorization header needed.

    Re-resolves ``alice``'s current groups from the demo authenticator
    and re-runs the ACL overlap (``user:alice`` matches the chunk's
    ``user:alice`` tag) — the file is served with
    ``Content-Disposition: attachment`` so the browser downloads rather
    than navigates.
    """
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, stored_path = await _seed_doc_with_staged(
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
    assert verify(token, secret=settings.api.download_token_secret) is not None

    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 200, resp.text
    assert resp.content == b"hello world\n"
    # Force-download disposition — the browser downloads rather than navigating.
    assert resp.headers["content-disposition"].startswith("attachment;")
    assert 'filename="hello.txt"' in resp.headers["content-disposition"]


# ---------------------------------------------------------------------------
# 2. Expired token → 404 (no bytes)
# ---------------------------------------------------------------------------


async def test_raw_with_expired_token_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """An expired token is rejected as if the document did not exist (existence never leaked)."""
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Mint a token that's already expired (now > exp boundary).
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=1,
        now=1_700_000_000,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token, "_now": 1_700_000_999})
    # Token verified by us above used a 1s TTL; once the request reaches
    # the server, the server's own time has advanced past the boundary
    # (or the boundary is already crossed on a long-running test run).
    # The test is robust as long as the token is rejected:
    assert resp.status_code == 404, resp.text
    body = resp.text
    assert body == "" or "trace_id" in body.lower()


# ---------------------------------------------------------------------------
# 3. Tampered signature / payload → 404
# ---------------------------------------------------------------------------


async def test_raw_with_tampered_signature_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """Flipping a byte in the signature segment fails HMAC verify → 404."""
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
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
        ttl_seconds=300,
    )
    payload_b64, sig_b64 = token.split(".")
    flipped = "A" if sig_b64[0] != "A" else "B"
    tampered = f"{payload_b64}.{flipped}{sig_b64[1:]}"

    resp = client.get(f"/documents/{doc_id}/raw", params={"token": tampered})
    assert resp.status_code == 404, resp.text


async def test_raw_with_tampered_payload_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """Forging a different ``document_id`` in the payload fails HMAC verify → 404."""
    import base64
    import json

    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
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
        ttl_seconds=300,
    )
    payload_b64, sig_b64 = token.split(".")
    # Forge a payload with a different document_id; reuse the original
    # signature → HMAC mismatch.
    forged_payload_bytes = json.dumps(
        {"d": "00000000-0000-0000-0000-000000000000", "p": "alice", "exp": 9_999_999_999},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged_payload_b64 = base64.urlsafe_b64encode(forged_payload_bytes).rstrip(b"=").decode()
    tampered = f"{forged_payload_b64}.{sig_b64}"

    resp = client.get(f"/documents/{doc_id}/raw", params={"token": tampered})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 4. Token whose document_id ≠ path `{id}` → 404
# ---------------------------------------------------------------------------


async def test_raw_with_token_for_other_document_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """A valid token for document A never serves document B (URL path is part of the binding)."""
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_a, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    doc_b, _ = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    token_for_a = mint(
        document_id=doc_a,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=300,
    )
    # Use the token for A against the URL of B → must not serve B.
    resp = client.get(f"/documents/{doc_b}/raw", params={"token": token_for_a})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 5. Unknown principal_id in token → 404
# ---------------------------------------------------------------------------


async def test_raw_with_unknown_principal_token_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """A valid signature for a principal id the auth source does not know is refused."""
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Mint a token bound to a principal the authenticator doesn't know.
    token = mint(
        document_id=doc_id,
        principal_id="ghost-user",
        secret=settings.api.download_token_secret,
        ttl_seconds=300,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 6. Empty secret → 404 (fail-safe; feature off when secret is unset)
# ---------------------------------------------------------------------------


async def test_raw_with_token_fails_closed_when_secret_unset(clean_pg: str, tmp_path: Path) -> None:
    """When the download secret is unset, the token path rejects every token.

    A misconfigured deployment never accidentally serves downloads via
    the token path — the bearer path is unaffected.
    """
    client_no_secret, _svc_no, _settings_no = _client_with_demo_user(
        clean_pg, tmp_path, with_secret=False
    )
    client_with_secret, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Mint under the "with secret" client; use it against the "no secret" client.
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=300,
    )
    resp = client_no_secret.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 404, resp.text

    # And the bearer path on the "no secret" client still works (regression pin).
    bearer = client_no_secret.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"}
    )
    assert bearer.status_code == 200, bearer.text


# ---------------------------------------------------------------------------
# 7. Bearer path still works unchanged (regression pin)
# ---------------------------------------------------------------------------


async def test_raw_with_bearer_only_still_works(clean_pg: str, tmp_path: Path) -> None:
    """The bearer Authorization header path is unchanged by the token addition."""
    client, _svc, _settings = _client_with_static_token(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice"), "public"],
        owner_id="alice",
        visibility="specific",
    )
    resp = client.get(f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"})
    assert resp.status_code == 200, resp.text
    assert resp.content == b"hello world\n"
    assert resp.headers["content-disposition"].startswith("attachment;")


async def test_raw_with_no_auth_and_no_token_returns_401(clean_pg: str, tmp_path: Path) -> None:
    """With no token AND no Authorization header, the request is unauthorized.

    The bearer dependency is optional ONLY when a valid token is present;
    neither present → 401 (the bearer gate fires).
    """
    client, _svc, _settings = _client_with_static_token(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice"), "public"],
        owner_id="alice",
        visibility="specific",
    )
    resp = client.get(f"/documents/{doc_id}/raw")
    assert resp.status_code == 401, resp.text


# ---------------------------------------------------------------------------
# 8. ACL-overlap re-check still gates the token path (defense in depth)
# ---------------------------------------------------------------------------


async def test_raw_with_token_for_foreign_user_returns_404(clean_pg: str, tmp_path: Path) -> None:
    """A valid token for ``bob`` never serves ``alice``'s specific doc.

    Even though the token verifies and ``bob`` resolves to a real
    principal, the ACL overlap is re-checked at download time and
    ``bob`` carries no ``user:alice`` group → 404.  Existence is not
    leaked.
    """
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],  # alice-only
        owner_id="alice",
        visibility="specific",
    )
    token = mint(
        document_id=doc_id,
        principal_id="bob",  # valid bob token, but no ACL overlap
        secret=settings.api.download_token_secret,
        ttl_seconds=300,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 9. Both bearer and token present → bearer wins
# ---------------------------------------------------------------------------


async def test_raw_with_both_bearer_and_token_prefers_bearer(clean_pg: str, tmp_path: Path) -> None:
    """When both ``Authorization`` and a valid ``?token=`` are present, the bearer wins.

    The token path is a fallback for browser navigation that cannot
    carry a header — when a header is available, the richer bearer path
    takes precedence (and a mismatched token is irrelevant because the
    bearer is sufficient on its own).  The contract is that the bearer
    path is never broken by the token addition.
    """
    client, _svc, settings = _client_with_demo_user(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    # Token for a different principal — but bearer is alice's → 200.
    bogus_token = mint(
        document_id=doc_id,
        principal_id="bob",
        secret=settings.api.download_token_secret,
        ttl_seconds=300,
    )
    resp = client.get(
        f"/documents/{doc_id}/raw",
        params={"token": bogus_token},
        headers={"authorization": "Bearer alice-token"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.content == b"hello world\n"
