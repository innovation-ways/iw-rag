"""Integration tests for the ``GET /documents/{id}/raw`` ACL-overlap matrix
(CR-00098 / S07).

The S03 integration tests in :mod:`tests.integration.test_documents_get_raw_acl`
exercise :meth:`DocumentService.get_raw` directly (the gate inside the
service).  The S04 integration tests in
:mod:`tests.integration.test_documents_raw_token` exercise the full route
on the bearer / token paths.

S07 is responsible for the **full route** under the ACL-overlap matrix —
``GET /documents/{id}/raw`` over the four ACL flavours, with a TestClient +
testcontainers + :class:`DocumentService` (the same shape as the S04
file, but the focus is the matrix, not the token plumbing).  The bearer
path AND the token path must both obey the same ACL-overlap gate
(defense in depth — the token is a capability, the gate is the
authoritative authorization).

Matrix (CR-00098 / AC4 — mirror the retrieval ACL):

* **Owner** (``user:<owner>``) → owner downloads.  Grant-only invariant.
* **Common / public** (``public``) → any authenticated caller downloads.
  Grant-only invariant (the old ``_visible_to`` gate authorised this;
  the new ACL-overlap MUST keep it authorised).
* **Group-tagged corpus doc** (``finance``) → a member of ``finance``
  downloads; a non-member is refused with a 404-shaped
  ``NotFoundError`` (existence is never leaked).
* **Session attachment** (``session:<id>``) → the session owner
  downloads (after server-side session-widening); a non-owner is
  refused.

Every test runs against a real ParadeDB test container (``clean_pg``
fixture) and a :class:`DocumentService` backed by the test infra; the
fixture is NOT a shared or live DB.
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
from iw_rag.api.download_tokens import mint
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import NotFoundError
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
# Fakes (embedder / chunker / query service) + a test-secret + helper builders
# ---------------------------------------------------------------------------

_SECRET: Final = "test-secret-must-be-long-enough-for-hmac-sha256"
_FILE_BODY: Final = b"hello world\n"


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
        raise RuntimeError("not used in S07 ACL-matrix tests")


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
    stored_filename: str = "hello.txt",
) -> tuple[str, Path]:
    """Seed a document + chunk + actual staged bytes; returns ``(id, path)``."""
    doc_uuid = uuid.uuid4()
    stored_dir = upload_dir / doc_uuid.hex
    stored_dir.mkdir(parents=True, exist_ok=True)
    stored_path = stored_dir / stored_filename
    stored_path.write_bytes(_FILE_BODY)
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
                    "mime": "text/plain",
                    "size_bytes": len(_FILE_BODY),
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
    """Insert an ephemeral-session ownership row (so the service grants the synthetic group)."""
    db = Database(DatabaseSettings(dsn=clean_pg))
    try:
        async with db.session() as session:
            repo = EphemeralSessionRepository(session)
            await repo.ensure(session_id, principal_id, datetime.now(UTC) + timedelta(hours=1))
            await session.commit()
    finally:
        await db.dispose()


def _client(
    clean_pg: str,
    tmp_path: Path,
) -> tuple[TestClient, Settings]:
    """Build a TestClient + Settings using a demo-user authenticator with two users.

    The demo authenticator exposes ``resolve_principal`` (S02) so both the
    bearer and the token path work end-to-end.  Settings are returned so
    the test can mint tokens with the same secret.
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
# 1. Owner — owner downloads (grant-only invariant)
# ---------------------------------------------------------------------------


async def test_route_owner_succeeds_bearer_and_token(clean_pg: str, tmp_path: Path) -> None:
    """An owner-uploaded doc is downloadable by its owner — bearer AND token paths.

    Grant-only invariant: the old ``_visible_to`` gate authorized this caller
    (owner = ``row.owner_id == ctx.principal_id``); the new ACL-overlap gate
    must keep it authorized.  A failure here is a REGRESSION.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )

    # Bearer path — Alice's Authorization header.
    bearer = client.get(f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"})
    assert bearer.status_code == 200, bearer.text
    assert bearer.content == _FILE_BODY

    # Token path — Alice mints her own token, no Authorization header.
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    token_resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert token_resp.status_code == 200, token_resp.text
    assert token_resp.content == _FILE_BODY


# ---------------------------------------------------------------------------
# 2. Common / public — any authenticated caller downloads (grant-only)
# ---------------------------------------------------------------------------


async def test_route_common_succeeds_for_any_authenticated_caller_both_paths(
    clean_pg: str, tmp_path: Path
) -> None:
    """A ``common`` (``public``-tagged) doc is downloadable by every auth'd caller.

    Grant-only invariant (headline): the old ``_visible_to`` gate authorized
    this caller (``row.visibility == "common"``); the new ACL-overlap gate
    must keep it authorized (the ``public`` group is in every principal's
    ``augment_groups`` set).  Tested for BOTH alice (the owner) and bob
    (a different principal with no special group) and for both bearer +
    token paths.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["public"],
        owner_id="alice",
        visibility="common",
    )

    # Alice (owner) — bearer.
    alice_bearer = client.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"}
    )
    assert alice_bearer.status_code == 200, alice_bearer.text

    # Bob (foreign user) — bearer.
    bob_bearer = client.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer bob-token"}
    )
    assert bob_bearer.status_code == 200, bob_bearer.text

    # Bob — token (his own).
    bob_token = mint(
        document_id=doc_id,
        principal_id="bob",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    bob_token_resp = client.get(f"/documents/{doc_id}/raw", params={"token": bob_token})
    assert bob_token_resp.status_code == 200, bob_token_resp.text
    assert bob_token_resp.content == _FILE_BODY


# ---------------------------------------------------------------------------
# 3. Group-tagged corpus doc — member downloads, non-member refused
# ---------------------------------------------------------------------------


async def test_route_group_tagged_corpus_doc_succeeds_for_member(
    clean_pg: str, tmp_path: Path
) -> None:
    """A corpus doc tagged ``finance`` is downloadable by ``alice`` (the finance member).

    The headline grant under the ACL-overlap gate: a corpus doc
    (``owner_id IS NULL``, ``visibility IS NULL``) with an
    ``acl_tags=['finance']`` row was REFUSED by the old ``_visible_to``
    gate (no owner match, not common) — but the new gate MUST grant it
    to anyone carrying the ``finance`` group.
    """
    client, _settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,  # corpus doc — no owner
        visibility=None,  # corpus doc — not "common"
    )

    # Alice carries the ``finance`` group → grant.
    alice_resp = client.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"}
    )
    assert alice_resp.status_code == 200, alice_resp.text
    assert alice_resp.content == _FILE_BODY


async def test_route_group_tagged_corpus_doc_denies_non_member(
    clean_pg: str, tmp_path: Path
) -> None:
    """A corpus doc tagged ``finance`` is refused for ``bob`` (no ``finance`` group).

    Defense in depth: a foreign user with no overlap is refused with a
    404 (the same shape as a missing document) — existence is never
    leaked.
    """
    client, _settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
    )

    # Bob carries only the synthetic groups — no overlap with {finance}.
    bob_resp = client.get(f"/documents/{doc_id}/raw", headers={"authorization": "Bearer bob-token"})
    assert bob_resp.status_code == 404, bob_resp.text


# ---------------------------------------------------------------------------
# 4. Session attachment — session owner downloads, non-owner refused
# ---------------------------------------------------------------------------


async def test_route_session_attachment_succeeds_for_session_owner_via_bearer(
    clean_pg: str, tmp_path: Path
) -> None:
    """A ``session:<id>``-tagged attachment is downloadable by the session owner
    via a bearer whose ``AccessContext`` was widened by the service.

    The /raw route does not take a session_id query parameter — the
    session-widening happens upstream at /query time (where the
    ``_scope_to_session`` call adds ``session:<id>`` to the
    principal_groups).  The widened AccessContext is what reaches
    ``DocumentService.get_raw``.

    In this test we simulate the same widening by stamping the
    document's ACL tag as ``user:alice`` (the owner's group, which
    alice's bearer ALREADY carries) — alice is the owner AND the
    session owner, so the bearer authorization is the same as any
    owner-uploaded doc.  The full session flow (mint token with
    session_id, S04 widens + re-validates) is exercised by the next
    two tests; this test pins the bearer-side ownership grant for the
    session attachment shape.
    """
    client, _settings = _client(clean_pg, tmp_path)
    session_id = "chat-s07-1"
    # The doc carries BOTH the session tag (so it's chat-scoped at the
    # ACL layer) AND the user tag (so the bearer ALONE authorizes
    # alice's download — the session widening is a *superset* of the
    # owner grant, never a *replacement* of it).
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}", user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    alice_resp = client.get(
        f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"}
    )
    assert alice_resp.status_code == 200, alice_resp.text


async def test_route_session_attachment_token_for_session_owner_downloads(
    clean_pg: str, tmp_path: Path
) -> None:
    """A session attachment token (bound to the session owner) authorizes the download.

    The S05 mint path binds the ``session_id`` to the token when the
    caller's AccessContext was widened by ``_scope_to_session``.  The
    S04 path re-validates the session ownership before serving bytes.
    This test pins the end-to-end: a token with the right
    ``session_id``, principal, and document_id is honored.

    Note: the S04 path checks session ownership via
    ``_scope_to_session`` which requires the session_id to still be
    live in the ephemeral-sessions table.  We seed that table here so
    the re-validation passes.
    """
    client, settings = _client(clean_pg, tmp_path)
    session_id = "chat-s07-2"
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}"],
        owner_id="alice",
        visibility="specific",
    )
    # Seed the session ownership row so the S04 path's _scope_to_session
    # re-validates the principal.
    await _seed_ephemeral_session(clean_pg, session_id=session_id, principal_id="alice")

    # Mint a token bound to (document_id, principal_id, session_id).
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
        session_id=session_id,
    )
    resp = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    # S04 re-validates session ownership and the ACL overlap widens with
    # ``session:<id>`` (alice owns the session).  The file is served.
    assert resp.status_code == 200, resp.text
    assert resp.content == _FILE_BODY


async def test_route_session_attachment_token_for_non_owner_refused(
    clean_pg: str, tmp_path: Path
) -> None:
    """A session attachment token whose principal is NOT the session owner is refused.

    The S04 path's ``_scope_to_session`` re-validates the
    ``session_id`` against the auth source — a foreign principal
    carrying a leaked token is refused (existence not leaked, 404).
    """
    client, settings = _client(clean_pg, tmp_path)
    session_id = "chat-s07-3"
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[f"session:{session_id}"],
        owner_id="alice",
        visibility="specific",
    )
    # The session is owned by alice.  Bob mints a token with the same
    # session_id — the S04 path re-validates and refuses (bob doesn't
    # own the session).
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
# 5. DocumentService.get_raw unit-of-the-service contract (pin)
# ---------------------------------------------------------------------------


async def test_get_raw_raises_notfound_on_no_overlap(clean_pg: str, tmp_path: Path) -> None:
    """The service-level ``get_raw`` raises ``NotFoundError`` (not a 403).

    Defense in depth: even with a leaked service call, the existence is
    never leaked — a foreign principal's call returns the same
    exception shape as a missing document.  This test pins the
    service-level contract; the route layer's 404 is the same code
    path.
    """
    svc = _make_service(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=["finance"],
        owner_id=None,
        visibility=None,
    )
    bob_ctx = AccessContext(
        principal_id="bob",
        principal_groups=("public", user_group("bob")),  # no "finance"
    )
    with pytest.raises(NotFoundError, match="document not found") as exc_info:
        await svc.get_raw(bob_ctx, doc_id)
    assert str(exc_info.value) == "document not found"


# ---------------------------------------------------------------------------
# 6. Cross-cutting ACL-overlap invariants — bearer AND token agree
# ---------------------------------------------------------------------------


async def test_route_bearer_and_token_paths_agree_on_owner(clean_pg: str, tmp_path: Path) -> None:
    """A successful bearer download == a successful token download for the same caller.

    The bearer and the token paths share the same ACL-overlap gate
    (``DocumentService.get_raw``) — both are subject to the same
    authorization.  This test pins the cross-cutting agreement on the
    owner-grant invariant.
    """
    client, settings = _client(clean_pg, tmp_path)
    doc_id, _stored = await _seed_doc_with_staged(
        clean_pg,
        upload_dir=tmp_path,
        acl_tags=[user_group("alice")],
        owner_id="alice",
        visibility="specific",
    )

    # Bearer path → 200.
    bearer = client.get(f"/documents/{doc_id}/raw", headers={"authorization": "Bearer alice-token"})
    assert bearer.status_code == 200, bearer.text

    # Token path → 200 (same bytes).
    token = mint(
        document_id=doc_id,
        principal_id="alice",
        secret=settings.api.download_token_secret,
        ttl_seconds=settings.api.download_token_ttl_seconds,
    )
    tok = client.get(f"/documents/{doc_id}/raw", params={"token": token})
    assert tok.status_code == 200, tok.text
    # Same bytes — the storage is the same on both paths.
    assert tok.content == bearer.content
