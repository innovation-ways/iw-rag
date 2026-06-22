"""Unit tests for :class:`iw_rag.api.conversation.ConversationService`.

Exercises the orchestration logic — ownership re-validation, the
conversation-as-source chunk shape, refusal filtering, and the
foreign/garbage-id fail-soft paths — against an in-memory fake repository
(no DB, no network).  The DB-backed repository itself is covered by the
testcontainers integration test.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from iw_rag.api.conversation import ConversationService, _coerce_uuid
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import CONVERSATION_CHUNK_KIND

_OWNER = AccessContext(
    principal_id="alice", principal_groups=("hr",), is_operator=False, request_id="r1"
)
_OTHER = AccessContext(
    principal_id="mallory", principal_groups=("hr",), is_operator=False, request_id="r2"
)


class _FakeSession:
    """A no-op async session (commit is a no-op; the fake repo holds state)."""

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def commit(self) -> None:
        return None


class _FakeDB:
    def session(self) -> _FakeSession:
        return _FakeSession()


class _FakeRepo:
    """In-memory ConversationRepository stand-in (shared store across instances)."""

    convs: dict[uuid.UUID, dict[str, Any]] = {}
    msgs: dict[uuid.UUID, list[dict[str, Any]]] = {}
    _seq: int = 0

    def __init__(self, _session: Any) -> None:  # noqa: D107 — matches real signature
        pass

    @classmethod
    def reset(cls) -> None:
        cls.convs = {}
        cls.msgs = {}
        cls._seq = 0

    async def create(self, principal_id: str, *, title: str | None = None) -> uuid.UUID:
        cid = uuid.UUID(int=len(self.convs) + 1)
        self.convs[cid] = {"id": cid, "principal_id": principal_id, "title": title}
        self.msgs[cid] = []
        return cid

    async def get(self, conversation_id: uuid.UUID) -> dict[str, Any] | None:
        return self.convs.get(conversation_id)

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        *,
        role: str,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        disposition: str | None = None,
    ) -> uuid.UUID:
        _FakeRepo._seq += 1
        mid = uuid.UUID(int=1000 + _FakeRepo._seq)
        self.msgs[conversation_id].append(
            {
                "id": mid,
                "role": role,
                "content": content,
                "citations": citations or [],
                "disposition": disposition,
                "created_at": f"2026-06-14T00:00:0{_FakeRepo._seq}Z",
            }
        )
        return mid

    async def recent_messages(
        self, conversation_id: uuid.UUID, *, limit: int
    ) -> list[dict[str, Any]]:
        return self.msgs.get(conversation_id, [])[-limit:] if limit > 0 else []

    async def delete(self, conversation_id: uuid.UUID) -> None:
        self.convs.pop(conversation_id, None)
        self.msgs.pop(conversation_id, None)


@pytest.fixture(autouse=True)
def _patch_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeRepo.reset()
    monkeypatch.setattr("iw_rag.api.conversation.ConversationRepository", _FakeRepo)


def _svc(**kw: Any) -> ConversationService:
    return ConversationService(_FakeDB(), **kw)


# ---------------------------------------------------------------------------
# _coerce_uuid
# ---------------------------------------------------------------------------


def test_coerce_uuid_rejects_garbage() -> None:
    assert _coerce_uuid(None) is None
    assert _coerce_uuid("") is None
    assert _coerce_uuid("not-a-uuid") is None
    good = str(uuid.uuid4())
    assert str(_coerce_uuid(good)) == good


# ---------------------------------------------------------------------------
# resolve — ownership re-validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_creates_when_none() -> None:
    svc = _svc()
    cid = await svc.resolve(_OWNER, None)
    # A real, owned conversation was created — it round-trips to itself.
    assert await svc.resolve(_OWNER, cid) == cid
    assert cid in {str(k) for k in _FakeRepo.convs}


@pytest.mark.asyncio
async def test_resolve_honours_owned_id() -> None:
    svc = _svc()
    cid = await svc.create(_OWNER)
    assert await svc.resolve(_OWNER, cid) == cid


@pytest.mark.asyncio
async def test_resolve_ignores_foreign_id_and_starts_fresh() -> None:
    """A conversation owned by someone else must NOT be reused (no cross-tenant)."""
    svc = _svc()
    cid = await svc.create(_OWNER)
    fresh = await svc.resolve(_OTHER, cid)
    assert fresh != cid
    # The fresh conversation is owned by the caller.
    assert (await svc.resolve(_OTHER, fresh)) == fresh


@pytest.mark.asyncio
async def test_resolve_garbage_id_starts_fresh() -> None:
    svc = _svc()
    cid = await svc.resolve(_OWNER, "garbage")
    # A malformed id is ignored and a real owned conversation is created instead.
    assert cid != "garbage"
    assert await svc.resolve(_OWNER, cid) == cid


# ---------------------------------------------------------------------------
# history_chunks — conversation-as-source shape + filtering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_history_chunks_builds_citable_user_evidence() -> None:
    svc = _svc()
    cid = await svc.create(_OWNER)
    await svc.record_turn(
        cid, user_text="My name is Sergio", answer_text="Noted.", disposition="answered"
    )

    chunks = await svc.history_chunks(_OWNER, cid)
    user_chunks = [c for c in chunks if c.metadata.get("role") == "user"]
    assert user_chunks, "the user turn must become evidence"
    uc = user_chunks[0]
    assert "My name is Sergio" in uc.content
    assert uc.metadata["kind"] == CONVERSATION_CHUNK_KIND
    assert uc.acl_applied is True
    assert uc.source_uri == f"conversation://{cid}"
    assert uc.chunk_id.startswith("conv:")


@pytest.mark.asyncio
async def test_history_chunks_skips_refused_assistant_turns() -> None:
    """A prior refusal carries no fact — it must not become evidence."""
    svc = _svc()
    cid = await svc.create(_OWNER)
    await svc.record_turn(
        cid, user_text="hello", answer_text="I can't answer that.", disposition="refused"
    )

    roles = [c.metadata.get("role") for c in await svc.history_chunks(_OWNER, cid)]
    assert "assistant" not in roles
    assert "user" in roles


@pytest.mark.asyncio
async def test_history_chunks_empty_for_foreign_conversation() -> None:
    svc = _svc()
    cid = await svc.create(_OWNER)
    await svc.record_turn(cid, user_text="secret", answer_text="ok", disposition="answered")
    assert await svc.history_chunks(_OTHER, cid) == []


@pytest.mark.asyncio
async def test_history_disabled_when_window_zero() -> None:
    svc = _svc(max_history_messages=0)
    cid = await svc.create(_OWNER)
    await svc.record_turn(cid, user_text="x", answer_text="y", disposition="answered")
    assert await svc.history_chunks(_OWNER, cid) == []


# ---------------------------------------------------------------------------
# get_history + clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_history_returns_turns_for_owner_only() -> None:
    svc = _svc()
    cid = await svc.create(_OWNER)
    await svc.record_turn(
        cid,
        user_text="My name is Sergio",
        answer_text="Hi Sergio [1].",
        citations=[{"marker": 1, "chunk_id": "conv:x", "source_uri": "conversation://c"}],
        disposition="answered",
    )

    owner_view = await svc.get_history(_OWNER, cid)
    assert [m.role for m in owner_view] == ["user", "assistant"]
    assert owner_view[1].citations and owner_view[1].citations[0]["marker"] == 1

    assert await svc.get_history(_OTHER, cid) == []


@pytest.mark.asyncio
async def test_clear_only_for_owner() -> None:
    svc = _svc()
    cid = await svc.create(_OWNER)
    await svc.record_turn(cid, user_text="x", answer_text="y", disposition="answered")

    # A non-owner cannot clear it.
    assert await svc.clear(_OTHER, cid) is False
    assert await svc.history_chunks(_OWNER, cid) != []

    # The owner can; afterward it is gone.
    assert await svc.clear(_OWNER, cid) is True
    assert await svc.history_chunks(_OWNER, cid) == []
    # Idempotent — clearing again is a no-op, not an error.
    assert await svc.clear(_OWNER, cid) is False
