"""Integration tests for ConversationRepository (testcontainers ParadeDB).

Verifies the chat-memory persistence round-trip against a real migrated
Postgres: create a conversation, append turns, read them back oldest-first
(capped + re-sorted), and confirm the ``ON DELETE CASCADE`` from the "Clear"
button purges the turns with the thread.  Exercises migration ``0010`` end to
end (the unit suite covers it with a fake repo; this proves the real DDL).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database
from iw_rag.storage.repositories import ConversationRepository


async def _round_trip(dsn: str) -> dict[str, Any]:
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            repo = ConversationRepository(session)
            conv_id = await repo.create("alice", title="greeting")
            await repo.add_message(conv_id, role="user", content="My name is Sergio")
            await repo.add_message(
                conv_id,
                role="assistant",
                content="Your name is Sergio [1].",
                citations=[{"marker": 1, "chunk_id": "conv:x", "source_uri": "conversation://c"}],
                disposition="answered",
            )
            await session.commit()

        async with db.session() as session:
            repo = ConversationRepository(session)
            row = await repo.get(conv_id)
            msgs = await repo.recent_messages(conv_id, limit=10)
            # limit smaller than the turn count keeps the most-recent, oldest-first.
            capped = await repo.recent_messages(conv_id, limit=1)

        async with db.session() as session:
            repo = ConversationRepository(session)
            await repo.delete(conv_id)
            await session.commit()

        async with db.session() as session:
            repo = ConversationRepository(session)
            after_delete = await repo.recent_messages(conv_id, limit=10)
            gone = await repo.get(conv_id)

        return {
            "owner": row["principal_id"] if row else None,
            "roles": [m["role"] for m in msgs],
            "first_content": msgs[0]["content"],
            "assistant_citations": msgs[1]["citations"],
            "assistant_disposition": msgs[1]["disposition"],
            "capped_roles": [m["role"] for m in capped],
            "after_delete": after_delete,
            "gone": gone,
        }
    finally:
        await db.dispose()


@pytest.mark.integration
def test_conversation_round_trip(clean_pg: str) -> None:
    out = asyncio.run(_round_trip(clean_pg))

    # Ownership + ordered history.
    assert out["owner"] == "alice"
    assert out["roles"] == ["user", "assistant"]
    assert out["first_content"] == "My name is Sergio"
    # JSONB citations + disposition survive the round-trip.
    assert out["assistant_citations"][0]["marker"] == 1
    assert out["assistant_disposition"] == "answered"
    # The cap keeps the most-recent turn (the assistant reply), not the first.
    assert out["capped_roles"] == ["assistant"]
    # The "Clear" delete cascades: turns gone, conversation gone.
    assert out["after_delete"] == []
    assert out["gone"] is None


@pytest.mark.integration
def test_recent_messages_zero_limit(clean_pg: str) -> None:
    async def _run(dsn: str) -> list[dict[str, Any]]:
        db = Database(DatabaseSettings(dsn=dsn))
        try:
            async with db.session() as session:
                repo = ConversationRepository(session)
                cid = await repo.create("bob")
                await repo.add_message(cid, role="user", content="hi")
                await session.commit()
                return await repo.recent_messages(cid, limit=0)
        finally:
            await db.dispose()

    assert asyncio.run(_run(clean_pg)) == []


@pytest.mark.integration
def test_get_unknown_conversation_returns_none(clean_pg: str) -> None:
    async def _run(dsn: str) -> Any:
        db = Database(DatabaseSettings(dsn=dsn))
        try:
            async with db.session() as session:
                return await ConversationRepository(session).get(uuid.uuid4())
        finally:
            await db.dispose()

    assert asyncio.run(_run(clean_pg)) is None
