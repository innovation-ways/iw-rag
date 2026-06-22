"""ConversationService — multi-turn chat memory orchestration behind ``/query``.

This is the seam between the HTTP layer and the persisted chat history
(``conversations`` + ``conversation_messages``).  It owns three jobs:

1. **Resolve** the conversation a turn belongs to — re-validating ownership
   server-side so a caller can never read or write a conversation it does not
   own (the ACL keystone: identity comes from the authenticated
   ``AccessContext``, never from request input).
2. **Replay** prior turns into the agent loop as citable, principal-scoped
   synthetic chunks (conversation-as-source).  The retrieve node prepends them;
   the grounding gate then lets a follow-up answer ("what is my name?") be
   answered *and cited* from the originating turn instead of bypassing the gate.
3. **Record** each completed turn (the user question + the assistant answer)
   so the next turn can see it.

The synthetic chunks are tagged ``metadata["kind"] == CONVERSATION_CHUNK_KIND``
so :func:`iw_rag.agent.nodes.sanitize_node` exempts them from PII redaction —
otherwise "Sergio" would be scrubbed out of the user's own "my name is Sergio"
and the memory would be useless.

The service is injectable (like ``DocumentService`` / ``QueryService``) so tests
substitute a fake without live DB wiring.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import CONVERSATION_CHUNK_KIND, RetrievedChunk
from iw_rag.storage.repositories import ConversationRepository

logger = logging.getLogger(__name__)

#: Default cap on how many of the most-recent turns are replayed as evidence.
#: Bounded (not "the whole thread") to keep the agent's context window safe —
#: datamark inflation + whole-doc chunks already pressure the window
#: (I-CTXOVERFLOW), so memory must not be the thing that blows it.
DEFAULT_MAX_HISTORY_MESSAGES = 16


@dataclass(frozen=True, slots=True)
class StoredMessage:
    """One persisted turn, as surfaced to the history endpoint."""

    role: str
    content: str
    citations: list[dict[str, Any]]
    disposition: str | None
    created_at: str


def _coerce_uuid(value: str | None) -> uuid.UUID | None:
    """Parse ``value`` as a UUID, or return ``None`` (foreign/garbage id → ignored)."""
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except (ValueError, AttributeError, TypeError):
        return None


class ConversationService:
    """Persisted chat memory: resolve, replay, and record conversation turns."""

    def __init__(
        self, db: Any, *, max_history_messages: int = DEFAULT_MAX_HISTORY_MESSAGES
    ) -> None:
        """Wire the database handle + the replay window.

        Args:
            db: A :class:`~iw_rag.storage.db.Database` (anything with an async
                ``session()`` context manager yielding an ``AsyncSession``).
            max_history_messages: Cap on the most-recent turns replayed as
                evidence (token-budget guard).
        """
        self._db = db
        self._max_history_messages = max(0, int(max_history_messages))

    async def create(self, ctx: AccessContext) -> str:
        """Create a new conversation owned by the caller; return its id."""
        async with self._db.session() as session:
            repo = ConversationRepository(session)
            conv_id = await repo.create(ctx.principal_id)
            await session.commit()
        return str(conv_id)

    async def resolve(self, ctx: AccessContext, conversation_id: str | None) -> str:
        """Return a conversation id the caller owns, creating one if needed.

        A provided id is honoured ONLY when it exists and is owned by the
        authenticated principal; a missing / foreign / malformed id silently
        yields a fresh conversation (never an error, never cross-tenant access).

        Args:
            ctx: The authenticated access context (sole ownership source).
            conversation_id: The client-supplied id, or ``None``.

        Returns:
            A conversation id (string UUID) owned by ``ctx.principal_id``.
        """
        async with self._db.session() as session:
            repo = ConversationRepository(session)
            parsed = _coerce_uuid(conversation_id)
            if parsed is not None:
                row = await repo.get(parsed)
                if row is not None and row.get("principal_id") == ctx.principal_id:
                    return str(parsed)
                # Foreign / unknown id — do NOT leak existence; start fresh.
                logger.debug("conversation resolve: id not owned/known → creating fresh")
            conv_id = await repo.create(ctx.principal_id)
            await session.commit()
            return str(conv_id)

    async def history_chunks(
        self, ctx: AccessContext, conversation_id: str
    ) -> list[RetrievedChunk]:
        """Build citable synthetic chunks from the recent turns of a conversation.

        Returns ``[]`` when the conversation is empty, not owned by the caller,
        or memory is disabled (``max_history_messages == 0``).  Only user turns
        and *answered* assistant turns become evidence — a prior refusal carries
        no fact worth grounding against.
        """
        parsed = _coerce_uuid(conversation_id)
        if parsed is None or self._max_history_messages == 0:
            return []
        async with self._db.session() as session:
            repo = ConversationRepository(session)
            row = await repo.get(parsed)
            if row is None or row.get("principal_id") != ctx.principal_id:
                return []
            messages = await repo.recent_messages(parsed, limit=self._max_history_messages)

        chunks: list[RetrievedChunk] = []
        for msg in messages:
            role = msg.get("role")
            content = (msg.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant" and msg.get("disposition") != "answered":
                # Skip refusals/rejections — no fact to ground a follow-up on.
                continue
            if role == "user":
                evidence = f'Earlier in this conversation, the user said: "{content}"'
            else:
                evidence = f'Earlier in this conversation, the assistant replied: "{content}"'
            chunks.append(
                RetrievedChunk(
                    chunk_id=f"conv:{msg.get('id')}",
                    document_id="",
                    content=evidence,
                    # High score so the grader's token-budget fit keeps memory
                    # ahead of the corpus tail (it is prepended anyway).
                    score=1.0,
                    source_uri=f"conversation://{conversation_id}",
                    metadata={
                        "kind": CONVERSATION_CHUNK_KIND,
                        "role": role,
                    },
                    retriever=CONVERSATION_CHUNK_KIND,
                    # Principal-scoped by construction (loaded only for the owner)
                    # — never flowed through the spine's ACL pre-filter.
                    acl_applied=True,
                )
            )
        return chunks

    async def record_turn(
        self,
        conversation_id: str,
        *,
        user_text: str,
        answer_text: str,
        citations: list[dict[str, Any]] | None = None,
        disposition: str | None = None,
    ) -> None:
        """Persist a completed turn (user question + assistant answer).

        Best-effort: a persistence failure is logged and swallowed — memory is
        an enhancement, never a reason to fail an already-answered query.
        """
        parsed = _coerce_uuid(conversation_id)
        if parsed is None:
            return
        try:
            async with self._db.session() as session:
                repo = ConversationRepository(session)
                await repo.add_message(parsed, role="user", content=user_text)
                await repo.add_message(
                    parsed,
                    role="assistant",
                    content=answer_text,
                    citations=citations or [],
                    disposition=disposition,
                )
                await session.commit()
        except Exception:
            logger.warning("conversation record_turn failed (memory not persisted)", exc_info=True)

    async def get_history(self, ctx: AccessContext, conversation_id: str) -> list[StoredMessage]:
        """Return a conversation's turns oldest-first, or ``[]`` if not owned."""
        parsed = _coerce_uuid(conversation_id)
        if parsed is None:
            return []
        async with self._db.session() as session:
            repo = ConversationRepository(session)
            row = await repo.get(parsed)
            if row is None or row.get("principal_id") != ctx.principal_id:
                return []
            messages = await repo.recent_messages(parsed, limit=self._max_history_messages)
        return [
            StoredMessage(
                role=m.get("role", ""),
                content=m.get("content", ""),
                citations=list(m.get("citations") or []),
                disposition=m.get("disposition"),
                created_at=str(m.get("created_at") or ""),
            )
            for m in messages
        ]

    async def clear(self, ctx: AccessContext, conversation_id: str) -> bool:
        """Delete a conversation the caller owns (cascades its turns).

        Returns ``True`` when a row owned by the caller was deleted, ``False``
        when the id is unknown / not owned (idempotent; never an error).
        """
        parsed = _coerce_uuid(conversation_id)
        if parsed is None:
            return False
        async with self._db.session() as session:
            repo = ConversationRepository(session)
            row = await repo.get(parsed)
            if row is None or row.get("principal_id") != ctx.principal_id:
                return False
            await repo.delete(parsed)
            await session.commit()
        return True


__all__ = ["ConversationService", "StoredMessage", "DEFAULT_MAX_HISTORY_MESSAGES"]
