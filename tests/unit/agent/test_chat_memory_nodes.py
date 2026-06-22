"""Unit tests for the chat-memory wiring in the retrieve + sanitize nodes.

Chat memory replays prior conversation turns into the agent loop as citable,
principal-scoped synthetic chunks (``metadata["kind"] == "conversation"``).  Two
node-level invariants make that safe and useful:

- ``make_retrieve_node`` **prepends** ``state["history_chunks"]`` to the spine's
  chunks (so memory is available to grade/generate and survives the grader's
  token-budget head-keep), without disturbing the ACL pass-through.
- ``sanitize_node`` **exempts** conversation chunks from injection-scan /
  PII-redaction / datamarking — otherwise "Sergio" would be scrubbed out of the
  user's own "my name is Sergio" and the memory would be useless.  Corpus chunks
  are still sanitized normally.

All tests use fakes — no DB, no live model, no network.
"""

from __future__ import annotations

import types
from typing import Any
from unittest.mock import patch

import pytest

from iw_rag.agent.nodes import output_guard_node
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import (
    CONVERSATION_CHUNK_KIND,
    Citation,
    RetrievalResult,
    RetrievedChunk,
    SanitizedContext,
)

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-chat-mem",
)

_CORPUS_CHUNK = RetrievedChunk(
    chunk_id="c1",
    document_id="doc-1",
    content="Leave policy: 20 days per year.",
    score=0.9,
    source_uri="corpus://hr/leave.md",
)

_MEMORY_CHUNK = RetrievedChunk(
    chunk_id="conv:abc",
    document_id="",
    content='Earlier in this conversation, the user said: "My name is Sergio"',
    score=1.0,
    source_uri="conversation://conv-1",
    metadata={"kind": CONVERSATION_CHUNK_KIND, "role": "user"},
    retriever=CONVERSATION_CHUNK_KIND,
    acl_applied=True,
)


class _FakeSpine:
    """Spine stub returning a single ACL-applied corpus chunk."""

    async def retrieve(self, query: str, ctx: AccessContext) -> RetrievalResult:
        return RetrievalResult(chunks=[_CORPUS_CHUNK], retriever="spine", acl_applied=True)


def _state(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "question": "what is my name?",
        "original_question": "what is my name?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# retrieve node — history_chunks are prepended
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieve_prepends_history_chunks() -> None:
    """``history_chunks`` lead the merged ``chunks`` list, before corpus chunks."""
    from iw_rag.agent.tools import make_retrieve_node

    node = make_retrieve_node(_FakeSpine())
    out = await node(_state(history_chunks=[_MEMORY_CHUNK]))

    chunks = out["chunks"]
    assert [c.chunk_id for c in chunks] == ["conv:abc", "c1"]
    # The retrieval verdict records the history count for audit.
    assert any("history=1" in v.reason for v in out["verdicts"])


@pytest.mark.asyncio
async def test_retrieve_without_history_is_unchanged() -> None:
    """No ``history_chunks`` → only the spine's chunks (back-compat)."""
    from iw_rag.agent.tools import make_retrieve_node

    node = make_retrieve_node(_FakeSpine())
    out = await node(_state())

    assert [c.chunk_id for c in out["chunks"]] == ["c1"]


# ---------------------------------------------------------------------------
# sanitize node — conversation chunks are exempt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sanitize_exempts_conversation_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Conversation chunks bypass ``sanitize_context``; corpus chunks do not.

    The fake sanitizer mutates every chunk it receives (simulating
    datamark/redaction).  The conversation chunk must come through verbatim —
    proving the user's own words are never scrubbed — while the corpus chunk is
    the mutated one returned by the sanitizer.
    """
    seen: list[str] = []

    async def _fake_sanitize_context(
        chunks: list[RetrievedChunk], ctx: Any, *, on_poison: str
    ) -> SanitizedContext:
        seen.extend(c.chunk_id for c in chunks)
        mutated = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                content="[REDACTED]",
                score=c.score,
                source_uri=c.source_uri,
            )
            for c in chunks
        ]
        return SanitizedContext(chunks=mutated, verdicts=[], dropped_chunk_ids=[])

    monkeypatch.setattr("iw_rag.guardrails.sanitize.sanitize_context", _fake_sanitize_context)

    from iw_rag.agent.nodes import sanitize_node

    out = await sanitize_node(_state(chunks=[_MEMORY_CHUNK, _CORPUS_CHUNK]))

    # The sanitizer only ever saw the corpus chunk.
    assert seen == ["c1"]

    by_id = {c.chunk_id: c for c in out["chunks"]}
    # Conversation chunk is verbatim (name preserved); corpus chunk is mutated.
    assert by_id["conv:abc"].content.endswith('"My name is Sergio"')
    assert by_id["c1"].content == "[REDACTED]"
    # Memory leads the merged order.
    assert [c.chunk_id for c in out["chunks"]] == ["conv:abc", "c1"]


# ---------------------------------------------------------------------------
# output_guard node — provenance-aware PII carve-out for the final answer
# ---------------------------------------------------------------------------

_GUARDS_ON = types.SimpleNamespace(guardrails=types.SimpleNamespace(enabled=True))


def _guard_state(graded: list[RetrievedChunk]) -> dict[str, Any]:
    return {
        "question": "what is my name?",
        "original_question": "what is my name?",
        "access_context": _CTX,
        "retries": 0,
        "draft": "Your name is Sergio [1].",
        "citations": [
            Citation(marker=1, chunk_id=graded[0].chunk_id, document_id="", source_uri="x")
        ],
        "graded": graded,
        "gen_verdict": "grounded",
        "verdicts": [],
    }


@pytest.mark.asyncio
@patch("iw_rag.agent.nodes.get_settings", lambda: _GUARDS_ON)
async def test_output_guard_threads_conversation_memory_as_pii_allowlist() -> None:
    """The recalled conversation text is passed to ``_presidio_redact`` as ``allow_text``.

    This is what lets the user's own recalled name survive redaction — proving
    the provenance carve-out is wired from graded conversation chunks into the
    redactor (the real Presidio call is mocked).
    """
    captured: dict[str, Any] = {}

    async def _fake_redact(text: str, _ctx: Any, *, allow_text: str | None = None) -> str:
        captured["allow_text"] = allow_text
        return text

    with patch("iw_rag.agent.nodes._presidio_redact", _fake_redact):
        out = await output_guard_node(_guard_state([_MEMORY_CHUNK]))

    assert out["output_guard_decision"] == "allow"
    # The carve-out received the user's own recalled words.
    assert captured["allow_text"] is not None
    assert "My name is Sergio" in captured["allow_text"]


@pytest.mark.asyncio
@patch("iw_rag.agent.nodes.get_settings", lambda: _GUARDS_ON)
async def test_output_guard_no_allowlist_without_conversation_memory() -> None:
    """No conversation chunks → redactor is called the legacy way (no ``allow_text``).

    Guarantees the strict (corpus-only) path is byte-for-byte unchanged, so every
    existing 2-arg ``_presidio_redact`` mock keeps working.
    """
    captured: dict[str, Any] = {"called": False}

    async def _fake_redact(text: str, _ctx: Any) -> str:  # legacy 2-arg signature
        captured["called"] = True
        return text

    with patch("iw_rag.agent.nodes._presidio_redact", _fake_redact):
        out = await output_guard_node(_guard_state([_CORPUS_CHUNK]))

    assert out["output_guard_decision"] == "allow"
    assert captured["called"] is True


def test_drop_allowlisted_keeps_user_supplied_pii() -> None:
    """The pure carve-out helper drops only spans present in the allow text."""
    from iw_rag.guardrails.sanitize import _drop_allowlisted

    text = "Your name is Sergio and your card is 4111 1111 1111 1111."
    sergio = types.SimpleNamespace(start=text.index("Sergio"), end=text.index("Sergio") + 6)
    card = types.SimpleNamespace(start=text.index("4111"), end=len(text) - 1)

    allow = 'the user said: "My name is Sergio"'
    kept = _drop_allowlisted(text, [sergio, card], allow)
    # The recalled name is exempt; the never-seen card number is still redacted.
    assert sergio not in kept
    assert card in kept
    # No allow text → nothing is exempt (legacy behaviour).
    assert _drop_allowlisted(text, [sergio, card], None) == [sergio, card]
