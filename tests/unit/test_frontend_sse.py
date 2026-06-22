"""Unit tests for the frontend SSE client — parse_sse_event and stream_query.

Tests the SSE line parser that consumes /query/stream events into typed dicts,
without any real network or Chainlit import.

TDD RED: parse_sse_event is expected in frontend/client.py but doesn't exist yet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Add the frontend directory to sys.path so we can import client.py directly
# (it's not an installed package, but a standalone Chainlit app module).
_FRONTEND = Path(__file__).resolve().parent.parent.parent / "frontend"
if str(_FRONTEND) not in sys.path:
    sys.path.insert(0, str(_FRONTEND))


# ---------------------------------------------------------------------------
# Helper: parse SSE lines into event dicts (pure function we are adding to client.py)
# ---------------------------------------------------------------------------


def _parse_sse_lines(lines: list[str]) -> list[dict[str, Any]]:
    """Parse SSE text lines into a list of event dicts.

    This mirrors the parse_sse_event helper that stream_query must use internally.
    """
    # Import the actual implementation once it exists
    from client import parse_sse_lines  # type: ignore[import-not-found]

    return parse_sse_lines(lines)


def test_parse_sse_step_event() -> None:
    """A 'step' SSE event is decoded into a typed dict."""
    data = {"node": "router", "status": "complete", "attempt": 0}
    lines = [
        "event: step",
        f"data: {json.dumps(data)}",
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "step"
    assert events[0]["data"]["node"] == "router"


def test_parse_sse_final_event() -> None:
    """A 'final' SSE event carries an answer and citations."""
    data = {
        "answer": "The policy is 20 days [1].",
        "citations": [{"marker": 1, "chunk_id": "c1", "source_uri": "u", "quote": "q"}],
        "trace_ref": "abc123",
        "acl_applied": True,
        "attempts": 2,
    }
    lines = [
        "event: final",
        f"data: {json.dumps(data)}",
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "final"
    assert events[0]["data"]["answer"] == "The policy is 20 days [1]."


def test_parse_sse_refusal_event() -> None:
    """A 'refusal' SSE event is decoded correctly."""
    data = {
        "reason_code": "no_grounded_answer",
        "message": "insufficient evidence",
        "trace_ref": "xyz",
    }
    lines = [
        "event: refusal",
        f"data: {json.dumps(data)}",
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "refusal"
    assert events[0]["data"]["reason_code"] == "no_grounded_answer"


def test_parse_sse_multiple_events() -> None:
    """Multiple SSE events in a stream are all decoded."""
    lines = [
        "event: step",
        'data: {"node": "router", "status": "complete", "attempt": 0}',
        "",
        "event: step",
        'data: {"node": "retrieve", "status": "complete", "attempt": 0}',
        "",
        "event: final",
        'data: {"answer": "ans", "citations": []}',
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 3
    assert events[0]["event"] == "step"
    assert events[1]["event"] == "step"
    assert events[2]["event"] == "final"


def test_parse_sse_ping_comment_skipped() -> None:
    """SSE comment lines (': ping') are skipped."""
    lines = [
        ": ping",
        "event: step",
        'data: {"node": "router", "status": "complete", "attempt": 0}',
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "step"


def test_parse_sse_verdict_event() -> None:
    """A 'verdict' event is decoded with kind/value/score/reason."""
    data = {"kind": "grade_documents", "value": "correct", "score": 0.9, "reason": "all relevant"}
    lines = [
        "event: verdict",
        f"data: {json.dumps(data)}",
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "verdict"
    assert events[0]["data"]["kind"] == "grade_documents"


def test_parse_sse_source_event() -> None:
    """A 'source' event carries chunk provenance."""
    data = {
        "chunk_id": "c1",
        "document_id": "doc1",
        "source_uri": "corpus://hr/policy.md",
        "score": 0.95,
        "snippet": "leave policy text",
    }
    lines = [
        "event: source",
        f"data: {json.dumps(data)}",
        "",
    ]
    events = _parse_sse_lines(lines)
    assert len(events) == 1
    assert events[0]["event"] == "source"
    assert events[0]["data"]["chunk_id"] == "c1"


def test_stream_query_is_importable() -> None:
    """stream_query async generator is available in client.py."""
    import inspect

    from client import stream_query  # type: ignore[import-not-found]

    assert callable(stream_query), "stream_query must be a callable (async generator function)"
    # Must be an async generator function (decorated with ``async def`` + ``yield``).
    assert inspect.isasyncgenfunction(stream_query), (
        "stream_query must be an async generator function, not a plain coroutine"
    )
