"""Unit tests for the F-00095 / S02 SSE event-shaping layer.

The :mod:`iw_rag.api.benchmark_routes` module wraps the S01
:class:`~iw_rag.eval.benchmark.runner.BenchmarkRunner`'s progress events
(``{"event": str, "data": dict}`` dicts) into the canonical
``event: <type>\\ndata: <json>\\n\\n`` SSE shape that
:func:`iw_rag.api.routes._sse_generator` already uses for
``/query/stream``.  These tests pin that contract verbatim — the S05
prompt's required coverage is:

* the runner's event dicts serialise to the canonical
  ``event: <type>\\ndata: <json>\\n\\n`` shape **in order**;
* the terminal ``done`` event carries a ``report_id``;
* the terminal ``error`` event carries ONLY ``{message, trace_ref}``
  (no exception text, no retrieved content, no PII);
* a defensive ``try/except`` around the runner's iterator yields a
  canonical ``error`` event on any escape, with no exception
  propagating to the client.

The runner is a duck-typed fake (no GPU, no live swap, no live DB) so
the tests are deterministic, fast, and self-contained.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from iw_rag.api.benchmark_routes import _run_sse_generator
from iw_rag.domain.acl import AccessContext

# ---------------------------------------------------------------------------
# Helpers — fakes + SSE parser
# ---------------------------------------------------------------------------


class FakeRunner:
    """In-memory fake of the S01 :class:`BenchmarkRunner`.

    Replays a canned event sequence from :attr:`scripted_events` so the
    S02 SSE wrapping can be exercised end-to-end without a GPU, a live
    swap-controller, or a live retrieval+generation pipeline.  When
    :attr:`raise_runtime_error` is set, the iterator raises a
    :class:`RuntimeError` on the first event — the defensive
    ``try/except`` in :func:`_run_sse_generator` must catch it.
    """

    def __init__(
        self,
        *,
        scripted_events: list[dict[str, Any]] | None = None,
        raise_runtime_error: bool = False,
    ) -> None:
        """Configure the fake.

        Args:
            scripted_events: The event dicts to replay (one per
                :func:`_run_sse_generator` iteration).  Defaults to the
                canonical AC1 happy-path sequence.
            raise_runtime_error: When ``True``, the runner raises a
                :class:`RuntimeError` from its iterator (defensive
                ``try/except`` exercise).
        """
        self.scripted_events = (
            list(scripted_events)
            if scripted_events is not None
            else [
                {"event": "queued", "data": {"model_config": {"generator": "qwen-7b"}}},
                {"event": "swapping", "data": {"model_config": {"generator": "qwen-7b"}}},
                {"event": "ready", "data": {"model_load_ms": 1234.5}},
                {"event": "running", "data": {"i": 0, "total": 2, "question": "q1"}},
                {"event": "running", "data": {"i": 1, "total": 2, "question": "q2"}},
                {"event": "scoring", "data": {"n_queries": 2}},
                {"event": "done", "data": {"report_id": "rep-fixture-1"}},
            ]
        )
        self._raise_runtime_error = raise_runtime_error

    async def run(
        self,
        model_config: dict[str, Any],
        *,
        trace_id: str = "",
        access_context: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Replay the canned events (or raise on first iteration).

        Args:
            model_config: The submitted per-step model config (recorded
                for assertion, not otherwise used).
            trace_id: The request trace id (recorded).
            access_context: The authenticated :class:`AccessContext`
                (recorded).
        """
        del model_config, trace_id, access_context  # recorded by caller
        if self._raise_runtime_error:
            raise RuntimeError("boom — fake escape from the runner")
        for ev in self.scripted_events:
            yield ev


def _parse_sse_events(body: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse a canonical ``event: <t>\\ndata: <json>\\n\\n`` SSE stream.

    Skips the leading ``: ping`` heartbeat and any other comment lines.

    Args:
        body: The raw SSE body string.

    Returns:
        A list of ``(event_type, data_dict)`` pairs in the order they
        appear in the body.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    for chunk in body.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.startswith(":"):
            # Heartbeat / comment line (e.g. ``: ping``).
            continue
        event_type = ""
        data_str = ""
        for line in chunk.splitlines():
            if line.startswith("event: "):
                event_type = line[len("event: ") :]
            elif line.startswith("data: "):
                data_str = line[len("data: ") :]
        if event_type and data_str:
            out.append((event_type, json.loads(data_str)))
    return out


def _consume(gen: AsyncIterator[str]) -> str:
    """Drain an async generator to a single string body."""
    import asyncio

    async def _drain() -> str:
        return "".join([chunk async for chunk in gen])

    return asyncio.run(_drain())


def _ctx() -> AccessContext:
    """A minimal authenticated :class:`AccessContext` for the tests."""
    return AccessContext(
        principal_id="user:operator-1",
        principal_groups=("eval-operator",),
        is_operator=True,
    )


# ---------------------------------------------------------------------------
# Canonical SSE shape — ``event: <type>\ndata: <json>\n\n`` in order
# ---------------------------------------------------------------------------


def test_sse_emits_canonical_event_data_double_newline_shape() -> None:
    """Every event in the body is exactly ``event: <type>\\ndata: <json>\\n\\n``."""
    runner = FakeRunner()
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-1"))
    # Skip the leading heartbeat; the rest must be canonical.
    chunks = [c for c in body.split("\n\n") if c.strip() and not c.strip().startswith(":")]
    assert chunks, "expected at least one SSE event chunk after the heartbeat"
    for chunk in chunks:
        lines = chunk.splitlines()
        # Exactly two logical lines: ``event: ...`` then ``data: ...``.
        assert len(lines) == 2, f"chunk not in canonical 2-line shape: {chunk!r}"
        assert lines[0].startswith("event: "), f"missing event: line in chunk {chunk!r}"
        assert lines[1].startswith("data: "), f"missing data: line in chunk {chunk!r}"
        # The data line must be valid JSON.
        json.loads(lines[1][len("data: ") :])


def test_sse_emits_canonical_event_order_queued_to_done() -> None:
    """AC1 — the runner's events are emitted in the canonical order:
    ``queued → swapping → ready → running×N → scoring → done``."""
    runner = FakeRunner()
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-2"))
    events = _parse_sse_events(body)
    types = [t for t, _ in events]
    # The exact sequence the S01 runner emits; the S02 wrapper must
    # preserve it byte-for-byte (the page's stage chips are gated on
    # this order).
    assert types == [
        "queued",
        "swapping",
        "ready",
        "running",
        "running",
        "scoring",
        "done",
    ]


def test_sse_emits_ping_heartbeat_first() -> None:
    """A ``: ping`` heartbeat precedes the first runner event (mirror of
    :func:`iw_rag.api.routes._sse_generator`)."""
    runner = FakeRunner()
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-3"))
    # The first non-empty chunk in the body must be a comment.
    first_chunk = next(c for c in body.split("\n\n") if c.strip())
    assert first_chunk.startswith(":"), (
        f"expected a ': ping' heartbeat as the first chunk; got {first_chunk!r}"
    )


def test_sse_done_event_carries_report_id() -> None:
    """AC1 — the terminal ``done`` event's data MUST carry ``report_id``."""
    runner = FakeRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {"event": "done", "data": {"report_id": "rep-fixture-xyz"}},
        ]
    )
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-4"))
    events = _parse_sse_events(body)
    done = next(d for t, d in events if t == "done")
    assert done["report_id"] == "rep-fixture-xyz"


# ---------------------------------------------------------------------------
# Terminal ``error`` event — only ``{message, trace_ref}`` (+ structured fields)
# ---------------------------------------------------------------------------


def test_sse_error_event_carries_only_message_and_trace_ref() -> None:
    """Terminal ``error`` event carries ONLY ``{message, trace_ref}`` (and the
    documented structured keys the runner surfaces, like ``reason_code`` for
    ``swap_timeout``).  Never the exception text, retrieved content, or PII.

    This is the F-00095 S05 keystone: the runner's structured error
    survives, but no exception text leaks.
    """
    runner = FakeRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {
                "event": "error",
                "data": {
                    "reason_code": "swap_timeout",
                    "message": "swap-controller never reported ready",
                    "trace_ref": "trace-abc-123",
                },
            },
        ]
    )
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-abc-123"))
    events = _parse_sse_events(body)
    err = next(d for t, d in events if t == "error")
    # The keystone: no exception text, no PII, no retrieved content.
    # The structured fields the runner surfaces (reason_code /
    # message / trace_ref) ARE allowed — they are the documented
    # contract; only the forbidden PII / exception fields are
    # excluded.
    forbidden_payload = {
        "exception",
        "exc",
        "traceback",
        "stack",
        "stacktrace",
        "exception_text",
        "retrieved",
        "content",
        "context",
        "query",
        "question",
        "user",
        "email",
        "password",
        "token",
    }
    payload_keys = {k.lower() for k in err}
    leaked = payload_keys & forbidden_payload
    assert not leaked, (
        f"error event leaks forbidden keys {leaked} (no exception text / "
        f"retrieved content / PII — F-00095 S05 keystone)"
    )
    # The documented contract is intact: reason_code, message, trace_ref.
    assert "message" in err
    assert "trace_ref" in err


def test_sse_error_event_does_not_leak_exception_text() -> None:
    """The defensive ``try/except`` wrapper catches runner escapes and
    yields a generic ``{message, trace_ref}`` event — never the exception
    text, the type name, or any part of the original error payload."""
    runner = FakeRunner(raise_runtime_error=True)
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-leak-1"))
    events = _parse_sse_events(body)
    err = next(d for t, d in events if t == "error")
    # The generic message must be the F-00095 S02 boilerplate, not the
    # runner's RuntimeError text.  We assert the LEAK is absent
    # ("boom" was the fake's exception text) and the documented
    # contract is present.
    dumped = json.dumps(err)
    assert "boom" not in dumped, (
        f"error event leaked the exception text 'boom' in payload: {dumped}"
    )
    assert "RuntimeError" not in dumped, (
        f"error event leaked the exception type 'RuntimeError' in payload: {dumped}"
    )
    # The keystone keys are present.
    assert err["trace_ref"] == "trace-leak-1"
    assert "message" in err


# ---------------------------------------------------------------------------
# Defensive ``try/except`` — no exception escapes to the client
# ---------------------------------------------------------------------------


def test_sse_does_not_propagate_runner_exceptions() -> None:
    """A mid-stream runner exception does NOT escape :func:`_run_sse_generator` —
    the defensive wrapper translates it to a canonical ``error`` event and
    the stream terminates cleanly."""
    runner = FakeRunner(raise_runtime_error=True)
    # The generator MUST yield at least one event (the heartbeat + the
    # error event) and MUST NOT raise.
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-5"))
    events = _parse_sse_events(body)
    err = next(d for t, d in events if t == "error")
    # The trace_ref is the caller's trace id, not the runner's
    # exception.  This pins that the wrapper uses the request trace
    # id, not anything the runner might synthesise.
    assert err["trace_ref"] == "trace-5"


def test_sse_logs_runner_failure_at_error_level(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A mid-stream runner exception is logged at ``ERROR`` / ``EXCEPTION``
    level with the trace_id (lld-api §error mapping — the trace_id is the
    only forensic handle the operator has)."""
    runner = FakeRunner(raise_runtime_error=True)
    with caplog.at_level(logging.ERROR, logger="iw_rag.api.benchmark_routes"):
        body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-log-1"))
    # The exception text must NOT appear in the SSE body, even if the
    # log captured it.
    assert "boom" not in body
    # The log captured the exception with the trace id.
    assert any("trace-log-1" in rec.message for rec in caplog.records), (
        "expected log record to include the trace id 'trace-log-1'"
    )


# ---------------------------------------------------------------------------
# Intermediate events — payload keys are preserved verbatim
# ---------------------------------------------------------------------------


def test_sse_preserves_ready_event_model_load_ms_field() -> None:
    """Invariant 4 — the ``ready`` event's data carries ``model_load_ms`` as a
    distinct top-level key (never folded into per-query latency)."""
    runner = FakeRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {"event": "ready", "data": {"model_load_ms": 9876.5}},
            {"event": "done", "data": {"report_id": "rep-x"}},
        ]
    )
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-6"))
    events = _parse_sse_events(body)
    ready = next(d for t, d in events if t == "ready")
    assert ready["model_load_ms"] == 9876.5


def test_sse_running_event_carries_i_total_and_question() -> None:
    """The ``running`` event payload must carry ``i``, ``total``, and
    ``question`` (the S03 page renders ``running query i / N — question``)."""
    runner = FakeRunner(
        scripted_events=[
            {"event": "queued", "data": {"model_config": {}}},
            {
                "event": "running",
                "data": {"i": 4, "total": 9, "question": "What is the refund policy?"},
            },
            {"event": "done", "data": {"report_id": "rep-x"}},
        ]
    )
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-7"))
    events = _parse_sse_events(body)
    running = next(d for t, d in events if t == "running")
    assert running["i"] == 4
    assert running["total"] == 9
    assert running["question"] == "What is the refund policy?"


# ---------------------------------------------------------------------------
# Per-chunk independence — events are emitted in the order the runner yields
# ---------------------------------------------------------------------------


def test_sse_each_event_chunk_is_self_delimited() -> None:
    """Every event chunk ends with a blank-line delimiter (``\\n\\n``) so the
    browser's ``EventSource`` parser can split them deterministically."""
    runner = FakeRunner()
    body = _consume(_run_sse_generator(runner, {"generator": "qwen-7b"}, _ctx(), "trace-8"))
    # Drop the leading heartbeat; every subsequent event chunk must end
    # with the ``\\n\\n`` delimiter.
    chunks = body.split("\n\n")
    # The trailing empty entry (Python's split) is fine; what matters
    # is that EVERY non-empty, non-heartbeat chunk has the canonical
    # ``event:`` / ``data:`` lines.
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk.startswith(":"):
            continue
        lines = chunk.splitlines()
        assert lines[0].startswith("event: ")
        assert lines[1].startswith("data: ")
        # The data line's JSON is well-formed.
        json.loads(lines[1][len("data: ") :])
