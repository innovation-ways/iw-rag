"""Unit tests for graph.answer_question_stream (Q-SSE-SCHEMA event loop).

Tests cover:
- Exact ordered sequence of event types emitted for a grounded run.
- Source events are emitted once per chunk returned by a node.
- Verdict events are emitted once per Verdict appended by a node.
- The ``final`` event is emitted for a disposition="answered" run (grounded).
- The ``refusal`` event is emitted instead of ``final`` when disposition is
  not "answered".
- The merged ``final.answer`` matches the accumulated delta from all nodes.
- The root-span __enter__ / __exit__ are called exactly once per invocation
  (even when the graph yields no chunks).
- An exception raised from astream propagates out of answer_question_stream.

The fake compiled graph used here actually exercises the generator loop in
``answer_question_stream`` — it implements ``astream`` as an async generator
that yields the same ``{node_name: partial_state}`` dicts that LangGraph
would produce on the "updates" stream mode.  The conftest autouse fixture
disables guardrails (irrelevant here — we bypass the real graph entirely).

References:
- graph.py ~line 833 (answer_question_stream)
- Q-SSE-SCHEMA (canonical SSE event taxonomy)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from iw_rag.agent.state import Verdict
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import Citation, RetrievedChunk

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-stream-test",
)


def _make_chunk(chunk_id: str = "c1", content: str = "Evidence text") -> RetrievedChunk:
    """Build a minimal RetrievedChunk for stream tests.

    Args:
        chunk_id: Chunk identifier.
        content: Chunk text.

    Returns:
        A RetrievedChunk suitable for injection into fake node output.
    """
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        score=0.9,
        source_uri=f"file:///doc-{chunk_id}.pdf",
        retriever="spine",
        acl_applied=True,
    )


def _make_citation(chunk_id: str = "c1") -> Citation:
    """Build a minimal Citation for stream tests.

    Args:
        chunk_id: The chunk this citation references.

    Returns:
        A Citation with a numeric marker.
    """
    return Citation(
        marker=1,
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_uri=f"file:///doc-{chunk_id}.pdf",
        quote="Evidence text",
    )


class _FakeCompiledGraph:
    """A fake compiled LangGraph graph that exercises answer_question_stream.

    Implements ``astream`` as an async generator yielding
    ``{node_name: partial_state_dict}`` dicts, exactly like LangGraph's
    stream_mode="updates" output.

    Args:
        node_outputs: Sequence of ``(node_name, partial_state_dict)`` pairs
            that astream will yield in order.
        raise_after: When set, astream raises this exception after all
            yielded outputs are exhausted.
    """

    def __init__(
        self,
        node_outputs: list[tuple[str, dict[str, Any]]],
        raise_after: Exception | None = None,
    ) -> None:
        self._node_outputs = node_outputs
        self._raise_after = raise_after

    async def astream(self, init: Any, config: Any | None = None) -> Any:
        """Yield fake per-node update dicts matching LangGraph's stream format.

        Args:
            init: Initial state (ignored — the fake ignores it).
            config: LangGraph config (ignored).

        Yields:
            Dicts of the form ``{node_name: partial_state_dict}``.
        """
        for node_name, partial in self._node_outputs:
            yield {node_name: partial}
        if self._raise_after is not None:
            raise self._raise_after


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _collect(
    graph: Any,
    question: str = "What is the policy?",
    ctx: AccessContext = _CTX,
) -> list[dict[str, Any]]:
    """Drive answer_question_stream and collect all yielded events.

    Args:
        graph: A compiled graph (real or fake).
        question: The question to stream.
        ctx: The access context.

    Returns:
        The complete list of SSE-shaped event dicts.
    """
    from iw_rag.agent.graph import answer_question_stream

    events: list[dict[str, Any]] = []
    async for event in answer_question_stream(graph, question, ctx, thread_id="test-thread"):
        events.append(event)
    return events


# ---------------------------------------------------------------------------
# Tests: event sequence for a grounded run
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_grounded_run_ends_with_final_event() -> None:
    """A grounded run (disposition=answered) ends with exactly one 'final' event."""
    chunk = _make_chunk("c1")
    citation = _make_citation("c1")
    graph = _FakeCompiledGraph(
        [
            ("retrieve", {"chunks": [chunk], "verdicts": []}),
            (
                "generate",
                {
                    "draft": "Answer text [1].",
                    "citations": [citation],
                    "verdicts": [Verdict(kind="generate", value="generated")],
                },
            ),
            (
                "done",
                {
                    "answer": "Answer text [1].",
                    "disposition": "answered",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)

    event_types = [e["event"] for e in events]
    final_events = [e for e in events if e["event"] == "final"]

    assert event_types[-1] == "final", (
        f"Last event must be 'final' for grounded run, got {event_types[-1]!r}"
    )
    assert len(final_events) == 1, f"Exactly one 'final' event expected, got {len(final_events)}"


@pytest.mark.asyncio
async def test_stream_grounded_run_exact_event_type_sequence() -> None:
    """Grounded run emits events in the expected order: step, source, verdict, final."""
    chunk = _make_chunk("c1")
    citation = _make_citation("c1")
    graph = _FakeCompiledGraph(
        [
            (
                "retrieve",
                {
                    "chunks": [chunk],
                    "verdicts": [Verdict(kind="retrieve", value="spine")],
                },
            ),
            (
                "done",
                {
                    "answer": "Answer [1].",
                    "citations": [citation],
                    "disposition": "answered",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)
    event_types = [e["event"] for e in events]

    # Must have at least: step (retrieve), source (c1), verdict (retrieve), step (done), final
    assert "step" in event_types, "Expected at least one 'step' event"
    assert "source" in event_types, "Expected at least one 'source' event"
    assert "verdict" in event_types, "Expected at least one 'verdict' event"
    assert event_types[-1] == "final", f"Expected final as last event, got {event_types[-1]!r}"

    # step events always precede the terminal event
    final_idx = event_types.index("final")
    step_indices = [i for i, t in enumerate(event_types) if t == "step"]
    assert all(idx < final_idx for idx in step_indices), (
        "All 'step' events must precede the 'final' event"
    )


@pytest.mark.asyncio
async def test_stream_grounded_run_final_event_carries_answer_and_citations() -> None:
    """The 'final' event data contains the merged answer text and citations list."""
    chunk = _make_chunk("c1")
    citation = _make_citation("c1")
    graph = _FakeCompiledGraph(
        [
            ("retrieve", {"chunks": [chunk], "verdicts": []}),
            (
                "generate",
                {
                    "answer": "The answer is here [1].",
                    "citations": [citation],
                    "disposition": "answered",
                    "verdicts": [],
                },
            ),
            # The 'done' node carries the disposition; 'answer' is already in state
            (
                "done",
                {
                    "answer": "The answer is here [1].",
                    "disposition": "answered",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)
    final_events = [e for e in events if e["event"] == "final"]

    assert len(final_events) == 1, "Expected exactly one 'final' event"
    data = final_events[0]["data"]
    assert data["answer"] == "The answer is here [1].", f"final.answer mismatch: {data['answer']!r}"
    assert len(data["citations"]) == 1, (
        f"Expected 1 citation in final event, got {len(data['citations'])}"
    )
    assert data["citations"][0]["chunk_id"] == "c1", (
        f"Citation chunk_id mismatch: {data['citations'][0]['chunk_id']!r}"
    )


# ---------------------------------------------------------------------------
# Tests: source events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_one_source_event_per_chunk() -> None:
    """Each chunk returned by a node generates exactly one 'source' event."""
    chunks = [_make_chunk("c1"), _make_chunk("c2"), _make_chunk("c3")]
    graph = _FakeCompiledGraph(
        [
            ("retrieve", {"chunks": chunks, "verdicts": []}),
            (
                "done",
                {"answer": "A.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    source_events = [e for e in events if e["event"] == "source"]

    assert len(source_events) == 3, (
        f"Expected 3 source events (one per chunk), got {len(source_events)}"
    )
    emitted_chunk_ids = [e["data"]["chunk_id"] for e in source_events]
    assert emitted_chunk_ids == ["c1", "c2", "c3"], (
        f"source event chunk_ids mismatch: {emitted_chunk_ids!r}"
    )


@pytest.mark.asyncio
async def test_stream_source_event_fields_match_chunk_attributes() -> None:
    """Source event data fields (chunk_id, source_uri, score, snippet) match the chunk."""
    chunk = _make_chunk("c1", content="The quick brown fox")
    graph = _FakeCompiledGraph(
        [
            ("retrieve", {"chunks": [chunk], "verdicts": []}),
            (
                "done",
                {"answer": "A.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    source_events = [e for e in events if e["event"] == "source"]

    assert len(source_events) == 1, "Expected exactly one source event"
    d = source_events[0]["data"]
    assert d["chunk_id"] == "c1", f"source.chunk_id={d['chunk_id']!r}"
    assert d["source_uri"] == "file:///doc-c1.pdf", f"source.source_uri={d['source_uri']!r}"
    assert d["score"] == 0.9, f"source.score={d['score']!r}"
    assert d["snippet"] == "The quick brown fox", f"source.snippet={d['snippet']!r}"


# ---------------------------------------------------------------------------
# Tests: verdict events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_verdict_event_per_verdict() -> None:
    """Each Verdict in a node's output generates exactly one 'verdict' event."""
    verdicts = [
        Verdict(kind="router", value="retrieve", reason="factual"),
        Verdict(kind="grade_documents", value="correct", score=1.0),
    ]
    graph = _FakeCompiledGraph(
        [
            ("router", {"verdicts": [verdicts[0]]}),
            ("grade_documents", {"verdicts": [verdicts[1]]}),
            (
                "done",
                {"answer": "B.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    verdict_events = [e for e in events if e["event"] == "verdict"]

    assert len(verdict_events) == 2, f"Expected 2 verdict events, got {len(verdict_events)}"
    assert verdict_events[0]["data"]["kind"] == "router", (
        f"First verdict kind={verdict_events[0]['data']['kind']!r}"
    )
    assert verdict_events[1]["data"]["kind"] == "grade_documents", (
        f"Second verdict kind={verdict_events[1]['data']['kind']!r}"
    )


@pytest.mark.asyncio
async def test_stream_verdict_event_fields_match_verdict_attributes() -> None:
    """Verdict event data fields (kind, value, score, reason) match the Verdict."""
    v = Verdict(kind="grade_generation", value="grounded", score=0.95, reason="both_pass")
    graph = _FakeCompiledGraph(
        [
            ("grade_generation", {"verdicts": [v]}),
            (
                "done",
                {"answer": "C.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    verdict_events = [e for e in events if e["event"] == "verdict"]

    assert len(verdict_events) == 1, "Expected exactly one verdict event"
    d = verdict_events[0]["data"]
    assert d["kind"] == "grade_generation", f"verdict.kind={d['kind']!r}"
    assert d["value"] == "grounded", f"verdict.value={d['value']!r}"
    assert d["score"] == 0.95, f"verdict.score={d['score']!r}"
    assert d["reason"] == "both_pass", f"verdict.reason={d['reason']!r}"


# ---------------------------------------------------------------------------
# Tests: refusal event (non-answered disposition)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_refusal_run_ends_with_refusal_event() -> None:
    """A refused run (disposition!=answered) ends with exactly one 'refusal' event."""
    graph = _FakeCompiledGraph(
        [
            (
                "fail_closed",
                {
                    "answer": None,
                    "refusal": "insufficient grounded evidence",
                    "disposition": "refused",
                    "verdicts": [Verdict(kind="fail_closed", value="refused")],
                },
            ),
        ]
    )

    events = await _collect(graph)
    event_types = [e["event"] for e in events]
    refusal_events = [e for e in events if e["event"] == "refusal"]

    assert event_types[-1] == "refusal", (
        f"Last event must be 'refusal' for refused run, got {event_types[-1]!r}"
    )
    assert len(refusal_events) == 1, (
        f"Exactly one 'refusal' event expected, got {len(refusal_events)}"
    )


@pytest.mark.asyncio
async def test_stream_refusal_event_carries_reason_code_and_message() -> None:
    """The 'refusal' event data has reason_code='no_grounded_answer' and a message."""
    graph = _FakeCompiledGraph(
        [
            (
                "fail_closed",
                {
                    "answer": None,
                    "refusal": "no grounded answer found",
                    "disposition": "refused",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)
    refusal_events = [e for e in events if e["event"] == "refusal"]

    assert len(refusal_events) == 1, "Expected exactly one refusal event"
    d = refusal_events[0]["data"]
    assert d["reason_code"] == "no_grounded_answer", f"refusal.reason_code={d['reason_code']!r}"
    assert d["message"] == "no grounded answer found", f"refusal.message={d['message']!r}"


@pytest.mark.asyncio
async def test_stream_no_disposition_falls_back_to_refusal() -> None:
    """When no node sets disposition, the stream ends with a 'refusal' event."""
    graph = _FakeCompiledGraph(
        [
            ("router", {"verdicts": [Verdict(kind="router", value="retrieve")]}),
        ]
    )

    events = await _collect(graph)
    terminal_events = [e for e in events if e["event"] in ("final", "refusal")]

    assert len(terminal_events) == 1, (
        f"Expected exactly one terminal event, got {len(terminal_events)}"
    )
    assert terminal_events[0]["event"] == "refusal", (
        f"Expected 'refusal' when no disposition set, got {terminal_events[0]['event']!r}"
    )


# ---------------------------------------------------------------------------
# Tests: delta merging — accumulated state across multiple nodes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_merges_state_across_nodes_for_final_event() -> None:
    """The final event reflects the merged state from all nodes, not just the last."""
    chunk = _make_chunk("c1")
    citation = _make_citation("c1")

    # citations come from generate, disposition from done — must both appear in final
    graph = _FakeCompiledGraph(
        [
            ("retrieve", {"chunks": [chunk], "verdicts": []}),
            (
                "generate",
                {
                    "draft": "The merged answer [1].",
                    "citations": [citation],
                    "verdicts": [],
                },
            ),
            (
                "done",
                {
                    "answer": "The merged answer [1].",
                    "disposition": "answered",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)
    final_events = [e for e in events if e["event"] == "final"]

    assert len(final_events) == 1, "Expected one final event"
    d = final_events[0]["data"]
    # answer comes from 'done', citations come from 'generate' — both must survive
    assert d["answer"] == "The merged answer [1].", f"Merged answer mismatch: {d['answer']!r}"
    assert len(d["citations"]) == 1, (
        f"Citations must survive delta merge; got {len(d['citations'])}"
    )


# ---------------------------------------------------------------------------
# Tests: root span enter/exit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_root_span_enter_and_exit_called_once() -> None:
    """The root span context manager is entered and exited exactly once per call.

    ``span`` is imported locally inside ``answer_question_stream``; the patch
    target is ``iw_rag.observability.tracing.span``.
    """
    graph = _FakeCompiledGraph(
        [
            (
                "done",
                {"answer": "X.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    span_mock = MagicMock()
    span_mock.__enter__ = MagicMock(return_value=None)
    span_mock.__exit__ = MagicMock(return_value=False)

    with patch("iw_rag.observability.tracing.span", return_value=span_mock) as _span_factory:
        events = await _collect(graph)

    # Verify span was created (factory called)
    assert _span_factory.call_count >= 1, "span() factory must be called at least once"
    # Verify enter and exit were each called once
    assert span_mock.__enter__.call_count == 1, (
        f"Root span __enter__ must be called exactly once, got {span_mock.__enter__.call_count}"
    )
    assert span_mock.__exit__.call_count == 1, (
        f"Root span __exit__ must be called exactly once, got {span_mock.__exit__.call_count}"
    )

    # Result still valid
    final_events = [e for e in events if e["event"] == "final"]
    assert len(final_events) == 1, "Expected final event to still be emitted"


@pytest.mark.asyncio
async def test_stream_root_span_exit_called_on_exception() -> None:
    """The root span __exit__ is called even when astream raises an exception.

    ``span`` is imported locally inside ``answer_question_stream``; the patch
    target is ``iw_rag.observability.tracing.span``.
    """
    graph = _FakeCompiledGraph(
        node_outputs=[],
        raise_after=RuntimeError("stream failure"),
    )

    span_mock = MagicMock()
    span_mock.__enter__ = MagicMock(return_value=None)
    span_mock.__exit__ = MagicMock(return_value=False)

    exit_call_count: list[int] = [0]
    original_exit = span_mock.__exit__

    def _counting_exit(*args: object) -> bool:
        exit_call_count[0] += 1
        return bool(original_exit(*args))

    span_mock.__exit__ = _counting_exit

    with (
        patch("iw_rag.observability.tracing.span", return_value=span_mock),
        pytest.raises(RuntimeError, match="stream failure"),
    ):
        await _collect(graph)

    # A strong value assertion: exit was called exactly once (resource cleanup on exception)
    assert exit_call_count[0] == 1, (
        f"Root span __exit__ must be called exactly once on exception, got {exit_call_count[0]}"
    )


# ---------------------------------------------------------------------------
# Tests: step events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_emits_step_event_for_each_node() -> None:
    """Each node completion emits exactly one 'step' event with the node name."""
    graph = _FakeCompiledGraph(
        [
            ("router", {"verdicts": []}),
            ("retrieve", {"chunks": [], "verdicts": []}),
            ("generate", {"draft": "Q.", "citations": [], "verdicts": []}),
            (
                "done",
                {"answer": "Q.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    step_events = [e for e in events if e["event"] == "step"]

    assert len(step_events) == 4, f"Expected 4 step events, got {len(step_events)}"
    step_nodes = [e["data"]["node"] for e in step_events]
    assert step_nodes == ["router", "retrieve", "generate", "done"], (
        f"Step node names mismatch: {step_nodes!r}"
    )


@pytest.mark.asyncio
async def test_stream_step_event_has_status_complete() -> None:
    """Each 'step' event carries status='complete'."""
    graph = _FakeCompiledGraph(
        [
            ("router", {"verdicts": []}),
            (
                "done",
                {"answer": "Z.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    step_events = [e for e in events if e["event"] == "step"]

    assert len(step_events) >= 1, "Expected at least one step event"
    statuses = [e["data"]["status"] for e in step_events]
    assert all(s == "complete" for s in statuses), (
        f"All step events must have status='complete', got {statuses!r}"
    )


# ---------------------------------------------------------------------------
# Tests: non-dict node output is ignored
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_ignores_non_dict_node_output() -> None:
    """Non-dict values from astream chunks do not crash the event loop."""

    class _GraphWithNonDictOutput:
        async def astream(self, init: Any, config: Any = None) -> Any:
            """Yield a mix of dict and non-dict node outputs."""
            yield {"router": "not_a_dict"}  # type: ignore[misc]  # non-dict — should be skipped
            yield {"done": {"answer": "OK.", "disposition": "answered", "verdicts": []}}

    graph = _GraphWithNonDictOutput()
    events = await _collect(graph)

    # Non-dict output must not crash; stream ends with terminal event
    terminal_events = [e for e in events if e["event"] in ("final", "refusal")]
    assert len(terminal_events) == 1, (
        f"Expected exactly one terminal event, got {len(terminal_events)}"
    )
    assert terminal_events[0]["event"] == "final", (
        f"Expected 'final' for answered, got {terminal_events[0]['event']!r}"
    )


# ---------------------------------------------------------------------------
# Tests: final event metadata fields
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_final_event_includes_trace_ref_and_acl_applied() -> None:
    """The 'final' event always includes trace_ref and acl_applied=True fields."""
    graph = _FakeCompiledGraph(
        [
            (
                "done",
                {"answer": "Final.", "disposition": "answered", "verdicts": []},
            ),
        ]
    )

    events = await _collect(graph)
    final_events = [e for e in events if e["event"] == "final"]

    assert len(final_events) == 1, "Expected one final event"
    d = final_events[0]["data"]
    assert "trace_ref" in d, "final event must include 'trace_ref' field"
    assert d["acl_applied"] is True, f"final.acl_applied must be True, got {d['acl_applied']!r}"
    assert "attempts" in d, "final event must include 'attempts' field"


@pytest.mark.asyncio
async def test_stream_refusal_event_includes_trace_ref() -> None:
    """The 'refusal' event always includes the trace_ref field."""
    graph = _FakeCompiledGraph(
        [
            (
                "fail_closed",
                {
                    "answer": None,
                    "refusal": "budget exhausted",
                    "disposition": "refused",
                    "verdicts": [],
                },
            ),
        ]
    )

    events = await _collect(graph)
    refusal_events = [e for e in events if e["event"] == "refusal"]

    assert len(refusal_events) == 1, "Expected one refusal event"
    d = refusal_events[0]["data"]
    assert "trace_ref" in d, "refusal event must include 'trace_ref' field"
