"""Unit tests for the per-request step timeline (the timing-breakdown data).

Pins: spans recorded while a timeline is active become ordered, described steps;
uncataloged/finer-grain spans are folded out of the user-facing list; attributes
are filtered to a non-identifying allowlist; and ``record_span`` is a safe no-op
when no timeline is active.  Pure logic — no graph, no network.
"""

from __future__ import annotations

from iw_rag.observability import tracing
from iw_rag.observability.timeline import (
    build_breakdown,
    record_span,
    start_timeline,
)


def test_record_span_without_active_timeline_is_noop() -> None:
    """record_span outside an active timeline is a no-op: does not raise and returns None."""
    # Must not raise when called outside start_timeline (e.g. spans in tests).
    result = record_span("iw_rag.agent.route", 1.0, 2.0, {"node": "router"})
    assert result is None, "record_span must return None (no-op outside a timeline)"


def test_start_timeline_collects_records_then_resets() -> None:
    with start_timeline() as timeline:
        record_span("iw_rag.agent.route", 0.0, 0.010, None)
        record_span("iw_rag.agent.retrieve", 0.010, 0.060, None)
    assert len(timeline.records) == 2
    # After the block the contextvar is reset → further records are dropped.
    record_span("iw_rag.agent.done", 0.0, 0.1, None)
    assert len(timeline.records) == 2


def test_build_breakdown_orders_by_start_and_describes_steps() -> None:
    with start_timeline() as timeline:
        # Record out of order; the breakdown must sort by start time.
        record_span("iw_rag.agent.retrieve", 0.020, 0.080, {"node": "retrieve"})
        record_span("iw_rag.agent.route", 0.000, 0.010, {"node": "router"})
    steps = build_breakdown(timeline)
    assert [s["key"] for s in steps] == ["iw_rag.agent.route", "iw_rag.agent.retrieve"]
    route = steps[0]
    assert route["label"] == "Route"
    assert route["what"] and route["how"]  # plain-language descriptions present
    assert route["duration_ms"] == 10.0
    assert steps[1]["duration_ms"] == 60.0


def test_build_breakdown_drops_uncataloged_and_detail_spans() -> None:
    with start_timeline() as timeline:
        record_span("iw_rag.agent.route", 0.0, 0.01, None)
        record_span("iw_rag.retrieval.dense", 0.01, 0.02, None)  # detail span
        record_span("iw_rag.agent.grade_batch", 0.02, 0.03, None)  # detail span
        record_span("something.uncataloged", 0.03, 0.04, None)
    keys = [s["key"] for s in build_breakdown(timeline)]
    assert keys == ["iw_rag.agent.route"]


def test_build_breakdown_filters_attributes_to_safe_allowlist() -> None:
    with start_timeline() as timeline:
        record_span(
            "iw_rag.agent.grade_documents",
            0.0,
            0.05,
            {"attempt": 1, "chunk_count": 8, "groups_hash": "SECRET", "node": "x"},
        )
    attrs = build_breakdown(timeline)[0]["attributes"]
    assert attrs == {"attempt": 1, "chunk_count": 8}
    assert "groups_hash" not in attrs  # never leak identifying attributes
    assert "node" not in attrs  # internal-only


def test_build_breakdown_none_timeline_is_empty() -> None:
    assert build_breakdown(None) == []


def test_repeated_step_occurrences_each_appear() -> None:
    # A corrective-retry loop runs retrieve/grade twice → two rows each.
    with start_timeline() as timeline:
        record_span("iw_rag.agent.retrieve", 0.00, 0.05, {"attempt": 0})
        record_span("iw_rag.agent.grade_documents", 0.05, 0.07, {"attempt": 0})
        record_span("iw_rag.agent.retrieve", 0.07, 0.12, {"attempt": 1})
        record_span("iw_rag.agent.grade_documents", 0.12, 0.14, {"attempt": 1})
    keys = [s["key"] for s in build_breakdown(timeline)]
    assert keys.count("iw_rag.agent.retrieve") == 2
    assert keys.count("iw_rag.agent.grade_documents") == 2


def test_span_contextmanager_feeds_active_timeline() -> None:
    """``tracing.span`` records into the active timeline even with OTel disabled."""
    tracing.reset_tracing_for_tests()  # ensure export path is off (no-op export)
    with start_timeline() as timeline, tracing.span("iw_rag.agent.route", {"node": "router"}):
        pass
    keys = [r.name for r in timeline.records]
    assert keys == ["iw_rag.agent.route"], (
        f"Exactly one record with name 'iw_rag.agent.route' expected, got {keys!r}"
    )
