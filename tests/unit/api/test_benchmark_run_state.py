"""Unit tests for the Benchmark Studio live-run state holder (F-00096).

:mod:`iw_rag.api.benchmark_run_state` decouples a benchmark run from the
client HTTP connection so the Operator Console's "Live progress" panel
survives a navigate-away-and-back: the run is driven by a background task
and every event is buffered for replay + fanned out to live subscribers.

These tests pin the keystones:

* a late subscriber **replays the full event history** (the panel
  restore on reconnect) and then streams live events to the terminal one;
* concurrent subscribers each receive the complete ordered stream with
  no gap and no duplication;
* :meth:`BenchmarkRunManager.is_active` flips ``False`` only at the
  terminal event, and a finished run lingers as the most-recent;
* the driver lands the run on a terminal event even when the source is
  exhausted without one (defensive — live subscribers never hang).

All deterministic: the event source is a plain async iterator, no GPU /
swap / DB.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from iw_rag.api.benchmark_run_state import BenchmarkRunManager, LiveBenchmarkRun

_HAPPY = [
    {"event": "queued", "data": {"model_config": {"generator": "g"}}},
    {"event": "swapping", "data": {"model_config": {"generator": "g"}}},
    {"event": "ready", "data": {"model_load_ms": 1234.5}},
    {"event": "running", "data": {"i": 0, "total": 2, "question": "q1"}},
    {"event": "running", "data": {"i": 1, "total": 2, "question": "q2"}},
    {"event": "scoring", "data": {"n_queries": 2}},
    {"event": "done", "data": {"report_id": "rep-1"}},
]


def _gated_source(
    events: list[dict[str, Any]], gate: asyncio.Event
) -> AsyncIterator[dict[str, Any]]:
    """An async event source that pauses on ``gate`` before the terminal event.

    Lets a test observe an *in-flight* run (``is_active() is True``) and a
    late subscriber's replay-then-live behaviour deterministically.
    """

    async def _gen() -> AsyncIterator[dict[str, Any]]:
        for ev in events[:-1]:
            yield ev
        await gate.wait()
        yield events[-1]

    return _gen()


async def _drain(run: LiveBenchmarkRun) -> list[dict[str, Any]]:
    return [ev async for ev in run.subscribe()]


# ---------------------------------------------------------------------------
# LiveBenchmarkRun — replay buffer + live fan-out
# ---------------------------------------------------------------------------


async def test_late_subscriber_replays_full_history_after_terminal() -> None:
    """A subscriber attaching AFTER the run finished replays the whole buffer.

    This is the reconnect keystone: returning to the Benchmark view must
    restore the terminal state (``done · report_id=…``), not show
    ``No run yet.``.
    """
    run = LiveBenchmarkRun(run_id="r1", trace_id="t1")
    for ev in _HAPPY:
        await run.publish(ev)
    assert run.terminal is True
    replayed = await _drain(run)
    assert [e["event"] for e in replayed] == [e["event"] for e in _HAPPY]
    assert replayed[-1]["data"]["report_id"] == "rep-1"


async def test_subscriber_attached_mid_run_gets_backlog_then_live() -> None:
    """A subscriber attaching mid-run replays the backlog, then streams live."""
    run = LiveBenchmarkRun(run_id="r2", trace_id="t2")
    for ev in _HAPPY[:4]:  # queued..running(0) already buffered
        await run.publish(ev)

    received: list[dict[str, Any]] = []

    async def consume() -> None:
        async for ev in run.subscribe():
            received.append(ev)

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)  # let the subscriber snapshot the backlog
    for ev in _HAPPY[4:]:  # running(1)..done published live
        await run.publish(ev)
    await asyncio.wait_for(task, timeout=2.0)

    # No gap, no duplication: exactly the full ordered sequence.
    assert [e["event"] for e in received] == [e["event"] for e in _HAPPY]


async def test_two_concurrent_subscribers_each_get_full_stream() -> None:
    """Two live subscribers each receive the complete ordered stream."""
    run = LiveBenchmarkRun(run_id="r3", trace_id="t3")
    out_a: list[dict[str, Any]] = []
    out_b: list[dict[str, Any]] = []

    async def consume(sink: list[dict[str, Any]]) -> None:
        async for ev in run.subscribe():
            sink.append(ev)

    ta = asyncio.create_task(consume(out_a))
    tb = asyncio.create_task(consume(out_b))
    await asyncio.sleep(0)
    for ev in _HAPPY:
        await run.publish(ev)
    await asyncio.wait_for(asyncio.gather(ta, tb), timeout=2.0)

    assert [e["event"] for e in out_a] == [e["event"] for e in _HAPPY]
    assert [e["event"] for e in out_b] == [e["event"] for e in _HAPPY]


async def test_status_tracks_latest_event() -> None:
    """``status`` reports the latest stage (``queued`` before any event)."""
    run = LiveBenchmarkRun(run_id="r4", trace_id="t4")
    assert run.status == "queued"
    await run.publish({"event": "swapping", "data": {}})
    assert run.status == "swapping"
    await run.publish({"event": "done", "data": {"report_id": "x"}})
    assert run.status == "done"


# ---------------------------------------------------------------------------
# BenchmarkRunManager — single active run + background driver
# ---------------------------------------------------------------------------


async def test_manager_start_drives_run_to_completion_in_background() -> None:
    """``start`` drives the source to a terminal event without a subscriber.

    The run survives with no client attached (the disconnect-survival
    keystone): we never subscribe, yet it reaches ``done`` and ``current``
    holds the finished run.
    """
    manager = BenchmarkRunManager()
    assert manager.is_active() is False

    async def source() -> AsyncIterator[dict[str, Any]]:
        for ev in _HAPPY:
            yield ev

    run = await manager.start(trace_id="t", event_source=source())
    assert run.task is not None
    await asyncio.wait_for(run.task, timeout=2.0)
    assert manager.is_active() is False  # terminal
    assert manager.current is run
    assert run.status == "done"


async def test_manager_is_active_true_until_terminal() -> None:
    """``is_active`` is ``True`` while a run is in-flight, ``False`` at terminal."""
    manager = BenchmarkRunManager()
    gate = asyncio.Event()
    run = await manager.start(trace_id="t", event_source=_gated_source(_HAPPY, gate))
    # Let the source emit everything up to the gated terminal event.
    await asyncio.sleep(0.05)
    assert manager.is_active() is True
    gate.set()
    assert run.task is not None
    await asyncio.wait_for(run.task, timeout=2.0)
    assert manager.is_active() is False


async def test_driver_publishes_terminal_when_source_exhausts_without_one() -> None:
    """If the source ends with no done/error, the driver publishes a terminal
    error so live subscribers are released rather than hanging forever."""
    manager = BenchmarkRunManager()

    async def truncated() -> AsyncIterator[dict[str, Any]]:
        yield {"event": "queued", "data": {}}
        yield {"event": "swapping", "data": {}}
        # no terminal event

    run = await manager.start(trace_id="t-trunc", event_source=truncated())
    assert run.task is not None
    await asyncio.wait_for(run.task, timeout=2.0)
    assert run.terminal is True
    assert run.events[-1]["event"] == "error"
    assert run.events[-1]["data"]["trace_ref"] == "t-trunc"
