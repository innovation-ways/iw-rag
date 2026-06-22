# ruff: noqa: E501
"""In-memory live-run state for the Benchmark Studio (single active run).

The Benchmark Studio drives a **single-GPU** box, so at most one
benchmark runs at a time.  This module holds the *current or
most-recent* run server-side, **decoupled from any client HTTP
connection**, so the Operator Console's "Live progress" panel survives a
navigate-away-and-back (the F-00096 console nav is full-page ``<a href>``
links — clicking away tears down the page and the SSE ``fetch``).

Without this holder the run was driven by the ``POST /benchmark/run``
request generator itself: a client disconnect cancelled the
``StreamingResponse`` and Starlette propagated the cancellation into the
runner, **killing the in-flight run** AND losing the progress DOM.  Here
the run is driven by a background :func:`asyncio.create_task` that keeps
going regardless of whether a client is attached, and every progress
event is appended to a **replay buffer** + fanned out to live
subscribers.  A reconnecting client replays the buffer (restoring the
panel) and then streams live to the terminal event.

The module is dependency-free (stdlib only): the *source* of progress
events is injected as an ``AsyncIterator[dict]`` by the route, so this
holder never imports the runner, the SSE formatter, or FastAPI — keeping
the api → (no cycle) layering clean and the unit tests pure.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

logger = logging.getLogger(__name__)

# Terminal event types — the run is finished once one of these is
# published (the buffer + the most-recent snapshot survive for replay).
_TERMINAL_EVENTS = frozenset({"done", "error"})

# Sentinel pushed onto each live subscriber's queue after the terminal
# event so the ``subscribe`` loop ends cleanly (``None`` is unambiguous —
# the runner's events are always dicts).
_SUBSCRIBER_CLOSE = None


class LiveBenchmarkRun:
    """A single in-flight-or-recently-finished benchmark run.

    Holds the ordered **replay buffer** of structured progress events
    (``{"event": str, "data": dict}`` — exactly the S01 runner's shape)
    and the set of live subscriber queues.  ``publish`` appends to the
    buffer and fans out to subscribers under a lock; ``subscribe``
    atomically snapshots the buffer (so a late subscriber replays the
    full history) and then streams subsequently-published events with no
    gap and no duplication.
    """

    def __init__(self, run_id: str, trace_id: str) -> None:
        """Create an empty run.

        Args:
            run_id: The opaque run id (uuid hex) — surfaced so a client
                can tell one run apart from the next.
            trace_id: The originating request's trace id (Phoenix
                correlation; used in the defensive terminal-error event).
        """
        self.run_id = run_id
        self.trace_id = trace_id
        self.events: list[dict[str, Any]] = []
        self.terminal = False
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._lock = asyncio.Lock()
        self.task: asyncio.Task[None] | None = None

    @property
    def status(self) -> str:
        """The latest stage name (the last published event's type, or ``"queued"``)."""
        return self.events[-1]["event"] if self.events else "queued"

    async def publish(self, event: dict[str, Any]) -> None:
        """Append ``event`` to the buffer and fan it out to live subscribers.

        Atomic w.r.t. :meth:`subscribe` (same lock), so a subscriber
        either sees an event in its snapshot OR receives it on its queue
        — never both, never neither.

        Args:
            event: A structured ``{"event": str, "data": dict}`` progress
                event from the runner.
        """
        async with self._lock:
            self.events.append(event)
            is_terminal = str(event.get("event", "")) in _TERMINAL_EVENTS
            if is_terminal:
                self.terminal = True
            for q in self._subscribers:
                q.put_nowait(event)
                if is_terminal:
                    q.put_nowait(_SUBSCRIBER_CLOSE)

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        """Yield the full event history, then stream live events to terminal.

        Yields:
            Structured ``{"event": str, "data": dict}`` events: first the
            buffered backlog (replay — this is what restores the panel on
            a reconnect), then any live events until the terminal one.
            Returns immediately after the backlog when the run already
            finished (the terminal event is in the backlog).
        """
        async with self._lock:
            backlog = list(self.events)
            already_terminal = self.terminal
            queue: asyncio.Queue[Any] = asyncio.Queue()
            if not already_terminal:
                self._subscribers.add(queue)
        for event in backlog:
            yield event
        if already_terminal:
            return
        try:
            while True:
                event = await queue.get()
                if event is _SUBSCRIBER_CLOSE:
                    break
                yield event
        finally:
            async with self._lock:
                self._subscribers.discard(queue)


class BenchmarkRunManager:
    """Singleton holder of the current/most-recent :class:`LiveBenchmarkRun`.

    Bound to ``app.state.benchmark_run_manager`` in
    :func:`iw_rag.api.app.create_app`.  Because the box is single-GPU,
    only one run exists at a time; a finished run lingers as the
    most-recent so a client returning to the Benchmark view still sees
    its terminal state (``done · report_id=…`` or the error), until the
    next run replaces it.
    """

    def __init__(self) -> None:
        self._current: LiveBenchmarkRun | None = None
        self._lock = asyncio.Lock()

    @property
    def current(self) -> LiveBenchmarkRun | None:
        """The current or most-recent run (``None`` before the first run)."""
        return self._current

    def is_active(self) -> bool:
        """``True`` when a run exists and has not yet reached a terminal event."""
        return self._current is not None and not self._current.terminal

    async def start(
        self,
        *,
        trace_id: str,
        event_source: AsyncIterator[dict[str, Any]],
    ) -> LiveBenchmarkRun:
        """Begin a new run, driving ``event_source`` in a background task.

        The background task is **not** tied to any client connection, so
        the run survives a client disconnect / navigation.  The caller is
        responsible for refusing the start when :meth:`is_active` is
        already ``True`` (single-GPU — one run at a time).

        Args:
            trace_id: The originating request's trace id.
            event_source: An async iterator of structured
                ``{"event": str, "data": dict}`` progress events (the
                route adapts the S01 runner into this shape, with the
                defensive try/except already applied).

        Returns:
            The freshly-created :class:`LiveBenchmarkRun` (subscribe to it
            for the SSE stream).
        """
        async with self._lock:
            run = LiveBenchmarkRun(run_id=uuid.uuid4().hex, trace_id=trace_id)
            self._current = run
        run.task = asyncio.create_task(self._drive(run, event_source))
        return run

    async def _drive(
        self, run: LiveBenchmarkRun, event_source: AsyncIterator[dict[str, Any]]
    ) -> None:
        """Consume ``event_source`` and publish each event into ``run``.

        Always lands ``run`` on a terminal event: ``event_source`` is the
        route's defensive adapter (it converts a runner escape into a
        terminal ``error``), but if the source is exhausted without a
        terminal event we publish a defensive one so live subscribers are
        released rather than hanging.
        """
        produced_terminal = False
        try:
            async for event in event_source:
                await run.publish(event)
                if str(event.get("event", "")) in _TERMINAL_EVENTS:
                    produced_terminal = True
        except asyncio.CancelledError:  # pragma: no cover - app shutdown
            raise
        except Exception:  # noqa: BLE001 - defensive; event_source already guards
            logger.exception("benchmark run driver crashed (trace_id=%s)", run.trace_id)
        finally:
            if not produced_terminal and not run.terminal:
                await run.publish(
                    {
                        "event": "error",
                        "data": {
                            "message": "benchmark run ended without a terminal event",
                            "trace_ref": run.trace_id,
                        },
                    }
                )


__all__ = ["BenchmarkRunManager", "LiveBenchmarkRun"]
