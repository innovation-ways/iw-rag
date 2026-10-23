"""Unit tests for the in-memory VRAM history ring buffer and sampler (F-00098)."""

from __future__ import annotations

import asyncio

import pytest

from iw_rag.serving.vram import VramSnapshot
from iw_rag.serving.vram_history import VramHistory, VramSample, run_sampler


def _s(ts: float, used: int = 100, free: int = 900, total: int = 1000) -> VramSample:
    return VramSample(ts=ts, used_mb=used, free_mb=free, total_mb=total)


def test_empty_history() -> None:
    h = VramHistory(max_points=10)
    assert len(h) == 0
    assert h.latest() is None
    assert h.window(60.0, now=100.0) == []


def test_ring_buffer_evicts_oldest_and_keeps_order() -> None:
    h = VramHistory(max_points=3)
    for t in range(5):
        h.add(_s(float(t)))
    assert len(h) == 3
    assert [x.ts for x in h.window(1000.0, now=10.0)] == [2.0, 3.0, 4.0]
    latest = h.latest()
    assert latest is not None and latest.ts == 4.0


def test_window_slices_by_age() -> None:
    h = VramHistory(max_points=100)
    for t in (0.0, 30.0, 60.0, 90.0):
        h.add(_s(t))
    assert [x.ts for x in h.window(45.0, now=90.0)] == [60.0, 90.0]


def test_window_larger_than_data_returns_everything() -> None:
    h = VramHistory(max_points=100)
    h.add(_s(10.0))
    h.add(_s(20.0))
    assert [x.ts for x in h.window(3600.0, now=30.0)] == [10.0, 20.0]


class _FakeReader:
    def __init__(self, snaps: list[VramSnapshot | Exception]) -> None:
        self._snaps = snaps
        self.calls = 0

    def read(self) -> VramSnapshot:
        snap = self._snaps[min(self.calls, len(self._snaps) - 1)]
        self.calls += 1
        if isinstance(snap, Exception):
            raise snap
        return snap


async def _run_for(coro_factory, seconds: float) -> None:  # type: ignore[no-untyped-def]
    task = asyncio.create_task(coro_factory())
    await asyncio.sleep(seconds)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_sampler_records_available_and_skips_unavailable_and_errors() -> None:
    good = VramSnapshot(available=True, used_mb=10, free_mb=90, total_mb=100)
    reader = _FakeReader(
        [
            VramSnapshot(available=False),
            RuntimeError("nvml hiccup"),
            VramSnapshot(available=True, used_mb=None, free_mb=None, total_mb=None),
            good,
        ]
    )
    h = VramHistory(max_points=10)
    await _run_for(lambda: run_sampler(reader, h, 0.01, clock=lambda: 42.0), 0.2)
    samples = h.window(1000.0, now=42.0)
    assert samples, "the good snapshot must be recorded"
    assert all((x.used_mb, x.free_mb, x.total_mb, x.ts) == (10, 90, 100, 42.0) for x in samples)
    assert reader.calls >= 4


async def test_sampler_cancellation_propagates() -> None:
    reader = _FakeReader([VramSnapshot(available=False)])
    task = asyncio.create_task(run_sampler(reader, VramHistory(max_points=10), 0.01))
    await asyncio.sleep(0.03)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
