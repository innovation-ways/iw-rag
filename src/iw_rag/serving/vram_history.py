"""In-memory VRAM history for the System view chart (F-00098).

A bounded ring buffer of ``(timestamp, used, free, total)`` samples plus the
background sampler that fills it.  Lives in the ``system-control`` sidecar,
which owns NVML; the API reads it over HTTP.  Deliberately not persisted:
history is lost on restart (no database, no migration).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from iw_rag.serving.vram import VramReader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VramSample:
    """One point of the VRAM time series.

    Attributes:
        ts: Wall-clock time of the reading, epoch seconds.
        used_mb: Total VRAM used on the device, MiB.
        free_mb: Total VRAM free on the device, MiB.
        total_mb: Device VRAM capacity, MiB.
    """

    ts: float
    used_mb: int
    free_mb: int
    total_mb: int


class VramHistory:
    """Bounded, ordered ring buffer of :class:`VramSample`.

    Args:
        max_points: Maximum samples kept; the oldest is evicted when full.
    """

    def __init__(self, max_points: int) -> None:
        self._samples: deque[VramSample] = deque(maxlen=max_points)

    def __len__(self) -> int:
        return len(self._samples)

    def add(self, sample: VramSample) -> None:
        """Append a sample, evicting the oldest when the buffer is full.

        Args:
            sample: The reading to record.
        """
        self._samples.append(sample)

    def latest(self) -> VramSample | None:
        """Return the newest sample, or ``None`` when empty."""
        return self._samples[-1] if self._samples else None

    def window(self, window_s: float, now: float | None = None) -> list[VramSample]:
        """Return the samples from the last ``window_s`` seconds, oldest first.

        A window larger than the data simply returns everything.

        Args:
            window_s: Look-back window in seconds.
            now: Reference time (epoch seconds); defaults to the wall clock.

        Returns:
            The matching samples in chronological order.
        """
        ref = time.time() if now is None else now
        cutoff = ref - window_s
        return [s for s in self._samples if s.ts >= cutoff]


async def run_sampler(
    reader: VramReader,
    history: VramHistory,
    interval_s: float,
    *,
    clock: Callable[[], float] = time.time,
) -> None:
    """Sample ``reader`` every ``interval_s`` seconds into ``history`` forever.

    NVML is blocking, so each read runs in a worker thread.  Unavailable or
    partial snapshots are skipped and reader errors are swallowed (logged at
    debug) — the sampler never raises except for cancellation.

    Args:
        reader: The VRAM reader (NVML in production, a fake in tests).
        history: Destination ring buffer.
        interval_s: Seconds between samples.
        clock: Time source for the sample timestamp.

    Raises:
        asyncio.CancelledError: When the task is cancelled (shutdown).
    """
    while True:
        try:
            snap = await asyncio.to_thread(reader.read)
            if (
                snap.available
                and snap.used_mb is not None
                and snap.free_mb is not None
                and snap.total_mb is not None
            ):
                history.add(
                    VramSample(
                        ts=clock(),
                        used_mb=snap.used_mb,
                        free_mb=snap.free_mb,
                        total_mb=snap.total_mb,
                    )
                )
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — the sampler must never die
            logger.debug("vram sampler tick failed; continuing", exc_info=True)
        await asyncio.sleep(interval_s)
