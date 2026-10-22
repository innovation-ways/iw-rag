"""Read-only GPU VRAM telemetry for the power-control sidecar.

A small, dependency-light NVML reader that returns a one-shot snapshot of
device VRAM (used / free / total) plus the per-process VRAM breakdown.  This
is what the console's System view renders as the live gauge, and what proves a
power-off actually freed VRAM (the ``used_mb`` total drops when a container is
stopped).

**Graceful degradation** (mirrors :class:`iw_rag.eval.benchmark.perf.GpuSampler`):
if ``pynvml`` (the module exposed by ``nvidia-ml-py``) cannot be imported or
``nvmlInit()`` raises — no GPU reservation, no driver, library absent — the
reader becomes a no-op that returns ``available=False`` with all numeric
fields ``None``.  Nothing propagates to the caller; the gauge just shows
"telemetry unavailable".

The reader is intentionally *separate* from the benchmark ``GpuSampler``:
``eval/benchmark`` is offline-only and must never be imported by a serving
module (layering / ``test_offline_only_import_guard``).  This is the serving
layer's own NVML touch — pure read, no sampling thread, no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class VramProcess:
    """One process holding VRAM on the device.

    Attributes:
        pid: The OS pid of the process (host pid namespace).
        used_mb: VRAM the process holds, in MiB.
        name: Best-effort process name (``nvmlSystemGetProcessName``), or
            ``None`` when the name could not be resolved.
    """

    pid: int
    used_mb: int
    name: str | None = None


@dataclass(frozen=True, slots=True)
class VramSnapshot:
    """A one-shot VRAM reading of the (single) GPU.

    Attributes:
        available: ``True`` when pynvml + a GPU were usable.  When ``False``
            every numeric field is ``None`` and ``processes`` is empty.
        used_mb: Total VRAM used on the device, MiB.
        free_mb: Total VRAM free on the device, MiB.
        total_mb: Device VRAM capacity, MiB.
        device_name: ``nvmlDeviceGetName`` for traceability.
        processes: Per-process VRAM breakdown (compute apps), largest first.
    """

    available: bool
    used_mb: int | None = None
    free_mb: int | None = None
    total_mb: int | None = None
    device_name: str | None = None
    processes: list[VramProcess] = field(default_factory=list)


class VramReader(Protocol):
    """Structural type for the VRAM reader (so tests can inject a fake)."""

    def read(self) -> VramSnapshot: ...


def _try_import_pynvml() -> Any | None:
    """Lazy-import ``pynvml``; return the module or ``None`` (never raises)."""
    try:
        import pynvml  # noqa: PLC0415

        return pynvml
    except (ImportError, OSError):
        return None


class NvmlVramReader:
    """NVML-backed :class:`VramReader` for device index 0 (the single GPU).

    Initialises NVML once on construction; if anything fails the instance is
    permanently in the ``available=False`` degraded state and :meth:`read`
    returns an empty snapshot.
    """

    def __init__(self, *, device_index: int = 0) -> None:
        """Initialise NVML and cache the device handle.

        Args:
            device_index: The NVML device index (0 on a single-GPU box).
        """
        self._pynvml = _try_import_pynvml()
        self._available = False
        self._handle: Any = None
        self._device_name: str | None = None
        if self._pynvml is None:
            return
        try:
            self._pynvml.nvmlInit()
            if int(self._pynvml.nvmlDeviceGetCount()) > self._index_guard(device_index):
                self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
                try:
                    self._device_name = str(self._pynvml.nvmlDeviceGetName(self._handle))
                except Exception:  # noqa: BLE001
                    self._device_name = None
                self._available = True
        except Exception:  # noqa: BLE001 — driver / library / no GPU
            self._available = False
            self._handle = None

    @staticmethod
    def _index_guard(device_index: int) -> int:
        """Return ``device_index`` so ``count > index`` is the presence check."""
        return device_index

    @property
    def available(self) -> bool:
        """``True`` when NVML + a GPU were usable for this reader."""
        return self._available

    def read(self) -> VramSnapshot:
        """Return a one-shot VRAM snapshot (never raises).

        Returns:
            A :class:`VramSnapshot`.  ``available=False`` with all numeric
            fields ``None`` when the reader is degraded or a read fails.
        """
        if not self._available or self._handle is None or self._pynvml is None:
            return VramSnapshot(available=False)
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            used_mb = int(info.used) // (1024 * 1024)
            free_mb = int(info.free) // (1024 * 1024)
            total_mb = int(info.total) // (1024 * 1024)
        except Exception:  # noqa: BLE001
            return VramSnapshot(available=False, device_name=self._device_name)
        processes = self._read_processes()
        return VramSnapshot(
            available=True,
            used_mb=used_mb,
            free_mb=free_mb,
            total_mb=total_mb,
            device_name=self._device_name,
            processes=processes,
        )

    def _read_processes(self) -> list[VramProcess]:
        """Return the per-process compute-app VRAM breakdown (best effort)."""
        assert self._pynvml is not None
        out: list[VramProcess] = []
        try:
            procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
        except Exception:  # noqa: BLE001
            return out
        for p in procs:
            used = getattr(p, "usedGpuMemory", None)
            # NVML reports None/UINT64_MAX when per-process accounting is off.
            used_mb = 0 if used is None else int(used) // (1024 * 1024)
            pid = int(getattr(p, "pid", 0))
            name: str | None = None
            try:
                raw = self._pynvml.nvmlSystemGetProcessName(pid)
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
            except Exception:  # noqa: BLE001
                name = None
            out.append(VramProcess(pid=pid, used_mb=used_mb, name=name))
        out.sort(key=lambda x: x.used_mb, reverse=True)
        return out
