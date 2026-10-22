"""Unit tests for the power-control NVML VRAM reader.

``pynvml`` is faked (no GPU, no driver) by patching the lazy import.  Tests
verify the snapshot math (MiB conversion), the per-process breakdown + sort,
and the graceful-degradation contract (no pynvml → ``available=False``).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from iw_rag.serving.vram import NvmlVramReader, VramSnapshot

_MIB = 1024 * 1024


class _Mem:
    def __init__(self, used: int, free: int, total: int) -> None:
        self.used = used
        self.free = free
        self.total = total


class _Proc:
    def __init__(self, pid: int, used: int | None) -> None:
        self.pid = pid
        self.usedGpuMemory = used  # noqa: N815 — mirrors the pynvml attribute name


class FakePynvml:
    """A minimal stand-in for the ``pynvml`` module (one GPU)."""

    def __init__(self, used_mib: int = 18000, total_mib: int = 32607) -> None:
        self._used = used_mib * _MIB
        self._total = total_mib * _MIB
        self._free = self._total - self._used

    def nvmlInit(self) -> None:  # noqa: N802
        return None

    def nvmlDeviceGetCount(self) -> int:  # noqa: N802
        return 1

    def nvmlDeviceGetHandleByIndex(self, idx: int) -> Any:  # noqa: N802
        return object()

    def nvmlDeviceGetName(self, handle: Any) -> str:  # noqa: N802
        return "NVIDIA GeForce RTX 5090"

    def nvmlDeviceGetMemoryInfo(self, handle: Any) -> _Mem:  # noqa: N802
        return _Mem(self._used, self._free, self._total)

    def nvmlDeviceGetComputeRunningProcesses(self, handle: Any) -> list[_Proc]:  # noqa: N802
        # Deliberately out of order so the reader's sort is exercised.
        return [_Proc(101, 4084 * _MIB), _Proc(202, 18410 * _MIB), _Proc(303, None)]

    def nvmlSystemGetProcessName(self, pid: int) -> bytes:  # noqa: N802
        return f"proc-{pid}".encode()


def _reader_with(fake: Any) -> NvmlVramReader:
    with patch("iw_rag.serving.vram._try_import_pynvml", return_value=fake):
        return NvmlVramReader()


def test_reader_available_reads_total_used_free() -> None:
    """A working NVML maps memory info to MiB with used+free==total."""
    reader = _reader_with(FakePynvml(used_mib=18000, total_mib=32607))
    snap = reader.read()
    assert snap.available is True
    assert snap.total_mb == 32607
    assert snap.used_mb == 18000
    assert snap.free_mb == 32607 - 18000
    assert snap.device_name == "NVIDIA GeForce RTX 5090"


def test_reader_processes_sorted_descending_and_named() -> None:
    """Per-process VRAM is returned largest-first with resolved names."""
    reader = _reader_with(FakePynvml())
    snap = reader.read()
    assert [p.pid for p in snap.processes] == [202, 101, 303]
    assert snap.processes[0].used_mb == 18410
    assert snap.processes[1].used_mb == 4084
    # None usedGpuMemory (accounting off) degrades to 0, not a crash.
    assert snap.processes[2].used_mb == 0
    assert snap.processes[0].name == "proc-202"


def test_reader_degrades_when_pynvml_absent() -> None:
    """No pynvml → available=False with all numeric fields None (never raises)."""
    reader = _reader_with(None)
    assert reader.available is False
    snap = reader.read()
    assert snap == VramSnapshot(available=False)


def test_reader_degrades_when_init_raises() -> None:
    """nvmlInit() raising (no driver) degrades to unavailable, no propagation."""

    class _Boom(FakePynvml):
        def nvmlInit(self) -> None:  # noqa: N802
            raise RuntimeError("driver missing")

    reader = _reader_with(_Boom())
    assert reader.available is False
    assert reader.read().available is False
