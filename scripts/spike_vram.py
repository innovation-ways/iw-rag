"""Phase 0.0 VRAM co-residence spike — the GATING de-risker (ADR-006, R-00103/T2).

Brings up Qwen3-14B-AWQ + FP8 KV (vLLM) alongside BGE-M3 + bge-reranker-v2-m3
(TEI) on the single RTX 5090 (32 GB), measures resident VRAM (true device memory
via ``pynvml`` / DCGM — NOT framework self-report) and a sample latency + TTFT,
and FAILS (exit 1) if the peak co-resident total exceeds the 32 GB card budget.

Run it twice: spine-only, then guards-on (``GUARDS_ON=1`` with a resident
quantized Granite Guardian 3.3-8B, ~5-6 GB) — both must fit 32 GB (dev = demo).
Implements the co-residence protocol in benchmarking-playbook §4.1: poll VRAM at
>=2 Hz during a simultaneous embed+rerank+generate step and keep the peak.

The pure threshold helpers (:func:`passes_budget`, :func:`headroom_low`) are unit
tested; the measurement itself is hardware-dependent and not unit tested.
"""

from __future__ import annotations

import asyncio
import json
import os
import time

import httpx

BUDGET_GB = float(os.environ.get("SPIKE_BUDGET_GB", "32"))  # single RTX 5090, 32 GB (dev = demo)
HEADROOM_FRAC = 0.10  # playbook §3.2: keep >=10% headroom
GUARDS_ON = os.environ.get("GUARDS_ON", "0") == "1"  # add resident Granite Guardian (~5-6 GB)
SAMPLE_HZ = 4.0  # >=2 Hz per playbook §4.1

VLLM = os.environ.get("VLLM_URL", "http://localhost:8001/v1")
GEN_MODEL = os.environ.get("SPIKE_GEN_MODEL", "Qwen/Qwen3-14B-AWQ")

# Embedder/reranker backend. TEI is canonical (tech-stack §3) but has no Blackwell
# (sm_120) build, so on the RTX 5090 we use the sanctioned Q-TEI fallback `infinity`.
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "tei").lower()  # "tei" | "infinity"
TEI_EMBED = os.environ.get("TEI_EMBED_URL", "http://localhost:8080")
TEI_RERANK = os.environ.get("TEI_RERANK_URL", "http://localhost:8081")
INFINITY = os.environ.get("INFINITY_URL", "http://localhost:7997")
EMBED_MODEL = os.environ.get("SPIKE_EMBED_MODEL", "BAAI/bge-m3")
RERANK_MODEL = os.environ.get("SPIKE_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def passes_budget(peak_gb: float, budget_gb: float = BUDGET_GB) -> bool:
    """Return whether the measured peak fits the card budget (pure threshold)."""
    return peak_gb <= budget_gb


def headroom_low(
    peak_gb: float,
    budget_gb: float = BUDGET_GB,
    headroom_frac: float = HEADROOM_FRAC,
) -> bool:
    """Return whether peak fits but leaves less than the target headroom."""
    ceiling = budget_gb * (1 - headroom_frac)
    return peak_gb <= budget_gb and peak_gb > ceiling


def vram(idx: int = 0) -> tuple[str, float]:
    """True device memory incl. CUDA context (NOT framework self-report).

    ``pynvml`` is imported lazily so the pure helpers above stay importable in
    the dev env without the heavy ``spike`` dependency group installed.
    """
    from pynvml import (  # noqa: PLC0415 — lazy: only the spike run needs pynvml
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
    )

    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(idx)
    name = nvmlDeviceGetName(handle)
    used = nvmlDeviceGetMemoryInfo(handle).used / 1024**3
    return (name if isinstance(name, str) else name.decode(), used)


async def sample_peak_vram(stop: asyncio.Event, hz: float = SAMPLE_HZ) -> float:
    """Poll VRAM at >=2 Hz while a load runs; keep the peak."""
    peak = 0.0
    while not stop.is_set():
        peak = max(peak, vram()[1])
        await asyncio.sleep(1.0 / hz)
    return peak


async def drive_step(client: httpx.AsyncClient) -> dict[str, float]:
    """One realistic agentic step: embed + rerank + a short streamed generation."""
    out: dict[str, float] = {}
    docs = ["the total due is 100", "unrelated"]
    t = time.perf_counter()
    if EMBED_BACKEND == "infinity":
        await client.post(
            f"{INFINITY}/embeddings",
            json={"model": EMBED_MODEL, "input": "what is the invoice total?"},
        )
    else:
        await client.post(f"{TEI_EMBED}/embed", json={"inputs": "what is the invoice total?"})
    out["embed_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    if EMBED_BACKEND == "infinity":
        await client.post(
            f"{INFINITY}/rerank",
            json={"model": RERANK_MODEL, "query": "invoice total", "documents": docs},
        )
    else:
        await client.post(
            f"{TEI_RERANK}/rerank",
            json={"query": "invoice total", "texts": docs},
        )
    out["rerank_ms"] = (time.perf_counter() - t) * 1000

    # Streamed generation → measure TTFT (first token) and full sample latency.
    t = time.perf_counter()
    ttft: float | None = None
    async with client.stream(
        "POST",
        f"{VLLM}/chat/completions",
        json={
            "model": GEN_MODEL,
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
            "max_tokens": 8,
            "temperature": 0.0,
            "seed": 0,
            "stream": True,
        },
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data:") and "[DONE]" not in line and ttft is None:
                ttft = (time.perf_counter() - t) * 1000
    out["generate_ttft_ms"] = ttft if ttft is not None else -1.0
    out["generate_ms"] = (time.perf_counter() - t) * 1000
    return out


async def main() -> int:
    """Run the co-residence measurement and return a process exit code."""
    name, baseline = vram()
    stop = asyncio.Event()
    async with httpx.AsyncClient(timeout=120) as client:
        sampler = asyncio.create_task(sample_peak_vram(stop))
        await drive_step(client)  # warmup (discarded)
        latency = await drive_step(client)  # measured step
        stop.set()
        peak = await sampler

    config = (
        "guards-on (spine + generator + Granite Guardian)"
        if GUARDS_ON
        else "spine-only (spine + generator)"
    )
    ceiling = BUDGET_GB * (1 - HEADROOM_FRAC)
    headroom = BUDGET_GB - peak
    ok = passes_budget(peak, BUDGET_GB)

    print(f"GPU: {name}")
    print(f"Config: {config}")
    print(f"Embed/rerank backend: {EMBED_BACKEND}")
    print(
        f"Baseline / peak co-resident VRAM: {baseline:.1f} / {peak:.1f} GB "
        f"(budget {BUDGET_GB:.0f} GB, RTX 5090 32 GB)"
    )
    print(
        f"Headroom: {headroom:.1f} GB ({headroom / BUDGET_GB * 100:.0f}%); "
        f">=10% target ceiling = {ceiling:.1f} GB"
    )
    print(f"TTFT: {latency['generate_ttft_ms']:.0f} ms; sample latencies: {json.dumps(latency)}")
    if headroom_low(peak, BUDGET_GB):
        print("WARN — fits 32 GB but headroom <10%; see scale-out trigger (playbook §5)")
    print("RESULT:", "PASS — co-residence confirmed" if ok else "FAIL — exceeds 32 GB card budget")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
