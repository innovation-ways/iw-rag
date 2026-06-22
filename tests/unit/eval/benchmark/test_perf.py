"""Unit tests for the perf collectors (eval/benchmark/perf.py, S05).

TDD RED-first: these tests are written before the perf.py implementation
(S05) and exercise the three collectors' public contracts per F-00093
Feature_Design §Scope item 3 / AC5 / Boundary Behavior row 3:

1. **vLLM ``/metrics`` Prometheus scraper** — parses a mocked Prometheus
   text payload (counter deltas, histogram quantiles, gauges) and returns
   the typed ``VllmServerMetrics`` result.  The HTTP fetch is injected so
   unit tests pass a fixture string with no network.
2. **``nvidia-ml-py`` background GPU/VRAM sampler** — records peak/mean/
   p95 utilization and peak − baseline incremental VRAM.  When ``pynvml``
   cannot be imported the sampler becomes a no-op and returns an empty
   /None result (never raises).  ``pynvml`` is mocked for the happy path
   (no real GPU).
3. **``model_load_ms`` timing helper** — wraps a callable and returns
   elapsed milliseconds.

The collectors live under ``eval/`` and are **offline only** (they never
belong on the hot path; the offline-only import guard already covers
this).  The collector interfaces are designed so unit tests stay
hermetic: no live GPU, no live vLLM, no network.
"""

from __future__ import annotations

import asyncio
import sys
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Module surface
# ---------------------------------------------------------------------------


def test_perf_module_exposes_three_collectors() -> None:
    """``perf.py`` MUST expose the three collectors + their typed results."""
    from iw_rag.eval.benchmark import perf as perf_mod

    # The three result dataclasses
    assert hasattr(perf_mod, "VllmServerMetrics")
    assert hasattr(perf_mod, "GpuMetrics")
    assert hasattr(perf_mod, "ModelLoadTiming")
    # The three collectors
    assert hasattr(perf_mod, "VllmMetricsScraper")
    assert hasattr(perf_mod, "GpuSampler")
    assert hasattr(perf_mod, "time_model_load_ms")


# ---------------------------------------------------------------------------
# 1. vLLM /metrics Prometheus scraper
# ---------------------------------------------------------------------------


# A representative vLLM /metrics text fixture.  Two snapshots — before/after
# a run — so the scraper must compute counter deltas.  The "after" snapshot
# includes:
#   - vllm:prompt_tokens_total counter increased from 100 to 250 (+150)
#   - vllm:generation_tokens_total counter increased from 50 to 200 (+150)
#   - vllm:time_to_first_token_seconds{quantile="0.99"} = 0.045
#   - vllm:inter_token_latency_seconds{quantile="0.99"}    = 0.012
#   - vllm:e2e_request_latency_seconds{quantile="0.99"}     = 0.430
#   - vllm:request_queue_time_seconds{quantile="0.99"}     = 0.008
#   - vllm:kv_cache_usage_perc                              = 0.62  (peak)
#   - vllm:prefix_cache_queries_total = 40
#   - vllm:prefix_cache_hits_total    = 28  → hit rate 0.70
_VLLM_METRICS_BEFORE = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 100
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 50
# HELP vllm:time_to_first_token_seconds Time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds summary
vllm:time_to_first_token_seconds{quantile="0.5"} 0.020
vllm:time_to_first_token_seconds{quantile="0.99"} 0.040
# HELP vllm:inter_token_latency_seconds Inter-token latency in seconds.
# TYPE vllm:inter_token_latency_seconds summary
vllm:inter_token_latency_seconds{quantile="0.5"} 0.008
vllm:inter_token_latency_seconds{quantile="0.99"} 0.010
# HELP vllm:e2e_request_latency_seconds End-to-end request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds summary
vllm:e2e_request_latency_seconds{quantile="0.99"} 0.410
# HELP vllm:request_queue_time_seconds Queue time in seconds.
# TYPE vllm:request_queue_time_seconds summary
vllm:request_queue_time_seconds{quantile="0.99"} 0.005
# HELP vllm:kv_cache_usage_perc KV-cache usage in percent (0..1).
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.55
# HELP vllm:prefix_cache_queries_total Total prefix cache lookups.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 20
# HELP vllm:prefix_cache_hits_total Total prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total 14
""".strip()

_VLLM_METRICS_AFTER = """
# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 250
# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total 200
# HELP vllm:time_to_first_token_seconds Time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds summary
vllm:time_to_first_token_seconds{quantile="0.5"} 0.025
vllm:time_to_first_token_seconds{quantile="0.99"} 0.045
# HELP vllm:inter_token_latency_seconds Inter-token latency in seconds.
# TYPE vllm:inter_token_latency_seconds summary
vllm:inter_token_latency_seconds{quantile="0.5"} 0.009
vllm:inter_token_latency_seconds{quantile="0.99"} 0.012
# HELP vllm:e2e_request_latency_seconds End-to-end request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds summary
vllm:e2e_request_latency_seconds{quantile="0.99"} 0.430
# HELP vllm:request_queue_time_seconds Queue time in seconds.
# TYPE vllm:request_queue_time_seconds summary
vllm:request_queue_time_seconds{quantile="0.99"} 0.008
# HELP vllm:kv_cache_usage_perc KV-cache usage in percent (0..1).
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 0.62
# HELP vllm:prefix_cache_queries_total Total prefix cache lookups.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 40
# HELP vllm:prefix_cache_hits_total Total prefix cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total 28
""".strip()


def test_vllm_scraper_parses_metrics_text_fixture() -> None:
    """vLLM scraper MUST parse a fixture payload and return the typed result.

    The fetcher is injectable — we pass an async lambda returning the text.
    Expected values come from ``_VLLM_METRICS_AFTER`` (a single-snapshot
    scrape) since the scraper also exposes a single-text parse path.
    """
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper, VllmServerMetrics

    async def fetcher() -> str:
        return _VLLM_METRICS_AFTER

    scraper = VllmMetricsScraper(fetcher=fetcher)
    res = asyncio.run(scraper.scrape())

    assert isinstance(res, VllmServerMetrics)
    # Histogram p99s from the fixture
    assert res.ttft_p99 == pytest.approx(0.045, abs=1e-6)
    assert res.tpot_p99 == pytest.approx(0.012, abs=1e-6)
    assert res.e2e_p99 == pytest.approx(0.430, abs=1e-6)
    assert res.queue_time_p99 == pytest.approx(0.008, abs=1e-6)
    # Gauge: kv_cache_peak_pct
    assert res.kv_cache_peak_pct == pytest.approx(0.62, abs=1e-6)
    # Prefix cache: 28 hits / 40 queries = 0.70
    assert res.prefix_cache_hit_rate == pytest.approx(0.70, abs=1e-6)
    # Counters are exposed as raw totals (and as deltas if before+after)
    assert res.prompt_tokens == 250
    assert res.generation_tokens == 200


def test_vllm_scraper_computes_counter_deltas_when_before_supplied() -> None:
    """When a before-text is supplied, the scraper MUST return token deltas."""
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    res = VllmMetricsScraper.parse(
        before_text=_VLLM_METRICS_BEFORE,
        after_text=_VLLM_METRICS_AFTER,
    )
    # Counter deltas
    assert res.prompt_tokens_delta == 150  # 250 - 100
    assert res.generation_tokens_delta == 150  # 200 - 50
    # Per-snapshot counters are also exposed
    assert res.prompt_tokens == 250
    assert res.generation_tokens == 200
    # prefix_cache deltas → hit rate computed from deltas
    assert res.prefix_cache_queries_delta == 20  # 40 - 20
    assert res.prefix_cache_hits_delta == 14  # 28 - 14
    assert res.prefix_cache_hit_rate == pytest.approx(14 / 20, abs=1e-6)


_VLLM_METRICS_HISTOGRAM = """
# HELP vllm:e2e_request_latency_seconds E2E request latency.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_bucket{le="0.5"} 0
vllm:e2e_request_latency_seconds_bucket{le="1.0"} 90
vllm:e2e_request_latency_seconds_bucket{le="2.0"} 100
vllm:e2e_request_latency_seconds_bucket{le="+Inf"} 100
vllm:e2e_request_latency_seconds_sum 95.0
vllm:e2e_request_latency_seconds_count 100
""".strip()


def test_vllm_scraper_computes_p99_from_histogram_buckets() -> None:
    """vLLM 0.11 emits latency as HISTOGRAMS (no summary quantiles); the scraper
    MUST compute the p99 from the cumulative buckets via histogram_quantile.

    p99 rank = 99 of 100.  The 99th obs falls in the (1.0, 2.0] bucket
    (cum 90 → 100): linear-interp 1.0 + (2.0-1.0) * (99-90)/(100-90) = 1.9.
    """
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    res = VllmMetricsScraper.parse(after_text=_VLLM_METRICS_HISTOGRAM)
    assert res.e2e_p99 == pytest.approx(1.9, abs=1e-6)


def test_histogram_quantile_edge_cases() -> None:
    """histogram_quantile handles empty / zero-count / +Inf-only inputs."""
    from iw_rag.eval.benchmark.perf import _histogram_quantile

    assert _histogram_quantile([], 0.99) is None
    assert _histogram_quantile([(1.0, 0.0), (float("inf"), 0.0)], 0.99) is None
    # All 100 obs in the (0, 1.0] bucket → p99 interpolates 0→1.0 at 99% = 0.99
    # (Prometheus assumes a uniform distribution within the bucket).
    assert _histogram_quantile([(1.0, 100.0), (float("inf"), 100.0)], 0.99) == pytest.approx(0.99)


def test_combine_vllm_deltas_from_before_after() -> None:
    """combine_vllm_deltas folds before→after counter deltas into ``after``."""
    from iw_rag.eval.benchmark.perf import VllmServerMetrics, combine_vllm_deltas

    before = VllmServerMetrics(
        prompt_tokens=100,
        generation_tokens=50,
        raw={
            "vllm:prefix_cache_queries_total": 20,
            "vllm:prefix_cache_hits_total": 14,
        },
    )
    after = VllmServerMetrics(
        prompt_tokens=250,
        generation_tokens=200,
        e2e_p99=1.2,
        prefix_cache_hit_rate=0.7,
        raw={
            "vllm:prefix_cache_queries_total": 40,
            "vllm:prefix_cache_hits_total": 28,
        },
    )
    res = combine_vllm_deltas(before, after)
    assert res.prompt_tokens_delta == 150  # 250 - 100
    assert res.generation_tokens_delta == 150  # 200 - 50
    assert res.prefix_cache_queries_delta == 20  # 40 - 20
    assert res.prefix_cache_hits_delta == 14  # 28 - 14
    assert res.prefix_cache_hit_rate == pytest.approx(14 / 20, abs=1e-6)
    # ``after``'s own fields are preserved unchanged.
    assert res.e2e_p99 == pytest.approx(1.2)
    assert res.prompt_tokens == 250


def test_combine_vllm_deltas_counter_reset_yields_none() -> None:
    """A counter that went backwards (vLLM restart) → delta is None, not negative."""
    from iw_rag.eval.benchmark.perf import VllmServerMetrics, combine_vllm_deltas

    before = VllmServerMetrics(prompt_tokens=500, generation_tokens=300)
    after = VllmServerMetrics(prompt_tokens=100, generation_tokens=50)  # reset
    res = combine_vllm_deltas(before, after)
    assert res.prompt_tokens_delta is None
    assert res.generation_tokens_delta is None


def test_vllm_scraper_handles_zero_prefix_cache_queries() -> None:
    """Zero queries + zero hits → hit rate is 0.0 (not NaN, not divide-by-zero)."""
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    text = """
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total 0
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total 0
""".strip()
    res = VllmMetricsScraper.parse(after_text=text)
    assert res.prefix_cache_hit_rate == 0.0


def test_vllm_scraper_handles_missing_metrics_gracefully() -> None:
    """If a metric is absent in the payload, the corresponding field is ``None``."""
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    text = """
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 10
""".strip()
    res = VllmMetricsScraper.parse(after_text=text)
    assert res.prompt_tokens == 10
    assert res.ttft_p99 is None
    assert res.tpot_p99 is None
    assert res.queue_time_p99 is None
    assert res.kv_cache_peak_pct is None
    assert res.prefix_cache_hit_rate is None


def test_vllm_scraper_fetcher_exception_returns_empty_result() -> None:
    """If the injected fetcher raises, the scraper returns an empty result,
    never propagates the exception (mirrors Boundary Behavior row 3 — graceful
    degradation so the report still persists without vLLM data)."""
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    async def boom() -> str:
        raise RuntimeError("connection refused")

    scraper = VllmMetricsScraper(fetcher=boom)
    res = asyncio.run(scraper.scrape())
    assert res.prompt_tokens is None
    assert res.ttft_p99 is None
    assert res.kv_cache_peak_pct is None


def test_vllm_scraper_degrades_when_prometheus_client_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing ``prometheus_client`` MUST NOT abort the run.

    Regression: the parser's lazy import lives outside the ``scrape``
    fetcher try/except, so when ``prometheus_client`` was a dev-only dep
    (absent from the ``--no-dev`` api image) the ``ModuleNotFoundError``
    escaped ``parse`` and killed the whole benchmark right after the
    ``ready`` event.  The scraper now degrades to an empty result
    (all-``None`` metrics) — its documented best-effort contract,
    mirroring the GpuSampler/pynvml behaviour — and the dep is a core
    runtime dependency so this only fires on a broken install.
    """
    from iw_rag.eval.benchmark.perf import VllmMetricsScraper

    # Make `from prometheus_client.parser import ...` raise ImportError.
    monkeypatch.setitem(sys.modules, "prometheus_client.parser", None)
    text = """
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total 42
# TYPE vllm:time_to_first_token_seconds summary
vllm:time_to_first_token_seconds{quantile="0.99"} 0.05
""".strip()
    res = VllmMetricsScraper.parse(after_text=text)
    # Empty/all-None result — never raised, never partial.
    assert res.prompt_tokens is None
    assert res.ttft_p99 is None
    assert res.raw == {}


# ---------------------------------------------------------------------------
# 2. nvidia-ml-py background GPU/VRAM sampler
# ---------------------------------------------------------------------------


def _install_mock_pynvml(
    monkeypatch: pytest.MonkeyPatch,
    *,
    vram_samples_mb: list[int],
    util_samples_pct: list[int],
    raise_on_init: bool = False,
) -> MagicMock:
    """Install a fake ``pynvml`` module in ``sys.modules`` for the sampler.

    The sampler lazy-imports ``pynvml`` (mirrors ``ragas_runner.py``), so a
    monkeypatched module in ``sys.modules['pynvml']`` is sufficient.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        vram_samples_mb: A list of VRAM-used values in MiB the fake device
            should return, in order, on successive ``DeviceGetMemoryInfo``
            calls.  If the list is exhausted the last value is returned
            indefinitely (mirrors a steady-state sampler).
        util_samples_pct: Same, for ``DeviceGetUtilizationRates`` percent.
        raise_on_init: When True, ``nvmlInit`` raises — the sampler must
            downgrade to a no-op.

    Returns:
        The mock module (handy for assertions in tests).
    """
    call_state: dict[str, int] = {"vram": 0, "util": 0, "shutdown": 0}

    vram_list = list(vram_samples_mb)
    util_list = list(util_samples_pct)
    vram_idx = {"i": 0}
    util_idx = {"i": 0}

    def nvml_init() -> None:
        if raise_on_init:
            raise RuntimeError("NVML: driver not loaded")

    def nvml_shutdown() -> None:
        call_state["shutdown"] += 1

    def nvml_device_get_count() -> int:
        return 1

    def nvml_device_get_handle_by_index(idx: int) -> int:
        return idx

    def nvml_device_get_memory_info(_handle: int) -> Any:
        i = vram_idx["i"]
        vram_idx["i"] = min(i + 1, len(vram_list) - 1)
        used = vram_list[i] if vram_list else 0
        # Return an object with .used, .total, .free in bytes
        return SimpleNamespace(used=used * 1024 * 1024, total=80 * 1024 * 1024 * 1024)

    def nvml_device_get_utilization_rates(_handle: int) -> Any:
        i = util_idx["i"]
        util_idx["i"] = min(i + 1, len(util_list) - 1)
        gpu = util_list[i] if util_list else 0
        return SimpleNamespace(gpu=gpu, memory=gpu)

    def nvml_device_get_name(_handle: int) -> str:
        return "Mock-GPU-A100"

    mock = MagicMock()
    mock.nvmlInit = nvml_init
    mock.nvmlShutdown = nvml_shutdown
    mock.nvmlDeviceGetCount = nvml_device_get_count
    mock.nvmlDeviceGetHandleByIndex = nvml_device_get_handle_by_index
    mock.nvmlDeviceGetMemoryInfo = nvml_device_get_memory_info
    mock.nvmlDeviceGetUtilizationRates = nvml_device_get_utilization_rates
    mock.nvmlDeviceGetName = nvml_device_get_name
    # Constants used by real pynvml; unused in this fake.
    mock.NVML_TEMPERATURE_GPU = 0
    mock._mock_state = call_state  # for assertions
    monkeypatch.setitem(sys.modules, "pynvml", mock)
    return mock


def test_gpu_sampler_records_peak_mean_p95_and_incremental_vram() -> None:
    """Happy path: peak, mean, p95 utilisation and peak−baseline VRAM are correct.

    Sample sequence:
      util % :  10, 30, 50, 70, 90
      VRAM MB : 1000, 1200, 1400, 1600, 1800
    Sampler is started AFTER the first sample is taken as the baseline
    (per R-00122 §5 — baseline snapshot after warmup, before measurement).
    Sampler is stopped and the aggregate is inspected.
    """
    from iw_rag.eval.benchmark.perf import GpuMetrics, GpuSampler

    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_mock_pynvml(
            monkeypatch,
            vram_samples_mb=[1000, 1200, 1400, 1600, 1800],
            util_samples_pct=[10, 30, 50, 70, 90],
        )

        sampler = GpuSampler(interval_s=0.001)
        baseline = sampler.snapshot_baseline()
        assert baseline is not None
        assert baseline.vram_used_mb == 1000

        sampler.start()
        # let the sampler take ~3 samples
        time.sleep(0.01)
        result = sampler.stop()
    finally:
        monkeypatch.undo()

    assert isinstance(result, GpuMetrics)
    # peak util = 90%
    assert result.util_peak_pct == pytest.approx(90.0, abs=1e-6)
    # mean util over the measurement samples (post-baseline) — between 10 and 90
    assert result.util_mean_pct is not None
    assert 10.0 <= result.util_mean_pct <= 90.0
    # p95 util ≤ peak
    assert result.util_p95_pct is not None
    assert result.util_p95_pct <= 90.0 + 1e-6
    # peak − baseline VRAM = 1800 − 1000 = 800
    assert result.vram_peak_mb == 1800
    assert result.vram_baseline_mb == 1000
    assert result.vram_incremental_mb == pytest.approx(800.0, abs=1e-6)
    assert result.device_name == "Mock-GPU-A100"


def test_gpu_sampler_incremental_is_footprint_over_swap_window() -> None:
    """``vram_incremental_mb`` is the resident model's footprint (run-peak −
    idle floor), positive even on a swap DOWN.

    Regression: the sampler now spans the whole swap. ``vram_baseline_mb``
    is the minimum VRAM across the FULL window (the trough while no
    generator is resident); ``vram_peak_mb`` and util come from the
    measurement window (after :meth:`mark_run_start`), so a swap-DOWN's
    *prior* (larger) model VRAM never masquerades as this run's peak and
    the load period never dilutes util.
    """
    from iw_rag.eval.benchmark.perf import GpuBaselineSnapshot, GpuSampler

    monkeypatch = pytest.MonkeyPatch()
    try:
        # Any usable mock device makes the sampler "available"; we drive the
        # aggregation directly via the sample buffer below (no live thread).
        _install_mock_pynvml(monkeypatch, vram_samples_mb=[1], util_samples_pct=[1])
        sampler = GpuSampler(interval_s=0.001)
        sampler._baseline = GpuBaselineSnapshot(  # noqa: SLF001 — drive aggregation
            available=True, vram_used_mb=24000, device_name="Mock-GPU-A100"
        )
        # Simulate a cold swap DOWN (14B → 8B): pre-swap old model resident,
        # then the idle trough, recorded BEFORE the run is marked.
        sampler._samples = [(5.0, 24000), (0.0, 11000)]  # noqa: SLF001
        sampler.mark_run_start()  # run window starts here (index 2)
        # Measurement-window samples: the new (smaller) 8B resident.
        sampler._samples.extend([(95.0, 17000), (99.0, 17500)])  # noqa: SLF001
        result = sampler.stop()
    finally:
        monkeypatch.undo()

    # Peak from the RUN window (8B), NOT the old 14B residual (24000).
    assert result.vram_peak_mb == 17500
    # Baseline = idle floor = min VRAM across the full window (the trough).
    assert result.vram_baseline_mb == 11000
    # Incremental = footprint of the resident model — positive on a swap-down.
    assert result.vram_incremental_mb == pytest.approx(6500.0)
    # Utilisation is measured only over the run window (95, 99) — the idle
    # load samples (5, 0) do not dilute it.
    assert result.util_mean_pct == pytest.approx(97.0)
    assert result.util_peak_pct == pytest.approx(99.0)


def test_gpu_sampler_no_op_when_pynvml_unavailable() -> None:
    """Boundary row 3: no GPU / pynvml unavailable → sampler is a no-op, no raise.

    When ``pynvml`` cannot be imported the sampler MUST produce an empty
    /None result and never raise into the caller.
    """
    from iw_rag.eval.benchmark.perf import GpuMetrics, GpuSampler

    monkeypatch = pytest.MonkeyPatch()
    try:
        # Force ``import pynvml`` to raise.  We achieve this by registering a
        # dummy module that raises on import via a custom MetaPathFinder.
        class _RaisingFinder:
            def find_spec(self, name: str, _path: Any = None, _target: Any = None) -> None:
                if name == "pynvml":
                    raise ImportError("No module named 'pynvml'")
                return None

        sys.meta_path.insert(0, _RaisingFinder())
        # Defensive: ensure no leftover pynvml module is visible to the
        # sampler at all.
        monkeypatch.delitem(sys.modules, "pynvml", raising=False)

        sampler = GpuSampler(interval_s=0.001)
        # snapshot_baseline() must not raise and must return a result
        # whose "available" flag is False.
        baseline = sampler.snapshot_baseline()
        assert baseline is not None
        assert baseline.available is False

        sampler.start()
        result = sampler.stop()
    finally:
        # Pop the meta-finder first, then undo the monkeypatch.
        try:
            sys.meta_path = [f for f in sys.meta_path if not isinstance(f, _RaisingFinder)]
        finally:
            monkeypatch.undo()

    assert isinstance(result, GpuMetrics)
    assert result.available is False
    # All numeric fields are None — never a real number from a fake GPU.
    assert result.util_peak_pct is None
    assert result.util_mean_pct is None
    assert result.util_p95_pct is None
    assert result.vram_peak_mb is None
    assert result.vram_baseline_mb is None
    assert result.vram_incremental_mb is None
    assert result.device_name is None


def test_gpu_sampler_no_op_when_nvml_init_raises() -> None:
    """If ``pynvml`` imports but ``nvmlInit()`` raises (driver missing),
    the sampler MUST downgrade to a no-op result, not propagate."""
    from iw_rag.eval.benchmark.perf import GpuSampler

    monkeypatch = pytest.MonkeyPatch()
    try:
        _install_mock_pynvml(
            monkeypatch,
            vram_samples_mb=[],
            util_samples_pct=[],
            raise_on_init=True,
        )

        sampler = GpuSampler(interval_s=0.001)
        baseline = sampler.snapshot_baseline()
        assert baseline is not None
        assert baseline.available is False

        sampler.start()
        result = sampler.stop()
    finally:
        monkeypatch.undo()

    assert result.available is False
    assert result.util_peak_pct is None
    assert result.vram_peak_mb is None


# ---------------------------------------------------------------------------
# 3. model_load_ms timing helper
# ---------------------------------------------------------------------------


async def _sleep_10ms() -> int:
    await asyncio.sleep(0.01)
    return 0


def test_time_model_load_ms_async_callable_returns_positive_ms() -> None:
    """``time_model_load_ms`` MUST time an async callable and return ms > 0."""
    from iw_rag.eval.benchmark.perf import ModelLoadTiming, time_model_load_ms

    timing: ModelLoadTiming = asyncio.run(time_model_load_ms(_sleep_10ms))
    assert isinstance(timing, ModelLoadTiming)
    # Sleep is 10 ms — the timing must be ≥ 5 ms (CI slack) and finite.
    assert timing.elapsed_ms is not None
    assert timing.elapsed_ms >= 5.0
    assert timing.elapsed_ms < 5000.0  # sanity upper bound


def test_time_model_load_ms_returns_result_of_callable() -> None:
    """The helper MUST return the wrapped callable's return value on success."""
    from iw_rag.eval.benchmark.perf import time_model_load_ms

    async def return_42() -> int:
        return 42

    timing = asyncio.run(time_model_load_ms(return_42))
    assert timing.result == 42
    assert timing.error is None


def test_time_model_load_ms_records_exception_does_not_raise() -> None:
    """If the wrapped callable raises, the helper records the error and
    returns the timing (no re-raise — same graceful pattern as the GPU
    sampler).  The caller decides what to do with ``timing.error``."""
    from iw_rag.eval.benchmark.perf import time_model_load_ms

    async def boom() -> None:
        raise RuntimeError("load failed")

    timing = asyncio.run(time_model_load_ms(boom))
    assert timing.result is None
    assert timing.error is not None
    assert "load failed" in str(timing.error)
    assert timing.elapsed_ms is not None
    assert timing.elapsed_ms >= 0.0


def test_time_model_load_ms_as_sync_helper() -> None:
    """A sync context manager flavour: ``time_model_load_blocking`` MUST
    also be exposed so callers that load a model with a blocking API
    (e.g. ``transformers.AutoModel.from_pretrained``) can time it."""
    from iw_rag.eval.benchmark.perf import time_model_load_blocking

    def slow_blocking() -> int:
        time.sleep(0.005)
        return 7

    timing = time_model_load_blocking(slow_blocking)
    assert timing.result == 7
    assert timing.error is None
    assert timing.elapsed_ms is not None
    assert timing.elapsed_ms >= 1.0
