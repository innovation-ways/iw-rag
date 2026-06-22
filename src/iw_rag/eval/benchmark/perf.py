"""Performance collectors for Benchmark Studio (F-00093, S05).

The three collectors that populate the ``latency`` and ``resource`` JSONB
columns of the ``benchmark_reports`` row (per
``F-00093_Feature_Design.md`` Scope item 3 / AC5 / Boundary Behavior
row 3 — *no GPU / pynvml unavailable*):

1. **vLLM ``/metrics`` Prometheus scraper** — :class:`VllmMetricsScraper`
   snapshots the vLLM Prometheus exposition text **before** and **after**
   a run, parses it with ``prometheus_client.parser.text_string_to_metric_families``,
   and computes the typed :class:`VllmServerMetrics` result.  The HTTP
   fetch is injectable (an async fetcher callable) so unit tests pass a
   fixture string with no network — the scraper is **offline-only** by
   contract, even though its production path talks to the local vLLM
   endpoint.
2. **``nvidia-ml-py`` background GPU/VRAM sampler** — :class:`GpuSampler`
   runs a daemon-thread sampler at a configurable interval (default
   500 ms) using the ``pynvml`` module exposed by ``nvidia-ml-py``.
   Records VRAM-used + GPU-utilization samples across the run and
   aggregates **peak**, **mean**, **p95** utilisation and **peak −
   baseline** incremental VRAM on stop.  **Graceful degradation**:
   if ``pynvml`` is missing or ``nvmlInit()`` raises, the sampler
   becomes a no-op and never propagates into the caller — the report
   still persists without GPU data.  ``pynvml`` is lazily imported
   inside the sampler (mirrors :mod:`iw_rag.eval.ragas_runner`).
3. **``model_load_ms`` timing helper** — :func:`time_model_load_ms` and
   :func:`time_model_load_blocking` wrap a callable and return a
   :class:`ModelLoadTiming` with the elapsed milliseconds plus the
   wrapped callable's return value (or error message on failure).
   No model-swap orchestration here (F-00094) — just the timing
   primitive.

The module is **offline-only** (CONVENTIONS §10 / lld-eval-observability
§5 — two-clocks rule) and MUST NOT be imported by any hot-path module
(``api/``, ``agent/``, ``retrieval/``, ``generation/``).  The
``tests/unit/test_offline_only_import_guard.py`` static check enforces
this.  All collectors are **local only** — the vLLM ``/metrics``
endpoint and the local GPU; **no cloud APIs** (Invariant 1).

Caveats documented here so callers (and the test suite) carry them
forward:

* **Baseline-subtraction** for VRAM is best-effort.  On a single-GPU
  box vLLM and Infinity share the card, so ``vram_incremental_mb`` is
  an *incremental over the warmup baseline*, not a clean per-process
  attribution.
* **No GPU / driver / library** — the GPU sampler degrades to a no-op
  (``available=False``, all numeric fields ``None``).  The report
  assembly path must still persist the row.
* **vLLM endpoint unreachable** — the scraper returns an empty
  :class:`VllmServerMetrics` (all fields ``None``); it never raises
  into the caller.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# A type alias for the injectable HTTP fetcher.  It returns the raw
# Prometheus exposition text.  Production wires an ``httpx.AsyncClient``
# GET; unit tests pass an async lambda returning a fixture string.
VllmFetcher = Callable[[], Awaitable[str]]


# ---------------------------------------------------------------------------
# 1. vLLM /metrics Prometheus scraper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VllmServerMetrics:
    """The typed result of :meth:`VllmMetricsScraper.scrape` / :meth:`parse`.

    Maps 1:1 onto the ``resource.vllm_server`` block of the
    ``benchmark_reports`` row per R-00122 (the *Recommendation* metric
    JSON shape).  Per-query vectors are not carried here — the
    scraper is a single-run aggregate of the server's view.

    Attributes:
        ttft_p99: p99 time-to-first-token in seconds, read from
            ``vllm:time_to_first_token_seconds{quantile="0.99"}``.
        tpot_p99: p99 inter-token latency in seconds (``...inter_token_latency_seconds``).
        e2e_p99: p99 end-to-end request latency in seconds.
        queue_time_p99: p99 request-queue time in seconds.
        kv_cache_peak_pct: Peak KV-cache usage as a fraction in ``[0, 1]``
            (read from the post-scrape ``vllm:kv_cache_usage_perc`` gauge).
        prefix_cache_hit_rate: ``hits / queries`` over the run; ``0.0``
            when there were no queries.
        prompt_tokens: Total prefill tokens at the post-scrape snapshot.
        generation_tokens: Total generation tokens at the post-scrape snapshot.
        prompt_tokens_delta: ``prompt_tokens`` increase during the run
            (only set when a before-snapshot is supplied).
        generation_tokens_delta: Same for generation tokens.
        prefix_cache_queries_delta: ``vllm:prefix_cache_queries_total`` delta.
        prefix_cache_hits_delta: ``vllm:prefix_cache_hits_total`` delta.
        raw: The parsed per-metric dictionary for traceability / tests.
    """

    ttft_p99: float | None = None
    tpot_p99: float | None = None
    e2e_p99: float | None = None
    queue_time_p99: float | None = None
    kv_cache_peak_pct: float | None = None
    prefix_cache_hit_rate: float | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None
    prompt_tokens_delta: int | None = None
    generation_tokens_delta: int | None = None
    prefix_cache_queries_delta: int | None = None
    prefix_cache_hits_delta: int | None = None
    raw: dict[str, Any] | None = None


def _coerce_float(v: Any) -> float | None:
    """Coerce a Prometheus sample value to ``float`` (None on NaN/inf/missing)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _parse_prometheus_text(text: str) -> dict[str, Any]:
    """Parse a Prometheus exposition text into a flat ``name -> value`` map.

    Uses the official ``prometheus_client.parser.text_string_to_metric_families``
    (per R-00122 §4 — never hand-roll a regex parser).  Summary metrics
    expose per-quantile children; we keep the ``quantile=0.99`` for the
    well-known latency histograms, and the raw scalar for counters /
    gauges.

    Args:
        text: Prometheus exposition format text.

    Returns:
        A dict mapping metric name to its extracted value (``float`` for
        summary quantiles / gauges / counters).  Absent metrics are not
        present in the dict — callers use ``.get()`` and get ``None``.
    """
    # Lazy import — keeping it inside the function body mirrors the
    # ragas_runner / pynvml pattern so the module stays importable when
    # the parser is unavailable.  prometheus_client is a core runtime
    # dependency (the scraper runs in the api on POST /benchmark/run),
    # but we still degrade gracefully if it is somehow absent: a missing
    # parser yields an empty metric map (all-None metrics downstream)
    # rather than aborting the whole benchmark run — the scraper's
    # documented "best-effort, never propagates" contract (matches the
    # GpuSampler/pynvml graceful-degradation behaviour above).
    try:
        from prometheus_client.parser import text_string_to_metric_families
    except ImportError:
        logger.warning(
            "prometheus_client unavailable — skipping vLLM /metrics parse "
            "(perf metrics will be omitted; the run still persists)."
        )
        return {}

    out: dict[str, Any] = {}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            labels = sample.labels or {}
            q = labels.get("quantile")
            name = sample.name
            if name.endswith("_bucket") and "le" in labels:
                # Histogram bucket: vLLM exposes the latency metrics
                # (time_to_first_token / inter_token_latency / e2e_request_latency
                # / request_queue_time) as HISTOGRAMS — cumulative
                # ``..._bucket{le="x"}`` counts, not summary quantiles — so the
                # p99 must be computed from the buckets (see
                # :func:`_histogram_quantile`).  Collect ``(le, count)`` under
                # the base metric name + ``.__buckets__``.
                le_raw = labels["le"]
                le = math.inf if le_raw in ("+Inf", "Inf", "inf") else _coerce_float(le_raw)
                if le is not None:
                    base = name[: -len("_bucket")]
                    out.setdefault(f"{base}.__buckets__", []).append(
                        (le, _coerce_float(sample.value) or 0.0)
                    )
            elif q is not None:
                # Summary quantile: keep the per-quantile child, e.g.
                # ``vllm:time_to_first_token_seconds.p99 == 0.045``.
                try:
                    qf = float(q)
                except (TypeError, ValueError):
                    continue
                # We tag quantile 0.99 under ``.p99`` for readability;
                # 0.5 -> ``.p50``.  Anything else is recorded numerically.
                key = f"{sample.name}.p{int(round(qf * 100))}"
                out[key] = _coerce_float(sample.value)
            else:
                # Counter or gauge scalar (incl. histogram ``_sum`` / ``_count``).
                out[sample.name] = _coerce_float(sample.value)
    return out


def _histogram_quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """Compute the ``q``-quantile from Prometheus cumulative histogram buckets.

    Mirrors Prometheus' ``histogram_quantile``: ``buckets`` is a list of
    ``(le, cumulative_count)`` pairs (the ``+Inf`` bucket carries the total
    count).  Finds the bucket the ``q``-rank falls into and linearly
    interpolates within it.

    Args:
        buckets: ``(upper_bound, cumulative_count)`` pairs (any order).
        q: The quantile in ``[0, 1]`` (e.g. ``0.99``).

    Returns:
        The interpolated quantile value (seconds for the vLLM latency
        histograms), or ``None`` when there are no observations.
    """
    if not buckets:
        return None
    ordered = sorted(buckets, key=lambda b: b[0])
    total = ordered[-1][1]  # the +Inf bucket holds the grand total
    if total <= 0:
        return None
    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in ordered:
        if count >= rank:
            if math.isinf(le):
                # Rank falls in the open-ended +Inf bucket — best estimate is
                # the largest finite bound we crossed (no upper edge to lerp to).
                return prev_le if prev_le > 0 else None
            if count <= prev_count:
                return le
            frac = (rank - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    last_le = ordered[-1][0]
    return prev_le if math.isinf(last_le) else last_le


def combine_vllm_deltas(before: VllmServerMetrics, after: VllmServerMetrics) -> VllmServerMetrics:
    """Fold before→after counter deltas into the ``after`` snapshot.

    The per-run token + prefix-cache deltas need a baseline: the runner scrapes
    the generator's ``/metrics`` once at run start (``before``) and once at run
    end (``after``).  This computes the deltas from the two snapshots and
    returns ``after`` enriched with them (the p99 latencies, gauges, and
    cumulative totals are taken from ``after`` unchanged).  A counter that went
    backwards (a vLLM restart between snapshots) yields ``None`` for that delta
    rather than a negative number.

    Args:
        before: The run-start snapshot.
        after: The run-end snapshot (its ``raw`` map carries the counters).

    Returns:
        A :class:`VllmServerMetrics` equal to ``after`` plus the
        ``*_delta`` fields and the per-run ``prefix_cache_hit_rate``.
    """
    from dataclasses import replace

    braw = before.raw or {}
    araw = after.raw or {}

    def _raw_delta(key: str) -> int | None:
        b, a = braw.get(key), araw.get(key)
        if b is None or a is None:
            return None
        d = float(a) - float(b)
        return int(d) if d >= 0 else None  # counter reset → no usable delta

    def _field_delta(field: str) -> int | None:
        b = getattr(before, field)
        a = getattr(after, field)
        if b is None or a is None or a < b:
            return None
        return int(a - b)

    q_delta = _raw_delta("vllm:prefix_cache_queries_total")
    h_delta = _raw_delta("vllm:prefix_cache_hits_total")
    if q_delta and q_delta > 0 and h_delta is not None:
        hit_rate: float | None = h_delta / q_delta
    elif q_delta == 0:
        hit_rate = 0.0
    else:
        hit_rate = after.prefix_cache_hit_rate

    return replace(
        after,
        prompt_tokens_delta=_field_delta("prompt_tokens"),
        generation_tokens_delta=_field_delta("generation_tokens"),
        prefix_cache_queries_delta=q_delta,
        prefix_cache_hits_delta=h_delta,
        prefix_cache_hit_rate=hit_rate,
    )


def _build_vllm_metrics(parsed: dict[str, Any]) -> VllmServerMetrics:
    """Shape a parsed Prometheus map into a :class:`VllmServerMetrics`.

    Helper used by both the single-snapshot and before+after paths.

    Args:
        parsed: Output of :func:`_parse_prometheus_text`.

    Returns:
        The :class:`VllmServerMetrics` with the relevant p99s, gauges,
        and counters populated (or ``None`` when absent).
    """
    p_queries_raw = parsed.get("vllm:prefix_cache_queries_total")
    p_hits_raw = parsed.get("vllm:prefix_cache_hits_total")
    p_queries = float(p_queries_raw) if p_queries_raw is not None else None
    p_hits = float(p_hits_raw) if p_hits_raw is not None else None
    if p_queries is None or p_queries == 0:
        # Zero queries + zero hits → hit rate is 0.0 (not NaN).
        hit_rate: float | None = 0.0 if (p_queries == 0 and p_hits == 0) else None
    else:
        assert p_hits is not None
        hit_rate = p_hits / p_queries

    def _p99(metric: str) -> float | None:
        """p99 for a latency metric: prefer a summary quantile, else compute
        it from the histogram buckets (vLLM 0.11 emits histograms)."""
        summary = parsed.get(f"{metric}.p99")
        if summary is not None:
            return _coerce_float(summary)
        buckets = parsed.get(f"{metric}.__buckets__")
        if isinstance(buckets, list):
            return _histogram_quantile(buckets, 0.99)
        return None

    return VllmServerMetrics(
        ttft_p99=_p99("vllm:time_to_first_token_seconds"),
        tpot_p99=_p99("vllm:inter_token_latency_seconds"),
        e2e_p99=_p99("vllm:e2e_request_latency_seconds"),
        queue_time_p99=_p99("vllm:request_queue_time_seconds"),
        kv_cache_peak_pct=parsed.get("vllm:kv_cache_usage_perc"),
        prefix_cache_hit_rate=hit_rate,
        prompt_tokens=int(parsed["vllm:prompt_tokens_total"])
        if parsed.get("vllm:prompt_tokens_total") is not None
        else None,
        generation_tokens=int(parsed["vllm:generation_tokens_total"])
        if parsed.get("vllm:generation_tokens_total") is not None
        else None,
        raw=parsed,
    )


class VllmMetricsScraper:
    """Async scraper for the vLLM ``/metrics`` Prometheus endpoint.

    The HTTP fetch is **injectable** — the constructor accepts an async
    :data:`VllmFetcher` that returns the raw text.  In production the
    caller wires an ``httpx.AsyncClient.get(<vllm>/metrics).text``; in
    tests the fetcher is an async lambda returning a fixture string.

    The class exposes two call paths:

    * :meth:`scrape` — single-snapshot scrape.  Calls the fetcher once,
      parses, returns a :class:`VllmServerMetrics`.  A fetcher exception
      is caught and produces an empty result (all fields ``None``).
    * :meth:`parse` — classmethod.  Parses either a single text (use
      ``after_text=...``) or a ``before+after`` pair (counter deltas are
      computed and exposed on the result).  Pure — no network.

    Attributes:
        _fetcher: The injectable async fetcher (``Awaitable[str]``).
    """

    def __init__(self, *, fetcher: VllmFetcher) -> None:
        """Bind to an async fetcher.

        Args:
            fetcher: An async callable returning the raw ``/metrics`` text.
                MUST NOT raise on transient errors — the scraper catches
                and degrades to an empty result.  (Unit tests are free
                to inject a fetcher that raises to exercise the
                graceful-degradation path.)
        """
        self._fetcher = fetcher

    async def scrape(self) -> VllmServerMetrics:
        """Single-snapshot scrape.

        Returns:
            A :class:`VllmServerMetrics`.  All fields are ``None`` when
            the fetcher raises or returns malformed text — the scraper
            never propagates into the caller.
        """
        try:
            text = await self._fetcher()
        except Exception:  # noqa: BLE001 — graceful degradation contract
            return VllmServerMetrics()
        return self.parse(after_text=text)

    @classmethod
    def parse(
        cls,
        *,
        before_text: str | None = None,
        after_text: str | None = None,
    ) -> VllmServerMetrics:
        """Parse a single or paired Prometheus text snapshot.

        Counter deltas are computed when both ``before_text`` and
        ``after_text`` are provided; the post-scrape values for
        ``prompt_tokens`` / ``generation_tokens`` always come from
        ``after_text`` (or ``before_text`` if ``after_text`` is not
        supplied — useful when a single-snapshot scrape is the only
        available signal).

        Args:
            before_text: Optional pre-run ``/metrics`` text.
            after_text: Optional post-run ``/metrics`` text.

        Returns:
            A :class:`VllmServerMetrics` with deltas computed when both
            inputs are present.
        """
        if not after_text and not before_text:
            return VllmServerMetrics()
        after = _parse_prometheus_text(after_text or before_text or "")
        metrics = _build_vllm_metrics(after)

        if before_text and after_text:
            before = _parse_prometheus_text(before_text)
            if (
                before.get("vllm:prompt_tokens_total") is not None
                and after.get("vllm:prompt_tokens_total") is not None
            ):
                q_delta = int(
                    after.get("vllm:prefix_cache_queries_total", 0)
                    - before.get("vllm:prefix_cache_queries_total", 0)
                )
                h_delta = int(
                    after.get("vllm:prefix_cache_hits_total", 0)
                    - before.get("vllm:prefix_cache_hits_total", 0)
                )
                if q_delta > 0:
                    new_hit_rate: float | None = h_delta / q_delta
                elif q_delta == 0:
                    new_hit_rate = 0.0
                else:
                    # Counter wraparound or non-monotonic input — keep the
                    # per-snapshot hit rate rather than reporting nonsense.
                    new_hit_rate = metrics.prefix_cache_hit_rate

                metrics = VllmServerMetrics(
                    ttft_p99=metrics.ttft_p99,
                    tpot_p99=metrics.tpot_p99,
                    e2e_p99=metrics.e2e_p99,
                    queue_time_p99=metrics.queue_time_p99,
                    kv_cache_peak_pct=metrics.kv_cache_peak_pct,
                    prefix_cache_hit_rate=new_hit_rate,
                    prompt_tokens=metrics.prompt_tokens,
                    generation_tokens=metrics.generation_tokens,
                    prompt_tokens_delta=int(
                        after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]
                    ),
                    generation_tokens_delta=int(
                        after["vllm:generation_tokens_total"]
                        - before["vllm:generation_tokens_total"]
                    ),
                    prefix_cache_queries_delta=q_delta,
                    prefix_cache_hits_delta=h_delta,
                    raw=metrics.raw,
                )
        return metrics


# ---------------------------------------------------------------------------
# 2. nvidia-ml-py background GPU/VRAM sampler
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GpuBaselineSnapshot:
    """The baseline VRAM reading captured at the start of a run.

    The sampler takes one of these after warmup and before measurement
    (per R-00122 §5 — baseline subtraction) and uses it to compute
    ``vram_incremental_mb = peak − baseline`` on stop.

    Attributes:
        available: ``False`` when ``pynvml`` is missing or
            ``nvmlInit()`` raised; all numeric fields are ``None``.
        vram_used_mb: VRAM used in MiB at baseline.
        device_name: ``nvmlDeviceGetName`` for traceability.
    """

    available: bool
    vram_used_mb: int | None = None
    device_name: str | None = None


@dataclass(frozen=True, slots=True)
class GpuMetrics:
    """The typed result of :meth:`GpuSampler.stop`.

    Maps 1:1 onto the ``resource.gpu`` block of the
    ``benchmark_reports`` row.  When ``available`` is ``False`` (no
    GPU / driver / library) every numeric field is ``None`` and the
    report still persists without GPU data.

    Attributes:
        available: ``True`` when pynvml was usable for the run.
        util_peak_pct: Peak GPU utilisation over the measurement samples.
        util_mean_pct: Mean GPU utilisation.
        util_p95_pct: 95th-percentile GPU utilisation.
        vram_peak_mb: Peak VRAM used (MiB) over the measurement samples.
        vram_baseline_mb: VRAM used (MiB) at the baseline snapshot.
        vram_incremental_mb: ``vram_peak_mb − vram_baseline_mb`` (MiB).
            Best-effort incremental attribution; the baseline is taken
            *after* warmup and *before* measurement per R-00122 §5.
        device_name: ``nvmlDeviceGetName`` for traceability.
    """

    available: bool
    util_peak_pct: float | None = None
    util_mean_pct: float | None = None
    util_p95_pct: float | None = None
    vram_peak_mb: int | None = None
    vram_baseline_mb: int | None = None
    vram_incremental_mb: float | None = None
    device_name: str | None = None


def _try_import_pynvml() -> tuple[Any, bool]:
    """Lazy-import ``pynvml`` (the module exposed by ``nvidia-ml-py``).

    Mirrors the pattern in :mod:`iw_rag.eval.ragas_runner` — the import
    happens inside the body, never at module top-level, so the eval
    package imports cleanly without the GPU library installed.

    Returns:
        ``(module, True)`` on success, ``(None, False)`` on any
        ``ImportError`` or ``OSError`` (driver missing, library
        missing).  Never raises.
    """
    try:
        import pynvml  # noqa: PLC0415

        return pynvml, True
    except (ImportError, OSError):
        return None, False


def _percentile_p95(values: list[float]) -> float:
    """Compute the 95th percentile of a list of floats (linear interp).

    ``numpy`` is in the dev dependency group, so we use the stdlib
    linear-interpolation percentile (matches ``numpy.percentile``'s
    default ``linear`` method) to avoid adding a runtime dependency
    for one call.
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = 0.95 * (len(s) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return s[lo]
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


class GpuSampler:
    """Background thread sampler for GPU utilisation and VRAM use.

    Records ``(util_pct, vram_used_mb)`` tuples at the configured
    interval (default 500 ms — R-00122 §5 background sampler) on a
    daemon thread.  The sampler is **cleanly stoppable**; calling
    :meth:`stop` joins the thread within the call and returns the
    aggregate :class:`GpuMetrics`.

    **Graceful degradation contract**: if ``pynvml`` cannot be imported
    or ``nvmlInit()`` raises, the sampler becomes a no-op.  All methods
    succeed, the returned metrics are flagged ``available=False``, and
    nothing propagates into the caller (F-00093 Boundary Behavior row 3).

    Attributes:
        _interval_s: The sampling interval in seconds.
        _pynvml: The lazy-imported ``pynvml`` module, or ``None``.
        _available: Whether the sampler is functional.
        _handle: The NVML device handle, or ``None``.
        _device_name: The cached ``nvmlDeviceGetName`` string.
        _baseline: The :class:`GpuBaselineSnapshot` captured by
            :meth:`snapshot_baseline`.
        _samples: The in-flight ``(util_pct, vram_used_mb)`` samples.
        _lock: Guards ``_samples`` against the worker thread.
        _stop_event: Signalled by :meth:`stop` to break the worker loop.
        _thread: The worker thread (None until :meth:`start`).
    """

    def __init__(self, *, interval_s: float = 0.5) -> None:
        """Initialise the sampler.

        Args:
            interval_s: The sampling interval in seconds.  Default 0.5 s
                (500 ms) per R-00122 §5.
        """
        self._interval_s = interval_s
        self._pynvml, _ok = _try_import_pynvml()
        self._available = False
        self._handle: Any = None
        self._device_name: str | None = None
        self._baseline: GpuBaselineSnapshot | None = None
        self._samples: list[tuple[float, int]] = []
        # Index into ``_samples`` marking the start of the measurement
        # window (set by :meth:`mark_run_start` once the model is ready).
        # ``None`` means "no mark" → the whole sample history is the window.
        self._run_start_index: int | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        if _ok and self._pynvml is not None:
            try:
                self._pynvml.nvmlInit()
                count = int(self._pynvml.nvmlDeviceGetCount())
                if count > 0:
                    self._handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
                    try:
                        self._device_name = str(self._pynvml.nvmlDeviceGetName(self._handle))
                    except Exception:  # noqa: BLE001
                        self._device_name = None
                    self._available = True
            except Exception:  # noqa: BLE001 — driver / library / no GPU
                self._available = False
                self._handle = None

    @property
    def available(self) -> bool:
        """``True`` when the sampler is functional (pynvml + GPU present)."""
        return self._available

    def snapshot_baseline(self) -> GpuBaselineSnapshot:
        """Take a one-shot baseline VRAM reading.

        The run-level incremental VRAM is computed as ``peak − baseline``
        in :meth:`stop`, where ``peak`` is the max VRAM observed by the
        sampler across the measurement run.  The benchmark runner takes
        this snapshot **before the generator swap/activation** (pre-swap),
        so ``vram_incremental_mb`` reflects the VRAM the resident model
        adds over the prior state — vLLM eagerly pre-allocates its
        KV-cache pool at load, so a post-load baseline would always equal
        the peak and the incremental would collapse to ~0.  The captured
        value is stored on the sampler so :meth:`stop` can subtract it.

        Returns:
            A :class:`GpuBaselineSnapshot`.  When the sampler is in a
            no-GPU state, ``available`` is ``False`` and the numeric
            fields are ``None``.  Never raises.
        """
        if not self._available or self._handle is None or self._pynvml is None:
            snap = GpuBaselineSnapshot(available=False)
            self._baseline = snap
            return snap
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            vram_mb = int(info.used) // (1024 * 1024)
            snap = GpuBaselineSnapshot(
                available=True,
                vram_used_mb=vram_mb,
                device_name=self._device_name,
            )
            self._baseline = snap
            return snap
        except Exception:  # noqa: BLE001
            snap = GpuBaselineSnapshot(available=False)
            self._baseline = snap
            return snap

    def mark_run_start(self) -> None:
        """Mark the start of the measurement window (call once the model is ready).

        The runner starts the sampler **before** the generator swap so the
        VRAM trough (the idle floor while the prior generator is unloaded
        during a cold swap) is captured for ``vram_baseline_mb``.  But the
        swap/load period must not dilute the **utilisation** stats, so
        ``util_*`` and the run-time ``vram_peak`` are computed only over the
        samples taken AFTER this mark.  No-op when the sampler is
        unavailable.
        """
        if not self._available:
            return
        with self._lock:
            self._run_start_index = len(self._samples)

    def start(self) -> None:
        """Start the background sampler thread (no-op when unavailable).

        The thread is a daemon — it never blocks interpreter exit.
        Multiple :meth:`start` calls without an intervening :meth:`stop`
        are a programming error; the second call is a no-op.
        """
        if not self._available or self._handle is None or self._pynvml is None:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="iw-rag-gpu-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Worker loop.  Records ``(util_pct, vram_mb)`` per tick."""
        assert self._pynvml is not None
        assert self._handle is not None
        while not self._stop_event.is_set():
            try:
                info = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                vram_mb = int(info.used) // (1024 * 1024)
                with self._lock:
                    self._samples.append((float(util.gpu), vram_mb))
            except Exception:  # noqa: BLE001 — single tick failure must not kill the loop
                pass
            # Sleep in small slices so stop() can join quickly.
            slept = 0.0
            slice_s = min(0.05, self._interval_s)
            while slept < self._interval_s and not self._stop_event.is_set():
                time.sleep(slice_s)
                slept += slice_s

    def stop(self) -> GpuMetrics:
        """Stop the sampler, join the thread, and return the aggregate.

        Returns:
            A :class:`GpuMetrics`.  When the sampler was unavailable
            (no pynvml / no GPU) every numeric field is ``None`` and
            ``available`` is ``False``.  Never raises.
        """
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval_s * 4))
            self._thread = None

        if not self._available or self._handle is None or self._pynvml is None:
            return GpuMetrics(available=False, device_name=None)

        with self._lock:
            samples = list(self._samples)
            run_start = self._run_start_index
        if not samples:
            return GpuMetrics(
                available=True,
                util_peak_pct=0.0,
                util_mean_pct=0.0,
                util_p95_pct=0.0,
                vram_peak_mb=self._baseline.vram_used_mb if self._baseline else 0,
                vram_baseline_mb=self._baseline.vram_used_mb if self._baseline else 0,
                vram_incremental_mb=0.0,
                device_name=self._device_name,
            )

        # The measurement window is the samples taken after mark_run_start
        # (the model-ready point); without a mark the whole history is the
        # window.  Utilisation and the run-time VRAM peak come from this
        # window so a cold-swap load period — sampled only to catch the VRAM
        # trough — neither dilutes util_mean nor lets the PRIOR model's
        # residual VRAM (a swap-down) masquerade as this run's peak.
        run_window = samples[run_start:] if run_start is not None else samples
        if not run_window:
            run_window = samples
        utils = [u for u, _ in run_window]
        vram_peak = max(v for _, v in run_window)
        # Baseline = the idle floor: the minimum VRAM seen across the FULL
        # window (the trough while no generator is resident mid-swap) plus
        # the pre-swap snapshot.  This makes vram_incremental_mb the resident
        # model's footprint (peak − floor) — positive and independent of
        # which model happened to be loaded before — rather than a signed
        # delta against the previous model.
        baseline_candidates = [v for _, v in samples]
        if self._baseline is not None and self._baseline.vram_used_mb is not None:
            baseline_candidates.append(self._baseline.vram_used_mb)
        baseline_mb = min(baseline_candidates)
        return GpuMetrics(
            available=True,
            util_peak_pct=float(max(utils)),
            util_mean_pct=float(sum(utils) / len(utils)),
            util_p95_pct=float(_percentile_p95(utils)),
            vram_peak_mb=int(vram_peak),
            vram_baseline_mb=int(baseline_mb),
            vram_incremental_mb=float(vram_peak - baseline_mb),
            device_name=self._device_name,
        )


# ---------------------------------------------------------------------------
# 3. model_load_ms timing helper
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ModelLoadTiming:
    """The typed result of :func:`time_model_load_ms` / :func:`time_model_load_blocking`.

    Carries the wrapped callable's return value (or ``None`` on error)
    AND the elapsed wall-clock time.  Errors do not propagate — they
    are recorded as :attr:`error` so the caller (e.g. report assembly
    in S06) can persist a partial row.

    Attributes:
        elapsed_ms: The wall-clock elapsed time in milliseconds (always
            set, even on error).
        result: The wrapped callable's return value, or ``None`` on error.
        error: An error message string when the wrapped callable raised,
            else ``None``.
    """

    elapsed_ms: float
    result: Any = None
    error: str | None = None


async def time_model_load_ms(
    coro_factory: Callable[[], Awaitable[Any]],
) -> ModelLoadTiming:
    """Time an async model-load / warmup callable and return ms + result.

    Mirrors the spirit of :func:`asyncio.wait_for` — wraps the coroutine
    with a wall-clock start/end and captures the result.  The wrapped
    coroutine is run **once** (no retries; the caller decides what to
    do on error).

    Args:
        coro_factory: A no-arg callable returning a fresh coroutine on
            each call.  We accept a factory (not a coroutine) so the
            timing helper can run the wrapped code inside its own
            ``try/except`` and record a failure cleanly.

    Returns:
        A :class:`ModelLoadTiming` with ``elapsed_ms >= 0``, the
        callable's return value (or ``None`` on error), and an
        ``error`` string when the callable raised.
    """
    start = time.perf_counter()
    try:
        result = await coro_factory()
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ModelLoadTiming(elapsed_ms=elapsed_ms, result=None, error=str(exc))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ModelLoadTiming(elapsed_ms=elapsed_ms, result=result, error=None)


def time_model_load_blocking(callable_: Callable[[], Any]) -> ModelLoadTiming:
    """Time a blocking (sync) model-load / warmup callable.

    Same contract as :func:`time_model_load_ms` but for sync callables
    (e.g. ``transformers.AutoModel.from_pretrained``).  Exposed so the
    caller does not have to wrap a sync call in an async coroutine.

    Args:
        callable_: A no-arg sync callable to time.

    Returns:
        A :class:`ModelLoadTiming` with ``elapsed_ms >= 0``, the
        callable's return value (or ``None`` on error), and an
        ``error`` string when the callable raised.
    """
    start = time.perf_counter()
    try:
        result = callable_()
    except Exception as exc:  # noqa: BLE001 — graceful degradation
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ModelLoadTiming(elapsed_ms=elapsed_ms, result=None, error=str(exc))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return ModelLoadTiming(elapsed_ms=elapsed_ms, result=result, error=None)


__all__ = [
    "GpuBaselineSnapshot",
    "GpuMetrics",
    "GpuSampler",
    "ModelLoadTiming",
    "VllmFetcher",
    "VllmMetricsScraper",
    "VllmServerMetrics",
    "time_model_load_blocking",
    "time_model_load_ms",
]
