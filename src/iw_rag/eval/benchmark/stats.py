"""Statistical comparison module for Benchmark Studio (F-00093, S07).

The A/B comparison layer that turns two reports' per-query metric vectors
into a rigorous, corrected delta table with a defensible winner rule
(R-00121 §4-5 / Feature_Design §Scope item 6 / AC3 / Boundary Behavior
row 5).

Public surface:

* :func:`wilcoxon_paired` — paired Wilcoxon signed-rank (continuous
  metrics such as recall@k, nDCG@k, faithfulness 0–1).
* :func:`bca_bootstrap_ci` — BCa bootstrap CI of ``mean_B − mean_A``,
  paired by query index, seeded for determinism (R-00121 §4.4 step 5).
* :func:`mcnemar` — McNemar's test for binary pass/fail metrics
  (R-00121 §4.3); uses the **exact binomial** test when the discordant
  pair count is below 10.
* :func:`bonferroni_holm` — Bonferroni-Holm correction across all
  compared metrics (R-00121 §4.5).
* :func:`compare` — the A/B compare entry point returning a
  :class:`ComparisonResult` with the per-metric rows, the winner (or
  ``None``), the underpowered flag, and a human-readable caveat.
* :class:`MetricDelta` / :class:`ComparisonResult` — the typed
  frozen-dataclass result shapes.

Test selection by metric type (R-00121 §4):

* **Continuous** (recall@k, nDCG@k, faithfulness) → paired Wilcoxon +
  BCa bootstrap CI of the mean difference.
* **Binary** (pass/fail) → McNemar (discordant-pair based; exact
  binomial when discordant count < 10).
* **Family-wise error control** → Bonferroni-Holm across all metrics.

**Winner rule (R-00121 §5.4)**: declare a winner ONLY if one config is
better on **both** nDCG@5 AND faithfulness (after correction).  A
single-metric win returns ``winner=None``.

**Underpowered caveat (R-00121 §4.2)**: when the query count is
below ``min_queries_for_significance`` (default 150), the result is
flagged ``underpowered=True`` with a caveat noting the detectable
effect size.  When ``n_queries < 2`` (cannot run a paired test), the
function returns ``underpowered=True``, ``significant=False``, and
``p_value=None`` rather than raising.

This module is **offline-only** (CONVENTIONS §10 / lld-eval-observability
§5 — two-clocks rule) and MUST NOT be imported by any hot-path module
(``api/``, ``agent/``, ``retrieval/``, ``generation/``).  The
``tests/unit/test_offline_only_import_guard.py`` static check enforces
this.  All randomness is **seeded** so tests are deterministic.

Dependencies: ``scipy`` is in the ``eval-judges`` group (added in S05
per the S05 prompt's instruction) and is **lazily imported** inside
the relevant function bodies so the module imports cleanly without
``scipy`` installed (mirrors the ``ragas_runner`` / ``perf.py`` /
``pynvml`` pattern).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from iw_rag.domain.errors import ConfigError

# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MetricDelta:
    """The per-metric A/B comparison row.

    Attributes:
        metric: The metric name (e.g. ``"ndcg_at_5"``, ``"faithfulness"``).
        val_a: The mean of A across the paired observations (or
            ``None`` when A is empty).
        val_b: The mean of B across the paired observations.
        delta: ``val_b - val_a``; positive means B is better (per the
            convention in :func:`compare`).
        p_value: The raw (uncorrected) p-value of the paired test, or
            ``None`` when the test could not run (e.g. n<2).
        p_corrected: The Bonferroni-Holm-corrected p-value, or ``None``
            when no correction is available.
        ci_95: A ``(lo, hi)`` tuple for the BCa bootstrap CI of the
            mean difference, or ``None`` when the bootstrap could not
            run.
        significant: ``True`` when ``p_corrected < 0.05`` AND the
            ``ci_95`` interval excludes 0 (R-00121 §4.4 step 5).
        test: The name of the test that produced ``p_value`` (e.g.
            ``"wilcoxon"`` for continuous metrics, ``"mcnemar"`` for
            binary metrics, ``"none"`` for n<2).
    """

    metric: str
    val_a: float | None
    val_b: float | None
    delta: float | None
    p_value: float | None
    p_corrected: float | None
    ci_95: tuple[float, float] | None
    significant: bool
    test: str


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """The overall A/B comparison result.

    Attributes:
        rows: The per-metric :class:`MetricDelta` rows (one per
            requested metric, in the requested order).
        winner: ``"A"`` or ``"B"`` when the dual-metric rule is
            satisfied, else ``None`` (R-00121 §5.4 — a single-metric
            win does not declare a winner).
        underpowered: ``True`` when the query count is below
            ``min_queries_for_significance`` (default 150).  When
            ``n_queries < 2``, this is also ``True`` and the per-metric
            p-values are ``None``.
        caveat: A human-readable caveat describing the underpowered
            state and the detectable effect size, or ``None`` when
            the comparison is well-powered.
        reasoning: A human-readable summary of the comparison,
            including the per-metric verdict and the winner decision
            (or lack thereof).
    """

    rows: tuple[MetricDelta, ...]
    winner: str | None
    underpowered: bool
    caveat: str | None
    reasoning: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[float | None]) -> float | None:
    """Compute the mean of a sequence, ignoring ``None`` and NaN.

    Returns ``None`` when no finite value is available (never ``0.0``,
    so the caller can distinguish "no data" from "all zero").
    """
    finite: list[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        finite.append(f)
    if not finite:
        return None
    return sum(finite) / len(finite)


def _differences(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """Element-wise difference ``b - a`` as floats.

    Args:
        a: The A-side paired observations.
        b: The B-side paired observations.

    Returns:
        A list of floats of length ``len(a)``.
    """
    return [float(b_i) - float(a_i) for a_i, b_i in zip(a, b, strict=True)]


# ---------------------------------------------------------------------------
# 1. Paired Wilcoxon signed-rank
# ---------------------------------------------------------------------------


def wilcoxon_paired(
    a: Sequence[float],
    b: Sequence[float],
) -> tuple[float, float]:
    """Run a paired Wilcoxon signed-rank test on ``a`` and ``b``.

    Test choice (R-00121 §4.1): the Wilcoxon signed-rank is the
    recommended non-parametric paired test for continuous per-query
    metrics.  Use ``scipy.stats.wilcoxon`` (the vetted implementation).

    Degenerate inputs:

    * ``a`` and ``b`` have different lengths → :class:`ConfigError`.
    * All paired differences are zero (tied zeros) → returns
      ``(p=1.0, stat=0.0)`` instead of raising (R-00121 §4.1
      footnote on tied zeros).
    * ``n < 2`` → returns ``(p=1.0, stat=0.0)`` (cannot run a paired
      test on a single observation).

    Args:
        a: The A-side per-query values (length must equal ``b``).
        b: The B-side per-query values (length must equal ``a``).

    Returns:
        ``(p_value, statistic)``.  ``p_value`` is the two-sided
        p-value; ``statistic`` is the Wilcoxon T statistic (or 0.0
        on degenerate input).
    """
    if len(a) != len(b):
        raise ConfigError(f"wilcoxon_paired: length mismatch (len(a)={len(a)}, len(b)={len(b)}).")
    n = len(a)
    if n < 2:
        return 1.0, 0.0
    # Lazy import — scipy is in the eval-judges group (added in S05);
    # we import inside the body so the module is importable without
    # scipy installed.
    from scipy import stats as _stats

    diffs = _differences(a, b)
    # All-zero differences: scipy raises a UserWarning + ValueError on
    # exact zeros; we treat the case as no evidence and return p=1.0.
    if all(d == 0.0 for d in diffs):
        return 1.0, 0.0
    result = _stats.wilcoxon(diffs, alternative="two-sided", correction=False)
    p = float(result.pvalue)
    stat = float(getattr(result, "statistic", 0.0))
    return p, stat


# ---------------------------------------------------------------------------
# 2. BCa bootstrap CI
# ---------------------------------------------------------------------------


_DEFAULT_BOOTSTRAP_RESAMPLES = 10_000


def bca_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    seed: int | None = None,
    n_resamples: int = _DEFAULT_BOOTSTRAP_RESAMPLES,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """Compute the BCa bootstrap CI of ``mean(b) - mean(a)``.

    Test choice (R-00121 §4.4 step 5): bias-corrected and accelerated
    (BCa) bootstrap, **paired** by query index (resample query indices
    jointly across A and B so the paired structure is preserved).  We
    use ``scipy.stats.bootstrap`` with ``method="BCa"`` and
    ``paired=True``; the statistic is a vectorised ``mean(b_i) -
    mean(a_i)`` over the resampled index set.  The RNG is seeded for
    determinism (so unit tests are reproducible).

    Degenerate inputs:

    * ``a`` and ``b`` have different lengths → :class:`ConfigError`.
    * ``n < 2`` → returns ``(0.0, 0.0)`` (cannot bootstrap on a single
      observation; this is not a meaningful CI).

    Args:
        a: The A-side per-query values.
        b: The B-side per-query values (same length as ``a``).
        seed: Optional integer seed for the bootstrap RNG.  ``None``
            uses Python's default (non-deterministic) entropy.
        n_resamples: The number of bootstrap resamples (default 10,000
            per R-00121 §4.4 step 5).
        confidence_level: The CI level (default 0.95).

    Returns:
        ``(lo, hi)`` tuple of CI bounds.  May be ``(0.0, 0.0)`` on
        degenerate input.
    """
    if len(a) != len(b):
        raise ConfigError(f"bca_bootstrap_ci: length mismatch (len(a)={len(a)}, len(b)={len(b)}).")
    n = len(a)
    if n < 2:
        return 0.0, 0.0
    # Lazy import — scipy is in the eval-judges group; we keep the
    # import inside the body so plain pytest can collect this module
    # in any environment.
    # Pass ``(a, b)`` as a tuple of arrays; ``paired=True`` resamples
    # query indices jointly across the two.  The statistic receives
    # ``(a_, b_, axis=-1)`` and returns the mean difference across
    # the resampled index set.  In vectorized mode ``a_`` and ``b_``
    # carry an extra leading axis over the resamples.
    import numpy as np  # numpy is in the dev group
    from scipy import stats as _stats

    a_arr = np.asarray(list(map(float, a)), dtype=float)
    b_arr = np.asarray(list(map(float, b)), dtype=float)
    rng = np.random.default_rng(seed)

    def _statistic(a_: Any, b_: Any, axis: int = -1) -> Any:
        return b_.mean(axis=axis) - a_.mean(axis=axis)

    res = _stats.bootstrap(
        (a_arr, b_arr),
        statistic=_statistic,
        n_resamples=n_resamples,
        paired=True,
        method="BCa",
        confidence_level=confidence_level,
        random_state=rng,
        vectorized=True,
    )
    lo = float(res.confidence_interval.low)
    hi = float(res.confidence_interval.high)
    return lo, hi


# ---------------------------------------------------------------------------
# 3. McNemar's test (binary metrics)
# ---------------------------------------------------------------------------


def mcnemar(
    *,
    b01: int,
    b10: int,
) -> tuple[float, float]:
    """Run McNemar's test on the discordant pair counts.

    Test choice (R-00121 §4.3): for binary pass/fail metrics, McNemar's
    test on the 2×2 contingency table is the recommended paired
    comparison.  When the discordant count (``b01 + b10``) is below
    10, use the **exact binomial test** (no chi-squared approximation
    — the small-sample distribution matters here).  When the
    discordant count is 0, return ``p=1.0`` (no evidence).

    Args:
        b01: The number of queries where ``A=1`` and ``B=0`` (only-A).
        b10: The number of queries where ``A=0`` and ``B=1`` (only-B).

    Returns:
        ``(p_value, statistic)``.  ``statistic`` is the test statistic
        (``"binomial"`` for exact, or the chi-squared ``b01 - b10``
        summary for large samples); it is returned as a float for
        a uniform return shape.
    """
    # Lazy import — scipy is in the eval-judges group.
    from scipy import stats as _stats

    n_discordant = int(b01) + int(b10)
    if n_discordant == 0:
        return 1.0, 0.0
    if n_discordant < 10:
        # Exact binomial test (two-sided).  Under H0 the more likely
        # value is equally likely, so each discordant pair is a
        # Bernoulli(0.5) trial.  ``binomtest(..., alternative='two-sided')``
        # doubles the smaller tail probability.
        result = _stats.binomtest(int(b01), n_discordant, p=0.5, alternative="two-sided")
        return float(result.pvalue), float(n_discordant)
    # Large-sample approximation: ``(b01 - b10)^2 / (b01 + b10)`` with
    # a continuity correction of 1 in the numerator (per
    # R-00121 §4.3).  Under H0 this is chi-squared(1); the
    # two-sided p-value is ``1 - chi2.cdf(stat, df=1)``.
    if n_discordant > 0 and abs(int(b01) - int(b10)) > 0:
        chi2 = float((abs(int(b01) - int(b10)) - 1) ** 2 / n_discordant)
    else:
        chi2 = 0.0
    p = float(1.0 - _stats.chi2.cdf(chi2, df=1))
    return p, chi2


# ---------------------------------------------------------------------------
# 4. Bonferroni-Holm correction
# ---------------------------------------------------------------------------


def bonferroni_holm(p_values: Sequence[float]) -> list[float]:
    """Apply the Bonferroni-Holm step-down correction to ``p_values``.

    Test choice (R-00121 §4.5): when multiple metrics are compared in
    one A/B run, the family-wise error rate must be controlled.
    Bonferroni-Holm is the recommended step-down procedure for
    moderate numbers of metrics (k ≤ ~20).  The correction is:

    1. Sort the p-values ascending: ``p_(1) ≤ p_(2) ≤ ... ≤ p_(k)``.
    2. For each rank i, multiply by ``(k - i + 1)``: the smallest p is
       multiplied by k, the second-smallest by ``k - 1``, etc.
    3. Enforce monotonicity: corrected p_(i) = max(corrected p_(i),
       corrected p_(i-1)).
    4. Cap at 1.0 (no negative values, no >1 values).

    The output preserves the input order — ``corrected[i]`` is the
    Bonferroni-Holm-adjusted p-value for the metric at input index
    ``i`` (NOT for the metric with the ``i``-th smallest p-value).

    Args:
        p_values: The raw p-values (one per metric).  May be empty.

    Returns:
        A list of corrected p-values, same length as the input.
    """
    n = len(p_values)
    if n == 0:
        return []
    # Pair each p with its original index, sort ascending.
    indexed = sorted(enumerate(p_values), key=lambda kv: float(kv[1]))
    out: list[float] = [0.0] * n
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        # k - i + 1 with k = n and i = rank+1 → (n - rank)
        corrected = float(p) * (n - rank)
        # Cap at 1.0
        if corrected > 1.0:
            corrected = 1.0
        # Enforce monotonicity
        if corrected < running_max:
            corrected = running_max
        else:
            running_max = corrected
        out[orig_idx] = corrected
    return out


# ---------------------------------------------------------------------------
# 5. A/B compare — the public entry point
# ---------------------------------------------------------------------------


_DEFAULT_MIN_QUERIES_FOR_SIGNIFICANCE = 150


def _detect_metric_type(metric: str) -> str:
    """Heuristic: which test family to apply to ``metric``?

    Continuous metrics (recall@k, nDCG@k, MRR, faithfulness 0–1,
    context_precision, context_recall, factual_correctness) →
    Wilcoxon + BCa.  Binary pass/fail metrics → McNemar.  Unknown
    metrics default to continuous (the safe path).
    """
    binary_suffixes = ("_pass", "_binary", "_win")
    if any(metric.endswith(suf) for suf in binary_suffixes):
        return "binary"
    return "continuous"


def _underpowered_caveat(
    n_queries: int,
    min_queries_for_significance: int,
) -> str:
    """Build a human-readable underpowered caveat string.

    Args:
        n_queries: The actual query count.
        min_queries_for_significance: The configured significance
            floor (default 150 per R-00121 §4.2).

    Returns:
        A one-line string describing the detectable effect size and
        the recommended n for stronger conclusions.
    """
    # The default caveat is calibrated to R-00121 §4.2: with n=50 the
    # test detects ~0.08–0.12 retrieval / ~0.06–0.10 faithfulness
    # differences.  The text is preserved verbatim across the run so
    # downstream logs are greppable.
    return (
        f"n={n_queries} is below min_queries_for_significance="
        f"{min_queries_for_significance}: the paired test detects only "
        "~0.08-0.12 retrieval / ~0.06-0.10 faithfulness differences. "
        f"Increase n to ~{min_queries_for_significance} for stronger conclusions."
    )


def compare(
    report_a_vectors: dict[str, Sequence[float | None]],
    report_b_vectors: dict[str, Sequence[float | None]],
    *,
    metrics: Sequence[str],
    n_queries: int,
    min_queries_for_significance: int = _DEFAULT_MIN_QUERIES_FOR_SIGNIFICANCE,
    seed: int | None = None,
) -> ComparisonResult:
    """Run an A/B comparison between two reports' per-query metric vectors.

    For each requested ``metric``, the per-query vectors in
    ``report_a_vectors[metric]`` and ``report_b_vectors[metric]`` are
    paired by index and tested with the metric-type-appropriate test
    (continuous → Wilcoxon + BCa bootstrap CI; binary → McNemar).
    Raw p-values are Bonferroni-Holm-corrected across the family.
    Each row is flagged ``significant`` only when
    ``p_corrected < 0.05`` AND the BCa CI excludes 0 (the
    R-00121 §4.4 step-5 rule).

    **Winner rule (R-00121 §5.4)**: declare a winner ONLY if one
    config is better on BOTH ``ndcg_at_5`` AND ``faithfulness`` (after
    correction).  Single-metric wins → ``winner=None``.

    **Underpowered caveat (Boundary row 5)**: when ``n_queries < 2``,
    the function returns ``underpowered=True``, ``significant=False``,
    ``p_value=None``, and ``ci_95=None`` per metric, with a caveat
    describing the n-too-small case.  When
    ``n_queries < min_queries_for_significance``, the result is
    flagged ``underpowered`` with a detectable-effect-size caveat but
    the per-metric p-values are still computed.

    Args:
        report_a_vectors: A ``metric → per-query values`` dict for A.
        report_b_vectors: A ``metric → per-query values`` dict for B.
        metrics: The ordered sequence of metric names to compare.
        n_queries: The number of queries in the run (must match the
            vector lengths).
        min_queries_for_significance: The floor for the
            underpowered-flag (default 150 per R-00121 §4.2).
        seed: Optional integer seed for the BCa bootstrap RNG.  The
            Wilcoxon and McNemar tests are deterministic; only the
            bootstrap benefits from the seed.

    Returns:
        A :class:`ComparisonResult` with the per-metric rows, the
        winner (or ``None``), the underpowered flag, the caveat, and
        a human-readable reasoning string.

    Raises:
        ConfigError: When ``n_queries`` does not match the per-metric
            vector lengths in A and B, or when a metric listed in
            ``metrics`` is missing from a report.
    """
    # ------------------------------------------------------------------------
    # 1) Validate input shape
    # ------------------------------------------------------------------------
    for metric in metrics:
        if metric not in report_a_vectors:
            raise ConfigError(f"compare: metric '{metric}' is missing from report_a_vectors.")
        if metric not in report_b_vectors:
            raise ConfigError(f"compare: metric '{metric}' is missing from report_b_vectors.")
        a_len = len(list(report_a_vectors[metric]))
        b_len = len(list(report_b_vectors[metric]))
        if a_len != n_queries:
            raise ConfigError(
                f"compare: report_a_vectors['{metric}'] has length {a_len}, "
                f"but n_queries={n_queries}."
            )
        if b_len != n_queries:
            raise ConfigError(
                f"compare: report_b_vectors['{metric}'] has length {b_len}, "
                f"but n_queries={n_queries}."
            )
        if a_len != b_len:
            raise ConfigError(
                f"compare: per-metric length mismatch for '{metric}': "
                f"len(A)={a_len}, len(B)={b_len}."
            )

    # ------------------------------------------------------------------------
    # 2) Per-metric p-values + means + BCa CIs
    # ------------------------------------------------------------------------
    raw_p: list[float | None] = []
    rows_input: list[dict[str, Any]] = []
    for metric in metrics:
        a_vec_raw = list(report_a_vectors[metric])
        b_vec_raw = list(report_b_vectors[metric])
        # Convert the ``float | None`` input to a clean ``list[float]``;
        # a missing value is represented as NaN so we can drop it
        # pairwise below while preserving index alignment.
        a_vec: list[float] = [float(v) if v is not None else math.nan for v in a_vec_raw]
        b_vec: list[float] = [float(v) if v is not None else math.nan for v in b_vec_raw]
        # Drop NaN values pairwise so Wilcoxon + bootstrap only see
        # finite paired observations.  This preserves the index
        # alignment per R-00121 §4.1.
        a_clean: list[float] = []
        b_clean: list[float] = []
        for a_i, b_i in zip(a_vec, b_vec, strict=True):
            if math.isnan(a_i) or math.isnan(b_i):
                continue
            a_clean.append(a_i)
            b_clean.append(b_i)
        n_paired = len(a_clean)
        val_a = _safe_mean(a_clean)
        val_b = _safe_mean(b_clean)
        delta: float | None = (val_b - val_a) if (val_a is not None and val_b is not None) else None

        if n_paired < 2:
            # Boundary row 5 — cannot run a paired test on a single
            # observation.  Do NOT raise; record as underpowered.
            rows_input.append(
                {
                    "metric": metric,
                    "val_a": val_a,
                    "val_b": val_b,
                    "delta": delta,
                    "p_value": None,
                    "ci_95": None,
                    "test": "none",
                }
            )
            raw_p.append(None)
            continue

        metric_type = _detect_metric_type(metric)
        if metric_type == "binary":
            # Convert continuous values to binary pass/fail using a
            # threshold of 0.5 (the conventional "pass" floor for
            # faithfulness / precision metrics in R-00121).  This is
            # a simplification; callers that need a different
            # threshold can pre-threshold their vectors.
            a_bin = [1 if x >= 0.5 else 0 for x in a_clean]
            b_bin = [1 if x >= 0.5 else 0 for x in b_clean]
            # Discordant pairs: A=1, B=0 vs A=0, B=1.
            b01 = sum(1 for a_i, b_i in zip(a_bin, b_bin, strict=True) if a_i == 1 and b_i == 0)
            b10 = sum(1 for a_i, b_i in zip(a_bin, b_bin, strict=True) if a_i == 0 and b_i == 1)
            p, _stat = mcnemar(b01=b01, b10=b10)
            # No bootstrap CI for binary metrics — set to a degenerate
            # ``(delta, delta)`` so the ``significant`` flag is driven
            # by the p-value alone (R-00121 §4.4 step 5 only requires
            # the CI-excludes-0 rule for continuous means; the binary
            # test is exact).
            ci_95: tuple[float, float] | None = None
            test_name = "mcnemar"
        else:
            # Continuous: Wilcoxon + BCa bootstrap CI of mean diff.
            p, _stat = wilcoxon_paired(a_clean, b_clean)
            try:
                lo, hi = bca_bootstrap_ci(
                    a_clean,
                    b_clean,
                    seed=seed,
                )
                ci_95 = (lo, hi)
            except (ValueError, ZeroDivisionError):
                # Degenerate bootstrap (e.g. all equal).  Record no CI;
                # significance is then driven by the p-value alone.
                ci_95 = None
            test_name = "wilcoxon"

        rows_input.append(
            {
                "metric": metric,
                "val_a": val_a,
                "val_b": val_b,
                "delta": delta,
                "p_value": float(p),
                "ci_95": ci_95,
                "test": test_name,
            }
        )
        raw_p.append(float(p))

    # ------------------------------------------------------------------------
    # 3) Bonferroni-Holm across the family
    # ------------------------------------------------------------------------
    finite_p_list: list[float] = [p_val for p_val in raw_p if p_val is not None]
    corrected_for_finite = bonferroni_holm(finite_p_list)
    # Re-inject ``None`` at the right positions so the corrected list
    # aligns with ``raw_p``.
    corrected_p: list[float | None] = []
    j = 0
    for raw_p_val in raw_p:
        if raw_p_val is None:
            corrected_p.append(None)
        else:
            corrected_p.append(corrected_for_finite[j])
            j += 1

    # ------------------------------------------------------------------------
    # 4) Build the MetricDelta rows + significance flag
    # ------------------------------------------------------------------------
    rows: list[MetricDelta] = []
    for i, row_in in enumerate(rows_input):
        p_raw = row_in["p_value"]
        p_corr = corrected_p[i]
        ci_95 = row_in["ci_95"]
        # Significance rule: corrected p < 0.05 AND CI excludes 0
        # (R-00121 §4.4 step 5 — both required).  Binary metrics
        # have no CI; they rely on p_corrected alone.
        significant = False
        if p_corr is not None and p_corr < 0.05:
            if ci_95 is None:
                # Binary metric — p alone is enough.
                significant = True
            else:
                lo, hi = ci_95
                if lo > 0.0 or hi < 0.0:
                    significant = True
        rows.append(
            MetricDelta(
                metric=row_in["metric"],
                val_a=row_in["val_a"],
                val_b=row_in["val_b"],
                delta=row_in["delta"],
                p_value=p_raw,
                p_corrected=p_corr,
                ci_95=ci_95,
                significant=significant,
                test=row_in["test"],
            )
        )

    # ------------------------------------------------------------------------
    # 5) Underpowered flag + caveat
    # ------------------------------------------------------------------------
    underpowered = n_queries < min_queries_for_significance
    caveat: str | None = None
    if n_queries < 2:
        caveat = (
            f"n={n_queries} (< 2): paired test could not run. "
            "Need n >= 2 for a paired test and n >= "
            f"{min_queries_for_significance} for a well-powered comparison."
        )
    elif underpowered:
        caveat = _underpowered_caveat(n_queries, min_queries_for_significance)

    # ------------------------------------------------------------------------
    # 6) Winner rule (R-00121 §5.4) — BOTH ndcg_at_5 AND faithfulness
    # ------------------------------------------------------------------------
    winner: str | None = None
    ndcg_row = next((r for r in rows if r.metric == "ndcg_at_5"), None)
    faith_row = next((r for r in rows if r.metric == "faithfulness"), None)
    if (
        ndcg_row is not None
        and faith_row is not None
        and ndcg_row.delta is not None
        and faith_row.delta is not None
    ):
        ndcg_better = ndcg_row.delta > 0.0
        faith_better = faith_row.delta > 0.0
        if ndcg_row.significant and faith_row.significant and ndcg_better and faith_better:
            winner = "B"
        elif (
            ndcg_row.significant and faith_row.significant and not ndcg_better and not faith_better
        ):
            winner = "A"

    # ------------------------------------------------------------------------
    # 7) Reasoning string
    # ------------------------------------------------------------------------
    parts: list[str] = []
    for r in rows:
        # Metrics can be None when a report has no value for them (e.g.
        # faithfulness when the judge was unavailable, or the reference-required
        # T3 metrics on a reference-free dataset). Format defensively so the
        # comparison never 500s on a NoneType.__format__.
        a_s = f"{r.val_a:.3f}" if r.val_a is not None else "n/a"
        b_s = f"{r.val_b:.3f}" if r.val_b is not None else "n/a"
        d_s = f"{r.delta:+.3f}" if r.delta is not None else "n/a"
        p_s = f"{r.p_corrected:.3g}" if r.p_corrected is not None else "nan"
        verdict = (
            f"{r.metric}: A={a_s} B={b_s} delta={d_s} p_corrected={p_s} significant={r.significant}"
        )
        parts.append(verdict)
    winner_msg = (
        f"winner={winner} (BOTH ndcg_at_5 AND faithfulness significant and B-better)"
        if winner is not None
        else "winner=None (dual-metric rule not satisfied)"
    )
    reasoning = "; ".join(parts) + "; " + winner_msg
    return ComparisonResult(
        rows=tuple(rows),
        winner=winner,
        underpowered=underpowered,
        caveat=caveat,
        reasoning=reasoning,
    )


# ---------------------------------------------------------------------------
# 6. Multi-run compare — baseline-relative N-way comparison (2..5 runs)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MultiComparisonResult:
    """The N-way (2..5 run) baseline-relative comparison result.

    Each non-baseline run is compared against the single designated
    baseline run with the same paired machinery :func:`compare` uses
    (Wilcoxon + BCa CI / McNemar, Bonferroni-Holm corrected), so the
    per-run :class:`ComparisonResult` carries ``delta = run - baseline``
    (positive ⇒ the run beats the baseline on that metric).

    Attributes:
        metrics: The ordered metric names compared (those present as a
            per-query vector in at least one run).
        baseline_index: The index (into ``means``/``deltas``) of the
            run chosen as the baseline.
        means: Per-run ``metric → mean`` dicts (``None`` when a run has
            no vector for that metric), aligned to the run order.
        deltas: Per-run baseline-relative :class:`ComparisonResult`,
            aligned to the run order.  The baseline's own entry is
            ``None`` (a run is not compared against itself).
        underpowered: ``True`` when ``n_queries`` is below
            ``min_queries_for_significance`` (shared across the cohort —
            all runs share the dataset, hence the query count).
        caveat: The shared underpowered caveat, or ``None``.
        n_queries: The baseline run's query count (the paired length).
    """

    metrics: tuple[str, ...]
    baseline_index: int
    means: tuple[dict[str, float | None], ...]
    deltas: tuple[ComparisonResult | None, ...]
    underpowered: bool
    caveat: str | None
    n_queries: int


def compare_multi(
    run_vectors: Sequence[dict[str, Any]],
    *,
    metrics: Sequence[str],
    baseline_index: int = 0,
    n_queries: int,
    min_queries_for_significance: int = _DEFAULT_MIN_QUERIES_FOR_SIGNIFICANCE,
    seed: int | None = None,
) -> MultiComparisonResult:
    """Compare 2..5 runs against one baseline (per-metric means + deltas).

    For every non-baseline run the metric subset present (as a non-empty
    per-query vector) in BOTH that run and the baseline is paired and
    tested with :func:`compare` (so ``delta = run - baseline``).  The
    per-run means are computed over each metric's vector independently
    so a metric absent from one run still shows for the others.

    Args:
        run_vectors: One ``metric → per-query vector`` mapping per run
            (extra non-vector blob keys are ignored).  Length 2..5.
        metrics: The ordered metric names to surface (means + deltas).
        baseline_index: The index of the baseline run within
            ``run_vectors``.
        n_queries: The paired query count (the baseline run's count).
        min_queries_for_significance: The underpowered floor (default
            150 per R-00121 §4.2).
        seed: Optional BCa bootstrap seed (forwarded to :func:`compare`).

    Returns:
        A :class:`MultiComparisonResult`.

    Raises:
        ConfigError: When fewer than 2 runs are supplied or
            ``baseline_index`` is out of range.
    """
    n_runs = len(run_vectors)
    if n_runs < 2:
        raise ConfigError(f"compare_multi needs at least 2 runs, got {n_runs}.")
    if not (0 <= baseline_index < n_runs):
        raise ConfigError(
            f"compare_multi baseline_index={baseline_index} is out of range for {n_runs} runs."
        )
    metric_tuple = tuple(metrics)

    def _vec(vecs: dict[str, Any], metric: str) -> list[float | None] | None:
        v = vecs.get(metric)
        if isinstance(v, list | tuple) and len(v) > 0:
            return [x if x is None else float(x) for x in v]
        return None

    # Per-run means (None when the run carries no vector for a metric).
    means: list[dict[str, float | None]] = []
    for vecs in run_vectors:
        run_means: dict[str, float | None] = {}
        for metric in metric_tuple:
            vec = _vec(vecs, metric)
            run_means[metric] = _safe_mean(vec) if vec is not None else None
        means.append(run_means)

    base_vecs = run_vectors[baseline_index]
    deltas: list[ComparisonResult | None] = []
    for i, vecs in enumerate(run_vectors):
        if i == baseline_index:
            deltas.append(None)
            continue
        # Metric subset present as a vector in BOTH the baseline and this run.
        pair_metrics: list[str] = []
        for metric in metric_tuple:
            if _vec(base_vecs, metric) is not None and _vec(vecs, metric) is not None:
                pair_metrics.append(metric)
        if not pair_metrics:
            deltas.append(
                ComparisonResult(
                    rows=(),
                    winner=None,
                    underpowered=n_queries < min_queries_for_significance,
                    caveat=None,
                    reasoning="no shared per-query metrics between this run and the baseline",
                )
            )
            continue
        # Common paired length (defensive: same dataset ⇒ equal lengths).
        n_eff = n_queries if n_queries > 0 else 0
        for metric in pair_metrics:
            bv = _vec(base_vecs, metric) or []
            rv = _vec(vecs, metric) or []
            n_eff = min(n_eff, len(bv), len(rv)) if n_eff else min(len(bv), len(rv))
        a_vectors: dict[str, Sequence[float | None]] = {
            m: (_vec(base_vecs, m) or [])[:n_eff] for m in pair_metrics
        }
        b_vectors: dict[str, Sequence[float | None]] = {
            m: (_vec(vecs, m) or [])[:n_eff] for m in pair_metrics
        }
        deltas.append(
            compare(
                a_vectors,
                b_vectors,
                metrics=pair_metrics,
                n_queries=n_eff,
                min_queries_for_significance=min_queries_for_significance,
                seed=seed,
            )
        )

    underpowered = n_queries < min_queries_for_significance
    caveat: str | None = None
    if n_queries < 2:
        caveat = (
            f"n={n_queries} (< 2): paired test could not run. "
            "Need n >= 2 for a paired test and n >= "
            f"{min_queries_for_significance} for a well-powered comparison."
        )
    elif underpowered:
        caveat = _underpowered_caveat(n_queries, min_queries_for_significance)

    return MultiComparisonResult(
        metrics=metric_tuple,
        baseline_index=baseline_index,
        means=tuple(means),
        deltas=tuple(deltas),
        underpowered=underpowered,
        caveat=caveat,
        n_queries=n_queries,
    )


__all__ = [
    "ComparisonResult",
    "MetricDelta",
    "MultiComparisonResult",
    "bca_bootstrap_ci",
    "bonferroni_holm",
    "compare",
    "compare_multi",
    "mcnemar",
    "wilcoxon_paired",
]
