"""Unit tests for the statistical comparison module (eval/benchmark/stats.py).

TDD RED-first: these tests are written before the stats implementation
(S07) and exercise its public contract per F-00093 Feature_Design
§Scope item 6 / AC3 / Boundary Behavior row 5 (n<2 for stats) and
R-00121 §4 (test selection by metric type, paired Wilcoxon, BCa
bootstrap CI, McNemar, Bonferroni-Holm, dual-metric winner rule,
underpowered caveat).

The tests are pure-Python (no DB, no network, no LLM) and use
deterministic fixtures.  All randomness is seeded so the BCa bootstrap
is reproducible across runs.
"""

from __future__ import annotations

import asyncio

import pytest

# All statistical tests in this file exercise ``iw_rag.eval.benchmark.stats``
# functions that lazy-import ``scipy`` (which lives in the optional
# ``eval-judges`` dep group — not the default dev venv).  Skip the entire
# module at collection time when scipy is absent so ``make test`` stays green
# in the default dev environment; the suite still runs (and tests the actual
# statistical behaviour) when invoked with the eval-judges group installed
# (e.g. ``uv run --group eval-judges pytest tests/unit/eval/benchmark/test_stats.py``).
pytest.importorskip("scipy")

# ---------------------------------------------------------------------------
# 1. Module surface
# ---------------------------------------------------------------------------


def test_stats_module_exposes_public_api() -> None:
    """``stats.py`` MUST expose the public API: ``MetricDelta``,
    ``ComparisonResult``, ``compare``, ``wilcoxon_paired``,
    ``bca_bootstrap_ci``, ``mcnemar``, ``bonferroni_holm``.
    """
    from iw_rag.eval.benchmark import stats as stats_mod

    for name in (
        "MetricDelta",
        "ComparisonResult",
        "compare",
        "wilcoxon_paired",
        "bca_bootstrap_ci",
        "mcnemar",
        "bonferroni_holm",
    ):
        assert hasattr(stats_mod, name), f"missing public symbol: {name}"


# ---------------------------------------------------------------------------
# 2. Paired Wilcoxon signed-rank (R-00121 §4.1)
# ---------------------------------------------------------------------------


def test_wilcoxon_paired_significant_difference() -> None:
    """A hand-constructed pair with a clear positive shift returns
    ``p < 0.05``.
    """
    from iw_rag.eval.benchmark.stats import wilcoxon_paired

    # 20 paired observations; A is 1.0 lower than B on every sample.
    a = [float(i) for i in range(1, 21)]
    b = [float(i) + 1.0 for i in range(1, 21)]
    p, stat = wilcoxon_paired(a, b)
    assert p < 0.001
    assert stat is not None


def test_wilcoxon_paired_no_difference() -> None:
    """A hand-constructed pair with no difference returns ``p > 0.05``."""
    from iw_rag.eval.benchmark.stats import wilcoxon_paired

    a = [0.50, 0.51, 0.49, 0.52, 0.50, 0.51, 0.49, 0.50, 0.52, 0.51]
    b = [0.50, 0.50, 0.50, 0.51, 0.50, 0.50, 0.49, 0.51, 0.50, 0.50]
    p, _ = wilcoxon_paired(a, b)
    assert p > 0.05


def test_wilcoxon_paired_handles_tied_zeros() -> None:
    """All-differences-zero is a degenerate case — return ``p=1.0`` and
    a stat of ``0.0`` rather than raising (per R-00121 §4.1 footnote
    on tied zeros)."""
    from iw_rag.eval.benchmark.stats import wilcoxon_paired

    a = [0.5, 0.6, 0.7, 0.4, 0.5]
    b = [0.5, 0.6, 0.7, 0.4, 0.5]
    p, stat = wilcoxon_paired(a, b)
    assert p == 1.0
    assert stat == 0.0


# ---------------------------------------------------------------------------
# 3. BCa bootstrap CI (R-00121 §4.4 step 5) — paired, seeded, deterministic
# ---------------------------------------------------------------------------


def test_bca_bootstrap_ci_seeded_returns_stable_bounds() -> None:
    """A BCa bootstrap CI with a fixed seed returns the same bounds on
    repeated calls (determinism requirement)."""
    from iw_rag.eval.benchmark.stats import bca_bootstrap_ci

    a = [
        0.40,
        0.42,
        0.45,
        0.50,
        0.55,
        0.60,
        0.62,
        0.65,
        0.70,
        0.75,
        0.78,
        0.80,
        0.82,
        0.84,
        0.86,
        0.88,
        0.90,
        0.92,
        0.94,
        0.96,
    ]
    # Heteroscedastic per-query differences so the bootstrap has
    # non-degenerate variance (a constant +0.05 shift gives zero
    # variance, which collapses the CI to a point).
    b = [a[i] + 0.05 + 0.01 * ((-1) ** i) for i in range(len(a))]
    lo1, hi1 = bca_bootstrap_ci(a, b, seed=42, n_resamples=2000)
    lo2, hi2 = bca_bootstrap_ci(a, b, seed=42, n_resamples=2000)
    assert lo1 == pytest.approx(lo2, abs=1e-12)
    assert hi1 == pytest.approx(hi2, abs=1e-12)
    # Mean_B - Mean_A = +0.05 → CI straddles that value.
    assert lo1 < 0.05 < hi1


def test_bca_bootstrap_ci_excludes_zero_for_clear_difference() -> None:
    """A clear +0.10 shift on 50 paired samples produces a CI that
    excludes 0."""
    from iw_rag.eval.benchmark.stats import bca_bootstrap_ci

    a = [0.10 + 0.01 * i for i in range(50)]
    b = [a[i] + 0.10 for i in range(50)]
    lo, hi = bca_bootstrap_ci(a, b, seed=123, n_resamples=2000)
    assert lo > 0.0
    assert hi > 0.0


def test_bca_bootstrap_ci_default_n_resamples_is_10000() -> None:
    """The default ``n_resamples`` is 10,000 per the S07 prompt and
    R-00121 §4.4 step 5."""
    from iw_rag.eval.benchmark.stats import bca_bootstrap_ci

    a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    b = [x + 0.05 for x in a]
    # Just smoke-test that the default runs without error at 10k.
    lo, hi = bca_bootstrap_ci(a, b, seed=7)
    assert lo is not None and hi is not None
    assert hi >= lo


# ---------------------------------------------------------------------------
# 4. McNemar's test (R-00121 §4.3) — binary metrics
# ---------------------------------------------------------------------------


def test_mcnemar_significant_for_clear_asymmetry() -> None:
    """When the off-diagonal cells of a 2x2 are very asymmetric, McNemar
    reports a small p-value."""
    from iw_rag.eval.benchmark.stats import mcnemar

    # Discordant: A=1/B=0 → 30; A=0/B=1 → 5
    p, stat = mcnemar(b01=30, b10=5)
    assert p < 0.001
    assert stat is not None


def test_mcnemar_uses_exact_binomial_for_low_discordant_count() -> None:
    """When the discordant count is < 10 the function MUST use the
    exact binomial test, not the chi-squared approximation."""
    from iw_rag.eval.benchmark.stats import mcnemar

    # 8 + 0 → exact binomial, not chi-squared
    p_exact, _ = mcnemar(b01=8, b10=0)
    # The exact two-sided binomial p of observing 8 successes out of 8
    # under p=0.5 is 2 * (1/2)^8 = 1/128 ≈ 0.0078.
    assert p_exact == pytest.approx(0.0078125, abs=1e-6)


def test_mcnemar_no_discordant_returns_p_one() -> None:
    """Zero discordant pairs → p=1.0 (no evidence of asymmetry)."""
    from iw_rag.eval.benchmark.stats import mcnemar

    p, _ = mcnemar(b01=0, b10=0)
    assert p == 1.0


# ---------------------------------------------------------------------------
# 5. Bonferroni-Holm correction (R-00121 §4.5)
# ---------------------------------------------------------------------------


def test_bonferroni_holm_rejects_some_keeps_others() -> None:
    """Bonferroni-Holm MUST reorder raw p-values and inflate them such
    that only the truly small ones survive α=0.05."""
    from iw_rag.eval.benchmark.stats import bonferroni_holm

    raw = [0.0001, 0.005, 0.02, 0.04]
    corrected = bonferroni_holm(raw)
    # Sorted raw: 0.0001, 0.005, 0.02, 0.04
    # Holm-corrected (k=4):
    #   0.0001 * 4 = 0.0004
    #   0.005 * 3 = 0.015
    #   0.02 * 2 = 0.04
    #   0.04 * 1 = 0.04 (capped at 1)
    # Round-trip ordering — the result keeps the input order
    assert corrected[0] == pytest.approx(0.0004, abs=1e-8)
    assert corrected[1] == pytest.approx(0.015, abs=1e-6)
    assert corrected[2] == pytest.approx(0.04, abs=1e-8)
    assert corrected[3] == pytest.approx(0.04, abs=1e-8)


def test_bonferroni_holm_caps_at_one() -> None:
    """Corrected p-values are capped at 1.0 (no negative or >1)."""
    from iw_rag.eval.benchmark.stats import bonferroni_holm

    raw = [0.5, 0.7]
    corrected = bonferroni_holm(raw)
    assert all(0.0 <= p <= 1.0 for p in corrected)


def test_bonferroni_holm_empty_input() -> None:
    """An empty list returns an empty list (no crash)."""
    from iw_rag.eval.benchmark.stats import bonferroni_holm

    assert bonferroni_holm([]) == []


# ---------------------------------------------------------------------------
# 6. A/B compare — AC3 (significant on nDCG@5, not on faithfulness, no winner)
# ---------------------------------------------------------------------------


def test_compare_ac3_significant_ndcg5_not_faithfulness_no_winner() -> None:
    """AC3: A pair where nDCG@5 differs significantly but faithfulness
    does not → ``compare`` marks nDCG@5 significant, faithfulness not,
    and ``winner is None`` (the dual-metric rule fails).

    Hand-constructed fixture: a known significant nDCG@5 difference and
    a deliberately noisy faithfulness comparison.
    """
    from iw_rag.eval.benchmark.stats import compare

    # 30 paired observations on nDCG@5 with a clear +0.12 shift.
    ndcg_a = [0.40 + 0.01 * i for i in range(30)]
    ndcg_b = [x + 0.12 for x in ndcg_a]
    # 30 paired observations on faithfulness with no real shift (noise only).
    faith_a = [
        0.70,
        0.72,
        0.68,
        0.71,
        0.69,
        0.70,
        0.72,
        0.68,
        0.71,
        0.69,
        0.70,
        0.71,
        0.69,
        0.70,
        0.72,
        0.68,
        0.71,
        0.69,
        0.70,
        0.72,
        0.68,
        0.71,
        0.69,
        0.70,
        0.72,
        0.68,
        0.71,
        0.69,
        0.70,
        0.71,
    ]
    faith_b = [
        0.70,
        0.72,
        0.69,
        0.70,
        0.70,
        0.71,
        0.71,
        0.69,
        0.70,
        0.70,
        0.71,
        0.70,
        0.70,
        0.71,
        0.71,
        0.69,
        0.70,
        0.70,
        0.71,
        0.71,
        0.69,
        0.70,
        0.70,
        0.71,
        0.71,
        0.69,
        0.70,
        0.70,
        0.71,
        0.70,
    ]
    report_a = {
        "ndcg_at_5": ndcg_a,
        "faithfulness": faith_a,
    }
    report_b = {
        "ndcg_at_5": ndcg_b,
        "faithfulness": faith_b,
    }
    res = compare(
        report_a_vectors=report_a,
        report_b_vectors=report_b,
        metrics=("ndcg_at_5", "faithfulness"),
        n_queries=30,
        seed=42,
    )
    # nDCG@5 is significant; faithfulness is not.
    ndcg_row = next(r for r in res.rows if r.metric == "ndcg_at_5")
    faith_row = next(r for r in res.rows if r.metric == "faithfulness")
    assert ndcg_row.significant is True
    assert ndcg_row.p_corrected < 0.05
    assert ndcg_row.ci_95[0] > 0.0
    assert faith_row.significant is False
    # The dual-metric rule fails → no winner.
    assert res.winner is None
    assert "ndcg_at_5" in (res.reasoning or "")
    assert "faithfulness" in (res.reasoning or "")


# ---------------------------------------------------------------------------
# 7. A/B compare — dual-metric winner (B better on BOTH nDCG@5 AND faithfulness)
# ---------------------------------------------------------------------------


def test_compare_dual_metric_winner_when_better_on_both() -> None:
    """When one config is better on BOTH nDCG@5 AND faithfulness, the
    winner is that config (R-00121 §5.4)."""
    from iw_rag.eval.benchmark.stats import compare

    # 30 paired observations; B better on both metrics by ~0.10.
    ndcg_a = [0.40 + 0.01 * i for i in range(30)]
    ndcg_b = [x + 0.10 for x in ndcg_a]
    faith_a = [0.60 + 0.005 * i for i in range(30)]
    faith_b = [x + 0.10 for x in faith_a]
    report_a = {"ndcg_at_5": ndcg_a, "faithfulness": faith_a}
    report_b = {"ndcg_at_5": ndcg_b, "faithfulness": faith_b}
    res = compare(
        report_a_vectors=report_a,
        report_b_vectors=report_b,
        metrics=("ndcg_at_5", "faithfulness"),
        n_queries=30,
        seed=7,
    )
    ndcg_row = next(r for r in res.rows if r.metric == "ndcg_at_5")
    faith_row = next(r for r in res.rows if r.metric == "faithfulness")
    assert ndcg_row.significant is True
    assert faith_row.significant is True
    # Winner is B (val_b > val_a on BOTH metrics).
    assert res.winner == "B"
    assert "ndcg_at_5" in res.reasoning
    assert "faithfulness" in res.reasoning


# ---------------------------------------------------------------------------
# 8. A/B compare — n<2 must not crash (Boundary Behavior row 5)
# ---------------------------------------------------------------------------


def test_compare_n_lt_2_returns_underpowered_no_crash() -> None:
    """When ``n_queries < 2``, ``compare`` MUST return a result with
    ``underpowered=True`` and ``significant=False`` per query — it must
    NOT raise."""
    from iw_rag.eval.benchmark.stats import compare

    report_a = {"ndcg_at_5": [0.5]}
    report_b = {"ndcg_at_5": [0.7]}
    res = compare(
        report_a_vectors=report_a,
        report_b_vectors=report_b,
        metrics=("ndcg_at_5",),
        n_queries=1,
        seed=0,
    )
    assert res.underpowered is True
    assert res.winner is None
    row = res.rows[0]
    assert row.significant is False
    # p_value is None when the test could not run.
    assert row.p_value is None
    assert row.ci_95 is None
    # Caveat carries a human-readable message.
    assert res.caveat is not None
    assert "n=" in res.caveat


# ---------------------------------------------------------------------------
# 9. A/B compare — n<min_queries_for_significance → underpowered caveat
# ---------------------------------------------------------------------------


def test_compare_underpowered_when_below_min_queries() -> None:
    """When ``n_queries < min_queries_for_significance`` (default ~150
    per R-00121 §4.2), the result is flagged underpowered but rows
    still produce a p-value (just with a caveat)."""
    from iw_rag.eval.benchmark.stats import compare

    n = 25
    ndcg_a = [0.40 + 0.01 * i for i in range(n)]
    ndcg_b = [x + 0.10 for x in ndcg_a]
    res = compare(
        report_a_vectors={"ndcg_at_5": ndcg_a},
        report_b_vectors={"ndcg_at_5": ndcg_b},
        metrics=("ndcg_at_5",),
        n_queries=n,
        min_queries_for_significance=150,
        seed=1,
    )
    assert res.underpowered is True
    assert res.caveat is not None
    assert "n=25" in res.caveat


# ---------------------------------------------------------------------------
# 10. A/B compare — length-mismatch between A and B raises ConfigError
# ---------------------------------------------------------------------------


def test_compare_length_mismatch_raises() -> None:
    """When report_a and report_b have a metric of different lengths
    (queries don't align), ``compare`` MUST raise ``ConfigError``."""
    from iw_rag.domain.errors import ConfigError
    from iw_rag.eval.benchmark.stats import compare

    with pytest.raises(ConfigError, match="compare: report_b_vectors"):
        compare(
            report_a_vectors={"ndcg_at_5": [0.4, 0.5, 0.6]},
            report_b_vectors={"ndcg_at_5": [0.5, 0.6]},
            metrics=("ndcg_at_5",),
            n_queries=3,
        )


# ---------------------------------------------------------------------------
# 11. A/B compare — missing metric in one report raises ConfigError
# ---------------------------------------------------------------------------


def test_compare_missing_metric_raises() -> None:
    """When a metric requested in ``metrics`` is missing from a report,
    ``compare`` MUST raise ``ConfigError``."""
    from iw_rag.domain.errors import ConfigError
    from iw_rag.eval.benchmark.stats import compare

    with pytest.raises(ConfigError, match=r"metric 'ndcg_at_5' is missing from report_b_vectors"):
        compare(
            report_a_vectors={"ndcg_at_5": [0.4, 0.5, 0.6]},
            report_b_vectors={"faithfulness": [0.5, 0.6, 0.7]},
            metrics=("ndcg_at_5",),
            n_queries=3,
        )


# ---------------------------------------------------------------------------
# 12. A/B compare — return shape (rows, winner, underpowered, caveat)
# ---------------------------------------------------------------------------


def test_compare_returns_frozen_dataclasses() -> None:
    """``compare`` MUST return ``MetricDelta`` and ``ComparisonResult``
    frozen dataclasses."""
    from dataclasses import FrozenInstanceError

    from iw_rag.eval.benchmark.stats import ComparisonResult, MetricDelta, compare

    res = compare(
        report_a_vectors={"ndcg_at_5": [0.4]},
        report_b_vectors={"ndcg_at_5": [0.5]},
        metrics=("ndcg_at_5",),
        n_queries=1,
    )
    assert isinstance(res, ComparisonResult)
    assert len(res.rows) == 1
    assert res.rows[0].metric == "ndcg_at_5"
    assert isinstance(res.rows[0], MetricDelta)
    # Frozen — assignment must raise FrozenInstanceError.
    with pytest.raises(FrozenInstanceError, match="cannot assign to field"):
        res.rows[0].metric = "x"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 13. Helper — public symbols of ``stats.py``
# ---------------------------------------------------------------------------


def test_stats_module_dunder_all() -> None:
    """``stats.py`` MUST declare ``__all__`` covering the public API."""
    from iw_rag.eval.benchmark import stats as stats_mod

    assert hasattr(stats_mod, "__all__")
    for name in (
        "MetricDelta",
        "ComparisonResult",
        "compare",
        "wilcoxon_paired",
        "bca_bootstrap_ci",
        "mcnemar",
        "bonferroni_holm",
    ):
        assert name in stats_mod.__all__


# ---------------------------------------------------------------------------
# 14. Sanity — Wilcoxon requires equal-length paired inputs
# ---------------------------------------------------------------------------


def test_wilcoxon_paired_length_mismatch_raises() -> None:
    """``wilcoxon_paired`` MUST raise when A and B have different lengths."""
    from iw_rag.domain.errors import ConfigError
    from iw_rag.eval.benchmark.stats import wilcoxon_paired

    with pytest.raises(ConfigError, match=r"wilcoxon_paired: length mismatch"):
        wilcoxon_paired([0.4, 0.5, 0.6], [0.5, 0.6])


# ---------------------------------------------------------------------------
# 15. BCa on degenerate input → NaN; ``compare``'s significance rule
# must treat NaN CI as "not significant" (R-00121 §4.4 step 5).
# S08 review Finding 1: scipy BCa returns (NaN, NaN) on degenerate input;
# the contract is "significant=False when the CI is unusable", and this
# test pins that contract end-to-end (so a future refactor cannot
# regress to "NaN CI → significant=True").
# ---------------------------------------------------------------------------


def test_compare_degenerate_bca_yields_not_significant() -> None:
    """When ``bca_bootstrap_ci`` returns ``(nan, nan)`` (constant input
    or zero variance), ``compare`` MUST mark the row ``significant=False``
    — the ``nan > 0`` and ``nan < 0`` checks must evaluate to ``False``.

    Without this guarantee, a downstream renderer would silently
    surface NaN as a "significant" verdict.
    """
    import math

    from iw_rag.eval.benchmark.stats import compare

    # 30 paired observations; both A and B are constant → BCa returns NaN.
    n = 30
    a = [0.5] * n
    b = [0.5] * n
    res = compare(
        report_a_vectors={"ndcg_at_5": a},
        report_b_vectors={"ndcg_at_5": b},
        metrics=("ndcg_at_5",),
        n_queries=n,
        seed=1,
    )
    row = res.rows[0]
    # The CI is NaN; the significance rule MUST short-circuit to False.
    assert row.ci_95 is not None, "BCa returned a CI shape, not None"
    assert math.isnan(row.ci_95[0])
    assert math.isnan(row.ci_95[1])
    assert row.significant is False, (
        "NaN CI must not flip significant to True (R-00121 §4.4 step 5 rule)"
    )
    # Winner is None — a single-metric NaN row never wins.
    assert res.winner is None


# ---------------------------------------------------------------------------
# 16. The assembly's ``created_by=None`` path is preserved end-to-end
# (the offline batch path has no authenticated principal).
# Mirrors the integration test for the same contract.
# ---------------------------------------------------------------------------


def test_assembly_created_by_none_round_trips() -> None:
    """``assemble_report`` MUST pass a ``None`` ``created_by`` through to
    the row payload verbatim (the schema marks the column nullable; the
    offline batch path uses this for runs without an authenticated
    principal).  Mirrors the integration test
    ``test_created_by_accepts_none_for_offline_batch_path``."""
    from iw_rag.eval.benchmark.assembly import assemble_report
    from tests.unit.eval.benchmark.test_assembly import (  # type: ignore[import-not-found]
        FakeBenchmarkReportRepository,
        _bm_full,
        _gpu,
        _model_load,
        _per_stage_ms,
        _vllm,
    )

    async def _run() -> None:
        repo = FakeBenchmarkReportRepository()
        await assemble_report(
            repository=repo,  # type: ignore[arg-type]
            run_id="run-1",
            dataset_id="ds-1",
            report_label="offline-batch",
            git_sha=None,
            created_by=None,  # the offline batch path
            model_config={},
            metrics=_bm_full(),
            per_stage_ms=_per_stage_ms(),
            vllm_metrics=_vllm(),
            gpu_metrics=_gpu(),
            model_load=_model_load(),
            ci_95={},
        )
        assert repo.last_create_kwargs is not None
        assert repo.last_create_kwargs["created_by"] is None

    asyncio.run(_run())
