"""Unit tests for ``stats.compare_multi`` — the N-way baseline-relative compare.

``compare_multi`` extends the paired :func:`stats.compare` machinery from
2 runs to 2..5: it computes per-run per-metric means and, for every
non-baseline run, a baseline-relative :class:`ComparisonResult`
(``delta = run - baseline``).  These tests pin:

* the means shape (per run, ``None`` for a metric a run lacks);
* the baseline entry is ``None`` (a run is not compared with itself);
* the delta sign convention (run − baseline);
* the per-pair metric subset (only metrics present in BOTH the run and
  the baseline are tested);
* the run-count + baseline-index guards;
* the shared underpowered flag/caveat.
"""

from __future__ import annotations

import pytest

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.benchmark import stats


def test_compare_multi_requires_at_least_two_runs() -> None:
    """Fewer than 2 runs raises ``ConfigError``."""
    with pytest.raises(ConfigError, match=r"at least 2 runs"):
        stats.compare_multi([{"ndcg_at_5": [0.1, 0.2]}], metrics=["ndcg_at_5"], n_queries=2)


def test_compare_multi_baseline_index_out_of_range_raises() -> None:
    """An out-of-range ``baseline_index`` raises ``ConfigError``."""
    runs = [{"ndcg_at_5": [0.1, 0.2]}, {"ndcg_at_5": [0.3, 0.4]}]
    with pytest.raises(ConfigError, match=r"baseline_index"):
        stats.compare_multi(runs, metrics=["ndcg_at_5"], baseline_index=5, n_queries=2)


def test_compare_multi_means_and_absent_metric_is_none() -> None:
    """Per-run means are computed; a metric absent from a run is ``None``."""
    runs = [
        {"ndcg_at_5": [0.4, 0.6], "recall_at_5": [0.2, 0.4]},
        {"ndcg_at_5": [0.6, 0.8]},  # no recall vector
    ]
    res = stats.compare_multi(
        runs, metrics=["ndcg_at_5", "recall_at_5"], baseline_index=0, n_queries=2
    )
    assert res.means[0]["ndcg_at_5"] == pytest.approx(0.5)
    assert res.means[0]["recall_at_5"] == pytest.approx(0.3)
    assert res.means[1]["ndcg_at_5"] == pytest.approx(0.7)
    assert res.means[1]["recall_at_5"] is None


def test_compare_multi_baseline_delta_is_none() -> None:
    """The baseline run's own delta entry is ``None`` (not self-compared)."""
    runs = [
        {"ndcg_at_5": [0.4, 0.6, 0.5]},
        {"ndcg_at_5": [0.6, 0.8, 0.7]},
    ]
    res = stats.compare_multi(runs, metrics=["ndcg_at_5"], baseline_index=0, n_queries=3)
    assert res.deltas[0] is None
    assert res.deltas[1] is not None


def test_compare_multi_delta_sign_is_run_minus_baseline() -> None:
    """``delta = run - baseline`` (positive ⇒ the run beats the baseline)."""
    pytest.importorskip("scipy", reason="scipy required for stats.compare")
    runs = [
        {"ndcg_at_5": [0.4, 0.5, 0.6]},  # baseline mean 0.5
        {"ndcg_at_5": [0.6, 0.7, 0.8]},  # run mean 0.7 → +0.2
        {"ndcg_at_5": [0.2, 0.3, 0.4]},  # run mean 0.3 → −0.2
    ]
    res = stats.compare_multi(runs, metrics=["ndcg_at_5"], baseline_index=0, n_queries=3)
    up = next(r for r in res.deltas[1].rows if r.metric == "ndcg_at_5")
    down = next(r for r in res.deltas[2].rows if r.metric == "ndcg_at_5")
    assert up.delta == pytest.approx(0.2)
    assert down.delta == pytest.approx(-0.2)


def test_compare_multi_pairs_only_shared_metrics() -> None:
    """A run is compared on metrics present in BOTH it and the baseline."""
    pytest.importorskip("scipy", reason="scipy required for stats.compare")
    runs = [
        {"ndcg_at_5": [0.4, 0.5], "faithfulness": [0.8, 0.9]},
        {"ndcg_at_5": [0.5, 0.6]},  # no faithfulness vector
    ]
    res = stats.compare_multi(
        runs, metrics=["ndcg_at_5", "faithfulness"], baseline_index=0, n_queries=2
    )
    metrics_tested = [row.metric for row in res.deltas[1].rows]
    assert "ndcg_at_5" in metrics_tested
    assert "faithfulness" not in metrics_tested


def test_compare_multi_no_shared_metrics_yields_empty_rows() -> None:
    """A run sharing no metric with the baseline yields an empty delta (not a crash)."""
    runs = [
        {"ndcg_at_5": [0.4, 0.5]},
        {"recall_at_5": [0.5, 0.6]},  # disjoint metric set
    ]
    res = stats.compare_multi(
        runs, metrics=["ndcg_at_5", "recall_at_5"], baseline_index=0, n_queries=2
    )
    assert res.deltas[1] is not None
    assert res.deltas[1].rows == ()


def test_compare_multi_underpowered_flag_and_caveat_for_small_n() -> None:
    """Small n flags ``underpowered`` and emits the detectable-effect caveat."""
    pytest.importorskip("scipy", reason="scipy required for stats.compare")
    runs = [
        {"ndcg_at_5": [0.4, 0.5, 0.6]},
        {"ndcg_at_5": [0.5, 0.6, 0.7]},
    ]
    res = stats.compare_multi(runs, metrics=["ndcg_at_5"], baseline_index=0, n_queries=3)
    assert res.underpowered is True
    assert res.caveat is not None
    assert "150" in res.caveat


def test_compare_multi_respects_explicit_baseline_index() -> None:
    """A non-zero ``baseline_index`` makes that run the reference."""
    runs = [
        {"ndcg_at_5": [0.4, 0.5, 0.6]},
        {"ndcg_at_5": [0.6, 0.7, 0.8]},
        {"ndcg_at_5": [0.2, 0.3, 0.4]},
    ]
    res = stats.compare_multi(runs, metrics=["ndcg_at_5"], baseline_index=1, n_queries=3)
    assert res.baseline_index == 1
    assert res.deltas[1] is None
    assert res.deltas[0] is not None and res.deltas[2] is not None


def test_compare_multi_exposes_public_symbol() -> None:
    """``compare_multi`` + ``MultiComparisonResult`` are part of the module surface."""
    assert "compare_multi" in stats.__all__
    assert "MultiComparisonResult" in stats.__all__
    # Neither symbol is accidentally absent — confirm exact membership count for multi symbols.
    multi_symbols = [s for s in stats.__all__ if "multi" in s.lower() or "Multi" in s]
    assert len(multi_symbols) == 2
