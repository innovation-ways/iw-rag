"""Regression anchor for the live headline numbers (2026-06-08, demo-scale).

These are pure-Python assertions over the committed
``tests/eval/baselines/live_headline_2026-06-08.json`` snapshot: they bite if the
frozen numbers are weakened in a commit, and — when a DeepEval/CI job re-measures an
experiment on a machine with the live stack — they are the thresholds the re-measured
values are checked against. No DB, no GPU, no judge here.

Maps each headline experiment to the existing CI gate where one applies
(``ci_gate.check_recall_floor`` / ``check_faithfulness_no_regress``) and to a plain
frozen-value assertion otherwise (κ floor, red-team ASR ceiling, the Phase-4 gate
outcomes). Full reports: ``eval/reports/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from iw_rag.eval.ci_gate import check_faithfulness_no_regress, check_recall_floor, load_baseline

_LIVE = Path(__file__).resolve().parent / "baselines" / "live_headline_2026-06-08.json"
_PHASE0 = Path(__file__).resolve().parent / "baselines" / "phase0.json"


def _live() -> dict:
    return json.loads(_LIVE.read_text())


def test_faithfulness_meets_the_gate() -> None:
    """Live faithfulness (1.000 demo + benchmark) passes the no-regression gate."""
    live = _live()
    baseline = load_baseline(_PHASE0)
    for corpus in ("demo", "benchmark"):
        observed_val = live["exp3_faithfulness"][corpus]
        result = check_faithfulness_no_regress(observed_val, baseline)
        assert result.passed, f"faithfulness[{corpus}] regressed: {result.message}"
        assert result.metric == "faithfulness"
        # Live value is 1.000 for both corpora; pin it so any weakening is caught.
        assert result.observed == pytest.approx(1.000)
        # threshold = baseline.faithfulness - tolerance = 1.0 - 0.03 = 0.97
        assert result.threshold == pytest.approx(
            baseline.faithfulness - baseline.faithfulness_regress_tolerance
        )


def test_demo_recall_passes_the_production_floor() -> None:
    """The demo-corpus retrieval result clears the 0.80 production floor."""
    live = _live()
    baseline = load_baseline(_PHASE0)
    r10 = live["exp1_retrieval"]["demo_corpus"]["recall_at_10"]
    result = check_recall_floor(r10, baseline)
    assert result.passed
    assert result.metric == "recall_at_10"
    # Live demo-corpus recall@10 is 1.000; pin it so any weakening is caught.
    assert result.observed == pytest.approx(1.000)
    # threshold = max(0.80, baseline.recall_at_10=0.80) = 0.80
    assert result.threshold == pytest.approx(0.80)


def test_benchmark_retrieval_lift_is_monotonic() -> None:
    """On the hard benchmark corpus the three-stage spine must keep its measured
    dense → hybrid → +reranker ordering (the reason the spine exists)."""
    bench = _live()["exp1_retrieval"]["benchmark_corpus"]
    # Frozen production numbers on the 108-doc corpus; the rerank arm must not drop
    # below the recorded recall (~0.78) / nDCG (~0.79).
    assert bench["recall_at_10"] >= 0.77
    assert bench["ndcg_at_10"] >= 0.78


def test_kappa_clears_the_floor() -> None:
    """Judge–human agreement stays at substantial-or-better (κ ≥ floor)."""
    live = _live()
    assert live["kappa_faithfulness"]["cohens_kappa"] >= live["kappa_floor"]


def test_redteam_asr_at_or_below_ceiling() -> None:
    """The standardized red-team sweep must not let any attack through (ASR ≤ 0)."""
    live = _live()
    assert live["exp5_redteam"]["garak_corrected_asr"] <= live["redteam_asr_ceiling"]
    assert live["exp5_redteam"]["acl_leak_rate"] == 0.0  # ACL leak is the non-negotiable zero


def test_phase4_gate_outcomes_frozen() -> None:
    """The measured Phase-4 verdicts are the frozen record (#6 off, #7 tie, #8 win)."""
    live = _live()
    assert live["exp6_graphrag"]["gate_passed"] is False  # graph route stays off by config
    assert live["exp7_visual"]["colvision_ndcg_at_5"] == 1.0
    assert live["exp8_longcontext"]["lc_needle_recovered"] is True
    assert live["exp8_longcontext"]["rag_needle_recovered"] is False
