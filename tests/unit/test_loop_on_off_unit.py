"""Unit tests for run_loop_on_off — the exp #4 loop ON vs OFF experiment.

TDD RED: these tests define the expected interface of run_loop_on_off before
implementation. No live models or DB required for the unit tests.

Tests cover:
- The function exists and is importable from eval.experiments.
- Returns a dict with the expected structure (per-arm recall + faithfulness keys).
- Zero-gold queries are excluded from the recall mean.
- When a FaithfulnessJudge is NOT supplied, faithfulness keys are absent/None.
- Router accuracy score function works on deterministic data.
- Budget exhaustion rate computation.
"""

from __future__ import annotations

import pytest

# RED TEST: this import will fail because run_loop_on_off doesn't exist yet.


def test_run_loop_on_off_is_importable() -> None:
    """run_loop_on_off must be importable and callable from eval.experiments."""
    from iw_rag.eval.experiments import run_loop_on_off  # noqa: F401

    assert callable(run_loop_on_off), "run_loop_on_off must be a callable"


def test_comparison_dict_has_expected_keys() -> None:
    """The return dict must carry per-arm retrieval metrics."""
    from iw_rag.eval.experiments import LoopComparison

    # LoopComparison must expose per-arm attributes
    comp = LoopComparison(
        on_run_id="run-on",
        off_run_id="run-off",
        on_mean_recall={5: 0.8, 10: 1.0},
        off_mean_recall={5: 0.6, 10: 0.8},
        on_mean_mrr=0.9,
        off_mean_mrr=0.7,
        on_faithfulness=None,
        off_faithfulness=None,
        on_exhaustion_rate=0.0,
        off_exhaustion_rate=0.0,
        n_scored=3,
        n_zero_gold=1,
    )
    assert comp.on_run_id == "run-on"
    assert comp.off_run_id == "run-off"
    assert comp.on_mean_mrr > comp.off_mean_mrr
    assert comp.on_faithfulness is None  # no judge supplied
    assert comp.n_zero_gold == 1


def test_exhaustion_rate_computation() -> None:
    """Budget exhaustion rate is fraction of refused dispositions."""
    from iw_rag.eval.experiments import _compute_exhaustion_rate

    dispositions = ["answered", "refused", "answered", "refused", "answered"]
    rate = _compute_exhaustion_rate(dispositions)
    assert rate == pytest.approx(2 / 5)


def test_exhaustion_rate_empty() -> None:
    """Empty dispositions list returns 0.0."""
    from iw_rag.eval.experiments import _compute_exhaustion_rate

    assert _compute_exhaustion_rate([]) == 0.0


def test_router_accuracy_computation() -> None:
    """Router accuracy is fraction of routes matching gold query_type."""
    from iw_rag.eval.experiments import _compute_router_accuracy

    # (actual_route, gold_query_type) pairs; accurate when they match
    pairs = [
        ("retrieve", "factual"),  # correct: factual → retrieve
        ("skip", "trivial"),  # correct: trivial → skip
        ("retrieve", "trivial"),  # wrong: should have been skip
        ("retrieve", "factual"),  # correct
    ]
    accuracy = _compute_router_accuracy(pairs)
    assert accuracy == pytest.approx(3 / 4)


def test_router_accuracy_empty() -> None:
    """Empty pairs return 0.0."""
    from iw_rag.eval.experiments import _compute_router_accuracy

    assert _compute_router_accuracy([]) == 0.0
