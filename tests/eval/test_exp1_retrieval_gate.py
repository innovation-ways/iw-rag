"""Experiment #1 retrieval gate — judge-free Recall@10 floor (CI gate).

This gate runs on every PR without a GPU or judge endpoint (two-clocks rule,
ADR-013).  It enforces two invariants from the committed Phase-0 baseline:

1. **Hard floor**: Recall@10 must be ≥ max(0.8, baseline.recall_at_10).
2. **No regression**: Recall@10 must not fall below the committed baseline
   value (so a spine change that improves recall can be re-baselined, but
   a regression cannot slip through silently).

The **live harness invocation** (calling ``run_experiment`` against real Postgres
+ the production spine) is marked ``@pytest.mark.live`` so it only runs on the
eval box (``pytest --run-live``).  The **gate logic** (compare a
``RunSummary``-shaped aggregate against the baseline floor) is exercised by
the pure meta-tests in ``tests/unit/test_ci_gate.py``, which always run.

References:
- task-1.4-ci-regression-gate (this task's spec)
- ADR-013 (DeepEval + promptfoo CI gates; Recall@10 ≥ 0.8 hard floor)
- lld-eval-observability §10 (CI gate thresholds)
- engineering-conventions §2 (RED-first TDD, live-DB write guard)
"""

from __future__ import annotations

import pytest

from iw_rag.eval.ci_gate import Baseline, GateResult, check_recall_floor

# ---------------------------------------------------------------------------
# Gate-logic smoke test — always runs (no DB, no GPU, no deepeval)
# ---------------------------------------------------------------------------


def test_gate_logic_passes_at_floor(baseline: Baseline) -> None:
    """Recall@10 exactly at the 0.8 floor passes the gate (boundary value)."""
    result: GateResult = check_recall_floor(0.80, baseline)
    assert result.passed, f"Expected gate to pass at Recall@10=0.80; got: {result.message}"
    assert result.metric == "recall_at_10"
    assert result.observed == pytest.approx(0.80)


def test_gate_logic_passes_above_floor(baseline: Baseline) -> None:
    """Recall@10 clearly above the floor passes the gate."""
    result: GateResult = check_recall_floor(0.95, baseline)
    assert result.passed, f"Expected gate to pass at Recall@10=0.95; got: {result.message}"
    assert result.metric == "recall_at_10"
    assert result.observed == pytest.approx(0.95)
    # threshold = max(0.8, baseline.recall_at_10=0.80) = 0.80
    assert result.threshold == pytest.approx(0.80)
    assert "OK" in result.message


def test_gate_logic_fails_below_floor(baseline: Baseline) -> None:
    """Recall@10 below the 0.8 floor fails the gate (gate has teeth)."""
    result: GateResult = check_recall_floor(0.50, baseline, fail_on_regression=True)
    assert not result.passed, (
        f"Expected gate to FAIL at Recall@10=0.50 but it passed; message: {result.message}"
    )
    assert result.observed == pytest.approx(0.50)


def test_gate_logic_warn_mode_does_not_fail_on_regression(baseline: Baseline) -> None:
    """``fail_on_regression=False`` downgrades a regression to a warning (passed=True)."""
    result: GateResult = check_recall_floor(0.50, baseline, fail_on_regression=False)
    assert result.passed, f"Warn-mode should pass even on regression; got: {result.message}"
    assert "WARNS" in result.message or "warn" in result.message.lower()


# ---------------------------------------------------------------------------
# Live harness gate — opt-in only (needs Postgres + ingest + golden set)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_recall_at_10_floor_live_deferred(baseline: Baseline) -> None:
    """Synthetic gate-logic check — NOT a live re-measurement; live measurement is DEFERRED.

    This placeholder is gated behind ``@pytest.mark.live`` (skipped unless
    ``--run-live`` is passed) and contains NO real spine invocation.  The
    real live harness call (``run_experiment`` against testcontainers Postgres +
    production spine) is deferred to the Phase-1 eval box run with a GPU.

    The gate logic is fully exercised — for both pass and fail cases — by the
    pure synthetic tests ``test_gate_logic_passes_at_floor``,
    ``test_gate_logic_passes_above_floor``, and ``test_gate_logic_fails_below_floor``
    above, which run unconditionally without any live backend.

    When the live harness is wired: remove this skip, call ``run_experiment``,
    extract ``summary.mean_recall[10]``, and pass it to ``check_recall_floor``.
    """
    pytest.skip(
        "Live Recall@10 re-measurement DEFERRED — no GPU in CI; "
        "see eval/CLAUDE.md and task-1.4 §validation for the eval-box wiring plan."
    )
