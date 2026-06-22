"""Faithfulness no-regress gate — judged, skip-guarded when judge is absent.

This gate checks that the mean faithfulness score from the current run has
not dropped more than ``faithfulness_regress_tolerance`` below the committed
Phase-0 baseline.  Unlike the Recall@10 gate it requires a live judge endpoint
(Lynx-8B / Selene-Mini-8B) and is therefore:

- Skipped with a warning when ``VLLM_JUDGE_URL`` is not set or unreachable.
- Also marked ``@pytest.mark.live`` so it only runs with ``--run-live``.

The **gate logic** (pure, no judge) is exercised by the meta-tests in
``tests/unit/test_ci_gate.py`` which always run and provide the RED evidence
that the gate bites when faithfulness drops.

The judge must be the **local offline judge** (Lynx-8B / Selene-Mini-8B via
``VLLM_JUDGE_URL``), **never the generator** (contamination rule; ADR-013,
lld-eval-observability §4/§9, CONVENTIONS §4).

DeepEval is lazily imported inside this module's functions and guarded by
``pytest.importorskip`` so collection does not error when deepeval is absent.

References:
- task-1.4-ci-regression-gate (spec step 5–6)
- lld-eval-observability §9 (uncalibrated → skip-with-warning)
- ADR-013 (contamination rule: judge ≠ generator)
- engineering-conventions §2 (RED-first TDD)
"""

from __future__ import annotations

import pytest

from iw_rag.eval.ci_gate import Baseline, GateResult, check_faithfulness_no_regress

from .conftest import _judge_endpoint_up

# ---------------------------------------------------------------------------
# Gate-logic smoke tests — always run (no judge, no DB, no deepeval)
# ---------------------------------------------------------------------------


def test_faithfulness_gate_passes_at_baseline(baseline: Baseline) -> None:
    """Faithfulness exactly at the baseline value passes the gate."""
    result: GateResult = check_faithfulness_no_regress(baseline.faithfulness, baseline)
    assert result.passed, f"Expected gate to pass at faithfulness=baseline; got: {result.message}"
    assert result.metric == "faithfulness"


def test_faithfulness_gate_passes_above_baseline(baseline: Baseline) -> None:
    """Faithfulness above the baseline passes the gate."""
    result: GateResult = check_faithfulness_no_regress(baseline.faithfulness + 0.10, baseline)
    assert result.passed, f"Expected gate to pass above baseline; got: {result.message}"
    assert result.metric == "faithfulness"
    # observed = baseline.faithfulness + 0.10 = 1.0 + 0.10 = 1.10 (baseline is 1.0)
    assert result.observed == pytest.approx(baseline.faithfulness + 0.10)
    # threshold = baseline.faithfulness - faithfulness_regress_tolerance = 1.0 - 0.03 = 0.97
    assert result.threshold == pytest.approx(
        baseline.faithfulness - baseline.faithfulness_regress_tolerance
    )
    assert "OK" in result.message


def test_faithfulness_gate_passes_within_tolerance(baseline: Baseline) -> None:
    """Faithfulness drop within the tolerance band passes the gate."""
    # Drop by exactly the tolerance — should still pass (threshold is exclusive
    # lower bound: observed >= baseline - tolerance).
    within = baseline.faithfulness - baseline.faithfulness_regress_tolerance
    result: GateResult = check_faithfulness_no_regress(within, baseline)
    assert result.passed, (
        f"Expected gate to pass at faithfulness=baseline-tolerance; got: {result.message}"
    )
    assert result.metric == "faithfulness"
    # observed == within == baseline - tolerance = 1.0 - 0.03 = 0.97
    assert result.observed == pytest.approx(within)
    # threshold == baseline - tolerance == 0.97
    assert result.threshold == pytest.approx(within)
    assert "OK" in result.message


def test_faithfulness_gate_fails_beyond_tolerance(baseline: Baseline) -> None:
    """Faithfulness drop beyond the tolerance fails the gate (gate has teeth).

    Using baseline=0.7 and tolerance=0.03 so the floor is 0.67; feeding 0.60
    triggers the regression.
    """
    synthetic_baseline = Baseline(
        recall_at_10=0.80,
        ndcg_at_10=0.0,
        faithfulness=0.70,
        faithfulness_regress_tolerance=0.03,
    )
    result: GateResult = check_faithfulness_no_regress(
        0.60, synthetic_baseline, fail_on_regression=True
    )
    assert not result.passed, (
        f"Expected gate to FAIL for faithfulness=0.60 (floor=0.67); message: {result.message}"
    )
    assert result.observed == pytest.approx(0.60)
    assert result.threshold == pytest.approx(0.67)


def test_faithfulness_warn_mode_does_not_fail_on_regression(baseline: Baseline) -> None:
    """``fail_on_regression=False`` downgrades a faithfulness regression to a warning."""
    synthetic_baseline = Baseline(
        recall_at_10=0.80,
        ndcg_at_10=0.0,
        faithfulness=0.70,
        faithfulness_regress_tolerance=0.03,
    )
    result: GateResult = check_faithfulness_no_regress(
        0.60, synthetic_baseline, fail_on_regression=False
    )
    assert result.passed, f"Warn-mode should pass even on regression; got: {result.message}"
    assert "WARNS" in result.message or "warn" in result.message.lower()


# ---------------------------------------------------------------------------
# Live judged gate — requires judge endpoint + live spine + --run-live
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(not _judge_endpoint_up(), reason="offline judge endpoint unavailable")
def test_faithfulness_no_regression_live_deferred(baseline: Baseline) -> None:
    """Synthetic gate-logic check — NOT a live re-measurement; live measurement is DEFERRED.

    This placeholder is gated behind ``@pytest.mark.live`` (skipped unless
    ``--run-live`` is passed) AND ``@pytest.mark.skipif`` (skipped unless
    ``VLLM_JUDGE_URL`` points at a reachable Lynx-8B / Selene-Mini-8B judge).
    It contains NO real judge invocation.

    The real live path calls ``run_experiment`` with a faithfulness judge, extracts
    ``summary.mean_faithfulness``, then calls ``check_faithfulness_no_regress``.
    That wiring is deferred to the Phase-1 eval box run (see task-1.4 §validation).

    The gate logic is fully exercised — for both pass and fail cases — by the
    pure synthetic tests above, which run unconditionally without any live backend.
    Live-measured faithfulness as of 2026-06-08: mean 1.000 (8/8 demo + 18/18
    benchmark corpora, calibrated κ=1.000) — see eval/reports/.

    When the live harness is wired: remove this skip, call ``run_experiment``
    with a judge, extract ``summary.mean_faithfulness``, pass to gate function.
    """
    pytest.skip(
        "Live faithfulness re-measurement DEFERRED — no GPU/judge in CI; "
        "see eval/CLAUDE.md and task-1.4 §validation for the eval-box wiring plan."
    )
