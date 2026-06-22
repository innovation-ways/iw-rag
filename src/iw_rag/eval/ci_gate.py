"""CI regression gate logic — pure, CPU-only, no DB or GPU required.

Provides two gate functions (one for judge-free retrieval, one for judged
faithfulness) plus a baseline loader.  All logic here is pure Python with no
I/O; it can be unit-tested without deepeval, without a Postgres container,
and without any model endpoint.

Design decision D16-A: the gate logic is extracted from the test layer so it
can be independently unit-tested with synthetic ``RunSummary``-shaped inputs
*before* the live eval harness exists — this is the RED-first evidence that the
gate actually bites (iw-ai-core-testing discipline, engineering-conventions §2).

Usage::

    from iw_rag.eval.ci_gate import (
        GateResult,
        check_recall_floor,
        check_faithfulness_no_regress,
        load_baseline,
    )

This module has ZERO imports from other iw_rag layers (only stdlib + domain
errors) so it is safe to import anywhere in the test suite without triggering
the lazy-import chain for ragas / deepeval.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from iw_rag.domain.errors import ConfigError

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateResult:
    """The outcome of a single CI gate check.

    Attributes:
        passed: True when the metric meets the required threshold.
        metric: The metric name that was checked (e.g. ``"recall_at_10"``).
        observed: The measured value.
        threshold: The threshold the observed value was compared against.
        message: Human-readable explanation of the outcome.
    """

    passed: bool
    metric: str
    observed: float
    threshold: float
    message: str


@dataclass(frozen=True, slots=True)
class Baseline:
    """Frozen thresholds loaded from the committed baseline JSON file.

    The trajectory fields default so a legacy four-field baseline (and the many
    synthetic test fixtures that build ``Baseline`` directly) stay valid; the
    committed JSON supplies the real measured values.

    Attributes:
        recall_at_10: Recall@10 hard floor (0.8 per ADR-013/R-00100).
        ndcg_at_10: nDCG@10 baseline (0.0 placeholder until live run).
        faithfulness: Faithfulness baseline (0.0 placeholder until live run).
        faithfulness_regress_tolerance: Maximum faithfulness drop tolerated.
        trajectory_validity_rate: Experiment-#9 structural-validity floor — must
            be 1.0 on the demo corpus (any I1-I4 violation is a real loop bug).
        route_accuracy: Experiment-#9 router-accuracy floor (measured on the
            live demo run; 0.0 default until measured).
    """

    recall_at_10: float
    ndcg_at_10: float
    faithfulness: float
    faithfulness_regress_tolerance: float
    trajectory_validity_rate: float = 1.0
    route_accuracy: float = 0.0


# ---------------------------------------------------------------------------
# Baseline loader
# ---------------------------------------------------------------------------


def load_baseline(path: Path) -> Baseline:
    """Load and parse the committed baseline JSON file.

    Fails loudly (raises ``ConfigError``) when the file is absent — a missing
    baseline is treated as a misconfiguration, not a silent pass.  The silent-
    pass failure mode is the most dangerous regression pattern: the gate appears
    green while enforcing nothing (engineering-conventions §2, RED-first rule).

    Args:
        path: Absolute or relative path to the baseline JSON (usually
            ``tests/eval/baselines/phase0.json``).

    Returns:
        A :class:`Baseline` with all threshold fields populated.

    Raises:
        ConfigError: If the file does not exist, is not valid JSON, or is
            missing required keys.
    """
    if not path.exists():
        raise ConfigError(
            f"CI regression baseline not found: '{path}'. "
            "A missing baseline is treated as a misconfiguration — the gate "
            "must never silently pass. "
            "Create the baseline with 'make eval' on the Phase-0 spine or "
            "restore it from the committed 'tests/eval/baselines/phase0.json'."
        )
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Baseline file '{path}' is not valid JSON: {exc}") from exc

    required = {"recall_at_10", "ndcg_at_10", "faithfulness", "faithfulness_regress_tolerance"}
    missing = required - set(raw)
    if missing:
        raise ConfigError(
            f"Baseline file '{path}' is missing required keys: {sorted(missing)}. "
            f"Present keys: {sorted(raw)}."
        )

    return Baseline(
        recall_at_10=float(raw["recall_at_10"]),
        ndcg_at_10=float(raw["ndcg_at_10"]),
        faithfulness=float(raw["faithfulness"]),
        faithfulness_regress_tolerance=float(raw["faithfulness_regress_tolerance"]),
        # Trajectory thresholds (experiment #9) are optional in legacy baselines:
        # absent → fall back to the dataclass defaults (validity floor 1.0,
        # route accuracy 0.0) so old baseline files remain loadable.
        trajectory_validity_rate=float(raw.get("trajectory_validity_rate", 1.0)),
        route_accuracy=float(raw.get("route_accuracy", 0.0)),
    )


# ---------------------------------------------------------------------------
# Gate functions — pure, no I/O
# ---------------------------------------------------------------------------


def check_recall_floor(
    recall10: float,
    baseline: Baseline,
    *,
    fail_on_regression: bool = True,
) -> GateResult:
    """Check that Recall@10 meets the hard floor and does not regress vs baseline.

    The effective threshold is ``max(0.8, baseline.recall_at_10)`` — never lower
    than the ADR-013 hard floor of 0.8, and never lower than the committed
    baseline (regressions are not allowed even when the baseline is above 0.8).

    When ``fail_on_regression=False`` the gate records the regression in the
    message but sets ``passed=True`` (warning-only mode), which honours the
    ``deepeval_fail_on_regression: false`` config toggle without silently
    discarding the diagnostic.

    Args:
        recall10: The measured Recall@10 from the current run.
        baseline: The committed :class:`Baseline` thresholds.
        fail_on_regression: When ``True`` (default), a regression sets
            ``passed=False``.  When ``False`` the gate downgrades to a warning.

    Returns:
        A :class:`GateResult` describing the outcome.
    """
    # Hard floor is the greater of the ADR-013 absolute minimum and the
    # committed baseline value — the gate cannot go below either.
    threshold = max(0.8, baseline.recall_at_10)
    regressed = recall10 < threshold

    if regressed:
        msg = (
            f"Recall@10 regression: observed {recall10:.4f} < "
            f"threshold {threshold:.4f} "
            f"(max(0.8, baseline={baseline.recall_at_10:.4f})). "
        )
        if fail_on_regression:
            msg += "Gate FAILS (deepeval_fail_on_regression=true)."
        else:
            msg += "Gate WARNS only (deepeval_fail_on_regression=false)."
        passed = not fail_on_regression
    else:
        msg = f"Recall@10 OK: observed {recall10:.4f} >= threshold {threshold:.4f}."
        passed = True

    return GateResult(
        passed=passed,
        metric="recall_at_10",
        observed=recall10,
        threshold=threshold,
        message=msg,
    )


def check_faithfulness_no_regress(
    faithfulness: float,
    baseline: Baseline,
    *,
    fail_on_regression: bool = True,
) -> GateResult:
    """Check that faithfulness has not regressed beyond the committed tolerance.

    The floor is ``baseline.faithfulness - baseline.faithfulness_regress_tolerance``.
    A faithfulness value below this floor is a regression (the generator has
    become meaningfully less faithful than the Phase-0 measurement).

    When ``fail_on_regression=False`` the gate records the regression in the
    message but sets ``passed=True`` (warning-only mode).

    Args:
        faithfulness: The measured faithfulness score from the current run.
        baseline: The committed :class:`Baseline` thresholds.
        fail_on_regression: When ``True`` (default), a regression sets
            ``passed=False``.  When ``False`` the gate downgrades to a warning.

    Returns:
        A :class:`GateResult` describing the outcome.
    """
    threshold = baseline.faithfulness - baseline.faithfulness_regress_tolerance
    regressed = faithfulness < threshold

    if regressed:
        msg = (
            f"Faithfulness regression: observed {faithfulness:.4f} < "
            f"threshold {threshold:.4f} "
            f"(baseline={baseline.faithfulness:.4f} - "
            f"tolerance={baseline.faithfulness_regress_tolerance:.4f}). "
        )
        if fail_on_regression:
            msg += "Gate FAILS (deepeval_fail_on_regression=true)."
        else:
            msg += "Gate WARNS only (deepeval_fail_on_regression=false)."
        passed = not fail_on_regression
    else:
        msg = f"Faithfulness OK: observed {faithfulness:.4f} >= threshold {threshold:.4f}."
        passed = True

    return GateResult(
        passed=passed,
        metric="faithfulness",
        observed=faithfulness,
        threshold=threshold,
        message=msg,
    )


def check_trajectory_validity_floor(
    validity_rate: float,
    baseline: Baseline,
    *,
    fail_on_regression: bool = True,
) -> GateResult:
    """Check that the trajectory-validity rate (experiment #9) does not regress.

    The threshold is the committed ``baseline.trajectory_validity_rate`` — set to
    1.0 on the demo corpus because every legal trajectory must satisfy the four
    structural invariants (I1-I4); any drop below it is a real loop bug (a
    grounded answer that skipped the grounding grade, a silent refusal, an
    over-budget retry, or a route/verdict ordering violation), not a tuning
    target.

    When ``fail_on_regression=False`` the gate records the regression in the
    message but sets ``passed=True`` (warning-only mode), honouring the
    ``deepeval_fail_on_regression: false`` config toggle.

    Args:
        validity_rate: The measured trajectory-validity rate from the current run.
        baseline: The committed :class:`Baseline` thresholds.
        fail_on_regression: When ``True`` (default), a regression sets
            ``passed=False``.  When ``False`` the gate downgrades to a warning.

    Returns:
        A :class:`GateResult` describing the outcome.
    """
    threshold = baseline.trajectory_validity_rate
    regressed = validity_rate < threshold

    if regressed:
        msg = (
            f"Trajectory-validity regression: observed {validity_rate:.4f} < "
            f"threshold {threshold:.4f} (baseline). A structural-invariant (I1-I4) "
            f"violation is a real loop bug. "
        )
        if fail_on_regression:
            msg += "Gate FAILS (deepeval_fail_on_regression=true)."
        else:
            msg += "Gate WARNS only (deepeval_fail_on_regression=false)."
        passed = not fail_on_regression
    else:
        msg = f"Trajectory-validity OK: observed {validity_rate:.4f} >= threshold {threshold:.4f}."
        passed = True

    return GateResult(
        passed=passed,
        metric="trajectory_validity_rate",
        observed=validity_rate,
        threshold=threshold,
        message=msg,
    )


def check_route_accuracy_floor(
    route_accuracy: float,
    baseline: Baseline,
    *,
    fail_on_regression: bool = True,
) -> GateResult:
    """Check that router accuracy (experiment #9) does not regress vs baseline.

    The threshold is the committed ``baseline.route_accuracy`` (measured on the
    live demo run).  A drop below it means the router is systematically
    misclassifying — even when the spine fallback masks it in the final answer.

    When ``fail_on_regression=False`` the gate records the regression in the
    message but sets ``passed=True`` (warning-only mode).

    Args:
        route_accuracy: The measured router accuracy from the current run.
        baseline: The committed :class:`Baseline` thresholds.
        fail_on_regression: When ``True`` (default), a regression sets
            ``passed=False``.  When ``False`` the gate downgrades to a warning.

    Returns:
        A :class:`GateResult` describing the outcome.
    """
    threshold = baseline.route_accuracy
    regressed = route_accuracy < threshold

    if regressed:
        msg = (
            f"Route-accuracy regression: observed {route_accuracy:.4f} < "
            f"threshold {threshold:.4f} (baseline). "
        )
        if fail_on_regression:
            msg += "Gate FAILS (deepeval_fail_on_regression=true)."
        else:
            msg += "Gate WARNS only (deepeval_fail_on_regression=false)."
        passed = not fail_on_regression
    else:
        msg = f"Route-accuracy OK: observed {route_accuracy:.4f} >= threshold {threshold:.4f}."
        passed = True

    return GateResult(
        passed=passed,
        metric="route_accuracy",
        observed=route_accuracy,
        threshold=threshold,
        message=msg,
    )
