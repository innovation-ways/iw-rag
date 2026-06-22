"""Experiment #9 trajectory gate — judge-free structural-validity + route floors.

This gate runs on every PR without a GPU or judge endpoint (the structural
invariants I1-I4 are deterministic Python).  It enforces two invariants from the
committed Phase-0 baseline:

1. **Trajectory-validity floor**: every legal trajectory must satisfy I1-I4 on
   the demo corpus, so the validity rate must hold at the committed 1.0 baseline.
2. **Route-accuracy floor**: router accuracy must not fall below the committed
   baseline (a mis-route the spine fallback masks is still a router regression).

The gate-logic smoke tests always run.  The end-to-end harness path is exercised
on a **mocked** sample set (a fake EvalRepository and hand-built per-query
samples — no live loop, no GPU), demonstrating that the harness → gate pipeline
bites on a synthetic regression.  The live route-accuracy re-measurement is
``@pytest.mark.live`` (deferred to the eval box; see scripts/run_trajectory_eval.py).

References:
- docs/build/03-tasks/trajectory-eval/03-design-and-implementation.md §3.3
- src/iw_rag/eval/ci_gate.py (check_trajectory_validity_floor / check_route_accuracy_floor)
- src/iw_rag/eval/experiments.py (run_experiment_9_trajectory)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.eval.ci_gate import (
    Baseline,
    GateResult,
    check_route_accuracy_floor,
    check_trajectory_validity_floor,
)
from iw_rag.eval.experiments import run_experiment_9_trajectory
from iw_rag.eval.repository import GoldenQuery

# ---------------------------------------------------------------------------
# Fixtures / mocked sample helpers
# ---------------------------------------------------------------------------


def _v(kind: str, value: str = "", attempt: int = 0) -> dict[str, Any]:
    """Build one trajectory verdict dict."""
    return {"kind": kind, "value": value, "score": None, "attempt": attempt}


def _valid_sample() -> dict[str, Any]:
    """A structurally valid answered+grounded retrieve-route sample."""
    return {
        "trajectory": [
            _v("input_guard", "allow"),
            _v("router", "retrieve"),
            _v("grade_documents", "correct"),
            _v("grade_generation", "grounded"),
            _v("output_guard", "allow"),
        ],
        "route": "retrieve",
        "retries": 0,
        "max_retries": 2,
        "disposition": "answered",
        "grounded": True,
    }


def _invalid_sample() -> dict[str, Any]:
    """A silent refusal (refused with no negative cause) — I2 violation."""
    return {
        "trajectory": [
            _v("input_guard", "allow"),
            _v("router", "retrieve"),
            _v("grade_documents", "correct"),
        ],
        "route": "retrieve",
        "retries": 0,
        "max_retries": 2,
        "disposition": "refused",
        "grounded": False,
    }


def _fake_repo() -> Any:
    """Build a fake EvalRepository (async create_run / add_results)."""
    repo = MagicMock()
    repo.create_run = AsyncMock(return_value="run-traj-gate")
    repo.add_results = AsyncMock(return_value=None)
    return repo


# ---------------------------------------------------------------------------
# Gate-logic smoke tests — always run (no DB, no GPU, no judge)
# ---------------------------------------------------------------------------


def test_validity_gate_passes_at_baseline(baseline: Baseline) -> None:
    """Validity rate at the committed 1.0 baseline passes the gate."""
    result: GateResult = check_trajectory_validity_floor(1.0, baseline)
    assert result.passed, f"Expected pass at validity=1.0; got: {result.message}"
    assert result.metric == "trajectory_validity_rate"


def test_validity_gate_fails_below_baseline(baseline: Baseline) -> None:
    """A validity rate below the 1.0 baseline fails the gate (it has teeth)."""
    result = check_trajectory_validity_floor(0.9, baseline, fail_on_regression=True)
    assert not result.passed, (
        f"Expected gate to FAIL at validity=0.9 but it passed: {result.message}"
    )


def test_route_gate_passes_at_or_above_baseline(baseline: Baseline) -> None:
    """Route accuracy at/above the committed baseline passes the gate."""
    result = check_route_accuracy_floor(baseline.route_accuracy, baseline)
    assert result.passed
    assert result.metric == "route_accuracy"


def test_route_gate_warn_mode_does_not_fail(baseline: Baseline) -> None:
    """``fail_on_regression=False`` downgrades a route-accuracy regression to a warning."""
    # Use a value below baseline only if the baseline is above 0; otherwise the
    # warn-mode contract is still exercised by check on a sub-baseline value.
    below = max(0.0, baseline.route_accuracy - 0.1)
    result = check_route_accuracy_floor(below, baseline, fail_on_regression=False)
    assert result.passed is True
    assert result.metric == "route_accuracy"
    assert result.observed == below


# ---------------------------------------------------------------------------
# End-to-end harness → gate (mocked samples; no GPU) — always runs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_harness_to_gate_passes_on_valid_trajectories(baseline: Baseline) -> None:
    """All-valid mocked samples → validity 1.0 → both gates pass."""
    repo = _fake_repo()
    queries = [
        GoldenQuery(id="q1", question="q1?", query_type="factual"),
        GoldenQuery(id="q2", question="q2?", query_type="factual"),
    ]
    samples = {
        "q1?": [_valid_sample() for _ in range(3)],
        "q2?": [_valid_sample() for _ in range(3)],
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )

    # All-valid samples must yield a perfect validity rate (the gate input).
    assert report.trajectory_validity_rate == pytest.approx(1.0)
    validity = check_trajectory_validity_floor(report.trajectory_validity_rate, baseline)
    route = check_route_accuracy_floor(report.route_accuracy, baseline)
    assert validity.passed, validity.message
    assert route.passed, route.message
    assert validity.metric == "trajectory_validity_rate"


@pytest.mark.asyncio
async def test_harness_to_gate_fails_on_structural_regression(baseline: Baseline) -> None:
    """A silent-refusal trajectory drops validity below 1.0 → the gate FAILS.

    This is the RED-evidence that the harness → gate pipeline bites: an I2
    violation (refused with no explained cause) must fail CI.
    """
    repo = _fake_repo()
    queries = [
        GoldenQuery(id="q1", question="ok?", query_type="factual"),
        GoldenQuery(id="q2", question="silent?", query_type="factual"),
    ]
    samples = {
        "ok?": [_valid_sample() for _ in range(3)],
        "silent?": [_invalid_sample() for _ in range(3)],
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )

    assert report.trajectory_validity_rate < 1.0
    result = check_trajectory_validity_floor(
        report.trajectory_validity_rate, baseline, fail_on_regression=True
    )
    assert not result.passed, (
        "A structural-invariant violation must fail the trajectory gate — "
        "the gate is meaningless if it passes here."
    )


# ---------------------------------------------------------------------------
# Live harness gate — opt-in only (needs the GPU stack + a live loop run)
# ---------------------------------------------------------------------------


@pytest.mark.live
def test_route_accuracy_live_deferred(baseline: Baseline) -> None:
    """Live route-accuracy re-measurement is DEFERRED to the eval box (no GPU in CI).

    The structural-validity gate runs fully on mocked samples above.  The live
    route-accuracy baseline is captured by scripts/run_trajectory_eval.py against
    the GPU stack and committed into tests/eval/baselines/phase0.json.
    """
    pytest.skip(
        "Live route-accuracy re-measurement DEFERRED — no GPU in CI; "
        "run scripts/run_trajectory_eval.py on the eval box and commit the baseline."
    )
