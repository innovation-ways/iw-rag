"""Unit tests for the CI regression gate logic (src/iw_rag/eval/ci_gate.py).

TDD RED-first: these tests were written before the gate implementation existed
to provide RED evidence that the gate actually bites when Recall@10 drops or
faithfulness regresses.  They run in plain ``pytest`` with no database, no GPU,
no deepeval, and no live endpoints.

Every test exercises a pure function with a synthetic input — the integration
with a real RunSummary is deferred to the ``@pytest.mark.live`` tests in
``tests/eval/``.

References:
- task-1.4-ci-regression-gate (meta-tests are the deliverable's teeth)
- engineering-conventions §2 (TDD RED-first, iw-ai-core-testing discipline)
- ADR-013 (gate must bite — not decorative)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from iw_rag.domain.errors import ConfigError
from iw_rag.eval.ci_gate import (
    Baseline,
    check_faithfulness_no_regress,
    check_recall_floor,
    check_route_accuracy_floor,
    check_trajectory_validity_floor,
    load_baseline,
)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

# Synthetic boundary-value baseline used for gate-logic unit tests.
# faithfulness=0.0 is a deliberate lower-boundary value that exercises
# edge cases of the gate arithmetic (floor = 0.0 - 0.03 = -0.03).
# NOTE: the committed tests/eval/baselines/phase0.json now has faithfulness=1.0
# (live-measured 2026-06-08, κ=1.000 calibrated). This constant is a test
# fixture, not a copy of the committed baseline file.
_PHASE0_BASELINE = Baseline(
    recall_at_10=0.80,
    ndcg_at_10=0.0,
    faithfulness=0.0,
    faithfulness_regress_tolerance=0.03,
)

_HIGH_BASELINE = Baseline(
    recall_at_10=0.85,
    ndcg_at_10=0.60,
    faithfulness=0.70,
    faithfulness_regress_tolerance=0.03,
)


def _write_baseline(path: Path, data: dict) -> None:  # type: ignore[type-arg]
    """Write a baseline JSON file at the given path.

    Args:
        path: Target file path.
        data: Dict to serialize as JSON.
    """
    path.write_text(json.dumps(data))


# ===========================================================================
# 1. Recall@10 gate — check_recall_floor
# ===========================================================================


class TestCheckRecallFloor:
    """Unit tests for :func:`iw_rag.eval.ci_gate.check_recall_floor`."""

    # --- happy path ---

    def test_passes_at_exact_floor_value(self) -> None:
        """Recall@10 exactly at 0.80 (the hard floor) passes the gate."""
        result = check_recall_floor(0.80, _PHASE0_BASELINE)
        assert result.passed
        assert result.metric == "recall_at_10"
        assert result.observed == pytest.approx(0.80)
        assert result.threshold == pytest.approx(0.80)

    def test_passes_above_floor(self) -> None:
        """Recall@10 clearly above the floor passes the gate."""
        result = check_recall_floor(0.93, _PHASE0_BASELINE)
        assert result.passed is True
        assert result.observed == pytest.approx(0.93)
        assert result.threshold == pytest.approx(0.80)
        assert result.metric == "recall_at_10"

    def test_passes_at_baseline_higher_than_hard_floor(self) -> None:
        """When baseline.recall_at_10 > 0.8, the effective threshold is the baseline."""
        result = check_recall_floor(0.86, _HIGH_BASELINE)
        assert result.passed
        assert result.threshold == pytest.approx(0.85)

    # --- failure path (gate has teeth) ---

    def test_fails_at_0_5_recall_ten(self) -> None:
        """Recall@10 = 0.5 FAILS the gate — RED evidence the gate bites.

        This is the canonical TDD RED-evidence test required by task-1.4.
        The gate is meaningless if it does not fail here.
        """
        result = check_recall_floor(0.50, _PHASE0_BASELINE, fail_on_regression=True)
        assert not result.passed, (
            "Gate must FAIL for Recall@10=0.5 — the gate must have real teeth, "
            "not just decorative assertions."
        )

    def test_fails_just_below_floor(self) -> None:
        """Recall@10 just below the hard floor (0.799) fails the gate."""
        result = check_recall_floor(0.799, _PHASE0_BASELINE)
        assert not result.passed

    def test_fails_below_higher_baseline(self) -> None:
        """Recall@10 below a baseline that is above 0.8 fails the gate."""
        result = check_recall_floor(0.82, _HIGH_BASELINE)
        # threshold = max(0.8, 0.85) = 0.85; 0.82 < 0.85 → fail
        assert not result.passed
        assert result.threshold == pytest.approx(0.85)

    # --- warning / no-fail mode ---

    def test_warn_mode_passes_even_on_regression(self) -> None:
        """``fail_on_regression=False`` passes the gate but records the regression."""
        result = check_recall_floor(0.50, _PHASE0_BASELINE, fail_on_regression=False)
        assert result.passed is True, (
            "Warn-mode must not fail the build — it only logs the regression."
        )
        assert result.observed == pytest.approx(0.50)
        assert result.threshold == pytest.approx(0.80)

    def test_warn_mode_message_contains_warn_indicator(self) -> None:
        """Warn-mode message includes a human-readable warning indicator."""
        result = check_recall_floor(0.50, _PHASE0_BASELINE, fail_on_regression=False)
        assert "WARNS" in result.message or "warn" in result.message.lower()

    def test_config_false_honours_fail_flag(self) -> None:
        """``deepeval_fail_on_regression: false`` scenario: gate does not raise."""
        # Simulate what the test file does when the config flag is false.
        result = check_recall_floor(0.60, _PHASE0_BASELINE, fail_on_regression=False)
        assert result.passed is True  # downgraded to warning — config honoured
        assert result.observed == pytest.approx(0.60)
        assert result.threshold == pytest.approx(0.80)

    # --- return contract ---

    def test_gate_result_has_correct_metric_name(self) -> None:
        """GateResult.metric is always 'recall_at_10'."""
        result = check_recall_floor(0.90, _PHASE0_BASELINE)
        assert result.metric == "recall_at_10"

    def test_gate_result_message_is_non_empty(self) -> None:
        """GateResult.message is always a non-empty string."""
        for val in (0.50, 0.80, 0.95):
            result = check_recall_floor(val, _PHASE0_BASELINE)
            assert isinstance(result.message, str), f"Expected str message for recall={val}"
            assert len(result.message) > 0, f"Expected non-empty message for recall={val}"
            assert "recall" in result.message.lower() or "Recall" in result.message, (
                f"Message should mention 'recall': {result.message!r}"
            )

    def test_gate_result_threshold_is_max_of_floor_and_baseline(self) -> None:
        """GateResult.threshold = max(0.8, baseline.recall_at_10)."""
        # Phase-0 baseline.recall_at_10 == 0.80 → threshold == 0.80
        assert check_recall_floor(0.90, _PHASE0_BASELINE).threshold == pytest.approx(0.80)
        # High baseline (0.85) → threshold == 0.85
        assert check_recall_floor(0.90, _HIGH_BASELINE).threshold == pytest.approx(0.85)
        # Below-0.8 baseline (would be unusual but tested) → floor 0.8 wins
        low_baseline = Baseline(
            recall_at_10=0.50,
            ndcg_at_10=0.0,
            faithfulness=0.0,
            faithfulness_regress_tolerance=0.03,
        )
        assert check_recall_floor(0.82, low_baseline).threshold == pytest.approx(0.80)


# ===========================================================================
# 2. Faithfulness gate — check_faithfulness_no_regress
# ===========================================================================


class TestCheckFaithfulnessNoRegress:
    """Unit tests for :func:`iw_rag.eval.ci_gate.check_faithfulness_no_regress`."""

    # --- happy path (synthetic boundary values) ---

    def test_passes_at_zero_faithfulness_baseline(self) -> None:
        """Synthetic baseline faithfulness=0.0; any non-negative observed value passes.

        This uses ``_PHASE0_BASELINE`` (a local synthetic fixture with
        ``faithfulness=0.0``) to test the gate's lower-boundary arithmetic:
        floor = 0.0 - 0.03 = -0.03, so any observed score ≥ 0 passes.

        Note: the committed ``tests/eval/baselines/phase0.json`` now has
        ``faithfulness=1.0`` (live-measured 2026-06-08; κ=1.000 calibrated).
        This test exercises a synthetic corner-case of the gate function, not
        the live baseline value.
        """
        result = check_faithfulness_no_regress(0.0, _PHASE0_BASELINE)
        assert result.passed is True
        # floor = 0.0 - 0.03 = -0.03; threshold is pinned exactly.
        assert result.threshold == pytest.approx(-0.03)
        assert result.observed == pytest.approx(0.0)
        assert result.metric == "faithfulness"

    def test_passes_above_baseline(self) -> None:
        """Faithfulness above the baseline passes the gate."""
        result = check_faithfulness_no_regress(0.80, _HIGH_BASELINE)
        assert result.passed is True
        assert result.observed == pytest.approx(0.80)
        # threshold = 0.70 - 0.03 = 0.67
        assert result.threshold == pytest.approx(0.67)
        assert result.metric == "faithfulness"

    def test_passes_within_tolerance(self) -> None:
        """Drop within tolerance (exactly at floor) passes the gate."""
        # floor = 0.70 - 0.03 = 0.67; observed = 0.67 → passes
        result = check_faithfulness_no_regress(0.67, _HIGH_BASELINE)
        assert result.passed
        assert result.threshold == pytest.approx(0.67)

    # --- failure path (gate has teeth) ---

    def test_fails_beyond_tolerance(self) -> None:
        """Drop beyond tolerance FAILS the gate — RED evidence.

        baseline=0.70, tolerance=0.03 → floor=0.67; observed=0.60 → fail.
        """
        result = check_faithfulness_no_regress(0.60, _HIGH_BASELINE, fail_on_regression=True)
        assert not result.passed, "Gate must FAIL when faithfulness drops 0.10 > tolerance 0.03."
        assert result.observed == pytest.approx(0.60)
        assert result.threshold == pytest.approx(0.67)

    def test_fails_just_below_floor(self) -> None:
        """Faithfulness just below the tolerance floor fails the gate."""
        # floor = 0.70 - 0.03 = 0.67; observed = 0.669 → fail
        result = check_faithfulness_no_regress(0.669, _HIGH_BASELINE)
        assert not result.passed

    # --- warning mode ---

    def test_warn_mode_passes_on_regression(self) -> None:
        """``fail_on_regression=False`` passes even on faithfulness regression."""
        result = check_faithfulness_no_regress(0.60, _HIGH_BASELINE, fail_on_regression=False)
        assert result.passed is True
        assert result.observed == pytest.approx(0.60)
        assert result.threshold == pytest.approx(0.67)

    def test_warn_mode_message_contains_warn_indicator(self) -> None:
        """Warn-mode message includes a warning indicator string."""
        result = check_faithfulness_no_regress(0.60, _HIGH_BASELINE, fail_on_regression=False)
        assert "WARNS" in result.message or "warn" in result.message.lower()

    # --- return contract ---

    def test_gate_result_metric_name_is_faithfulness(self) -> None:
        """GateResult.metric is always 'faithfulness'."""
        result = check_faithfulness_no_regress(0.80, _HIGH_BASELINE)
        assert result.metric == "faithfulness"

    def test_gate_result_threshold_equals_baseline_minus_tolerance(self) -> None:
        """GateResult.threshold = baseline.faithfulness - faithfulness_regress_tolerance."""
        result = check_faithfulness_no_regress(0.80, _HIGH_BASELINE)
        assert result.threshold == pytest.approx(0.70 - 0.03)

    def test_gate_result_message_non_empty(self) -> None:
        """GateResult.message is always a non-empty string."""
        result = check_faithfulness_no_regress(0.80, _HIGH_BASELINE)
        assert isinstance(result.message, str)
        assert len(result.message) > 0
        assert "faithfulness" in result.message.lower() or "Faithfulness" in result.message


# ===========================================================================
# 3. Baseline loader — load_baseline
# ===========================================================================


class TestLoadBaseline:
    """Unit tests for :func:`iw_rag.eval.ci_gate.load_baseline`."""

    def test_loads_valid_baseline_file(self) -> None:
        """A well-formed baseline JSON file is loaded into a Baseline dataclass."""
        data = {
            "recall_at_10": 0.80,
            "ndcg_at_10": 0.0,
            "faithfulness": 0.0,
            "faithfulness_regress_tolerance": 0.03,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phase0.json"
            _write_baseline(p, data)
            b = load_baseline(p)
        assert b.recall_at_10 == pytest.approx(0.80)
        assert b.ndcg_at_10 == pytest.approx(0.0)
        assert b.faithfulness == pytest.approx(0.0)
        assert b.faithfulness_regress_tolerance == pytest.approx(0.03)

    def test_missing_baseline_raises_config_error_loudly(self) -> None:
        """A missing baseline file raises ConfigError — never silently passes.

        This is the anti-pattern guard: if the baseline file is deleted or not
        committed, the gate must fail loudly rather than treating absence as
        'all OK' (which would make the gate decorative).
        """
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nonexistent_baseline.json"
            with pytest.raises(ConfigError, match="baseline"):
                load_baseline(missing)

    def test_invalid_json_raises_config_error(self) -> None:
        """A baseline file with invalid JSON raises ConfigError."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text("{not valid json")
            with pytest.raises(ConfigError, match="not valid JSON"):
                load_baseline(p)

    def test_missing_key_raises_config_error(self) -> None:
        """A baseline file missing a required key raises ConfigError."""
        incomplete = {
            "recall_at_10": 0.80,
            # ndcg_at_10, faithfulness, faithfulness_regress_tolerance missing
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "incomplete.json"
            _write_baseline(p, incomplete)
            with pytest.raises(ConfigError, match="missing required keys"):
                load_baseline(p)

    def test_loads_committed_phase0_baseline(self) -> None:
        """The committed tests/eval/baselines/phase0.json is loadable and matches live values.

        The committed baseline reflects the live-measured Phase-0 values
        (2026-06-08, RTX 5090, κ=1.000 calibrated):
        - recall_at_10 = 0.80 (hard floor, ADR-013)
        - faithfulness  = 1.0  (live-measured; resolved from the prior 0.0 placeholder)
        """
        committed = (
            Path(__file__).resolve().parent.parent  # tests/
            / "eval"
            / "baselines"
            / "phase0.json"
        )
        b = load_baseline(committed)
        # Hard floor invariant from ADR-013
        assert b.recall_at_10 >= 0.80, (
            f"Committed baseline recall_at_10={b.recall_at_10} must be >= 0.80 (ADR-013 hard floor)"
        )
        assert b.faithfulness_regress_tolerance > 0, "Committed baseline tolerance must be positive"
        # Live-measured faithfulness resolved from the old 0.0 placeholder (2026-06-08).
        assert b.faithfulness == pytest.approx(1.0), (
            f"Committed baseline faithfulness={b.faithfulness} must be 1.0 "
            "(live-measured 2026-06-08; update this assertion if re-baselined)"
        )

    def test_extra_keys_in_baseline_are_ignored(self) -> None:
        """Extra keys in the baseline file are silently ignored (forward compat)."""
        data = {
            "recall_at_10": 0.80,
            "ndcg_at_10": 0.0,
            "faithfulness": 0.0,
            "faithfulness_regress_tolerance": 0.03,
            "future_metric": 0.99,  # unknown key — must not error
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "phase0.json"
            _write_baseline(p, data)
            b = load_baseline(p)
        assert b.recall_at_10 == pytest.approx(0.80)


# ===========================================================================
# 3b. Trajectory gate — check_trajectory_validity_floor + check_route_accuracy_floor
# ===========================================================================

_TRAJ_BASELINE = Baseline(
    recall_at_10=0.80,
    ndcg_at_10=0.0,
    faithfulness=1.0,
    faithfulness_regress_tolerance=0.03,
    trajectory_validity_rate=1.0,
    route_accuracy=0.75,
)


class TestCheckTrajectoryValidityFloor:
    """Unit tests for :func:`check_trajectory_validity_floor`."""

    def test_passes_at_baseline(self) -> None:
        """Validity rate exactly at the committed baseline passes the gate."""
        result = check_trajectory_validity_floor(1.0, _TRAJ_BASELINE)
        assert result.passed is True
        assert result.metric == "trajectory_validity_rate"
        assert result.observed == pytest.approx(1.0)
        assert result.threshold == pytest.approx(1.0)

    def test_fails_below_baseline(self) -> None:
        """A structural-invariant violation (rate < 1.0 baseline) FAILS the gate."""
        result = check_trajectory_validity_floor(0.9, _TRAJ_BASELINE, fail_on_regression=True)
        assert not result.passed, (
            "A trajectory-validity drop below the 1.0 baseline is a real loop bug — "
            "the gate must have teeth."
        )
        assert result.observed == pytest.approx(0.9)

    def test_warn_mode_does_not_fail(self) -> None:
        """``fail_on_regression=False`` downgrades a validity regression to a warning."""
        result = check_trajectory_validity_floor(0.5, _TRAJ_BASELINE, fail_on_regression=False)
        assert result.passed is True
        assert "WARNS" in result.message or "warn" in result.message.lower()

    def test_message_non_empty(self) -> None:
        """GateResult.message is always a non-empty string mentioning trajectory."""
        result = check_trajectory_validity_floor(1.0, _TRAJ_BASELINE)
        assert isinstance(result.message, str) and len(result.message) > 0
        # At validity 1.0 the floor is met, so the message is the OK form that
        # names the metric and reports the observed/threshold values verbatim.
        assert result.message.startswith("Trajectory-validity OK:")
        assert "trajectory" in result.message.lower()


class TestCheckRouteAccuracyFloor:
    """Unit tests for :func:`check_route_accuracy_floor`."""

    def test_passes_at_baseline(self) -> None:
        """Route accuracy exactly at the committed baseline passes the gate."""
        result = check_route_accuracy_floor(0.75, _TRAJ_BASELINE)
        assert result.passed is True
        assert result.metric == "route_accuracy"
        assert result.threshold == pytest.approx(0.75)

    def test_passes_above_baseline(self) -> None:
        """Route accuracy above the baseline passes the gate."""
        result = check_route_accuracy_floor(0.9, _TRAJ_BASELINE)
        assert result.passed is True
        assert result.observed == pytest.approx(0.9)

    def test_fails_below_baseline(self) -> None:
        """Route accuracy below the committed baseline FAILS the gate."""
        result = check_route_accuracy_floor(0.5, _TRAJ_BASELINE, fail_on_regression=True)
        assert not result.passed
        assert result.observed == pytest.approx(0.5)

    def test_warn_mode_does_not_fail(self) -> None:
        """``fail_on_regression=False`` downgrades a route-accuracy regression to a warning."""
        result = check_route_accuracy_floor(0.5, _TRAJ_BASELINE, fail_on_regression=False)
        assert result.passed is True
        assert "WARNS" in result.message or "warn" in result.message.lower()


class TestTrajectoryBaselineLoading:
    """The committed baseline and the loader carry the trajectory thresholds."""

    def test_baseline_defaults_are_backward_compatible(self) -> None:
        """Baseline can still be built with only the original four fields (defaults apply)."""
        b = Baseline(
            recall_at_10=0.80,
            ndcg_at_10=0.0,
            faithfulness=0.0,
            faithfulness_regress_tolerance=0.03,
        )
        # Defaults: structural validity floor at 1.0, route accuracy at 0.0.
        assert b.trajectory_validity_rate == pytest.approx(1.0)
        assert b.route_accuracy == pytest.approx(0.0)

    def test_loader_parses_trajectory_fields_when_present(self) -> None:
        """load_baseline reads trajectory_validity_rate + route_accuracy when present."""
        data = {
            "recall_at_10": 0.80,
            "ndcg_at_10": 0.0,
            "faithfulness": 1.0,
            "faithfulness_regress_tolerance": 0.03,
            "trajectory_validity_rate": 1.0,
            "route_accuracy": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "b.json"
            _write_baseline(p, data)
            b = load_baseline(p)
        assert b.trajectory_validity_rate == pytest.approx(1.0)
        assert b.route_accuracy == pytest.approx(0.5)

    def test_loader_defaults_trajectory_fields_when_absent(self) -> None:
        """A legacy baseline file without the trajectory keys still loads (defaults)."""
        data = {
            "recall_at_10": 0.80,
            "ndcg_at_10": 0.0,
            "faithfulness": 1.0,
            "faithfulness_regress_tolerance": 0.03,
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "legacy.json"
            _write_baseline(p, data)
            b = load_baseline(p)
        assert b.trajectory_validity_rate == pytest.approx(1.0)
        assert b.route_accuracy == pytest.approx(0.0)

    def test_committed_phase0_baseline_has_trajectory_fields(self) -> None:
        """The committed phase0.json carries trajectory_validity_rate=1.0 + route_accuracy."""
        committed = Path(__file__).resolve().parent.parent / "eval" / "baselines" / "phase0.json"
        b = load_baseline(committed)
        assert b.trajectory_validity_rate == pytest.approx(1.0), (
            "Structural-validity floor must be 1.0 on the demo corpus (any violation is a bug)"
        )
        assert 0.0 <= b.route_accuracy <= 1.0


# ===========================================================================
# 4. EvalCiSettings — config model smoke tests
# ===========================================================================


class TestEvalCiSettings:
    """Smoke tests for the EvalCiSettings pydantic model in settings.py."""

    def test_eval_ci_settings_defaults(self) -> None:
        """EvalCiSettings has the correct defaults from the spec."""
        from iw_rag.settings import EvalCiSettings

        s = EvalCiSettings()
        assert s.deepeval_fail_on_regression is True
        assert s.faithfulness_regress_tolerance == pytest.approx(0.03)

    def test_eval_ci_settings_flag_false(self) -> None:
        """EvalCiSettings.deepeval_fail_on_regression can be set to False."""
        from iw_rag.settings import EvalCiSettings

        s = EvalCiSettings(deepeval_fail_on_regression=False)
        assert s.deepeval_fail_on_regression is False

    def test_eval_settings_has_ci_field(self) -> None:
        """EvalSettings has a nested 'ci' field of type EvalCiSettings."""
        from iw_rag.settings import EvalCiSettings, EvalSettings

        s = EvalSettings()
        assert isinstance(s.ci, EvalCiSettings)
        assert s.ci.deepeval_fail_on_regression is True
        assert s.ci.faithfulness_regress_tolerance == pytest.approx(0.03)
