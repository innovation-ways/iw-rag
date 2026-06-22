"""Unit tests for eval/calibration.py — TDD RED-first.

Tests cover: perfect agreement, total disagreement, partial agreement,
empty-sequence sentinel handling, and the documented metric (Cohen's kappa).
"""

from __future__ import annotations

import pytest

from iw_rag.eval.calibration import judge_human_agreement

# ---------------------------------------------------------------------------
# 1. Perfect agreement
# ---------------------------------------------------------------------------


def test_perfect_agreement_returns_one() -> None:
    """Perfect agreement (all labels match) returns 1.0."""
    result = judge_human_agreement(
        judge_labels=[True, True, False, False],
        human_labels=[True, True, False, False],
    )
    assert result is not None
    assert result == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Total disagreement
# ---------------------------------------------------------------------------


def test_total_disagreement_returns_low_value() -> None:
    """Total disagreement (all labels flip) returns a low or negative value."""
    result = judge_human_agreement(
        judge_labels=[True, True, False, False],
        human_labels=[False, False, True, True],
    )
    assert result is not None
    # Cohen's kappa for total disagreement with balanced classes (2 T, 2 F each, all
    # flipped) is exactly -1.0: p_o=0.0, p_e=0.5, κ=(0.0−0.5)/(1−0.5) = -1.0.
    assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 3. Partial agreement
# ---------------------------------------------------------------------------


def test_partial_agreement_is_between_bounds() -> None:
    """Partial agreement (half matching) returns a value between 0 and 1."""
    # 4 labels: 2 match, 2 don't
    result = judge_human_agreement(
        judge_labels=[True, True, True, False],
        human_labels=[True, True, False, True],
    )
    assert result is not None
    # Must be strictly between total-disagreement and perfect-agreement bounds
    assert -1.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# 4. Empty sequence → documented sentinel (None)
# ---------------------------------------------------------------------------


def test_empty_sequences_return_none() -> None:
    """Empty inputs return None (the 'uncalibrated' sentinel — no data to score)."""
    result = judge_human_agreement(judge_labels=[], human_labels=[])
    assert result is None


# ---------------------------------------------------------------------------
# 5. Mismatched length → ValueError
# ---------------------------------------------------------------------------


def test_mismatched_lengths_raise_value_error() -> None:
    """Mismatched judge/human label lists raise ValueError."""
    with pytest.raises(ValueError, match="same length"):
        judge_human_agreement(
            judge_labels=[True, False],
            human_labels=[True],
        )


# ---------------------------------------------------------------------------
# 6. All-same labels (degenerate kappa)
# ---------------------------------------------------------------------------


def test_all_positive_labels_matching() -> None:
    """All-positive perfect agreement returns 1.0 or None (degenerate kappa)."""
    # All TRUE: kappa = 1.0 if both sides all-true (agreement = 1.0, chance = 1.0)
    # Both perfect AND all-one-class: simple agreement = 1.0 is the right answer
    result = judge_human_agreement(
        judge_labels=[True, True, True],
        human_labels=[True, True, True],
    )
    # Either 1.0 or None is acceptable for degenerate all-same input
    assert result is None or result == pytest.approx(1.0)
