"""Judge–human calibration — agreement metric over the held-out slice.

Computes how well the local offline judge (Lynx-8B / Selene-Mini-8B) agrees
with human-labelled ground truth on a small held-out **calibration slice**.
This number is part of the published trust story — every judged metric is
reported alongside it (lld-eval-observability §5, R-00110/F5, ADR-013).

**Metric: Cohen's kappa** — chosen over simple accuracy because it corrects for
chance agreement, which is important when the label distribution is skewed (a
judge that always says "FAITHFUL" can achieve high accuracy on a faithful corpus
by chance; kappa penalises this).  Cohen's kappa formula:

    κ = (p_o - p_e) / (1 - p_e)

where p_o is observed agreement and p_e is expected chance agreement for two
binary annotators.  κ = 1.0 → perfect agreement; κ = 0.0 → chance; κ < 0 →
worse than chance.  Returns ``None`` for edge-cases where kappa is undefined
(empty input, all-same labels with degenerate chance-agreement denominator).

The ``None`` sentinel is the "**uncalibrated**" flag referenced in
lld-eval-observability §9: callers must surface it in CI warnings rather than
silently trusting the judge.
"""

from __future__ import annotations

from collections.abc import Sequence


def judge_human_agreement(
    judge_labels: Sequence[bool],
    human_labels: Sequence[bool],
) -> float | None:
    """Compute Cohen's kappa agreement between judge and human labels.

    Operates on binary (True/False) labels from the held-out calibration slice.
    Returns ``None`` (the "uncalibrated" sentinel) when no data is available or
    when the expected-agreement denominator is zero (degenerate all-same-class
    case), so callers can surface an ``uncalibrated`` warning rather than an
    invalid number.

    Args:
        judge_labels: Binary verdicts from the offline judge (True = positive).
        human_labels: Ground-truth human labels for the same samples.

    Returns:
        Cohen's kappa in [-1.0, 1.0], or ``None`` when the input is empty or
        the kappa is undefined (all labels identical on both sides, making
        expected agreement = 1.0 and the denominator zero).

    Raises:
        ValueError: If ``judge_labels`` and ``human_labels`` have different
            lengths.
    """
    if len(judge_labels) != len(human_labels):
        raise ValueError(
            f"judge_labels and human_labels must be the same length; "
            f"got {len(judge_labels)} vs {len(human_labels)}."
        )
    n = len(judge_labels)
    if n == 0:
        # No data → uncalibrated sentinel
        return None

    # Observed agreement (fraction of identical labels); strict → equal lengths.
    agree = sum(1 for j, h in zip(judge_labels, human_labels, strict=True) if j == h)
    p_o = agree / n

    # Expected agreement by chance (product of marginals for each class)
    # p_e = p(judge=T) * p(human=T) + p(judge=F) * p(human=F)
    judge_pos = sum(1 for j in judge_labels if j) / n
    judge_neg = 1.0 - judge_pos
    human_pos = sum(1 for h in human_labels if h) / n
    human_neg = 1.0 - human_pos

    p_e = judge_pos * human_pos + judge_neg * human_neg

    denominator = 1.0 - p_e
    if abs(denominator) < 1e-12:
        # Degenerate case: all labels the same on both sides (p_e → 1).
        # If p_o == 1.0 as well (perfect agreement), return 1.0.
        # Otherwise return None (undefined kappa).
        return 1.0 if p_o >= 1.0 - 1e-12 else None

    return (p_o - p_e) / denominator
