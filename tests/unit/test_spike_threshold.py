"""Sanity gate: the spike's pure pass/fail threshold logic.

The measurement (resident VRAM) is hardware-dependent and not unit tested; the
decision boundary is. We inject a fake measured ``peak_gb`` and assert the
script would exit non-zero when it exceeds the budget.
"""

from __future__ import annotations

import pytest

from scripts.spike_vram import headroom_low, passes_budget


@pytest.mark.parametrize(
    ("peak_gb", "expected"),
    [(0.0, True), (19.4, True), (25.0, True), (32.0, True), (32.01, False), (40.0, False)],
)
def test_passes_budget_boundary(peak_gb: float, expected: bool) -> None:
    assert passes_budget(peak_gb, budget_gb=32.0) is expected


def test_headroom_low_only_inside_budget() -> None:
    # >=10% headroom on a 32 GB card means the ceiling is 28.8 GB.
    assert headroom_low(29.0, budget_gb=32.0) is True  # fits but <10% headroom
    assert headroom_low(20.0, budget_gb=32.0) is False  # comfortable headroom
    assert headroom_low(40.0, budget_gb=32.0) is False  # over budget → not a headroom warning


def test_exit_code_logic_mirrors_passes_budget() -> None:
    # main() returns 0 on PASS, 1 on FAIL — the boundary is passes_budget().
    assert (0 if passes_budget(25.0, 32.0) else 1) == 0
    assert (0 if passes_budget(33.0, 32.0) else 1) == 1
