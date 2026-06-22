"""Cross-cutting unit tests for the F-00096 S08 VRAM budget math.

The VRAM math lives in
:func:`iw_rag.api.models_routes._compute_vram_budget` (S08).  The
S08 unit tests in ``tests/unit/api/test_models_routes.py`` exercise
the math end-to-end through the ``GET /models`` HTTP route.  This
file is the **cross-cutting math keystone test** the S17 prompt pins
separately so the S17 audit can stand on its own without re-asserting
the S08 sibling — the tests call ``_compute_vram_budget`` directly
(no HTTP round-trip, no fakes) so the boundary conditions the design
names are pinned with the **exact** numeric contract.

The two named module constants (one obvious place):

* ``RESIDENT_INFINITY_GB = 4.0`` — Infinity is ALWAYS resident
  (~4 GB VRAM; per ``config/model_catalog.yaml`` header).
* ``VRAM_CEILING_GB = 32.0`` — single 32 GB GPU (CLAUDE.md, F-00096
  §Scope).

Formula (mirror of the S08 docstring + the S08 unit test):

* ``budget_used_gb = resident_infinity_gb + expected_vram_gb``
* ``headroom_gb    = ceiling_gb - budget_used_gb``
* ``fits           = budget_used_gb <= ceiling_gb``

Boundary cases (AC1 / Boundary row "VRAM over budget"):

* A candidate at the ceiling exactly → ``fits=True``,
  ``headroom_gb == 0.0`` (the bar touches the right edge).
* A candidate one GB over the ceiling → ``fits=False``,
  ``headroom_gb < 0`` (red zone).
* Zero-budget / unreachable placeholder → ``fits=True`` (the
  reachable-but-unmeasured state; the UI suppresses the red zone).
"""

from __future__ import annotations

import pytest

from iw_rag.api.models_routes import (
    RESIDENT_INFINITY_GB,
    VRAM_CEILING_GB,
    _compute_vram_budget,
)

# ---------------------------------------------------------------------------
# A. Constants are pinned to the design values
# ---------------------------------------------------------------------------


def test_vram_constants_match_design() -> None:
    """The two module constants are pinned to the F-00096 §Scope values.

    * ``RESIDENT_INFINITY_GB = 4.0`` — Infinity is ALWAYS resident.
    * ``VRAM_CEILING_GB = 32.0`` — single 32 GB GPU.

    A future change here must be a reviewed decision (the S08 model
    catalog header documents the same value).
    """
    assert RESIDENT_INFINITY_GB == 4.0
    assert VRAM_CEILING_GB == 32.0


# ---------------------------------------------------------------------------
# B. In-budget candidate — exact numeric contract
# ---------------------------------------------------------------------------


def test_vram_budget_under_ceiling_exact_numeric() -> None:
    """A candidate under the ceiling reports the exact ``fits / budget / headroom`` triple.

    For ``expected_vram_gb=5.0`` the math must compute:

        budget_used_gb = 4.0 (resident) + 5.0 = 9.0
        headroom_gb    = 32.0 - 9.0 = 23.0
        fits           = (9.0 <= 32.0) = True

    Pinned with ``==`` (not ``approx``) because the inputs are exact
    floats and the production catalog carries 1-decimal GB values
    that round-trip cleanly.
    """
    budget = _compute_vram_budget(5.0)
    assert budget.fits is True
    assert budget.budget_used_gb == 9.0
    assert budget.headroom_gb == 23.0


# ---------------------------------------------------------------------------
# C. At-ceiling candidate — boundary "bar touches right edge" (AC1)
# ---------------------------------------------------------------------------


def test_vram_budget_at_ceiling_exact_fits_true_zero_headroom() -> None:
    """A candidate at the ceiling exactly reports ``fits=True`` and ``headroom_gb=0.0``.

    Boundary case (AC1): a pick that *exactly* fits is GREEN, not
    red.  The bar fills 100% but the operator sees "fits" (the
    red-zone wiring in the S12 Benchmark Runner view is keyed off
    ``fits === false``, not off the bar fill percentage).
    """
    budget = _compute_vram_budget(28.0)  # 4.0 + 28.0 == 32.0 (ceiling)
    assert budget.fits is True
    assert budget.budget_used_gb == 32.0
    assert budget.headroom_gb == 0.0


# ---------------------------------------------------------------------------
# D. Over-budget candidate — red zone, negative headroom (AC1 / Boundary "VRAM over budget")
# ---------------------------------------------------------------------------


def test_vram_budget_one_gb_over_ceiling_reports_fits_false() -> None:
    """A pick 1 GB over the ceiling reports ``fits=False`` and ``headroom_gb=-1.0``.

    Boundary row "VRAM over budget" + AC1: the red zone triggers when
    ``fits is False``.  A 1-GB overshoot is a clean, exact float
    (avoids the IEEE-754 binary-representation drift the S08 unit
    test noted; the real catalog uses 1-decimal GB values).
    """
    budget = _compute_vram_budget(29.0)  # 4.0 + 29.0 == 33.0 (1 GB over)
    assert budget.fits is False
    assert budget.budget_used_gb == 33.0
    assert budget.headroom_gb == -1.0


# ---------------------------------------------------------------------------
# E. Per-candidate isolation — one over-budget pick does not poison the others
# ---------------------------------------------------------------------------


def test_vram_budget_isolated_per_candidate_call() -> None:
    """The math is per-call: one over-budget pick does not influence a subsequent call.

    Pinned here so a future refactor that caches a stale
    ``budget_used_gb`` across calls breaks loudly (the S08 unit test
    asserts this end-to-end via the ``GET /models`` route; this is
    the unit-level mirror).
    """
    a = _compute_vram_budget(5.0)
    b = _compute_vram_budget(29.0)  # over budget
    c = _compute_vram_budget(10.0)  # back under
    assert a.fits is True
    assert a.budget_used_gb == 9.0
    assert b.fits is False
    assert b.budget_used_gb == 33.0
    # The third call sees a clean state (no carry-over from b).
    assert c.fits is True
    assert c.budget_used_gb == 14.0
    assert c.headroom_gb == 18.0


# ---------------------------------------------------------------------------
# F. Zero-budget / unmeasured placeholder — fits=True (no measurement)
# ---------------------------------------------------------------------------


def test_vram_budget_zero_value_placeholder_fits_true() -> None:
    """The zero-budget placeholder (``expected_vram_gb=0.0``) reports ``fits=True``.

    The S08 report notes: "the unreachable placeholder row carries
    the zero-budget marker (the design prompt explicitly allows a
    zero budget for the unreachable placeholder)".  The math is
    plain arithmetic: ``4.0 + 0.0 == 4.0 <= 32.0`` so the placeholder
    row does NOT enter the red zone.
    """
    budget = _compute_vram_budget(0.0)
    assert budget.fits is True
    assert budget.budget_used_gb == 4.0
    assert budget.headroom_gb == 28.0


# ---------------------------------------------------------------------------
# G. Idempotence — the same call with the same input returns the same triple
# ---------------------------------------------------------------------------


def test_vram_budget_is_idempotent_for_same_input() -> None:
    """The math is a pure function — same input → same output, every time.

    Pinned as a sanity check against any future refactor that
    introduces hidden state (e.g. an instance-cached previous
    candidate).
    """
    a = _compute_vram_budget(5.0)
    b = _compute_vram_budget(5.0)
    assert a.fits == b.fits
    assert a.budget_used_gb == b.budget_used_gb
    assert a.headroom_gb == b.headroom_gb


# ---------------------------------------------------------------------------
# H. Schema field set — the VramBudgetOut shape the S14 page consumes
# ---------------------------------------------------------------------------


def test_vram_budget_out_schema_field_set() -> None:
    """``VramBudgetOut`` exposes exactly the documented triple (no extra fields).

    The S14 Scoreboard view (and S12 Benchmark Runner view) consume
    this shape verbatim; a future addition of fields would be a
    contract break.  Pinned here so the change is loud.
    """
    from iw_rag.api.schemas import VramBudgetOut

    fields = set(VramBudgetOut.model_fields.keys())
    assert fields == {"fits", "budget_used_gb", "headroom_gb"}


# ---------------------------------------------------------------------------
# I. The exact-arithmetic contract — no client-side recomputation hint
# ---------------------------------------------------------------------------


def test_vram_budget_returns_exact_floats_no_rounding_artefacts() -> None:
    """The triple values are Python floats with the exact bit pattern.

    Pinned here so a future refactor that rounds / quantises the
    result (e.g. ``round(budget_used_gb, 2)``) breaks loudly — the
    API contract is exact float (the front-end reads the triple
    verbatim).
    """
    budget = _compute_vram_budget(5.0)
    # 4.0 + 5.0 is exact in IEEE-754 (both exact; sum is exact).
    assert budget.budget_used_gb == pytest.approx(9.0, rel=0, abs=0)
    assert budget.headroom_gb == pytest.approx(23.0, rel=0, abs=0)
