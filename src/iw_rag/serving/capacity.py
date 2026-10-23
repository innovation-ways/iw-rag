"""Capacity pre-check: will a start fit in the GPU's free VRAM? (F-00098).

The card is shared with other projects, so a power-on, mode switch, profile
change or model activation must be refused *before* it starts when it cannot
fit — an OOM here can take down someone else's process.  This module is the
pure arithmetic plus the structured error; the :class:`PowerController` gathers
the inputs (live free VRAM, component budgets, what is already running).

All arithmetic is in MiB; budgets are configured in GiB and converted in one
place (:func:`gb_to_mb`).
"""

from __future__ import annotations

from dataclasses import dataclass, field


def gb_to_mb(gb: float) -> int:
    """Convert GiB to MiB (1 GiB = 1024 MiB), rounded to the nearest MiB.

    Args:
        gb: Size in GiB.

    Returns:
        The size in MiB.
    """
    return round(gb * 1024)


@dataclass(frozen=True, slots=True)
class CapacityVerdict:
    """The outcome of a capacity check.

    Attributes:
        ok: ``True`` when the action may proceed.
        needed_mb: VRAM the action would add (components not already on).
        free_mb: Live free VRAM before the action, or ``None`` when unknown.
        projected_free_mb: Free VRAM after the action's own stops, or ``None``.
        margin_mb: Free VRAM that must remain after the start.
        shortfall_mb: How much is missing (``0`` when ok).
        skipped: Why the check did not gate the action (``"nvml_unavailable"``,
            ``"disabled"`` or ``"forced"``), else ``None``.
        components: Component keys the action would start.
    """

    ok: bool
    needed_mb: int
    free_mb: int | None
    projected_free_mb: int | None
    margin_mb: int
    shortfall_mb: int
    skipped: str | None = None
    components: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Serialise to the JSON body shared by the sidecar, client and routes."""
        return {
            "ok": self.ok,
            "needed_mb": self.needed_mb,
            "free_mb": self.free_mb,
            "projected_free_mb": self.projected_free_mb,
            "margin_mb": self.margin_mb,
            "shortfall_mb": self.shortfall_mb,
            "skipped": self.skipped,
            "components": list(self.components),
        }


class InsufficientVram(Exception):
    """Raised when a start would not fit in the free VRAM (maps to HTTP 409).

    Args:
        verdict: The failing :class:`CapacityVerdict`.
    """

    def __init__(self, verdict: CapacityVerdict) -> None:
        self.verdict = verdict
        free = "unknown" if verdict.free_mb is None else f"{verdict.free_mb} MiB"
        super().__init__(
            f"insufficient VRAM: need {verdict.needed_mb} MiB + {verdict.margin_mb} MiB margin, "
            f"free {free}, short by {verdict.shortfall_mb} MiB "
            f"(components: {verdict.components})"
        )


def check_capacity(
    *,
    starts_mb: int,
    freed_mb: int,
    free_mb: int | None,
    margin_mb: int,
    enabled: bool = True,
    force: bool = False,
    components: list[str] | None = None,
) -> CapacityVerdict:
    """Decide whether ``starts_mb`` fits once ``freed_mb`` has been released.

    Allowed iff ``free_mb + freed_mb - starts_mb >= margin_mb`` (the exact
    limit is allowed; one MiB short is refused).  An action that starts
    nothing (``starts_mb == 0``) is always allowed.

    Args:
        starts_mb: VRAM the action adds (components not already on).
        freed_mb: VRAM the same action frees first (its own stops).
        free_mb: Live free VRAM, or ``None`` when NVML is unavailable.
        margin_mb: Free VRAM that must remain afterwards.
        enabled: ``False`` disables the check (always ok, ``skipped="disabled"``).
        force: Operator override (always ok, ``skipped="forced"``).
        components: Component keys being started (for the verdict / UI).

    Returns:
        The :class:`CapacityVerdict`.
    """
    comps = list(components or [])
    projected = None if free_mb is None else free_mb + freed_mb
    if not enabled:
        return CapacityVerdict(True, starts_mb, free_mb, projected, margin_mb, 0, "disabled", comps)
    if force:
        return CapacityVerdict(True, starts_mb, free_mb, projected, margin_mb, 0, "forced", comps)
    if starts_mb <= 0:
        # Nothing new is loaded, so there is nothing to refuse.
        return CapacityVerdict(True, 0, free_mb, projected, margin_mb, 0, None, comps)
    if projected is None:
        return CapacityVerdict(
            True, starts_mb, free_mb, None, margin_mb, 0, "nvml_unavailable", comps
        )
    shortfall = max(0, starts_mb + margin_mb - projected)
    return CapacityVerdict(
        shortfall == 0, starts_mb, free_mb, projected, margin_mb, shortfall, None, comps
    )
