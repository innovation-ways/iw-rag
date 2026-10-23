"""Operating profiles: which managed components stay on or off (F-00098).

A profile names components that must be running (``on``) and components that
must be stopped (``off``).  Components it does not name are left untouched, so
applying ``light`` never stops a benchmark judge.  Planning is pure; applying
lives in :meth:`PowerController.apply_profile`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from iw_rag.settings import PowerComponentCfg, PowerControlSettings, PowerProfileCfg

_HOLDING = ("on", "partial")


@dataclass(frozen=True, slots=True)
class ProfilePlan:
    """What applying a profile would do from the current state.

    Attributes:
        stop: Component keys to stop (named in ``off`` and currently holding memory).
        start: Component keys to start (named in ``on`` and currently not running).
        budget_gb: Total configured VRAM budget of the profile's ``on`` components.
    """

    stop: list[str] = field(default_factory=list)
    start: list[str] = field(default_factory=list)
    budget_gb: float = 0.0


def resolve_profile(name: str, settings: PowerControlSettings) -> PowerProfileCfg:
    """Return the configured profile called ``name``.

    Args:
        name: Profile key (``normal`` / ``light`` / ``yield`` by default).
        settings: The power settings holding the profiles.

    Returns:
        The :class:`PowerProfileCfg`.

    Raises:
        KeyError: If no such profile is configured.
    """
    try:
        return settings.profiles[name]
    except KeyError:
        raise KeyError(
            f"unknown power profile {name!r}; known: {sorted(settings.profiles)}"
        ) from None


def plan_profile(
    profile: PowerProfileCfg,
    components: dict[str, PowerComponentCfg],
    states: dict[str, str],
) -> ProfilePlan:
    """Plan the stops and starts needed to reach ``profile`` from ``states``.

    Pure and side-effect free.  A ``partial`` component holds memory already,
    so it is stopped when the profile wants it off but never "started".

    Args:
        profile: The target profile.
        components: The managed components (for budgets).
        states: Current ``component key -> state`` (``on`` / ``off`` / ``partial`` / ...).

    Returns:
        The :class:`ProfilePlan`.
    """
    stop = [k for k in profile.off if states.get(k) in _HOLDING]
    start = [k for k in profile.on if states.get(k) not in _HOLDING and k in components]
    budget = sum(components[k].vram_gb for k in profile.on if k in components)
    return ProfilePlan(stop=stop, start=start, budget_gb=budget)
