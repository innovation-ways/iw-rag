"""API-side power policy: activity tracking, idle auto-unload, block-and-warm.

The privilege split (design): the **sidecar** executes ``docker stop/start``;
the **API** owns policy.  Policy lives here:

* :class:`PowerActivityTracker` — records the wall-clock of the last query so
  the idle monitor knows how long the box has been idle.
* :func:`should_idle_unload` — the **pure** decision: given the last-activity
  time, now, the threshold, the current power status, and the mode, return the
  component keys to auto-unload (empty = do nothing).  Unit-tested in isolation.
* :func:`power_idle_monitor` — the thin background loop that calls the pure
  decision and proxies a ``power_off`` to the sidecar when it fires.
* :func:`maybe_warm_generator` — block-and-warm: a query arriving while the
  generator is powered off transparently powers it back on and waits.

Every public entry point is **fail-safe**: power control is an optional
deployment feature (``Settings.power.enabled`` + a wired ``power_client``).
When it is absent, or the sidecar is unreachable, these helpers no-op and the
query path behaves exactly as before — power policy never breaks a query.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from fastapi import HTTPException, Request

from iw_rag.serving.capacity import gb_to_mb
from iw_rag.serving.power_client import PowerCapacityError, PowerClient, PowerControlUnavailable

logger = logging.getLogger(__name__)


class PowerActivityTracker:
    """Tracks the monotonic time of the last user query.

    The idle monitor compares ``last_activity`` against ``time.monotonic()``;
    every query :meth:`touch` -es it.  Monotonic (not wall-clock) so a system
    clock change never makes the box look idle for hours.
    """

    def __init__(self) -> None:
        # Initialise to "now" so a freshly booted box is not instantly idle.
        self._last_activity = time.monotonic()

    def touch(self) -> None:
        """Record that activity just happened (called on every query)."""
        self._last_activity = time.monotonic()

    @property
    def last_activity(self) -> float:
        """The monotonic timestamp of the most recent :meth:`touch`."""
        return self._last_activity


def is_serving_mode(settings: Any) -> bool:
    """Return ``True`` when the box is in serving mode (not benchmark).

    Mirrors :func:`iw_rag.api.benchmark_routes.benchmark_mode`: the mode is
    inferred from the generator endpoint — a ``llama-swap`` endpoint means
    benchmark mode (llama-swap owns the GPU, so power control + idle-unload
    are disabled; the swap-controller is the authority there).

    Args:
        settings: The ``Settings`` instance (or ``None``).

    Returns:
        ``True`` for serving mode (or when the endpoint can't be read —
        fail toward serving so a misread never silently disables the switch
        on the real serving box).
    """
    if settings is None:
        return True
    try:
        endpoint = settings.models.get(settings.generation.model).endpoint
    except Exception:  # noqa: BLE001 — best-effort introspection
        return True
    return "llama-swap" not in (endpoint or "")


def resolve_mode(state: Any) -> str:
    """Return the box's current mode (``"serving"`` | ``"benchmark"``).

    Prefers the runtime mode set by the in-UI mode toggle
    (``app.state.runtime_mode``); falls back to the static, endpoint-derived
    :func:`is_serving_mode` when no toggle has happened this process.

    Args:
        state: The app state object (``request.app.state``).

    Returns:
        ``"serving"`` or ``"benchmark"``.
    """
    rt = getattr(state, "runtime_mode", None)
    if rt in ("serving", "benchmark"):
        return str(rt)
    return "serving" if is_serving_mode(getattr(state, "settings", None)) else "benchmark"


def should_idle_unload(
    *,
    last_activity: float,
    now: float,
    threshold_s: float,
    idle_enabled: bool,
    serving_mode: bool,
    status: dict[str, Any] | None,
    idle_components: list[str],
) -> list[str]:
    """Pure decision: which components to auto-unload right now (or none).

    Returns the subset of ``idle_components`` that are currently ``on`` once
    the box has been idle past ``threshold_s`` — but only when idle-unload is
    enabled, the box is in serving mode, and no transition is already in
    flight.  Returns ``[]`` (do nothing) otherwise.

    Args:
        last_activity: Monotonic time of the last query.
        now: Monotonic time now.
        threshold_s: Idle threshold in seconds.
        idle_enabled: ``Settings.power.idle_unload_enabled``.
        serving_mode: Whether the box is in serving mode (benchmark → never).
        status: The latest power status payload (``None`` when unknown).
        idle_components: Component keys whose config opts into idle-unload.

    Returns:
        The component keys to unload (a subset of ``idle_components``), or
        ``[]`` when nothing should happen.
    """
    if not idle_enabled or not serving_mode or not idle_components:
        return []
    if (now - last_activity) < threshold_s:
        return []
    if not isinstance(status, dict):
        return []
    # Never act while a transition is in flight.
    if status.get("transition") is not None:
        return []
    on_keys = {
        c.get("key")
        for c in status.get("components", [])
        if isinstance(c, dict) and c.get("state") == "on"
    }
    # Only unload idle-eligible components that are actually on right now.
    return [k for k in idle_components if k in on_keys]


def _idle_component_keys(settings: Any) -> list[str]:
    """Return the component keys whose config opts into idle-unload."""
    try:
        return [k for k, cfg in settings.power.components.items() if cfg.idle_unload]
    except Exception:  # noqa: BLE001
        return []


async def power_idle_monitor(app: Any, *, check_interval_s: float = 30.0) -> None:
    """Background loop: auto-unload idle components after the threshold.

    Started at app startup (see :func:`iw_rag.api.app.create_app`).  Each tick
    reads the current status from the sidecar, runs the pure
    :func:`should_idle_unload` decision, and issues a ``power_off`` for any
    components it returns.  Every tick is wrapped so a transient sidecar
    outage (or any error) never kills the monitor or the app.

    Args:
        app: The FastAPI app (for ``app.state``).
        check_interval_s: Seconds between idle checks.
    """
    state = app.state
    settings = getattr(state, "settings", None)
    client: PowerClient | None = getattr(state, "power_client", None)
    tracker: PowerActivityTracker | None = getattr(state, "power_activity", None)
    if settings is None or client is None or tracker is None:
        return
    if not getattr(settings.power, "enabled", False):
        return
    threshold_s = float(settings.power.idle_unload_minutes) * 60.0
    idle_components = _idle_component_keys(settings)
    while True:
        try:
            await asyncio.sleep(check_interval_s)
            status = await client.get_status()
            keys = should_idle_unload(
                last_activity=tracker.last_activity,
                now=time.monotonic(),
                threshold_s=threshold_s,
                idle_enabled=bool(settings.power.idle_unload_enabled),
                serving_mode=(resolve_mode(state) == "serving"),
                status=status,
                idle_components=idle_components,
            )
            if keys:
                logger.info("idle %.0fs → auto-unloading %s", threshold_s, keys)
                await client.power_off(keys)
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except PowerControlUnavailable:
            logger.debug("idle monitor: system-control unreachable; will retry")
        except Exception:  # noqa: BLE001 — the monitor must never die
            logger.warning("idle monitor tick failed; continuing", exc_info=True)


# ---------------------------------------------------------------------------
# Reactive yield (F-00098) — release memory when OTHER processes squeeze the card
# ---------------------------------------------------------------------------


class PowerEventLog:
    """Bounded in-memory log of GPU-aware events shown on the System view.

    Kinds: ``yield`` (components released), ``yield_deferred`` (pressure seen
    but a benchmark run is active), ``warm_refused`` (a re-warm did not fit),
    ``forced`` (an operator bypassed the capacity pre-check).

    Args:
        max_events: Maximum events kept; the oldest is dropped when full.
    """

    def __init__(self, max_events: int = 100) -> None:
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

    def add(
        self,
        kind: str,
        *,
        ts: float | None = None,
        free_mb: int | None = None,
        components: list[str] | None = None,
        detail: str = "",
    ) -> None:
        """Record one event.

        Args:
            kind: Event kind (see the class docstring).
            ts: Epoch seconds; defaults to now.
            free_mb: Free VRAM at the time, MiB, if known.
            components: Component keys involved.
            detail: Short human-readable context.
        """
        self._events.append(
            {
                "ts": time.time() if ts is None else ts,
                "kind": kind,
                "free_mb": free_mb,
                "components": list(components or []),
                "detail": detail,
            }
        )

    def recent(self, n: int = 20) -> list[dict[str, Any]]:
        """Return the newest ``n`` events, oldest first.

        Args:
            n: How many events to return.

        Returns:
            Up to ``n`` event dicts in chronological order.
        """
        return list(self._events)[-n:] if n > 0 else []


@dataclass(frozen=True, slots=True)
class YieldDecision:
    """The pure outcome of one yield-monitor tick.

    Attributes:
        action: ``"none"``, ``"yield"`` (power ``components`` off) or
            ``"defer"`` (pressure, but a benchmark run is active).
        components: Component keys to release (empty for ``none``).
        below_since: Monotonic time free VRAM first went below the threshold
            (carry into the next tick), or ``None`` when not under pressure.
        reason: Short explanation for logs.
    """

    action: str
    components: list[str] = field(default_factory=list)
    below_since: float | None = None
    reason: str = ""


def should_yield(
    *,
    enabled: bool,
    free_mb: int | None,
    below_since: float | None,
    now: float,
    threshold_mb: int,
    sustain_s: float,
    cooldown_s: float,
    last_yield_at: float | None,
    status: dict[str, Any] | None,
    benchmark_active: bool,
    yield_components: list[str],
) -> YieldDecision:
    """Pure decision: should iw-rag release memory right now?

    Free VRAM must stay below the threshold for ``sustain_s``.  Never acts
    while a power transition is in flight, during the cooldown after a
    previous yield, or when no yield-eligible component is on.  While a
    benchmark run is active the decision is ``defer`` (stopping a judge mid-run
    would corrupt the run).  Only components flagged ``yield_unload`` are ever
    released, so the embedder stays up.

    Args:
        enabled: ``Settings.power.yield_enabled``.
        free_mb: Live free VRAM in MiB, or ``None`` when NVML is unavailable.
        below_since: Carry-over from the previous tick.
        now: Monotonic time now.
        threshold_mb: Free-VRAM level below which the card is under pressure.
        sustain_s: How long the pressure must last.
        cooldown_s: Minimum gap after a yield.
        last_yield_at: Monotonic time of the previous yield, if any.
        status: The latest power status payload (``None`` when unknown).
        benchmark_active: Whether a benchmark run is in progress.
        yield_components: Component keys eligible for yield.

    Returns:
        The :class:`YieldDecision`.
    """
    if not enabled or free_mb is None:
        return YieldDecision("none", reason="disabled or no vram reading")
    if free_mb >= threshold_mb:
        return YieldDecision("none", reason="enough free vram")
    since = now if below_since is None else below_since
    if (now - since) < sustain_s:
        return YieldDecision("none", below_since=since, reason="pressure not sustained yet")
    if not isinstance(status, dict) or status.get("transition") is not None:
        return YieldDecision("none", below_since=since, reason="transition in flight")
    if last_yield_at is not None and (now - last_yield_at) < cooldown_s:
        return YieldDecision("none", below_since=since, reason="cooldown")
    on_keys = {
        c.get("key")
        for c in status.get("components", [])
        if isinstance(c, dict) and c.get("state") in ("on", "partial")
    }
    eligible = [k for k in yield_components if k in on_keys]
    if not eligible:
        return YieldDecision("none", below_since=since, reason="nothing eligible is on")
    if benchmark_active:
        return YieldDecision("defer", eligible, since, "benchmark run active")
    return YieldDecision("yield", eligible, None, "sustained pressure")


def _benchmark_active(state: Any) -> bool:
    """Return ``True`` when a benchmark run is in progress (fail toward protecting it)."""
    manager = getattr(state, "benchmark_run_manager", None)
    if manager is None:
        return False
    try:
        return bool(manager.is_active())
    except Exception:  # noqa: BLE001 — unknown ⇒ protect the run
        return True


async def power_yield_monitor(app: Any) -> None:
    """Background loop: release generator/judges when other processes squeeze the card.

    Each tick reads the power status (components + VRAM) from the sidecar, runs
    the pure :func:`should_yield` decision and powers the chosen components
    off, recording a ``yield`` / ``yield_deferred`` event.  Fail-safe like the
    idle monitor: a sidecar outage or any error never kills the loop.

    Args:
        app: The FastAPI app (for ``app.state``).

    Raises:
        asyncio.CancelledError: On shutdown.
    """
    state = app.state
    settings = getattr(state, "settings", None)
    client: PowerClient | None = getattr(state, "power_client", None)
    if settings is None or client is None:
        return
    power = settings.power
    if not getattr(power, "enabled", False) or not getattr(power, "yield_enabled", False):
        return
    events: PowerEventLog | None = getattr(state, "power_events", None)
    yield_keys = [k for k, c in power.components.items() if getattr(c, "yield_unload", False)]
    threshold_mb = gb_to_mb(power.yield_free_gb_threshold)
    below_since: float | None = None
    last_yield_at: float | None = None
    deferred_logged = False
    while True:
        try:
            await asyncio.sleep(power.yield_check_interval_s)
            status = await client.get_status()
            vram = status.get("vram") or {}
            free_mb = vram.get("free_mb") if vram.get("available") else None
            now = time.monotonic()
            decision = should_yield(
                enabled=True,
                free_mb=free_mb,
                below_since=below_since,
                now=now,
                threshold_mb=threshold_mb,
                sustain_s=float(power.yield_sustain_s),
                cooldown_s=float(power.yield_cooldown_s),
                last_yield_at=last_yield_at,
                status=status,
                benchmark_active=_benchmark_active(state),
                yield_components=yield_keys,
            )
            below_since = decision.below_since
            if decision.below_since is None:
                deferred_logged = False
            if decision.action == "yield":
                logger.info("yield: free %s MiB → releasing %s", free_mb, decision.components)
                await client.power_off(decision.components)
                last_yield_at = now
                if events is not None:
                    events.add(
                        "yield",
                        free_mb=free_mb,
                        components=decision.components,
                        detail="free VRAM stayed below the threshold (other processes)",
                    )
            elif decision.action == "defer" and not deferred_logged:
                deferred_logged = True
                if events is not None:
                    events.add(
                        "yield_deferred",
                        free_mb=free_mb,
                        components=decision.components,
                        detail="benchmark run active — not releasing",
                    )
        except asyncio.CancelledError:  # pragma: no cover - shutdown
            raise
        except PowerControlUnavailable:
            logger.debug("yield monitor: system-control unreachable; will retry")
        except Exception:  # noqa: BLE001 — the monitor must never die
            logger.warning("yield monitor tick failed; continuing", exc_info=True)


async def maybe_warm_generator(request: Request) -> None:
    """Block-and-warm: if the generator is powered off, power it on and wait.

    Called at the top of the query path.  When power control is enabled, the
    box is in serving mode, and the generator component is currently ``off``,
    this powers it back on (``power_on(["generator"])``) and blocks until the
    sidecar reports it ready — so the first query after an idle auto-unload
    transparently pays the cold-start and then answers, instead of failing.

    Always **fail-safe**: any missing wiring, benchmark mode, or a sidecar
    error makes this a no-op (the downstream generation call surfaces the real
    error if the model genuinely is not up).  Also :meth:`PowerActivityTracker.touch`
    -es the activity tracker so the idle clock resets on every query.

    Args:
        request: The FastAPI request (for ``app.state``).
    """
    state = request.app.state
    tracker: PowerActivityTracker | None = getattr(state, "power_activity", None)
    if tracker is not None:
        tracker.touch()
    settings = getattr(state, "settings", None)
    client: PowerClient | None = getattr(state, "power_client", None)
    if settings is None or client is None:
        return
    power = getattr(settings, "power", None)
    if power is None or not power.enabled or not power.block_and_warm:
        return
    if resolve_mode(state) != "serving":
        return
    try:
        status = await client.get_status()
    except PowerControlUnavailable:
        return
    if not _generator_needs_warm(status):
        return
    events: PowerEventLog | None = getattr(state, "power_events", None)
    try:
        verdict = await client.get_capacity(["generator"])
    except PowerControlUnavailable:
        verdict = {"ok": True}
    if not verdict.get("ok", True):
        _refuse_warm(events, verdict)
    logger.info("block-and-warm: generator is off — powering on before serving the query")
    try:
        await client.power_on(["generator"])
    except PowerCapacityError as exc:
        _refuse_warm(events, exc.body)
    except PowerControlUnavailable:
        logger.warning("block-and-warm: power-on failed; serving will surface the real error")


def _refuse_warm(events: PowerEventLog | None, verdict: dict[str, Any]) -> None:
    """Fail the query fast with 503 ``gpu_busy`` instead of risking an OOM.

    Args:
        events: Event log to record the refusal in (optional).
        verdict: The sidecar's capacity verdict / ``insufficient_vram`` body.

    Raises:
        HTTPException: Always — 503 with ``error="gpu_busy"`` and a ``Retry-After``.
    """
    mb = 1024.0
    if events is not None:
        events.add(
            "warm_refused",
            free_mb=verdict.get("free_mb"),
            components=["generator"],
            detail="not enough free VRAM to re-warm the generator",
        )
    raise HTTPException(
        status_code=503,
        detail={
            "error": "gpu_busy",
            "message": "The GPU is busy with other workloads; the generator cannot be loaded yet.",
            "needed_gb": round((verdict.get("needed_mb") or 0) / mb, 1),
            "free_gb": round((verdict.get("free_mb") or 0) / mb, 1),
            "shortfall_gb": round((verdict.get("shortfall_mb") or 0) / mb, 1),
        },
        headers={"Retry-After": "30"},
    )


def _generator_needs_warm(status: dict[str, Any] | None) -> bool:
    """Return ``True`` when the ``generator`` component is off/partial.

    A transition already in flight (another warm racing us) is treated as
    "do not start a second one" — we let the in-flight power-on finish.
    """
    if not isinstance(status, dict):
        return False
    if status.get("transition") is not None:
        return False
    for c in status.get("components", []):
        if isinstance(c, dict) and c.get("key") == "generator":
            return c.get("state") in ("off", "partial")
    return False
