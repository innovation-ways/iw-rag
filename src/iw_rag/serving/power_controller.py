"""Power state machine for the VRAM power switch (system-control sidecar).

The :class:`PowerController` is the single power-state authority for the
serving box.  It maps named *components* (``generator`` / ``embedder`` /
``guards``) to Docker containers and exposes three operations:

* :meth:`status` — inspect every managed container + read VRAM → a
  :class:`PowerStatus` snapshot (the console gauge + toggles render this).
* :meth:`power_off` — ``docker stop`` a component set (or all) → frees VRAM.
* :meth:`power_on` — ``docker start`` a component set (or all) and poll each
  container to ready (running + healthy) → reloads the models.

**State machine** (overall, derived from the component set):

    OFF ──power_on──▶ POWERING_ON ──ready──▶ ON
     ▲                                        │
     └────────── POWERING_OFF ◀──power_off────┘
    (mixed component states → PARTIAL; daemon unreachable → UNKNOWN)

A single :class:`asyncio.Lock` serialises mutating operations so two
concurrent power flips cannot interleave.  An ``_inflight`` marker (read
*without* the lock) lets :meth:`status` report ``POWERING_ON`` / ``POWERING_OFF``
*during* a transition — the same split-brain-safe pattern as the
swap-controller (the lock guards the mutation; the marker is the live view).

This module is pure orchestration: it owns NO socket and NO NVML handle — it
depends on injected :class:`~iw_rag.serving.docker_engine.DockerEngineClient`
and :class:`~iw_rag.serving.vram.VramReader` protocols, so the whole state
machine is unit-testable with fakes (no daemon, no GPU).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum

from iw_rag.settings import PowerComponentCfg, PowerProfileCfg

from .capacity import CapacityVerdict, InsufficientVram, check_capacity, gb_to_mb
from .docker_engine import ContainerState, DockerEngineClient, DockerUnavailable
from .profiles import plan_profile
from .vram import VramReader, VramSnapshot


class PowerState(StrEnum):
    """The overall power state of the serving box.

    Attributes:
        ON: Every managed (present) container is running.
        OFF: Every managed (present) container is stopped.
        PARTIAL: A mix — some components on, some off.
        POWERING_ON: A power-on transition is in flight.
        POWERING_OFF: A power-off transition is in flight.
        UNKNOWN: The Docker daemon was unreachable; state is indeterminate.
    """

    ON = "on"
    OFF = "off"
    PARTIAL = "partial"
    POWERING_ON = "powering_on"
    POWERING_OFF = "powering_off"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    """The resolved state of one managed component.

    Attributes:
        key: The component key (``"generator"`` / ``"embedder"`` / ``"guards"``).
        label: Human-facing label.
        description: One-line explanation.
        idle_unload: Whether the idle monitor auto-unloads this component.
        containers: Per-container state (``absent`` rows for missing names).
        present: Count of containers that exist on the daemon.
        running: Count of present containers currently running.
        state: ``"on"`` (all present running), ``"off"`` (none running),
            ``"partial"`` (mixed), ``"absent"`` (no container exists), or
            ``"unreachable"`` (daemon down).
    """

    key: str
    label: str
    description: str
    idle_unload: bool
    containers: list[ContainerState]
    present: int
    running: int
    state: str


@dataclass(frozen=True, slots=True)
class PowerStatus:
    """A snapshot of the whole power surface (rendered by the System view).

    Attributes:
        state: The overall :class:`PowerState`.
        components: Per-component status, in declared order.
        vram: The current VRAM snapshot (may be ``available=False``).
        transition: ``None`` when idle, else the in-flight op label
            (``"power_on"`` / ``"power_off"``) and its target keys.
        transition_keys: The component keys the in-flight op targets.
        estimated_held_mb: Sum of the configured budgets of the components
            that are on — an *estimate* of what iw-rag holds (per-process VRAM
            is unavailable inside the sidecar container).
        estimated_other_mb: ``used - estimated_held_mb`` (never negative), the
            estimated share held by other processes; ``None`` without NVML.
    """

    state: PowerState
    components: list[ComponentStatus]
    vram: VramSnapshot
    transition: str | None = None
    transition_keys: list[str] = field(default_factory=list)
    estimated_held_mb: int = 0
    estimated_other_mb: int | None = None


class PowerController:
    """The power state machine — owned by the system-control sidecar.

    Args:
        components: The managed component → container mapping (from
            ``Settings.power.components``).
        docker: The Docker Engine client (real socket client in production;
            a fake in tests).
        vram: The VRAM reader (NVML in production; a fake in tests).
        ready_timeout_s: Per power-on readiness budget in seconds.
        poll_interval_s: Sleep between readiness polls in seconds.
        stop_timeout_s: Graceful-stop seconds passed to ``docker stop``.
        capacity_precheck_enabled: Whether starts are checked against live
            free VRAM and refused with :class:`InsufficientVram` when they
            would not fit (F-00098).
        capacity_margin_gb: Free VRAM that must remain after a start.
    """

    def __init__(
        self,
        components: dict[str, PowerComponentCfg],
        docker: DockerEngineClient,
        vram: VramReader,
        *,
        ready_timeout_s: int = 300,
        poll_interval_s: float = 1.0,
        stop_timeout_s: int = 10,
        capacity_precheck_enabled: bool = False,
        capacity_margin_gb: float = 0.0,
    ) -> None:
        self._components = components
        self._docker = docker
        self._vram = vram
        self._ready_timeout_s = ready_timeout_s
        self._poll_interval_s = poll_interval_s
        self._stop_timeout_s = stop_timeout_s
        self._precheck_enabled = capacity_precheck_enabled
        self._margin_mb = gb_to_mb(capacity_margin_gb)
        self._lock = asyncio.Lock()
        # Live transition marker, read WITHOUT the lock so status() reflects an
        # in-flight flip.  Tuple of (op_label, target_keys) or None.
        self._inflight: tuple[str, list[str]] | None = None

    # ------------------------------------------------------------------
    # Key resolution
    # ------------------------------------------------------------------

    def _resolve_keys(self, keys: list[str] | None) -> list[str]:
        """Return the requested component keys, or all keys when ``None``.

        Args:
            keys: Requested component keys, or ``None`` for "all managed".

        Returns:
            The validated keys in declared order.

        Raises:
            KeyError: If any requested key is not a managed component.
        """
        if keys is None:
            return list(self._components)
        unknown = [k for k in keys if k not in self._components]
        if unknown:
            raise KeyError(
                f"unknown power component(s): {unknown}; known: {list(self._components)}"
            )
        # Preserve declared order for stable, deterministic operation order.
        return [k for k in self._components if k in set(keys)]

    async def _resolve_names(self, entries: list[str]) -> list[str]:
        """Expand any ``prefix*`` pattern entries to concrete running names.

        A literal entry (``"iw-rag-vllm-1"``) passes through.  A pattern entry
        (``"vllm-*"``) is expanded to the running containers whose name starts
        with the prefix — this is how the single component map works in BOTH
        modes: the generator resolves to ``iw-rag-vllm-1`` in serving mode and
        to llama-swap's dynamic ``vllm-<id>`` in benchmark mode.  Order +
        dedup is stable (literals first, in declared order, then sorted matches).

        Args:
            entries: The raw container entries from a component config.

        Returns:
            The concrete container names (patterns expanded).
        """
        out: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if entry.endswith("*"):
                try:
                    matches = await self._docker.list_matching(entry[:-1])
                except DockerUnavailable:
                    matches = []
                for m in matches:
                    if m not in seen:
                        seen.add(m)
                        out.append(m)
            elif entry not in seen:
                seen.add(entry)
                out.append(entry)
        return out

    async def _containers_for(self, keys: list[str]) -> list[tuple[str, str]]:
        """Flatten ``keys`` to ``(component_key, container_name)`` pairs (patterns expanded)."""
        out: list[tuple[str, str]] = []
        for key in keys:
            for name in await self._resolve_names(self._components[key].containers):
                out.append((key, name))
        return out

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self) -> PowerStatus:
        """Inspect all managed containers + read VRAM → a snapshot.

        Never raises: if the Docker daemon is unreachable, the overall state
        is :attr:`PowerState.UNKNOWN`, each component is ``"unreachable"``,
        and the VRAM snapshot is read independently (it may still be present).

        Returns:
            The current :class:`PowerStatus`.
        """
        vram = self._read_vram_safe()
        daemon_down = False
        comp_status: list[ComponentStatus] = []
        for key, cfg in self._components.items():
            states: list[ContainerState] = []
            for name in await self._resolve_names(cfg.containers):
                try:
                    states.append(await self._docker.inspect(name))
                except DockerUnavailable:
                    daemon_down = True
                    states.append(
                        ContainerState(
                            name=name,
                            exists=False,
                            running=False,
                            status="unreachable",
                            health=None,
                        )
                    )
            comp_status.append(self._component_status(key, cfg, states, daemon_down))

        held_mb = sum(self._held_mb(c) for c in comp_status)
        other_mb = None if vram.used_mb is None else max(0, vram.used_mb - held_mb)
        inflight = self._inflight
        if inflight is not None:
            op, t_keys = inflight
            overall = PowerState.POWERING_ON if op == "power_on" else PowerState.POWERING_OFF
            return PowerStatus(
                state=overall,
                components=comp_status,
                vram=vram,
                transition=op,
                transition_keys=list(t_keys),
                estimated_held_mb=held_mb,
                estimated_other_mb=other_mb,
            )
        overall = self._overall_state(comp_status, daemon_down)
        return PowerStatus(
            state=overall,
            components=comp_status,
            vram=vram,
            estimated_held_mb=held_mb,
            estimated_other_mb=other_mb,
        )

    # ------------------------------------------------------------------
    # Capacity pre-check (F-00098)
    # ------------------------------------------------------------------

    def _budget_mb(self, key: str) -> int:
        """Return a component's configured VRAM budget in MiB (0 if unbudgeted)."""
        return gb_to_mb(self._components[key].vram_gb)

    def _held_mb(self, c: ComponentStatus) -> int:
        """Estimate the VRAM a component holds: full budget when on, pro-rated when partial.

        A ``partial`` component has only some of its containers running (e.g. a
        CPU-only guard without its GPU model), so counting its whole budget would
        overstate what iw-rag holds.
        """
        if c.state == "on":
            return self._budget_mb(c.key)
        if c.state == "partial" and c.present:
            return round(self._budget_mb(c.key) * c.running / c.present)
        return 0

    async def _keys_for_containers(self, names: list[str]) -> list[str]:
        """Map raw container names back to component keys (unmapped names dropped).

        Args:
            names: Container names (as passed to :meth:`orchestrate`).

        Returns:
            The distinct component keys owning those names, in declared order.
            Unmapped containers (e.g. the llama-swap stack) have no budget.
        """
        owners = {name: key for key, name in await self._containers_for(list(self._components))}
        wanted = {owners[n] for n in names if n in owners}
        return [k for k in self._components if k in wanted]

    async def _verdict(
        self, start_keys: list[str], stop_keys: list[str], *, force: bool = False
    ) -> CapacityVerdict:
        """Compute the capacity verdict for starting ``start_keys`` after ``stop_keys``.

        Only components not already running contribute to the memory needed;
        only stopped components that are currently running free memory.

        Args:
            start_keys: Component keys the action would start.
            stop_keys: Component keys the same action stops first.
            force: Operator override (bypasses the gate).

        Returns:
            The :class:`CapacityVerdict` (never raises).
        """
        status = await self.status()
        state = {c.key: c.state for c in status.components}
        to_start = [k for k in start_keys if state.get(k) not in ("on", "partial")]
        starts_mb = sum(self._budget_mb(k) for k in to_start)
        freed_mb = sum(self._budget_mb(k) for k in stop_keys if state.get(k) in ("on", "partial"))
        return check_capacity(
            starts_mb=starts_mb,
            freed_mb=freed_mb,
            free_mb=status.vram.free_mb if status.vram.available else None,
            margin_mb=self._margin_mb,
            enabled=self._precheck_enabled,
            force=force,
            components=to_start,
        )

    async def check(
        self, keys: list[str] | None = None, *, stop: list[str] | None = None
    ) -> CapacityVerdict:
        """Dry-run the capacity pre-check for starting ``keys`` (after ``stop``).

        Args:
            keys: Component keys to start, or ``None`` for all managed.
            stop: Component keys the same action would stop first.

        Returns:
            The :class:`CapacityVerdict`; nothing is started or stopped.

        Raises:
            KeyError: If a requested key is unknown.
        """
        return await self._verdict(self._resolve_keys(keys), self._resolve_keys(stop or []))

    async def _require_capacity(
        self, start_keys: list[str], stop_keys: list[str], *, force: bool
    ) -> None:
        """Raise :class:`InsufficientVram` unless the action fits (or is forced/disabled)."""
        verdict = await self._verdict(start_keys, stop_keys, force=force)
        if not verdict.ok:
            raise InsufficientVram(verdict)

    @staticmethod
    def _component_status(
        key: str, cfg: PowerComponentCfg, states: list[ContainerState], daemon_down: bool
    ) -> ComponentStatus:
        """Derive one component's ``state`` from its container states."""
        present = [s for s in states if s.exists]
        running = [s for s in present if s.running]
        # "Ready" = running AND (no healthcheck OR healthy).  A container that
        # is running but still ``starting`` / ``unhealthy`` is mid-warmup, not
        # fully on — it counts as a transitional ``partial`` so the console
        # shows "starting" rather than a premature green.
        ready = [s for s in running if s.health in (None, "healthy")]
        if daemon_down and not present:
            state = "unreachable"
        elif not present:
            state = "absent"
        elif len(ready) == len(present):
            state = "on"
        elif not running:
            state = "off"
        else:
            state = "partial"
        return ComponentStatus(
            key=key,
            label=cfg.label or key,
            description=cfg.description,
            idle_unload=cfg.idle_unload,
            containers=states,
            present=len(present),
            running=len(running),
            state=state,
        )

    @staticmethod
    def _overall_state(comp_status: list[ComponentStatus], daemon_down: bool) -> PowerState:
        """Fold per-component states into the overall :class:`PowerState`."""
        # Only components that actually have a present container vote.
        votes = [c.state for c in comp_status if c.state not in ("absent", "unreachable")]
        if not votes:
            return PowerState.UNKNOWN if daemon_down else PowerState.OFF
        if all(v == "on" for v in votes):
            return PowerState.ON
        if all(v == "off" for v in votes):
            return PowerState.OFF
        return PowerState.PARTIAL

    def _read_vram_safe(self) -> VramSnapshot:
        """Read VRAM without ever raising into status()."""
        try:
            return self._vram.read()
        except Exception:  # noqa: BLE001 — telemetry must never break status
            return VramSnapshot(available=False)

    # ------------------------------------------------------------------
    # Power off
    # ------------------------------------------------------------------

    async def power_off(self, keys: list[str] | None = None) -> PowerStatus:
        """Stop the given components (or all) and return the new status.

        Args:
            keys: Component keys to stop, or ``None`` for the master OFF
                (every managed component).

        Returns:
            The :class:`PowerStatus` after the stop.

        Raises:
            KeyError: If a requested key is unknown.
            DockerUnavailable: If the daemon is unreachable mid-stop.
        """
        resolved = self._resolve_keys(keys)
        async with self._lock:
            self._inflight = ("power_off", resolved)
            try:
                for _key, name in await self._containers_for(resolved):
                    await self._docker.stop(name, timeout_s=self._stop_timeout_s)
            finally:
                self._inflight = None
        return await self.status()

    # ------------------------------------------------------------------
    # Power on
    # ------------------------------------------------------------------

    async def power_on(self, keys: list[str] | None = None, *, force: bool = False) -> PowerStatus:
        """Start the given components (or all) and wait until each is ready.

        For every container: issue ``docker start``, then poll its state until
        it is running and (if it declares a healthcheck) healthy, or the
        ready budget elapses.  A container name that does not resolve (404) is
        skipped — so the optional guard containers being absent is not an
        error.

        Before starting anything the capacity pre-check runs (when enabled):
        a start that would not fit in the free VRAM is refused and nothing
        is started.

        Args:
            keys: Component keys to start, or ``None`` for the master ON.
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The :class:`PowerStatus` after the start (components that timed
            out reading not-ready will show as ``partial``/``off``).

        Raises:
            KeyError: If a requested key is unknown.
            InsufficientVram: If the start would not fit and ``force`` is false.
            DockerUnavailable: If the daemon is unreachable mid-start.
        """
        resolved = self._resolve_keys(keys)
        async with self._lock:
            await self._require_capacity(resolved, [], force=force)
            self._inflight = ("power_on", resolved)
            try:
                started: list[str] = []
                for _key, name in await self._containers_for(resolved):
                    ok = await self._docker.start(name)
                    if ok:
                        started.append(name)
                for name in started:
                    await self._poll_until_ready(name)
            finally:
                self._inflight = None
        return await self.status()

    # ------------------------------------------------------------------
    # Mode-switch orchestration (raw container lists)
    # ------------------------------------------------------------------

    async def orchestrate(
        self, stop: list[str], start: list[str], *, force: bool = False
    ) -> PowerStatus:
        """Stop one set of containers and start (+ ready-wait) another.

        The container-lifecycle half of a mode switch (the API does the
        generator-endpoint re-point separately).  Operates on RAW container
        names, not component keys, because the swap stack (llama-swap /
        swap-controller) is not a power *component*.  Absent containers are
        skipped (start returns False → not waited on), so an optional guard
        container being absent is not an error.

        The capacity pre-check runs first (when enabled), counting the memory
        the stops free; a switch that would not fit is refused and nothing is
        stopped or started.

        Args:
            stop: Container names to stop first (e.g. the outgoing mode's stack).
            start: Container names to start + poll to ready (the incoming stack).
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The :class:`PowerStatus` after the switch.

        Raises:
            InsufficientVram: If the start would not fit and ``force`` is false.
            DockerUnavailable: If the daemon is unreachable mid-switch.
        """
        async with self._lock:
            await self._require_capacity(
                await self._keys_for_containers(start),
                await self._keys_for_containers(stop),
                force=force,
            )
            self._inflight = ("switch", [*stop, *start])
            try:
                for name in stop:
                    await self._docker.stop(name, timeout_s=self._stop_timeout_s)
                started: list[str] = []
                for name in start:
                    if await self._docker.start(name):
                        started.append(name)
                for name in started:
                    await self._poll_until_ready(name)
            finally:
                self._inflight = None
        return await self.status()

    async def apply_profile(
        self, name: str, profiles: dict[str, PowerProfileCfg], *, force: bool = False
    ) -> PowerStatus:
        """Move the box to the named operating profile.

        The plan (stops, then starts) is computed from the live status; the
        capacity pre-check runs up front against the projected free memory
        *including* the plan's own stops, so a refused profile stops nothing.
        Components the profile does not name are left untouched.

        Args:
            name: Profile key.
            profiles: The configured profiles (``Settings.power.profiles``).
            force: Operator override — bypass the capacity pre-check.

        Returns:
            The :class:`PowerStatus` after the change.

        Raises:
            KeyError: If the profile is unknown.
            InsufficientVram: If the starts would not fit and ``force`` is false.
            DockerUnavailable: If the daemon is unreachable mid-change.
        """
        if name not in profiles:
            raise KeyError(f"unknown power profile {name!r}; known: {sorted(profiles)}")
        async with self._lock:
            status = await self.status()
            states = {c.key: c.state for c in status.components}
            plan = plan_profile(profiles[name], self._components, states)
            await self._require_capacity(plan.start, plan.stop, force=force)
            self._inflight = ("switch", [*plan.stop, *plan.start])
            try:
                for _key, cname in await self._containers_for(plan.stop):
                    await self._docker.stop(cname, timeout_s=self._stop_timeout_s)
                started: list[str] = []
                for _key, cname in await self._containers_for(plan.start):
                    if await self._docker.start(cname):
                        started.append(cname)
                for cname in started:
                    await self._poll_until_ready(cname)
            finally:
                self._inflight = None
        return await self.status()

    async def _poll_until_ready(self, name: str) -> bool:
        """Poll a container until running+healthy or the ready budget elapses.

        Args:
            name: The container name.

        Returns:
            ``True`` iff the container reached running + (healthy | no
            healthcheck) within the budget.
        """
        deadline = time.monotonic() + self._ready_timeout_s
        while True:
            try:
                st = await self._docker.inspect(name)
            except DockerUnavailable:
                st = ContainerState(
                    name=name, exists=False, running=False, status="unreachable", health=None
                )
            if st.running and st.health in (None, "healthy"):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._poll_interval_s)
