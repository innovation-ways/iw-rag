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

from iw_rag.settings import PowerComponentCfg

from .docker_engine import ContainerState, DockerEngineClient, DockerUnavailable
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
    """

    state: PowerState
    components: list[ComponentStatus]
    vram: VramSnapshot
    transition: str | None = None
    transition_keys: list[str] = field(default_factory=list)


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
    ) -> None:
        self._components = components
        self._docker = docker
        self._vram = vram
        self._ready_timeout_s = ready_timeout_s
        self._poll_interval_s = poll_interval_s
        self._stop_timeout_s = stop_timeout_s
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
            )
        overall = self._overall_state(comp_status, daemon_down)
        return PowerStatus(state=overall, components=comp_status, vram=vram)

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

    async def power_on(self, keys: list[str] | None = None) -> PowerStatus:
        """Start the given components (or all) and wait until each is ready.

        For every container: issue ``docker start``, then poll its state until
        it is running and (if it declares a healthcheck) healthy, or the
        ready budget elapses.  A container name that does not resolve (404) is
        skipped — so the optional guard containers being absent is not an
        error.

        Args:
            keys: Component keys to start, or ``None`` for the master ON.

        Returns:
            The :class:`PowerStatus` after the start (components that timed
            out reading not-ready will show as ``partial``/``off``).

        Raises:
            KeyError: If a requested key is unknown.
            DockerUnavailable: If the daemon is unreachable mid-start.
        """
        resolved = self._resolve_keys(keys)
        async with self._lock:
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

    async def orchestrate(self, stop: list[str], start: list[str]) -> PowerStatus:
        """Stop one set of containers and start (+ ready-wait) another.

        The container-lifecycle half of a mode switch (the API does the
        generator-endpoint re-point separately).  Operates on RAW container
        names, not component keys, because the swap stack (llama-swap /
        swap-controller) is not a power *component*.  Absent containers are
        skipped (start returns False → not waited on), so an optional guard
        container being absent is not an error.

        Args:
            stop: Container names to stop first (e.g. the outgoing mode's stack).
            start: Container names to start + poll to ready (the incoming stack).

        Returns:
            The :class:`PowerStatus` after the switch.

        Raises:
            DockerUnavailable: If the daemon is unreachable mid-switch.
        """
        async with self._lock:
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
