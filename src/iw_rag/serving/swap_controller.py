"""Activation state machine for the Benchmark Studio swap-controller (F-00094).

The :class:`SwapController` is the **single swap-state authority** for the
Benchmark Studio generator-swap infrastructure.  It:

  1. Loads the user-facing catalog (``config/model_catalog.yaml``).
  2. Triggers a generator swap via the local ``llama-swap`` proxy (whose
     ``exclusive: true`` group guarantees exactly one resident generator).
  3. Polls the new model's ``/v1/models`` readiness endpoint until HTTP
     200 — readiness is the AC2 / Boundary-row-1 health gate; the
     controller never reports ``ready`` before a 200.
  4. Exposes the current state and active model id via :meth:`status`,
     and a catalog annotated with ``{available, loading, active,
     unreachable}`` for the public API's ``GET /models`` (S03).

**State machine** (mirrors F-00094 AC1 / AC2):

    idle → swapping → ready   (success)
              ↓
            failed            (timeout or upstream error)

**vLLM sleep-mode fallback** (R-00123 Limitations 1): vLLM sleep mode
(``--enable-sleep-mode``, ``POST /sleep|/wake_up``, requiring
``VLLM_SERVER_DEV_MODE=1``) is a **narrow** fallback for parking the
**same** model's VRAM — it cannot cross-swap to a different base model
and ``VLLM_SERVER_DEV_MODE`` must **never** be exposed to the network /
users.  It is **not** the swap path; the exclusive ``generators`` group
in ``config/llama-swap.yaml`` is.  This controller never sets the dev-mode
flag and never calls the sleep endpoints.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

from iw_rag.domain.errors import NotFoundError

from .catalog import Catalog, CatalogModel
from .llama_swap_client import LlamaSwapUnavailable


class ActivationState(StrEnum):
    """The activation state of the swap-controller.

    Attributes:
        IDLE: No activation in flight, no active model.
        SWAPPING: An activation is in flight (calling llama-swap, polling
            readiness).  Public API generation requests while in this
            state MUST be served 503 (Boundary row 4 / AC4 guard).
        READY: An activation completed successfully; ``active_model_id``
            names the resident generator.
        FAILED: The last activation failed.  ``failure_reason`` names the
            cause (``"timeout"`` or ``"error"``).
    """

    IDLE = "idle"
    SWAPPING = "swapping"
    READY = "ready"
    FAILED = "failed"


class LlamaSwapClientProtocol(Protocol):
    """Structural type for the llama-swap client used by the controller.

    Defined so tests can pass a MagicMock that quacks like a
    :class:`~.llama_swap_client.LlamaSwapClient` without subclassing it.
    """

    async def list_models(self) -> Any: ...
    async def model_health(self, model_id: str) -> bool: ...
    async def activate(self, model_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class SwapStatus:
    """A snapshot of the swap-controller's state.

    Attributes:
        state: The current :class:`ActivationState`.
        active_model_id: The id of the currently resident generator, or
            ``None`` when no model is active.
        failure_reason: When ``state`` is ``FAILED``, the reason
            (``"timeout"`` or ``"error"``); ``None`` otherwise.
        started_at: Unix timestamp when the current activation started
            (``None`` when no activation has ever started).
    """

    state: ActivationState
    active_model_id: str | None = None
    failure_reason: str | None = None
    started_at: float | None = None


@dataclass(frozen=True, slots=True)
class CatalogRow:
    """One catalog entry annotated with its current state.

    Attributes:
        id: The catalog id.
        served_name: The HF repo id (mirrors :class:`CatalogModel`).
        quantization: Quantisation scheme (mirrors :class:`CatalogModel`).
        expected_vram_gb: VRAM budget (mirrors :class:`CatalogModel`).
        max_context: vLLM ``--max-model-len`` (mirrors :class:`CatalogModel`).
        state: One of ``"available"``, ``"loading"``, ``"active"``,
            ``"unreachable"``.  ``unreachable`` is set for every row when
            llama-swap is down (Boundary row 3).
    """

    id: str
    served_name: str
    quantization: str
    expected_vram_gb: float
    max_context: int
    state: str


class SwapController:
    """The activation state machine — single swap-state authority (F-00094).

    Holds the loaded catalog, a llama-swap client, and the readiness-poll
    budget (``health_timeout_s``).  The instance is owned by the
    :mod:`iw_rag.serving.app` FastAPI sidecar; the public API holds NO
    in-process instance — it proxies to the sidecar over HTTP via
    :class:`~.swap_controller_client.SwapControllerClient`.
    """

    def __init__(
        self,
        catalog: Catalog,
        client: LlamaSwapClientProtocol,
        *,
        health_timeout_s: int = 300,
        poll_interval_s: float = 0.5,
    ) -> None:
        """Configure the controller.

        Args:
            catalog: The validated :class:`~.catalog.Catalog`.
            client: A typed llama-swap client (or a structurally-compatible
                mock in tests).
            health_timeout_s: Total readiness-poll budget in seconds
                (default 300; matches ``llama-swap.yaml``
                ``healthCheckTimeout: 300`` and
                ``Settings.serving.health_timeout_s``).
            poll_interval_s: Sleep between readiness polls in seconds
                (default 0.5 — bounded to keep the loop responsive
                without hammering the proxy).
        """
        self._catalog = catalog
        self._client = client
        self._health_timeout_s = health_timeout_s
        self._poll_interval_s = poll_interval_s
        self._status = SwapStatus(state=ActivationState.IDLE)
        # Mutex serialises activations so two simultaneous
        # POST /catalog/activate calls cannot interleave their state
        # updates and yield a split-brain ``active_model_id``.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Status access
    # ------------------------------------------------------------------

    def status(self) -> SwapStatus:
        """Return the current state snapshot.

        Returns:
            The current :class:`SwapStatus`.
        """
        return self._status

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    async def activate(self, model_id: str) -> SwapStatus:
        """Activate ``model_id`` and block until ready / failed.

        State transitions:

        * Unknown ``model_id`` → :class:`NotFoundError`; status unchanged.
        * Idempotent: activating the already-active+ready model is a fast
          no-op returning the current ``ready`` status.
        * ``SWAPPING`` → poll ``client.model_health`` until HTTP 200 or
          ``health_timeout_s`` elapses → ``READY`` or ``FAILED``.
        * Client connection error → :class:`LlamaSwapUnavailable`
          propagates; status flips to ``FAILED`` with reason ``"error"``.

        Args:
            model_id: The catalog id to activate.

        Returns:
            The :class:`SwapStatus` after the activation attempt.

        Raises:
            NotFoundError: If ``model_id`` is not in the catalog.
            LlamaSwapUnavailable: If the client cannot reach llama-swap
                (caller maps to HTTP 503).
        """
        # Idempotent fast path — avoids an HTTP round-trip + poll loop
        # for repeated activation of the same resident model.
        if self._status.state == ActivationState.READY and self._status.active_model_id == model_id:
            return self._status

        # Validate against the catalog BEFORE acquiring the lock so a
        # bad id surfaces a typed NotFoundError without serialising.
        try:
            self._catalog.get(model_id)
        except NotFoundError:
            raise

        async with self._lock:
            # Re-check after lock acquisition: a concurrent caller may
            # have completed the activation while we were waiting.
            if (
                self._status.state == ActivationState.READY
                and self._status.active_model_id == model_id
            ):
                return self._status

            self._status = SwapStatus(
                state=ActivationState.SWAPPING,
                active_model_id=None,
                failure_reason=None,
                started_at=time.monotonic(),
            )
            try:
                await self._client.activate(model_id)
            except LlamaSwapUnavailable:
                self._status = SwapStatus(
                    state=ActivationState.FAILED,
                    active_model_id=None,
                    failure_reason="error",
                    started_at=self._status.started_at,
                )
                raise

            ready = await self._poll_until_ready(model_id)
            if ready:
                self._status = SwapStatus(
                    state=ActivationState.READY,
                    active_model_id=model_id,
                    failure_reason=None,
                    started_at=self._status.started_at,
                )
            else:
                self._status = SwapStatus(
                    state=ActivationState.FAILED,
                    active_model_id=None,
                    failure_reason="timeout",
                    started_at=self._status.started_at,
                )
            return self._status

    async def _poll_until_ready(self, model_id: str) -> bool:
        """Poll ``client.model_health`` until True or the timeout elapses.

        Args:
            model_id: The catalog id to poll readiness for.

        Returns:
            ``True`` iff the model reported HTTP 200 within the budget.
        """
        deadline = time.monotonic() + self._health_timeout_s
        while True:
            try:
                ok = await self._client.model_health(model_id)
            except LlamaSwapUnavailable:
                # Transient proxy hiccup during activation: keep polling
                # until the budget elapses, then surface as timeout.
                ok = False
            if ok:
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(self._poll_interval_s)

    # ------------------------------------------------------------------
    # Catalog annotation
    # ------------------------------------------------------------------

    async def catalog_with_state(self) -> list[CatalogRow]:
        """Return the catalog annotated with per-model state.

        State derivation:

        * If ``client.list_models()`` raises :class:`LlamaSwapUnavailable`
          (proxy down), every model is annotated ``"unreachable"`` (no
          crash) — Boundary row 3.
        * The active model (per ``self._status.active_model_id``) is
          annotated ``"active"`` when ``state`` is ``READY`` or
          ``SWAPPING``; ``"loading"`` during ``SWAPPING`` is reserved for
          the moment between ``activate()`` and the first successful
          health poll (we currently collapse SWAPPING onto ``active`` for
          the *public* annotation; the sidecar's ``GET /catalog/status``
          exposes the finer-grained ``state`` field).
        * Models llama-swap currently reports as resident are
          ``"available"``; everything else is ``"available"`` too (the
          exclusive group makes at most one resident; non-resident models
          in the same group are still in the catalog and still callable
          for activation).

        Returns:
            A list of :class:`CatalogRow` records, one per catalog entry.
        """
        try:
            live = await self._client.list_models()
        except LlamaSwapUnavailable:
            live = None
        live_ids = {m.id for m in (live or [])}
        active = self._status.active_model_id
        out: list[CatalogRow] = []
        for entry in self._catalog.models.values():
            state = self._row_state(entry, active, live_ids, live is None)
            out.append(
                CatalogRow(
                    id=entry.id,
                    served_name=entry.served_name,
                    quantization=entry.quantization,
                    expected_vram_gb=entry.expected_vram_gb,
                    max_context=entry.max_context,
                    state=state,
                )
            )
        return out

    @staticmethod
    def _row_state(
        entry: CatalogModel,
        active: str | None,
        live_ids: set[str],
        proxy_unreachable: bool,
    ) -> str:
        """Compute the annotation state for a single catalog row.

        Args:
            entry: The catalog entry to annotate.
            active: The currently active model id (``None`` when idle).
            live_ids: The set of model ids llama-swap currently reports
                as resident.
            proxy_unreachable: ``True`` if the proxy is unreachable;
                every row degrades to ``"unreachable"``.

        Returns:
            The annotation string: one of ``{"active", "loading",
            "available", "unreachable"}``.
        """
        if proxy_unreachable:
            return "unreachable"
        if active == entry.id:
            return "active"
        if entry.id in live_ids:
            return "available"
        return "available"
