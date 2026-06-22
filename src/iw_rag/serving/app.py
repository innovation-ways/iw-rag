"""Thin FastAPI sidecar for the Benchmark Studio swap-controller (F-00094).

This module is the **single swap-state authority** for the Benchmark
Studio generator-swap infrastructure.  The :class:`SwapController` is
instantiated exactly once per process — inside this sidecar — and is the
sole source of truth for ``active_model_id`` / ``swapping`` state.  The
main API's ``GET /models`` and ``POST /models/activate`` (S03) are thin
HTTP facades that proxy to this sidecar over
``settings.serving.swap_controller_base_url`` via
:class:`~.swap_controller_client.SwapControllerClient`; they do **NOT**
instantiate their own in-process ``SwapController``.

**Boundary rationale.**  Two processes each holding their own swap state
would yield split-brain: each would see the other's in-flight swap as
"no swap in progress" and let generation requests through to a
non-resident model.  The sidecar being the only ``SwapController`` owner
keeps the in-flight-swap 503 guard (Boundary row 4 / AC4) coherent.

**Sleep-mode fallback** (R-00123 Limitations 1): vLLM sleep mode
(``--enable-sleep-mode``, ``POST /sleep|/wake_up``, requiring
``VLLM_SERVER_DEV_MODE=1``) is a **narrow** fallback for parking the
**same** model's VRAM — it cannot cross-swap to a different base model
and ``VLLM_SERVER_DEV_MODE`` must **never** be exposed to the network /
users.  This sidecar never sets the dev-mode flag and never calls the
sleep endpoints; the exclusive ``generators`` group in
``config/llama-swap.yaml`` is the swap mechanism.

**Network exposure.**  The sidecar binds to ``:8090`` and is reachable
only on the internal Compose network — never published to the host.
Auth/ACL is the public API's responsibility; this sidecar is internal-only
and unauthenticated.

Routes:

* ``GET /catalog`` — catalog annotated with ``{available, loading,
  active, unreachable}`` state.
* ``POST /catalog/activate`` — trigger an activation; returns the
  resulting status.
* ``GET /catalog/status`` — current ``SwapStatus`` snapshot.
* ``GET /health`` — liveness; returns ``{"status": "ok"}`` when the
  sidecar itself is up (does NOT probe llama-swap — the public API
  reports per-model state via ``GET /catalog``).
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .catalog import load_catalog
from .llama_swap_client import LlamaSwapClient, LlamaSwapUnavailable
from .swap_controller import (
    ActivationState,
    CatalogRow,
    SwapController,
    SwapStatus,
)


class ActivateRequest(BaseModel):
    """The body of ``POST /catalog/activate``.

    Attributes:
        model_id: The catalog id to activate.
    """

    model_id: str = Field(min_length=1, max_length=128)


def create_app(
    catalog_path: str | None = None,
    llama_swap_base_url: str | None = None,
    health_timeout_s: int = 300,
    poll_interval_s: float = 0.5,
) -> FastAPI:
    """Build the FastAPI sidecar app.

    Args:
        catalog_path: Path to the catalog YAML.  ``None`` = defer to
            the caller (production wires this from
            ``Settings.serving.catalog_path``).
        llama_swap_base_url: Base URL of the llama-swap proxy.  ``None``
            = defer to the caller.
        health_timeout_s: Readiness poll budget in seconds.
        poll_interval_s: Sleep between readiness polls in seconds.

    Returns:
        A configured ``FastAPI`` instance owning the single
        :class:`SwapController`.
    """
    app = FastAPI(title="iw-rag-swap-controller", version="0.0.0")

    # Lazy initialise the controller on first request so test fixtures
    # can override ``catalog_path`` / ``llama_swap_base_url`` via
    # ``app.dependency_overrides`` without rebuilding the app.
    @app.on_event("startup")
    async def _init_controller() -> None:  # pragma: no cover - exercised via ASGI test
        if getattr(app.state, "controller", None) is not None:
            return
        resolved_catalog_path = catalog_path
        resolved_llama_swap_base_url = llama_swap_base_url
        if resolved_catalog_path is None or resolved_llama_swap_base_url is None:
            # Production wiring: `uvicorn ...:create_app --factory` calls
            # create_app() with no args, so fall back to Settings
            # (IW_RAG__SERVING__CATALOG_PATH / __LLAMA_SWAP_BASE_URL).
            from iw_rag.settings import get_settings

            serving = get_settings().serving
            if resolved_catalog_path is None:
                resolved_catalog_path = serving.catalog_path
            if resolved_llama_swap_base_url is None:
                resolved_llama_swap_base_url = serving.llama_swap_base_url
        if resolved_catalog_path is None or resolved_llama_swap_base_url is None:
            raise RuntimeError(
                "create_app() requires catalog_path and llama_swap_base_url "
                "(or app.state.controller must be pre-set by tests)."
            )
        catalog = load_catalog(resolved_catalog_path)
        # The activation trigger (POST /v1/chat/completions, max_tokens=1) is
        # held open by llama-swap for the FULL cold-start of the upstream vLLM
        # container (~60-180 s, longer on first-pull). A 5 s client timeout
        # fails it mid-swap and the controller reports 503 while the model is
        # still loading. Give the trigger client and the readiness budget room
        # to cover a real cold swap on the single GPU.
        swap_load_budget_s = max(int(health_timeout_s), 1200)
        client = LlamaSwapClient(
            base_url=resolved_llama_swap_base_url, timeout_s=float(swap_load_budget_s)
        )
        app.state.controller = SwapController(
            catalog=catalog,
            client=client,
            health_timeout_s=swap_load_budget_s,
            poll_interval_s=poll_interval_s,
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Liveness probe — does NOT probe llama-swap."""
        return {"status": "ok"}

    @app.get("/catalog")
    async def get_catalog() -> dict[str, Any]:
        """Return the catalog annotated with per-model state.

        Returns:
            ``{"models": [{"id": ..., "served_name": ..., ...,
                "state": "active"|"available"|"loading"|"unreachable"}, ...]}``

        Raises:
            HTTPException: 500 if the controller is not initialised.
        """
        controller = _get_controller(app)
        rows = await controller.catalog_with_state()
        return {"models": [_row_to_dict(r) for r in rows]}

    @app.post("/catalog/activate")
    async def post_activate(body: ActivateRequest) -> dict[str, Any]:
        """Trigger an activation.  Blocks until ready / failed / timeout.

        Returns:
            The post-activation :class:`SwapStatus` payload.

        Raises:
            HTTPException: 404 on unknown model id, 503 on llama-swap
                unreachable, 500 on internal error.
        """
        from iw_rag.domain.errors import NotFoundError

        controller = _get_controller(app)
        try:
            status = await controller.activate(body.model_id)
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except LlamaSwapUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return _status_to_dict(status)

    @app.get("/catalog/status")
    async def get_status() -> dict[str, Any]:
        """Return the current :class:`SwapStatus` snapshot.

        Returns:
            The status payload (``state``, ``active_model_id``,
            ``failure_reason``, ``started_at``).

        Raises:
            HTTPException: 500 if the controller is not initialised.
        """
        controller = _get_controller(app)
        return _status_to_dict(controller.status())

    return app


def _get_controller(app: FastAPI) -> SwapController:
    """Return the controller stored on ``app.state`` or raise 500."""
    ctrl = getattr(app.state, "controller", None)
    if ctrl is None:
        raise HTTPException(status_code=500, detail="swap-controller not initialised")
    return ctrl  # type: ignore[no-any-return]


def _row_to_dict(row: CatalogRow) -> dict[str, Any]:
    """Serialise a :class:`CatalogRow` to a JSON-friendly dict."""
    return {
        "id": row.id,
        "served_name": row.served_name,
        "quantization": row.quantization,
        "expected_vram_gb": row.expected_vram_gb,
        "max_context": row.max_context,
        "state": row.state,
    }


def _status_to_dict(status: SwapStatus) -> dict[str, Any]:
    """Serialise a :class:`SwapStatus` to a JSON-friendly dict."""
    return {
        "state": status.state.value
        if isinstance(status.state, ActivationState)
        else str(status.state),
        "active_model_id": status.active_model_id,
        "failure_reason": status.failure_reason,
        "started_at": status.started_at,
    }
