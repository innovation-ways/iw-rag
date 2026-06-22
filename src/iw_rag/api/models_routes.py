"""HTTP routes for the Benchmark Studio model-swap catalog (F-00094).

Public surface for the swap infrastructure: ``GET /models`` and
``POST /models/activate``.  The main API is a **thin HTTP facade** that
proxies to the swap-controller sidecar (``:8090``) via
:class:`~iw_rag.serving.swap_controller_client.SwapControllerClient` —
the sidecar is the **sole swap-state authority**; the api process holds
NO in-process :class:`~iw_rag.serving.swap_controller.SwapController`.
The ACL gate is enforced on the main API **before** the proxy call (the
sidecar is internal-only and unauthenticated, so this gate is the single
choke point where the eval-operator group is checked).

**Auth-only ACL keystone.**  The ``eval-operator`` group required to
activate a model is read from the resolved
:class:`~iw_rag.domain.acl.AccessContext` (auth-only), NEVER from the
request body.  The :class:`ActivateRequest` body carries only
``model_id`` — structurally identical to :class:`QueryRequest`'s
"no acl field" pattern.

**In-flight-swap guard seam.**  The public ``GET /models`` response
always carries ``active_model_id`` (or ``None``) so a future caller
(F-00095 orchestration, the Run Full Test button) can gate generation
on the swap-controller's state.  We do not implement the 503-on-
swapping guard for ``/query`` here — that is the orchestration layer's
concern (F-00095).  This step only exposes the state.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from iw_rag.api.routes import AccessCtx
from iw_rag.api.schemas import (
    ActivateRequest,
    ActivateResponse,
    CatalogModelOut,
    ModelsResponse,
    VramBudgetOut,
)
from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError, ValidationError
from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable
from iw_rag.serving.swap_controller_client import SwapControllerClient

models_router = APIRouter()


# ---------------------------------------------------------------------------
# VRAM budget math (Invariant 7)
# ---------------------------------------------------------------------------
#
# The console renders the VRAM bar from the server-computed triple
# ``(fits, budget_used_gb, headroom_gb)`` — the math NEVER lives in
# the frontend.  The two named constants are the inputs:
#
# * ``RESIDENT_INFINITY_GB`` — the always-resident Infinity (BGE-M3)
#   footprint.  Infinity is pinned as a persistent llama-swap entry
#   (config/llama-swap.yaml ``persistent``); the model_catalog.yaml
#   header documents "Infinity is ALWAYS resident (~4 GB VRAM)".
# * ``VRAM_CEILING_GB`` — the single 32 GB GPU ceiling (CLAUDE.md,
#   F-00096 §Scope).  Settings does not (yet) expose these as fields
#   — they are documented constants here so the math has one obvious
#   home.  When the serving settings grow a ``vram_ceiling_gb`` field
#   this is the place to switch to ``settings.serving.vram_ceiling_gb``.

RESIDENT_INFINITY_GB: float = 4.0
VRAM_CEILING_GB: float = 32.0


def _compute_vram_budget(expected_vram_gb: float) -> VramBudgetOut:
    """Compute the VRAM budget assessment for one candidate (Invariant 7).

    The math is plain arithmetic over the candidate's catalog
    ``expected_vram_gb`` + the always-resident Infinity footprint +
    the 32 GB GPU ceiling:

    * ``budget_used_gb = resident_infinity_gb + expected_vram_gb``
    * ``headroom_gb    = ceiling_gb - budget_used_gb``
    * ``fits           = budget_used_gb <= ceiling_gb`` (equivalently
      ``headroom_gb >= 0``)

    Boundary cases (AC1 / Boundary row *"VRAM over budget"*):

    * A candidate at the ceiling exactly → ``fits=True``,
      ``headroom_gb == 0.0``.
    * A candidate one cent over the ceiling → ``fits=False``,
      ``headroom_gb < 0`` (negative).

    Args:
        expected_vram_gb: The candidate's catalog expected VRAM figure
            (rounded to the same float precision the catalog carries).

    Returns:
        A :class:`VramBudgetOut` carrying the server-computed triple.
    """
    budget_used = RESIDENT_INFINITY_GB + float(expected_vram_gb)
    headroom = VRAM_CEILING_GB - budget_used
    return VramBudgetOut(
        fits=(budget_used <= VRAM_CEILING_GB),
        budget_used_gb=budget_used,
        headroom_gb=headroom,
    )


# ---------------------------------------------------------------------------
# Dependency wiring
# ---------------------------------------------------------------------------


def _swap_controller_client(request: Request) -> SwapControllerClient:
    """Resolve the swap-controller client from ``app.state`` (503 if unwired)."""
    client: SwapControllerClient | None = getattr(request.app.state, "swap_controller_client", None)
    if client is None:
        raise ValidationError(
            "model-swap catalog is not enabled for this deployment "
            "(swap_controller_client is not wired on app.state)."
        )
    return client


SwapControllerClientDep = Annotated[SwapControllerClient, Depends(_swap_controller_client)]


# Local Settings proxy — the activation group is read from Settings, which
# is the lowest-priority yaml+env source.  Avoids a hard import on
# ``iw_rag.settings`` here so the module stays importable in slim tests.
def _activation_group(request: Request) -> str:
    """Return ``settings.serving.activation_group`` (default ``"eval-operator"``)."""
    settings = getattr(request.app.state, "settings", None)
    if settings is None:
        return "eval-operator"
    return getattr(getattr(settings, "serving", None), "activation_group", "eval-operator")


# ---------------------------------------------------------------------------
# ACL: require the activation group in AccessContext.principal_groups
# ---------------------------------------------------------------------------


async def require_eval_operator(
    request: Request,
    ctx: AccessCtx,
) -> AccessContext:
    """Require the ``eval-operator`` group in the caller's ACL (→ 403 otherwise).

    Mirrors :func:`iw_rag.api.routes.require_operator`: a thin ACL gate
    that reads groups from the resolved ``AccessContext`` only (the
    keystone — never from the request body or query string).  The
    activation group is sourced from
    :class:`~iw_rag.settings.ServingSwapSettings.activation_group` so
    operators can rename the gate without a code change.

    Args:
        request: The raw FastAPI request (used to read the configured
            activation group from ``app.state.settings``).
        ctx: The authenticated ``AccessContext`` (from auth only).

    Returns:
        The same :class:`AccessContext` on success.

    Raises:
        AuthorizationError: When the caller's ``principal_groups`` does
            not contain the configured activation group (→ 403).
    """
    needed = _activation_group(request)
    if needed not in ctx.principal_groups:
        raise AuthorizationError(f"'{needed}' group required to activate a model")
    return ctx


EvalOperatorCtx = Annotated[AccessContext, Depends(require_eval_operator)]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _row_from_payload(payload: dict[str, Any]) -> CatalogModelOut:
    """Map a sidecar catalog row dict to the API response model.

    Args:
        payload: The sidecar's per-model dict (``id``, ``served_name``,
            ``quantization``, ``expected_vram_gb``, ``max_context``,
            ``state``).

    Returns:
        A :class:`CatalogModelOut` with the same fields plus the
        server-computed :class:`VramBudgetOut` (Invariant 7: the
        console only renders the triple — the math lives here).
        Missing ``state`` is normalised to ``"unreachable"``
        (defensive — the sidecar always sets it).
    """
    expected_vram_gb = float(payload.get("expected_vram_gb", 0.0))
    return CatalogModelOut(
        id=str(payload.get("id", "")),
        served_name=str(payload.get("served_name", "")),
        quantization=str(payload.get("quantization", "")),
        expected_vram_gb=expected_vram_gb,
        max_context=int(payload.get("max_context", 0)),
        state=str(payload.get("state", "unreachable")),
        vram_budget=_compute_vram_budget(expected_vram_gb),
    )


@models_router.get("/models", response_model=ModelsResponse)
async def list_models(
    request: Request,
    ctx: AccessCtx,
    client: SwapControllerClientDep,
) -> ModelsResponse:
    """Return the catalog annotated with per-model state.

    Open to any authenticated principal (the catalog itself reveals
    nothing sensitive — model ids, served names, and public quantisation
    metadata).  When the sidecar (or, behind it, llama-swap) is
    unreachable, returns HTTP 200 with every model annotated
    ``"unreachable"`` and ``active_model_id=None`` — F-00094 Boundary
    row 3 (never 5xx on a transient dependency outage).

    The ``active_model_id`` field doubles as the **in-flight-swap guard
    seam** for F-00095: a caller can poll ``GET /models`` and gate
    generation on the sidecar's resident-model id (or, once the sidecar
    adds an explicit ``state`` field, on the ``swapping`` flag).

    Args:
        request: The raw FastAPI request.
        ctx: The authenticated ``AccessContext`` (any principal).
        client: The injected swap-controller HTTP client.

    Returns:
        The annotated catalog and the currently active model id.
    """
    try:
        payload = await client.get_catalog()
        status = await client.get_status()
    except LlamaSwapUnavailable:
        # Sidecar (or llama-swap behind it) is down — degrade to all
        # unreachable + no active model.  Per F-00094 Boundary row 3:
        # never 5xx on a transient dependency outage.  The placeholder
        # row carries the zero-budget "no measurement" marker (the
        # design prompt explicitly allows a zero budget for the
        # unreachable placeholder: the UI treats the row as
        # unmeasured, suppresses the red zone, and tells the
        # operator the swap-controller is down).
        return ModelsResponse(
            models=[
                CatalogModelOut(
                    id="",
                    served_name="",
                    quantization="",
                    expected_vram_gb=0.0,
                    max_context=0,
                    state="unreachable",
                    vram_budget=VramBudgetOut(
                        fits=True,
                        budget_used_gb=0.0,
                        headroom_gb=0.0,
                    ),
                )
            ],
            active_model_id=None,
        )
    rows = payload.get("models", []) if isinstance(payload, dict) else []
    return ModelsResponse(
        models=[_row_from_payload(r) for r in rows if isinstance(r, dict)],
        active_model_id=(status.get("active_model_id") if isinstance(status, dict) else None),
    )


@models_router.post("/models/activate", response_model=ActivateResponse)
async def activate_model(
    body: ActivateRequest,
    request: Request,
    ctx: EvalOperatorCtx,
    client: SwapControllerClientDep,
) -> ActivateResponse:
    """Activate a catalog model — gated to the ``eval-operator`` group.

    The ACL check (:func:`require_eval_operator`) runs **before** the
    proxy call: a caller without the activation group never reaches the
    sidecar (returns 403, no swap is triggered).  On success, the
    response mirrors the sidecar's status payload.

    Args:
        body: The activation request — ``model_id`` only (no groups).
        request: The raw FastAPI request.
        ctx: The authenticated ``AccessContext`` (must contain the
            ``eval-operator`` group, verified by the dependency).
        client: The injected swap-controller HTTP client.

    Returns:
        The post-activation status (state + active model id + reason).

    Raises:
        AuthorizationError: When the caller's groups do not include the
            configured activation group (→ 403, raised by the
            ``EvalOperatorCtx`` dependency).
        ValidationError: When the swap-controller client is not wired on
            ``app.state`` (→ 400, raised by the client dependency).
        NotFoundError: When the sidecar rejects an unknown ``model_id``
            (→ 404).  Surfaces when the typed client raises a 404-bearing
            error, currently mapped through ``LlamaSwapUnavailable``;
            callers see 503/404 per the sidecar's response.
        LlamaSwapUnavailable: When the sidecar is unreachable (→ 503,
            via the existing ``_ERROR_STATUS`` handler in
            :mod:`iw_rag.api.app`).
    """
    payload = await client.activate(body.model_id)
    if not isinstance(payload, dict):
        # Defensive: the sidecar should always return a dict.
        return ActivateResponse(state="unknown", active_model_id=body.model_id, reason=None)
    state = str(payload.get("state", "unknown"))
    reason = payload.get("failure_reason")
    # On success the sidecar reports ``active_model_id``; on failure it
    # is None.  Echo the requested id so clients can correlate.
    active_model_id = payload.get("active_model_id") or body.model_id
    return ActivateResponse(state=state, active_model_id=active_model_id, reason=reason)
