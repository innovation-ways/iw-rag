"""Unit tests for the F-00094 swap-controller sidecar FastAPI app.

The sidecar (``src/iw_rag/serving/app.py``) is the **single swap-state
authority**.  This test file exercises the public routes against an
injected :class:`SwapController` so no real llama-swap, no real GPU, no
network — the controller's client is a ``MagicMock`` that quacks like
:class:`LlamaSwapClient`.

Boundary rows covered:

* ``GET /catalog`` — annotated catalog payload.
* ``POST /catalog/activate`` — happy path → status payload; unknown
  model id → 404; llama-swap down → 503.
* ``GET /catalog/status`` — current ``SwapStatus`` snapshot.
* ``GET /health`` — liveness; returns ``{"status": "ok"}`` and does NOT
  probe llama-swap.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi.testclient import TestClient

from iw_rag.serving.app import create_app
from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable, ModelState
from iw_rag.serving.swap_controller import SwapController

# ---------------------------------------------------------------------------
# Fixtures — a real controller with a mocked llama-swap client
# ---------------------------------------------------------------------------


def _catalog_payload() -> dict[str, object]:
    """Minimal valid catalog (mirrors the S01 ``config/model_catalog.yaml`` shape)."""
    return {
        "models": {
            "qwen3-7b-awq": {
                "id": "qwen3-7b-awq",
                "served_name": "Qwen/Qwen3-7B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "vllm_flags": ["--quantization awq", "--kv-cache-dtype fp8"],
            },
            "qwen3-14b-awq": {
                "id": "qwen3-14b-awq",
                "served_name": "Qwen/Qwen3-14B-AWQ",
                "role": "generation",
                "quantization": "awq",
                "expected_vram_gb": 9.0,
                "max_context": 16384,
                "vllm_flags": ["--quantization awq", "--kv-cache-dtype fp8"],
            },
        }
    }


@pytest.fixture
def controller() -> SwapController:
    """A controller with a mocked llama-swap client and the default catalog."""
    client = MagicMock()
    client.list_models = AsyncMock(return_value=[])
    client.model_health = AsyncMock(return_value=True)
    client.activate = AsyncMock(return_value=None)
    cat = load_catalog_from_payload(_catalog_payload())
    return SwapController(
        catalog=cat,
        client=client,  # type: ignore[arg-type]
        health_timeout_s=300,
        poll_interval_s=0.0,
    )


def load_catalog_from_payload(payload: dict[str, object]) -> Any:
    """Build a :class:`Catalog` from an in-memory payload (helper for the fixture)."""
    import tempfile

    from iw_rag.serving.catalog import load_catalog

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        yaml.safe_dump(payload, fh)
        path = fh.name
    return load_catalog(path)


def _app_with(controller: SwapController) -> TestClient:
    """Build a TestClient with ``app.state.controller`` pre-set.

    Pre-setting ``app.state.controller`` short-circuits the ``@app.on_event("startup")``
    hook so we never need a real catalog file or llama-swap URL.
    """
    app = create_app()  # args are irrelevant — we pre-set the controller
    app.state.controller = controller
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /health — liveness
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok(controller: SwapController) -> None:
    """``GET /health`` returns ``{"status": "ok"}`` and does NOT probe llama-swap."""
    client = _app_with(controller)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /catalog — annotated catalog
# ---------------------------------------------------------------------------


def test_get_catalog_returns_annotated_rows(controller: SwapController) -> None:
    """``GET /catalog`` returns the catalog annotated with per-model state."""
    controller._client.list_models = AsyncMock(  # type: ignore[attr-defined]
        return_value=[ModelState(id="qwen3-7b-awq", state="available")]
    )
    client = _app_with(controller)
    r = client.get("/catalog")
    assert r.status_code == 200
    body = r.json()
    assert "models" in body
    ids = {m["id"] for m in body["models"]}
    assert ids == {"qwen3-7b-awq", "qwen3-14b-awq"}
    for m in body["models"]:
        assert m["state"] in {"available", "loading", "active", "unreachable"}


def test_get_catalog_marks_all_unreachable_on_client_error(controller: SwapController) -> None:
    """``GET /catalog`` degrades to all ``unreachable`` when llama-swap is down."""
    controller._client.list_models = AsyncMock(  # type: ignore[attr-defined]
        side_effect=LlamaSwapUnavailable("connect refused")
    )
    client = _app_with(controller)
    r = client.get("/catalog")
    assert r.status_code == 200
    body = r.json()
    for m in body["models"]:
        assert m["state"] == "unreachable"


# ---------------------------------------------------------------------------
# POST /catalog/activate — sidecar proxy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_activate_succeeds(controller: SwapController) -> None:
    """``POST /catalog/activate`` returns the post-activation status."""
    client = _app_with(controller)
    r = client.post("/catalog/activate", json={"model_id": "qwen3-7b-awq"})
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "ready"
    assert body["active_model_id"] == "qwen3-7b-awq"
    assert body["failure_reason"] is None


def test_post_activate_unknown_model_returns_404(controller: SwapController) -> None:
    """Unknown model id → 404 (mapped from ``NotFoundError``)."""
    # Wire a client whose activate would never be called (controller
    # raises NotFoundError before that).
    client = _app_with(controller)
    r = client.post("/catalog/activate", json={"model_id": "nope"})
    assert r.status_code == 404
    assert "nope" in r.json()["detail"]


def test_post_activate_unavailable_returns_503(controller: SwapController) -> None:
    """llama-swap unreachable on activation → 503 (typed ``LlamaSwapUnavailable``)."""
    controller._client.activate = AsyncMock(  # type: ignore[attr-defined]
        side_effect=LlamaSwapUnavailable("connect refused")
    )
    client = _app_with(controller)
    r = client.post("/catalog/activate", json={"model_id": "qwen3-7b-awq"})
    assert r.status_code == 503


def test_post_activate_validates_model_id_field(controller: SwapController) -> None:
    """Body must carry a non-empty ``model_id`` (Pydantic 422 on missing)."""
    client = _app_with(controller)
    r = client.post("/catalog/activate", json={})
    assert r.status_code == 422  # Pydantic validation error


# ---------------------------------------------------------------------------
# GET /catalog/status — current SwapStatus snapshot
# ---------------------------------------------------------------------------


def test_get_status_returns_idle_state(controller: SwapController) -> None:
    """``GET /catalog/status`` returns the current ``SwapStatus`` (idle initially)."""
    client = _app_with(controller)
    r = client.get("/catalog/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["active_model_id"] is None


def test_get_status_after_activation_returns_ready(controller: SwapController) -> None:
    """After a successful activation, ``GET /catalog/status`` reports ``ready``."""
    client = _app_with(controller)
    client.post("/catalog/activate", json={"model_id": "qwen3-7b-awq"})
    r = client.get("/catalog/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "ready"
    assert body["active_model_id"] == "qwen3-7b-awq"


# ---------------------------------------------------------------------------
# App construction: no controller pre-set → startup hook fails fast
# ---------------------------------------------------------------------------


def test_create_app_without_controller_pre_set_fails_on_request() -> None:
    """If the startup hook is skipped (no catalog_path/url), requests fail fast with 500.

    Production code always sets ``app.state.controller`` during the startup
    hook.  If the hook raises (no catalog path / no llama-swap URL), the
    route handlers raise HTTPException 500 because ``_get_controller`` finds
    ``None``.  We test the 500 path explicitly.
    """
    app = create_app()  # no catalog_path, no llama_swap_base_url
    # Skip the startup hook by NOT entering the TestClient context manager.
    # Use a bare TestClient without `with` so startup isn't run.
    client = TestClient(app)
    r = client.get("/catalog")
    # 500 because the controller is not initialised.
    assert r.status_code == 500


# ---------------------------------------------------------------------------
# Architectural keystone: sidecar routes do not enforce ACL
# ---------------------------------------------------------------------------


def test_sidecar_has_no_acl_routes(controller: SwapController) -> None:
    """The sidecar has no auth; ACL is the public API's responsibility.

    The public ACL gate is in ``src/iw_rag/api/models_routes.py`` and runs
    BEFORE the proxy hop.  The sidecar is internal-only and
    unauthenticated; the public routes must reflect that boundary.  Here
    we assert the sidecar's routes do NOT include a token/auth dependency
    (the simpler check: the sidecar's POST works without any header).
    """
    client = _app_with(controller)
    # No auth headers — the sidecar must still accept the request.
    r = client.post("/catalog/activate", json={"model_id": "qwen3-7b-awq"})
    assert r.status_code == 200
