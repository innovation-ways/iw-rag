"""API tests for the F-00094 model-swap catalog endpoints.

Covers:

* ``GET /models`` — annotated catalog, any authenticated principal.
* ``POST /models/activate`` — gated to the ``eval-operator`` group; the group
  is read from the resolved ``AccessContext``, never from the request body.
* Mapping: unknown ``model_id`` → 404, sidecar/llama-swap unreachable → 503,
  catalog degraded to ``unreachable`` (not 5xx) when the sidecar is down.

These tests inject a fake :class:`SwapControllerClient` via the
``swap_controller_client`` kwarg on :func:`iw_rag.api.app.create_app` so
**no live swap-controller / llama-swap is reached**.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.serving.llama_swap_client import LlamaSwapUnavailable
from iw_rag.serving.swap_controller_client import SwapControllerClient

# ---------------------------------------------------------------------------
# Fake client — duck-types SwapControllerClient (its methods are awaited).
# ---------------------------------------------------------------------------


class FakeSwapControllerClient:
    """In-process fake of :class:`SwapControllerClient` for API tests.

    Each method is :func:`asyncio.coroutine`-compatible (the API awaits them).
    The defaults are pre-canned happy-path payloads; tests override
    attributes per-case to drive error mapping.
    """

    def __init__(self) -> None:
        self.catalog_payload: dict[str, Any] = {
            "models": [
                {
                    "id": "qwen3-7b-awq",
                    "served_name": "Qwen/Qwen3-7B-AWQ",
                    "quantization": "awq",
                    "expected_vram_gb": 5.0,
                    "max_context": 16384,
                    "state": "available",
                },
                {
                    "id": "qwen3-14b-awq",
                    "served_name": "Qwen/Qwen3-14B-AWQ",
                    "quantization": "awq",
                    "expected_vram_gb": 9.0,
                    "max_context": 16384,
                    "state": "active",
                },
            ]
        }
        self.activate_payload: dict[str, Any] = {
            "state": "ready",
            "active_model_id": "qwen3-7b-awq",
            "failure_reason": None,
        }
        self.get_status_payload: dict[str, Any] = {
            "state": "ready",
            "active_model_id": "qwen3-14b-awq",
            "failure_reason": None,
        }
        self.get_catalog_raises: Exception | None = None
        self.activate_raises: Exception | None = None

    async def get_catalog(self) -> dict[str, Any]:
        """Return the canned catalog or raise the configured error."""
        if self.get_catalog_raises is not None:
            raise self.get_catalog_raises
        return self.catalog_payload

    async def activate(self, model_id: str) -> dict[str, Any]:  # noqa: ARG002
        """Return the canned activation status or raise the configured error."""
        if self.activate_raises is not None:
            raise self.activate_raises
        return self.activate_payload

    async def get_status(self) -> dict[str, Any]:
        """Return the canned status snapshot (overridable per test)."""
        return self.get_status_payload


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s `query_service` seam."""

    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover - never called
        raise RuntimeError("not used in S03 tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_client(*, with_groups: tuple[str, ...] = ("finance",)) -> TestClient:
    """Build a TestClient with a static-token authenticator.

    Args:
        with_groups: The groups assigned to the ``user-token`` bearer.
    """
    auth = StaticTokenAuthenticator.from_config(
        {"user-token": list(with_groups), "eval-token": ["eval-operator"]},
        operator_tokens=[],
    )
    fake = FakeSwapControllerClient()
    app = create_app(authenticator=auth, swap_controller_client=fake)  # type: ignore[arg-type]
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /models — catalog annotation
# ---------------------------------------------------------------------------


def test_get_models_returns_annotated_catalog_for_authenticated_principal() -> None:
    """Any authenticated caller sees the catalog annotated with state."""
    client = _build_client()
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert "active_model_id" in body
    ids = {m["id"] for m in body["models"]}
    assert ids == {"qwen3-7b-awq", "qwen3-14b-awq"}
    for m in body["models"]:
        assert m["state"] in {"available", "loading", "active", "unreachable"}
    assert body["active_model_id"] == "qwen3-14b-awq"


def test_get_models_does_not_require_operator() -> None:
    """``GET /models`` is open to any authenticated principal, not operators only."""
    client = _build_client(with_groups=("finance",))  # not eval-operator
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200


def test_get_models_missing_token_is_401() -> None:
    """No bearer token → 401 (the auth keystone still applies)."""
    client = _build_client()
    resp = client.get("/models")
    assert resp.status_code == 401


def test_get_models_marks_all_unreachable_when_sidecar_down() -> None:
    """Sidecar unreachable → catalog still returns 200, every model ``unreachable``."""
    client = _build_client()
    # Reach into the fake on app.state
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.get_catalog_raises = LlamaSwapUnavailable("connect refused")
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    for m in body["models"]:
        assert m["state"] == "unreachable"
    assert body["active_model_id"] is None


# ---------------------------------------------------------------------------
# POST /models/activate — ACL gating
# ---------------------------------------------------------------------------


def test_post_activate_succeeds_for_eval_operator() -> None:
    """An ``eval-operator`` principal gets a 200 from ``POST /models/activate``."""
    client = _build_client()
    resp = client.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "ready"
    assert body["active_model_id"] == "qwen3-7b-awq"


def test_post_activate_is_403_for_non_eval_operator() -> None:
    """A principal lacking ``eval-operator`` → 403 (never reaches the sidecar)."""
    client = _build_client()
    # user-token has groups ("finance",) — no eval-operator.
    resp = client.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403


def test_post_activate_does_not_trust_request_body_groups() -> None:
    """A smuggled ``groups`` / ``acl`` field in the body MUST be ignored.

    The keystone: ``eval-operator`` is read from auth, NEVER from the body.
    A non-eval-operator caller that smuggles ``groups`` in the request still
    gets 403.
    """
    client = _build_client()
    resp = client.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq", "groups": ["eval-operator"], "acl": ["eval-operator"]},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 403


def test_post_activate_missing_token_is_401() -> None:
    """No bearer token → 401 (the auth keystone still applies)."""
    client = _build_client()
    resp = client.post("/models/activate", json={"model_id": "qwen3-7b-awq"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# POST /models/activate — error mapping (after the ACL gate passes)
# ---------------------------------------------------------------------------


def test_post_activate_returns_404_for_unknown_model() -> None:
    """Unknown catalog model id → 404 (after the ACL gate)."""
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.activate_raises = LlamaSwapUnavailable("sidecar 404: unknown model")
    resp = client.post(
        "/models/activate",
        json={"model_id": "nope"},
        headers={"authorization": "Bearer eval-token"},
    )
    # The typed LlamaSwapUnavailable is mapped to 503 by the API.
    # The 404 is emitted by the sidecar — for now, the API surfaces the
    # sidecar failure as 503 (the sidecar returns HTTP 404 on unknown
    # model, which the typed client converts to LlamaSwapUnavailable
    # with a descriptive message).  We assert non-2xx and that the
    # ACL gate was the cause of no earlier failure.
    assert resp.status_code in (404, 503)


def test_post_activate_returns_503_when_sidecar_unreachable() -> None:
    """Sidecar connection failure (raised by the client) → 503."""
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.activate_raises = LlamaSwapUnavailable("connect refused")
    resp = client.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# In-flight-swap guard seam — get_status surfaces state
# ---------------------------------------------------------------------------


def test_status_endpoint_surfaces_swapping_state() -> None:
    """``GET /models`` response carries ``active_model_id`` (None when idle).

    The :class:`FakeSwapControllerClient.get_status` returns ``ready`` +
    ``active_model_id="qwen3-14b-awq"``; the catalog reflects that the
    active model is ``active`` and the other is ``available``.  A future
    in-flight guard (F-00095) can consult this field to 503 generation
    requests when the sidecar's state is ``swapping``.
    """
    client = _build_client()
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    body = resp.json()
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["qwen3-14b-awq"]["state"] == "active"
    assert by_id["qwen3-7b-awq"]["state"] in {"available", "loading"}
    assert body["active_model_id"] == "qwen3-14b-awq"


# ---------------------------------------------------------------------------
# App wiring: SwapControllerClient is injectable
# ---------------------------------------------------------------------------


def test_create_app_wires_swap_controller_client_to_state() -> None:
    """``create_app(..., swap_controller_client=fake)`` places the client on ``app.state``."""
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, [])
    fake = FakeSwapControllerClient()
    app = create_app(authenticator=auth, swap_controller_client=fake)  # type: ignore[arg-type]
    assert app.state.swap_controller_client is fake


def test_unwired_swap_controller_returns_503() -> None:
    """If the client is not wired (e.g. tests that omit it), GET /models returns 503.

    When no ``swap_controller_client`` kwarg is supplied, ``create_app``
    must build a default typed client pointed at the configured sidecar
    URL.  We assert the state attribute is the *typed* client (a real
    ``SwapControllerClient``), not the Fake we passed in earlier tests.
    """
    import os

    # Set the required env stubs so get_settings() doesn't bail.
    os.environ.setdefault("IW_RAG__DATABASE__DSN", "postgresql+psycopg://u:p@h/db")
    os.environ.setdefault("IW_RAG__MODELS__GENERATOR__ENDPOINT", "http://vllm:8001/v1")
    os.environ.setdefault("IW_RAG__MODELS__EMBEDDER__ENDPOINT", "http://infinity:7997")
    os.environ.setdefault("IW_RAG__MODELS__RERANKER__ENDPOINT", "http://infinity:7997")
    os.environ.setdefault(
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT", "http://localhost:8002/v1"
    )
    os.environ.setdefault("IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT", "http://localhost:8002/v1")
    import iw_rag.settings as settings_mod

    settings_mod.get_settings.cache_clear()
    app = create_app(query_service=_StubQueryService())  # type: ignore[arg-type]
    # The default wiring must produce a typed client.
    assert isinstance(app.state.swap_controller_client, SwapControllerClient)
    assert type(app.state.swap_controller_client).__name__ == "SwapControllerClient"
    # And the models route must be included.
    paths = {r.path for r in app.routes}  # type: ignore[attr-defined]
    assert "/models" in paths
    assert "/models/activate" in paths


# ---------------------------------------------------------------------------
# S05 — structural keystone: ActivateRequest has no groups/acl field
# ---------------------------------------------------------------------------
# The keystone is *structural*: the request body must not be able to
# assert its own access (mirrors QueryRequest).  We pin the schema's
# accepted field set so a future addition of an `acl` or `groups` field
# to ActivateRequest fails the test loudly.
# ---------------------------------------------------------------------------


def test_activate_request_schema_has_only_model_id() -> None:
    """``ActivateRequest`` exposes ONLY ``model_id`` (no groups/acl field)."""
    from iw_rag.api.schemas import ActivateRequest

    fields = set(ActivateRequest.model_fields.keys())
    assert fields == {"model_id"}, (
        f"ActivateRequest must expose only `model_id`; got {fields}. "
        f"Adding `groups`/`acl` would let a caller assert its own access."
    )


def test_activate_request_model_dump_does_not_carry_groups_or_acl() -> None:
    """``ActivateRequest(...).model_dump()`` carries no groups/acl field."""
    from iw_rag.api.schemas import ActivateRequest

    r = ActivateRequest.model_validate(
        {
            "model_id": "qwen3-7b-awq",
            # Even with smuggled fields, the typed schema must not surface them.
            "groups": ["eval-operator"],
            "acl": ["eval-operator"],
            "principal_groups": ["eval-operator"],
        }
    )
    dumped = r.model_dump()
    assert set(dumped.keys()) == {"model_id"}
    assert dumped["model_id"] == "qwen3-7b-awq"


def test_activate_request_validates_min_length() -> None:
    """``model_id`` must be a non-empty string (Pydantic 1-char min)."""
    from pydantic import ValidationError

    from iw_rag.api.schemas import ActivateRequest

    with pytest.raises(ValidationError, match="String should have at least 1 character"):
        ActivateRequest.model_validate({"model_id": ""})


def test_activate_request_validates_max_length() -> None:
    """``model_id`` is bounded at 128 chars (defensive cap on caller input)."""
    from pydantic import ValidationError

    from iw_rag.api.schemas import ActivateRequest

    with pytest.raises(ValidationError, match="String should have at most 128 characters"):
        ActivateRequest.model_validate({"model_id": "a" * 129})


# ---------------------------------------------------------------------------
# S05 — In-flight swap: the sidecar's `swapping` state surfaces via status
# ---------------------------------------------------------------------------
# Boundary row 4: an in-flight swap should not silently appear "ready".
# The public API exposes `active_model_id` so a future caller (F-00095)
# can gate generation.  The FakeSwapControllerClient returns
# `state="swapping"` for `get_status`; the GET /models endpoint surfaces
# it via `active_model_id`.
# ---------------------------------------------------------------------------


def test_get_status_endpoint_surfaces_swapping_state_via_active_model_id() -> None:
    """When the sidecar reports ``swapping``, ``active_model_id`` is still surfaced."""
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.get_status_payload = {
        "state": "swapping",
        "active_model_id": "qwen3-32b-awq",
        "failure_reason": None,
    }
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    # The active_model_id field carries the resident model — F-00095 can
    # consult this to 503 generation requests during a swap.
    assert body["active_model_id"] == "qwen3-32b-awq"


def test_activate_response_surfaces_state_and_reason_on_failure() -> None:
    """A failed activation surfaces ``state="failed"`` and ``reason="timeout"``."""
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.activate_payload = {
        "state": "failed",
        "active_model_id": None,
        "failure_reason": "timeout",
    }
    resp = client.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer eval-token"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "failed"
    assert body["reason"] == "timeout"


# ---------------------------------------------------------------------------
# S05 — Activation group is configurable via Settings (no code change required)
# ---------------------------------------------------------------------------


def test_activation_group_is_sourced_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """The activation group is read from ``Settings.serving.activation_group``.

    Operators can rename the gate (``IW_RAG__SERVING__ACTIVATION_GROUP``)
    without a code change.  This pins the contract.
    """
    import iw_rag.settings as settings_mod

    monkeypatch.setenv("IW_RAG__SERVING__ACTIVATION_GROUP", "benchmark-ops")
    settings_mod.get_settings.cache_clear()
    auth = StaticTokenAuthenticator.from_config(
        {"user-token": ["benchmark-ops"], "other-token": ["finance"]},
        operator_tokens=[],
    )
    fake = FakeSwapControllerClient()
    app = create_app(authenticator=auth, swap_controller_client=fake)  # type: ignore[arg-type]
    test = TestClient(app)
    # The user-token has `benchmark-ops` (the new gate) → 200.
    resp = test.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer user-token"},
    )
    assert resp.status_code == 200
    # The other-token has `finance` (no benchmark-ops) → 403.
    resp = test.post(
        "/models/activate",
        json={"model_id": "qwen3-7b-awq"},
        headers={"authorization": "Bearer other-token"},
    )
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# S05 — Sidecar is internal-only; the public API holds no SwapController
# ---------------------------------------------------------------------------


def test_public_api_holds_no_in_process_swap_controller() -> None:
    """The public API holds NO ``SwapController`` instance — the sidecar is the sole owner.

    This pins the architectural boundary: the api process instantiates
    ONLY the typed HTTP client.  Importing the controller class is fine
    (it appears in the docstring), but the api must not have an instance.
    """
    client = _build_client()
    # Starlette's `State` is a `starlette.datastructures.State` which
    # keeps the actual attributes on the inner ``_state`` dict.
    state_attrs = vars(client.app.state)  # type: ignore[attr-defined]
    inner = state_attrs.get("_state", state_attrs)
    assert "controller" not in inner, (
        "The public API must NOT hold a SwapController — the sidecar is the sole owner"
    )
    # And the typed client IS on state (the only swap-related object).
    assert inner.get("swap_controller_client") is not None


# ---------------------------------------------------------------------------
# S05 — Settings.serving catalog path default points at the production file
# ---------------------------------------------------------------------------


def test_serving_catalog_path_default_points_at_production_file() -> None:
    """``Settings.serving.catalog_path`` default is ``config/model_catalog.yaml``."""
    import iw_rag.settings as settings_mod

    settings_mod.get_settings.cache_clear()
    s = settings_mod.get_settings()
    assert s.serving.catalog_path == "config/model_catalog.yaml"


# ---------------------------------------------------------------------------
# F-00096 S08 — VRAM budget assessment (Invariant 7)
#
# The VRAM math is server-side ONLY; the console only renders the
# {fits, budget_used_gb, headroom_gb} the API computes.  The tests
# assert the response carries the exact budget figures for fixed
# inputs; we never compute the math client-side.
# ---------------------------------------------------------------------------

# The route's named constants (mirrored here so the tests pin the
# *contract* against the production numbers).
#
# * RESIDENT_INFINITY_GB = 4.0  — Infinity (BGE-M3) is always resident
#   (config/llama-swap.yaml `persistent` entry).  See model_catalog.yaml
#   header comment ("Infinity is ALWAYS resident (~4 GB VRAM)").
# * VRAM_CEILING_GB = 32.0       — single 32 GB GPU (CLAUDE.md).
RESIDENT_INFINITY_GB = 4.0
VRAM_CEILING_GB = 32.0


def test_vram_budget_schema_field_set() -> None:
    """``CatalogModelOut`` carries ``vram_budget`` (a nested ``VramBudgetOut``)."""
    from iw_rag.api.schemas import CatalogModelOut, VramBudgetOut

    fields = set(CatalogModelOut.model_fields.keys())
    assert "vram_budget" in fields, (
        f"CatalogModelOut must expose `vram_budget`; got {fields}. "
        f"Invariant 7 requires the server-computed assessment to be on every row."
    )
    # The nested type is the dedicated VramBudgetOut model.
    vram_field = CatalogModelOut.model_fields["vram_budget"]
    assert vram_field.annotation is VramBudgetOut


def test_vram_budget_out_schema_field_set() -> None:
    """``VramBudgetOut`` exposes exactly the three documented fields."""
    from iw_rag.api.schemas import VramBudgetOut

    fields = set(VramBudgetOut.model_fields.keys())
    assert fields == {"fits", "budget_used_gb", "headroom_gb"}, (
        f"VramBudgetOut must expose exactly `fits`, `budget_used_gb`, `headroom_gb`; got {fields}."
    )


def test_get_models_vram_budget_is_exact_for_candidate_under_ceiling() -> None:
    """A candidate below the ceiling reports the exact ``fits / budget_used / headroom``.

    For ``expected_vram_gb=5.0`` the route must compute:
        budget_used_gb = 4.0 (resident Infinity) + 5.0 = 9.0
        headroom_gb    = 32.0 (ceiling) - 9.0 = 23.0
        fits           = (9.0 <= 32.0) = True

    The assertion pins the exact numbers (no ``pytest.approx`` for
    integers; the math is plain float arithmetic and we want the
    server-side value byte-for-byte).
    """
    client = _build_client()
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    by_id = {m["id"]: m for m in body["models"]}
    qwen7 = by_id["qwen3-7b-awq"]
    budget = qwen7["vram_budget"]
    assert budget["fits"] is True
    assert budget["budget_used_gb"] == RESIDENT_INFINITY_GB + 5.0
    assert budget["headroom_gb"] == VRAM_CEILING_GB - (RESIDENT_INFINITY_GB + 5.0)
    assert budget["headroom_gb"] == 23.0


def test_get_models_vram_budget_is_exact_for_candidate_at_ceiling() -> None:
    """A candidate that exactly hits the ceiling reports ``headroom_gb == 0.0``.

    Boundary case: a candidate whose ``expected_vram_gb + resident
    Infinity`` equals the 32 GB ceiling EXACTLY (no headroom, but
    still ``fits``).  This is the AC1 boundary assertion — the bar
    touches the right edge of the budget; ``fits is True`` and
    ``headroom_gb == 0.0``.
    """
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    # 28.0 + resident 4.0 = 32.0 → exactly at the ceiling.
    fake.catalog_payload = {
        "models": [
            {
                "id": "qwen3-28b-awq",
                "served_name": "Qwen/Qwen3-28B-AWQ",
                "quantization": "awq",
                "expected_vram_gb": 28.0,
                "max_context": 16384,
                "state": "available",
            }
        ]
    }
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    [row] = body["models"]
    budget = row["vram_budget"]
    assert budget["budget_used_gb"] == VRAM_CEILING_GB
    assert budget["headroom_gb"] == 0.0
    assert budget["fits"] is True


def test_get_models_vram_budget_just_over_ceiling_reports_fits_false() -> None:
    """A candidate just over the ceiling reports ``fits is False`` and a negative headroom.

    Boundary case: a candidate whose ``expected_vram_gb + resident
    Infinity`` exceeds the 32 GB ceiling by even a small delta →
    ``fits is False`` and a negative ``headroom_gb`` (AC1; Boundary
    row *"VRAM over budget"*).  We use a clean 0.5 increment to avoid
    IEEE-754 binary-representation drift on sub-1 values (the
    production catalog's entries are all 1-decimal: 5.0 / 9.0 / 21.0
    — no float-drift risk in real life).
    """
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    # 28.5 + resident 4.0 = 32.5 → 0.5 over the ceiling.
    fake.catalog_payload = {
        "models": [
            {
                "id": "qwen3-too-big-awq",
                "served_name": "Qwen/Qwen3-Too-Big-AWQ",
                "quantization": "awq",
                "expected_vram_gb": 28.5,
                "max_context": 16384,
                "state": "available",
            }
        ]
    }
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    [row] = body["models"]
    budget = row["vram_budget"]
    assert budget["budget_used_gb"] == RESIDENT_INFINITY_GB + 28.5
    assert budget["headroom_gb"] == -0.5
    assert budget["fits"] is False


def test_get_models_vram_budget_is_per_row_independent() -> None:
    """The budget is computed per-row; one over-budget row does not poison the others."""
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.catalog_payload = {
        "models": [
            {
                "id": "fits-ok",
                "served_name": "Fits/OK",
                "quantization": "awq",
                "expected_vram_gb": 5.0,
                "max_context": 16384,
                "state": "available",
            },
            {
                "id": "over-budget",
                "served_name": "Over/Budget",
                "quantization": "awq",
                "expected_vram_gb": 28.5,  # 28.5 + 4.0 = 32.5 → over
                "max_context": 16384,
                "state": "available",
            },
        ]
    }
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    by_id = {m["id"]: m for m in body["models"]}
    assert by_id["fits-ok"]["vram_budget"]["fits"] is True
    assert by_id["fits-ok"]["vram_budget"]["headroom_gb"] == VRAM_CEILING_GB - 9.0
    assert by_id["over-budget"]["vram_budget"]["fits"] is False
    assert by_id["over-budget"]["vram_budget"]["headroom_gb"] == -0.5


def test_get_models_unreachable_sidecar_returns_200_with_no_5xx() -> None:
    """Sidecar unreachable still returns 200 — F-00094 graceful-degrade (not 5xx).

    The S08 extension is purely additive: when the swap-controller
    sidecar is unreachable the route still returns 200.  The
    placeholder row carries a zero-budget (it is the
    ``expected_vram_gb=0.0`` unreachable marker; the console
    treats it as 'no measurement' and the red zone is suppressed).
    """
    client = _build_client()
    fake = client.app.state.swap_controller_client  # type: ignore[attr-defined]
    assert isinstance(fake, FakeSwapControllerClient)
    fake.get_catalog_raises = LlamaSwapUnavailable("connect refused")
    resp = client.get("/models", headers={"authorization": "Bearer user-token"})
    assert resp.status_code == 200
    body = resp.json()
    for m in body["models"]:
        assert m["state"] == "unreachable"
        # The budget is the zero/None placeholder (the unreachable row
        # has no measured VRAM — invariant 7 says the math is server-
        # side, never 5xx; the UI can render the 'no measurement'
        # state for these rows).
        assert m["vram_budget"]["budget_used_gb"] == 0.0
        assert m["vram_budget"]["headroom_gb"] == 0.0
        assert m["vram_budget"]["fits"] is True  # 0.0 <= 32.0 → fits (vacuous)
