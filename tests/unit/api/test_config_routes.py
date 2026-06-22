"""Unit tests for the F-00096 S07 Config API.

The S07 routes are a **thin HTTP facade** over the S03 ``ConfigService``
(``GET /api/config`` reads ``service.read_active()``; ``PUT
/api/config/{key}`` calls ``service.apply_edit(...)``).  These tests
inject a fake ``ConfigService`` via ``app.state.config_service`` so no
live DB / live Settings validation / live overlay state is reached
— the routes are exercised end-to-end against the typed fake.

Covers:

* ``GET /api/config`` — surfaces the active entries with editable /
  locked / pending_restart flags and per-key explanations; open to any
  authenticated principal.
* ``PUT /api/config/{key}`` — operator-gated (``eval-operator``),
  validated, applied, audited.  The fake records the exact call so we
  pin: the in-bounds hot path returns 200 + ``applied_mode``; a locked
  key is refused (4xx, no write); an out-of-bounds value is refused
  (4xx, no write); a non-operator is 403 (AC7).
* The principal_id sourced from auth is the one the fake received —
  the keystone that the body never carries an acl field (Invariant 3,
  AC2, AC7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.domain.errors import ValidationError

# ---------------------------------------------------------------------------
# Fake ConfigService — duck-typed
# ---------------------------------------------------------------------------


@dataclass
class _RecordedCall:
    """A captured call to the fake service's ``apply_edit``."""

    key: str
    new_value: Any
    principal_id: str
    trace_id: str | None = None


@dataclass
class _FakeActiveEntry:
    """A canned entry the fake returns from ``read_active``."""

    key: str
    description: str
    active_value: Any
    editable: bool
    locked: bool
    pending_restart: bool


@dataclass
class _FakeApplyResult:
    """A canned result the fake returns from ``apply_edit``."""

    config_key: str
    old_value: Any
    new_value: Any
    applied_mode: str
    principal_id: str


class FakeConfigService:
    """In-process fake of :class:`~iw_rag.api.config_service.ConfigService`.

    Tests configure ``active_entries`` (the GET payload) and
    ``apply_result`` (the canned PUT response) per-case; tests that
    drive the error paths set ``apply_raises`` to the typed
    :class:`ValidationError` so the route's propagation can be
    observed.  All ``apply_edit`` calls are recorded into
    ``recorded_calls`` so the assertion-strength pins verify the
    exact principal_id / key / value the route forwarded.

    Plain class (not a ``@dataclass``) so :attr:`recorded_calls` is
    initialised per-instance in ``__init__`` (a ``@dataclass`` with a
    ``field(default_factory=list)`` annotation is fine but
    unnecessary and obscures the in-__init__ wiring — the field
    cannot be a mutable default like ``[]``).
    """

    def __init__(self) -> None:
        self.active_entries: list[_FakeActiveEntry] = []
        self.apply_result: _FakeApplyResult | None = None
        self.apply_raises: Exception | None = None
        self.recorded_calls: list[_RecordedCall] = []

    def read_active(self) -> list[_FakeActiveEntry]:
        """Return the canned active entries list."""
        return list(self.active_entries)

    async def apply_edit(
        self,
        *,
        key: str,
        new_value: Any,
        principal_id: str,
        trace_id: str | None = None,
    ) -> _FakeApplyResult:
        """Record the call and return the canned result (or raise)."""
        self.recorded_calls.append(
            _RecordedCall(
                key=key,
                new_value=new_value,
                principal_id=principal_id,
                trace_id=trace_id,
            )
        )
        if self.apply_raises is not None:
            raise self.apply_raises
        assert self.apply_result is not None, (
            "test must configure either apply_result or apply_raises"
        )
        return self.apply_result


# ---------------------------------------------------------------------------
# Stub query service — satisfies create_app's `query_service` seam
# ---------------------------------------------------------------------------


class _StubQueryService:
    async def answer(self, query: str, ctx: Any) -> Any:  # pragma: no cover - never called
        raise RuntimeError("not used in S07 config tests")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _build_client(
    *,
    fake: FakeConfigService,
    user_groups: tuple[str, ...] = ("finance",),
    eval_groups: tuple[str, ...] = ("eval-operator",),
) -> TestClient:
    """Build a TestClient with a static-token authenticator + a fake service.

    Args:
        fake: The fake ConfigService to inject on ``app.state``.
        user_groups: The groups assigned to ``user-token``.
        eval_groups: The groups assigned to ``eval-token`` (the operator
            gate required by ``require_eval_operator``).
    """
    auth = StaticTokenAuthenticator.from_config(
        {
            "user-token": list(user_groups),
            "eval-token": list(eval_groups),
        },
        operator_tokens=[],
    )
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        config_service=fake,  # type: ignore[arg-type]
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /api/config
# ---------------------------------------------------------------------------


def test_get_config_returns_active_entries_with_explanations_and_flags() -> None:
    """GET /api/config surfaces the service's entries with editable / locked / pending flags."""
    fake = FakeConfigService()
    fake.active_entries = [
        _FakeActiveEntry(
            key="guardrails.input.threshold",
            description="Block threshold for the input guard.",
            active_value=0.5,
            editable=True,
            locked=False,
            pending_restart=False,
        ),
        _FakeActiveEntry(
            key="retrieval.top_k",
            description="Final top-k returned to the generator.",
            active_value=10,
            editable=True,
            locked=False,
            pending_restart=True,
        ),
        _FakeActiveEntry(
            key="guardrails.enabled",
            description="(locked) — security-critical / infra; not editable.",
            active_value=True,
            editable=False,
            locked=True,
            pending_restart=False,
        ),
    ]
    client = _build_client(fake=fake)

    resp = client.get("/api/config", headers={"authorization": "Bearer user-token"})

    assert resp.status_code == 200
    body = resp.json()
    assert "items" in body
    by_key = {item["key"]: item for item in body["items"]}
    assert by_key["guardrails.input.threshold"]["editable"] is True
    assert by_key["guardrails.input.threshold"]["locked"] is False
    assert by_key["guardrails.input.threshold"]["pending_restart"] is False
    assert by_key["guardrails.input.threshold"]["explanation"].startswith("Block threshold")
    assert by_key["guardrails.input.threshold"]["value"] == 0.5
    assert by_key["retrieval.top_k"]["pending_restart"] is True
    assert by_key["retrieval.top_k"]["editable"] is True
    assert by_key["guardrails.enabled"]["editable"] is False
    assert by_key["guardrails.enabled"]["locked"] is True


def test_get_config_does_not_require_operator() -> None:
    """``GET /api/config`` is open to any authenticated principal.

    Mirrors the :func:`iw_rag.api.benchmark_routes.list_benchmark_reports`
    openness — the active config list reveals no sensitive content
    because the S03 service redacts secret-shaped locked keys.
    """
    fake = FakeConfigService()
    fake.active_entries = []
    client = _build_client(fake=fake, user_groups=("finance",))

    resp = client.get("/api/config", headers={"authorization": "Bearer user-token"})

    assert resp.status_code == 200
    assert resp.json() == {"items": []}


def test_get_config_missing_token_is_401() -> None:
    """No bearer token → 401 (the auth keystone still applies)."""
    fake = FakeConfigService()
    client = _build_client(fake=fake)

    resp = client.get("/api/config")

    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# PUT /api/config/{key} — happy path
# ---------------------------------------------------------------------------


def test_put_config_in_bounds_editable_key_returns_200_and_audits() -> None:
    """PUT on an in-bounds editable key → 200 + applied_mode echoed; fake records the call."""
    fake = FakeConfigService()
    fake.apply_result = _FakeApplyResult(
        config_key="guardrails.input.threshold",
        old_value=0.5,
        new_value=0.7,
        applied_mode="hot",
        principal_id="eval-tok",
    )
    client = _build_client(fake=fake)

    resp = client.put(
        "/api/config/guardrails.input.threshold",
        json={"value": 0.7},
        headers={"authorization": "Bearer eval-token"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["key"] == "guardrails.input.threshold"
    assert body["applied_mode"] == "hot"
    assert body["new_value"] == 0.7
    assert body["old_value"] == 0.5
    # The fake recorded exactly one call with the auth-supplied principal.
    assert len(fake.recorded_calls) == 1
    call = fake.recorded_calls[0]
    assert call.key == "guardrails.input.threshold"
    assert call.new_value == 0.7
    assert call.principal_id == "eval-tok"


def test_put_config_sources_principal_id_from_auth_never_body() -> None:
    """A smuggled ``principal_id`` / ``groups`` field in the body MUST be ignored.

    The keystone: the route sources the principal from the resolved
    :class:`AccessContext`, never from the request body.  The fake
    records the exact ``principal_id`` it received; we assert that
    it equals the auth-supplied value, NOT the body's smuggled value.
    """
    fake = FakeConfigService()
    fake.apply_result = _FakeApplyResult(
        config_key="guardrails.input.threshold",
        old_value=0.5,
        new_value=0.7,
        applied_mode="hot",
        principal_id="eval-tok",
    )
    client = _build_client(fake=fake)

    resp = client.put(
        "/api/config/guardrails.input.threshold",
        json={"value": 0.7, "principal_id": "attacker", "groups": ["eval-operator"]},
        headers={"authorization": "Bearer eval-token"},
    )

    assert resp.status_code == 200
    assert len(fake.recorded_calls) == 1
    assert fake.recorded_calls[0].principal_id == "eval-tok"


# ---------------------------------------------------------------------------
# PUT /api/config/{key} — error mapping
# ---------------------------------------------------------------------------


def test_put_config_locked_key_returns_4xx_and_records_no_write() -> None:
    """PUT on a locked key → 4xx (ValidationError → 400) and the fake records no write.

    AC3 / boundary "Config edit to a locked key".
    """
    fake = FakeConfigService()
    fake.apply_raises = ValidationError(
        "Config key 'guardrails.enabled' is locked — it cannot be edited. "
        "Locked keys cover security/infra invariants (F-00096 Invariant 5)."
    )
    client = _build_client(fake=fake)

    resp = client.put(
        "/api/config/guardrails.enabled",
        json={"value": False},
        headers={"authorization": "Bearer eval-token"},
    )

    assert resp.status_code == 400
    assert len(fake.recorded_calls) == 1
    assert fake.recorded_calls[0].key == "guardrails.enabled"


def test_put_config_out_of_bounds_returns_4xx_and_records_no_write() -> None:
    """PUT out-of-bounds value → 4xx; the fake raised — the route must propagate (no overlay)."""
    fake = FakeConfigService()
    fake.apply_raises = ValidationError(
        "Value for 'retrieval.top_k' is out of declared bounds (kind=int_range, min=1, max=100)."
    )
    client = _build_client(fake=fake)

    resp = client.put(
        "/api/config/retrieval.top_k",
        json={"value": -1},
        headers={"authorization": "Bearer eval-token"},
    )

    assert resp.status_code == 400
    assert len(fake.recorded_calls) == 1
    assert fake.recorded_calls[0].key == "retrieval.top_k"


def test_put_config_pending_restart_applied_mode_is_echoed() -> None:
    """A pending_restart edit returns 200 with applied_mode='pending_restart' (AC4)."""
    fake = FakeConfigService()
    fake.apply_result = _FakeApplyResult(
        config_key="retrieval.top_k",
        old_value=10,
        new_value=12,
        applied_mode="pending_restart",
        principal_id="eval-tok",
    )
    client = _build_client(fake=fake)

    resp = client.put(
        "/api/config/retrieval.top_k",
        json={"value": 12},
        headers={"authorization": "Bearer eval-token"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["applied_mode"] == "pending_restart"
    assert body["new_value"] == 12


# ---------------------------------------------------------------------------
# PUT /api/config/{key} — ACL gating
# ---------------------------------------------------------------------------


def test_put_config_non_operator_returns_403_and_records_no_call() -> None:
    """A non-eval-operator principal → 403 (AC7); the fake is NEVER called."""
    fake = FakeConfigService()
    client = _build_client(fake=fake, user_groups=("finance",))

    resp = client.put(
        "/api/config/guardrails.input.threshold",
        json={"value": 0.7},
        headers={"authorization": "Bearer user-token"},
    )

    assert resp.status_code == 403
    # The ACL gate ran BEFORE the service — no call recorded.
    assert fake.recorded_calls == []


def test_put_config_missing_token_is_401() -> None:
    """No bearer token → 401; the fake is NEVER called."""
    fake = FakeConfigService()
    client = _build_client(fake=fake)

    resp = client.put("/api/config/guardrails.input.threshold", json={"value": 0.7})

    assert resp.status_code == 401
    assert fake.recorded_calls == []


# ---------------------------------------------------------------------------
# Schemas — structural keystone: ConfigUpdateRequest has no acl/groups field
# ---------------------------------------------------------------------------


def test_config_update_request_schema_has_only_value() -> None:
    """``ConfigUpdateRequest`` exposes ONLY ``value`` (no acl/groups/principal_id field)."""
    from iw_rag.api.schemas import ConfigUpdateRequest

    fields = set(ConfigUpdateRequest.model_fields.keys())
    assert fields == {"value"}, (
        f"ConfigUpdateRequest must expose only `value`; got {fields}. "
        f"Adding `principal_id`/`groups`/`acl` would let a caller assert its own access."
    )


def test_config_update_request_model_dump_does_not_carry_acl_fields() -> None:
    """``ConfigUpdateRequest(...).model_dump()`` carries no acl/principal_id/groups field."""
    from iw_rag.api.schemas import ConfigUpdateRequest

    r = ConfigUpdateRequest.model_validate(
        {
            "value": 0.7,
            # Smuggled fields must be ignored by the typed schema.
            "principal_id": "attacker",
            "groups": ["eval-operator"],
            "acl": ["eval-operator"],
        }
    )
    dumped = r.model_dump()
    assert set(dumped.keys()) == {"value"}
    assert dumped["value"] == 0.7


# ---------------------------------------------------------------------------
# App wiring: config_service is injectable on app.state
# ---------------------------------------------------------------------------


def test_create_app_wires_config_service_to_state() -> None:
    """``create_app(..., config_service=fake)`` places the service on app.state."""
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, [])
    fake = FakeConfigService()
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        config_service=fake,  # type: ignore[arg-type]
    )
    assert app.state.config_service is fake
    paths = {r.path for r in app.routes}  # type: ignore[attr-defined]
    assert "/api/config" in paths


def test_get_config_unwired_service_returns_400() -> None:
    """When ``config_service`` is not wired on ``app.state``, GET returns 400 (not 5xx).

    Mirrors the ``_benchmark_service`` / ``_swap_controller_client``
    seam — an unwired deployment surfaces a typed ``ValidationError``
    that ``_ERROR_STATUS`` maps to 400, not 5xx.
    """
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, [])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
    )
    # No config_service kwarg — state is left unwired.
    client = TestClient(app)
    resp = client.get("/api/config", headers={"authorization": "Bearer u"})
    assert resp.status_code == 400
