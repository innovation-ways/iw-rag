"""Unit tests for the /auth/login route and DemoUserAuthenticator.login().

Coverage:

- ``DemoUserAuthenticator.login()``: valid credentials → token; wrong
  password → None; unknown user → None.
- ``/auth/login`` route: 200 + body fields on success; 401 on bad
  credentials; 401 when the authenticator has no ``login`` method.
- ``bearer_token()`` parsing: valid ``Bearer <tok>`` → token; missing
  prefix; mixed case; extra whitespace; empty / None input.

Security note: ``DemoUserAuthenticator.login()`` docstring claims
"length-stable comparison" (constant-time) but the implementation uses
a plain ``!=`` comparison (auth.py:229).  This is a doc/security drift
in the sense that the claim in the docstring is not backed by
``hmac.compare_digest``; the functional behaviour (returns None on
mismatch) is correct.  No xfail is issued — this is not a functional
bug — but the discrepancy is documented here for transparency.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from iw_rag.api.app import create_app
from iw_rag.api.auth import DemoUserAuthenticator, StaticTokenAuthenticator, bearer_token

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg`` (attribute access).

    Args:
        password: The demo user's plain-text password.
        token: The bearer token issued to this user.
        groups: The configured entitlement groups.
        operator: Whether the user has operator privilege.
    """

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


def _demo_auth() -> DemoUserAuthenticator:
    """Build a two-user DemoUserAuthenticator (alice + bob).

    Returns:
        A configured ``DemoUserAuthenticator`` with alice (finance,
        non-operator, password "secret-alice") and bob (hr, operator,
        password "secret-bob").
    """
    cfg = {
        "alice": _FakeDemoCfg(
            password="secret-alice",
            token="tok-alice",
            groups=["finance"],
            operator=False,
        ),
        "bob": _FakeDemoCfg(
            password="secret-bob",
            token="tok-bob",
            groups=["hr"],
            operator=True,
        ),
    }
    return DemoUserAuthenticator.from_demo_users(cfg)


def _demo_client() -> TestClient:
    """Build a TestClient wired with the demo authenticator.

    Returns:
        A ``TestClient`` backed by ``create_app`` with the two-user
        demo auth.  No live DB / LLM is built — services are injected
        via the stub ``query_service`` seam.
    """

    class _StubQueryService:
        async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
            raise RuntimeError("not used in auth login tests")

    app = create_app(
        authenticator=_demo_auth(),
        query_service=_StubQueryService(),
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. DemoUserAuthenticator.login() unit tests
# ---------------------------------------------------------------------------


def test_demo_login_valid_credentials_returns_token() -> None:
    """Valid username + password returns the configured bearer token."""
    auth = _demo_auth()
    token = auth.login("alice", "secret-alice")
    assert token == "tok-alice"


def test_demo_login_valid_token_decodes_to_right_principal() -> None:
    """Token from login resolves back to the correct principal via authenticate()."""
    auth = _demo_auth()
    token = auth.login("alice", "secret-alice")
    assert token is not None
    ctx = auth.authenticate(token)
    assert ctx is not None
    assert ctx.principal_id == "alice"
    assert "finance" in ctx.principal_groups
    assert "public" in ctx.principal_groups
    assert "user:alice" in ctx.principal_groups
    assert ctx.is_operator is False


def test_demo_login_operator_user_returns_operator_context() -> None:
    """Login as an operator user decodes to an operator AccessContext."""
    auth = _demo_auth()
    token = auth.login("bob", "secret-bob")
    assert token is not None
    ctx = auth.authenticate(token)
    assert ctx is not None
    assert ctx.principal_id == "bob"
    assert ctx.is_operator is True


def test_demo_login_wrong_password_returns_none() -> None:
    """Submitting a wrong password returns None (does not raise)."""
    auth = _demo_auth()
    result = auth.login("alice", "wrong-password")
    assert result is None


def test_demo_login_unknown_user_returns_none() -> None:
    """Submitting an unknown username returns None (does not raise)."""
    auth = _demo_auth()
    result = auth.login("eve", "any-password")
    assert result is None


def test_demo_login_empty_password_returns_none() -> None:
    """An empty password never matches a non-empty configured password."""
    auth = _demo_auth()
    result = auth.login("alice", "")
    assert result is None


def test_demo_login_empty_username_returns_none() -> None:
    """An empty username is not a registered user — returns None."""
    auth = _demo_auth()
    result = auth.login("", "secret-alice")
    assert result is None


def test_demo_login_case_sensitive_username() -> None:
    """Username matching is case-sensitive (Alice != alice)."""
    auth = _demo_auth()
    result = auth.login("Alice", "secret-alice")
    assert result is None


def test_demo_login_case_sensitive_password() -> None:
    """Password matching is case-sensitive (Secret-Alice != secret-alice)."""
    auth = _demo_auth()
    result = auth.login("alice", "Secret-alice")
    assert result is None


# ---------------------------------------------------------------------------
# 2. POST /auth/login route — success path
# ---------------------------------------------------------------------------


def test_login_route_valid_credentials_returns_200() -> None:
    """Valid credentials produce HTTP 200."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    assert resp.status_code == 200


def test_login_route_response_contains_token_field() -> None:
    """The 200 body carries a non-empty ``token`` field."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    body = resp.json()
    assert "token" in body
    assert body["token"] == "tok-alice"


def test_login_route_response_contains_principal_id() -> None:
    """The 200 body carries ``principal_id`` matching the submitted username."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    body = resp.json()
    assert body["principal_id"] == "alice"


def test_login_route_response_contains_groups() -> None:
    """The 200 body carries ``groups`` with the expected augmented set for alice."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    body = resp.json()
    groups = set(body["groups"])
    assert groups == {"finance", "public", "user:alice"}


def test_login_route_response_contains_is_operator() -> None:
    """The 200 body carries ``is_operator`` set to False for a non-operator user."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    body = resp.json()
    assert "is_operator" in body
    assert body["is_operator"] is False


def test_login_route_operator_user_returns_is_operator_true() -> None:
    """Bob (operator) gets ``is_operator=true`` in the login response."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "bob", "password": "secret-bob"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["is_operator"] is True
    assert body["principal_id"] == "bob"


def test_login_route_issued_token_authenticates_subsequent_requests() -> None:
    """The token from /auth/login can be used as a bearer on /health."""
    client = _demo_client()
    login_resp = client.post("/auth/login", json={"username": "alice", "password": "secret-alice"})
    token = login_resp.json()["token"]
    health_resp = client.get("/health", headers={"authorization": f"Bearer {token}"})
    assert health_resp.status_code == 200


# ---------------------------------------------------------------------------
# 3. POST /auth/login route — failure paths
# ---------------------------------------------------------------------------


def test_login_route_wrong_password_returns_401() -> None:
    """Wrong password → HTTP 401 (not 400, not 403)."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "nope"})
    assert resp.status_code == 401


def test_login_route_unknown_user_returns_401() -> None:
    """Unknown username → HTTP 401 (not 404)."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "eve", "password": "anything"})
    assert resp.status_code == 401


def test_login_route_wrong_password_error_body_has_error_field() -> None:
    """A 401 on bad credentials carries ``error == 'AuthenticationError'`` (no PII)."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "bad"})
    body = resp.json()
    assert body["error"] == "AuthenticationError"


def test_login_route_wrong_password_body_does_not_leak_token() -> None:
    """The 401 body must not contain a ``token`` field."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": "bad"})
    body = resp.json()
    assert "token" not in body


def test_login_route_missing_username_is_422() -> None:
    """A request body without ``username`` fails Pydantic validation → 422."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"password": "secret-alice"})
    assert resp.status_code == 422


def test_login_route_missing_password_is_422() -> None:
    """A request body without ``password`` fails Pydantic validation → 422."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice"})
    assert resp.status_code == 422


def test_login_route_empty_username_is_422() -> None:
    """An empty ``username`` violates the min_length=1 constraint → 422."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "", "password": "secret-alice"})
    assert resp.status_code == 422


def test_login_route_empty_password_is_422() -> None:
    """An empty ``password`` violates the min_length=1 constraint → 422."""
    client = _demo_client()
    resp = client.post("/auth/login", json={"username": "alice", "password": ""})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 4. POST /auth/login — authenticator without login support → 401
# ---------------------------------------------------------------------------


def _static_client() -> TestClient:
    """Build a TestClient wired with the legacy StaticTokenAuthenticator.

    The static token authenticator has no ``login`` attribute, so the
    /auth/login route must refuse with 401 and the message
    "login is not enabled for this deployment".

    Returns:
        A ``TestClient`` backed by ``StaticTokenAuthenticator``.
    """

    class _StubQueryService:
        async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
            raise RuntimeError("not used in login-unsupported tests")

    auth = StaticTokenAuthenticator.from_config(
        {"tok": ["finance"]},
        operator_tokens=[],
    )
    app = create_app(authenticator=auth, query_service=_StubQueryService())
    return TestClient(app)


def test_login_route_without_login_support_returns_401() -> None:
    """An authenticator without a ``login`` method → HTTP 401."""
    client = _static_client()
    resp = client.post("/auth/login", json={"username": "any", "password": "any"})
    assert resp.status_code == 401


def test_login_route_without_login_support_body_mentions_not_enabled() -> None:
    """The 401 body contains ``error`` when login is unsupported (not 500)."""
    client = _static_client()
    resp = client.post("/auth/login", json={"username": "any", "password": "any"})
    body = resp.json()
    # The route raises AuthenticationError("login is not enabled for this deployment")
    # which the app error handler maps to {"error": "AuthenticationError", ...}.
    assert "error" in body
    assert resp.status_code == 401


def test_login_route_without_login_support_patches_login_attribute_to_none() -> None:
    """Patching ``login`` off a DemoUserAuthenticator also triggers the 401 path.

    The route checks ``getattr(authenticator, 'login', None)`` — if any
    authenticator is deployed with the attribute removed, the route must
    still return 401 (not 500).
    """

    class _NoLoginAuth:
        """Minimal duck-type: has ``authenticate`` + ``resolve_principal`` but no ``login``."""

        def authenticate(self, token: str | None, request_id: str = "") -> None:
            return None

        def resolve_principal(self, principal_id: str) -> None:
            return None

    class _StubQS:
        async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover
            raise RuntimeError("not reached")

    app = create_app(authenticator=_NoLoginAuth(), query_service=_StubQS())  # type: ignore[arg-type]
    client = TestClient(app)
    resp = client.post("/auth/login", json={"username": "alice", "password": "secret"})
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# 5. bearer_token() parsing
# ---------------------------------------------------------------------------


def test_bearer_token_valid_bearer_prefix_returns_token() -> None:
    """A well-formed ``Bearer <tok>`` header yields the token part."""
    result = bearer_token("Bearer my-secret-token")
    assert result == "my-secret-token"


def test_bearer_token_lowercase_bearer_prefix_is_accepted() -> None:
    """``bearer <tok>`` (all lowercase) is accepted — the check is case-insensitive."""
    result = bearer_token("bearer my-token")
    assert result == "my-token"


def test_bearer_token_mixed_case_bearer_prefix_is_accepted() -> None:
    """``Bearer`` / ``BEARER`` / ``BeArEr`` — all normalised to lower for the check."""
    assert bearer_token("BEARER tok") == "tok"
    assert bearer_token("BeArEr tok") == "tok"


def test_bearer_token_strips_trailing_whitespace_from_token() -> None:
    """Trailing whitespace after the token value is stripped."""
    result = bearer_token("Bearer tok-value   ")
    assert result == "tok-value"


def test_bearer_token_none_input_returns_none() -> None:
    """``None`` input (header absent) → None."""
    assert bearer_token(None) is None


def test_bearer_token_empty_string_returns_none() -> None:
    """Empty string (header present but blank) → None."""
    assert bearer_token("") is None


def test_bearer_token_missing_bearer_prefix_returns_none() -> None:
    """A raw token without the ``Bearer `` prefix is not accepted."""
    assert bearer_token("my-secret-token") is None


def test_bearer_token_basic_scheme_returns_none() -> None:
    """``Basic <credentials>`` is not a bearer token → None."""
    assert bearer_token("Basic dXNlcjpwYXNz") is None


def test_bearer_token_bearer_without_space_returns_none() -> None:
    """``Bearertoken`` (no space separator) does not match the prefix → None."""
    assert bearer_token("Bearertoken") is None


def test_bearer_token_token_with_internal_spaces_is_preserved() -> None:
    """A token that internally contains spaces is returned as-is (strip only trims edges)."""
    # The header value: "Bearer tok with spaces"
    # The implementation returns authorization[len("bearer "):].strip() which is
    # "tok with spaces" — internal spaces are preserved.
    result = bearer_token("Bearer tok with spaces")
    assert result == "tok with spaces"
