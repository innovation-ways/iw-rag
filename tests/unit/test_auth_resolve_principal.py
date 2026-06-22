"""Unit tests for ``Authenticator.resolve_principal`` (CR-00098, S02).

The S04 download-by-token path needs to recover a principal's **current**
groups from a ``principal_id`` carried in the signed download token.  This
is what re-resolves entitlements at download time so a revoked group takes
effect immediately (defense in depth — the token is just a capability, the
ACL re-check is authoritative).

The keystone is preserved: ``principal_groups`` flow only from the auth
source, never from the request body or the token's payload.  These tests
pin:

- ``DemoUserAuthenticator.resolve_principal`` returns the current groups for
  a known principal (with the same ``public`` / ``user:<id>`` augmentation
  ``authenticate`` applies), and ``None`` for an unknown one
- ``StaticTokenAuthenticator.resolve_principal`` always returns ``None``
  (the legacy static-token auth derives ``principal_id`` from token bytes
  and is not reversible)
- The default ``Authenticator`` Protocol-level ``resolve_principal`` is
  ``None``-returning (it must never be invoked accidentally on an
  authenticator that has no index by principal)
"""

from __future__ import annotations

from iw_rag.api.auth import (
    DemoUserAuthenticator,
    StaticTokenAuthenticator,
    augment_groups,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDemoCfg:
    """Structural stand-in for ``settings.DemoUserCfg``."""

    def __init__(self, password: str, token: str, groups: list[str], operator: bool) -> None:
        self.password = password
        self.token = token
        self.groups = groups
        self.operator = operator


def _demo_auth() -> DemoUserAuthenticator:
    """A two-user demo authenticator (alice in finance, bob in hr)."""
    cfg = {
        "alice": _FakeDemoCfg(password="p1", token="tok-alice", groups=["finance"], operator=False),
        "bob": _FakeDemoCfg(password="p2", token="tok-bob", groups=["hr"], operator=True),
    }
    return DemoUserAuthenticator.from_demo_users(cfg)


# ---------------------------------------------------------------------------
# 1. DemoUserAuthenticator — known principal returns current groups
# ---------------------------------------------------------------------------


def test_demo_resolve_principal_known_returns_current_groups() -> None:
    """``resolve_principal('alice')`` returns alice's groups, augmented."""
    auth = _demo_auth()
    ctx = auth.resolve_principal("alice")
    assert ctx is not None
    # Same augmentation as authenticate: base + 'public' + 'user:<id>'.
    assert ctx.principal_id == "alice"
    assert ctx.principal_groups == augment_groups("alice", ("finance",))
    assert "finance" in ctx.principal_groups
    assert "public" in ctx.principal_groups
    assert "user:alice" in ctx.principal_groups


def test_demo_resolve_principal_preserves_operator_flag() -> None:
    """The is_operator flag follows the configured value, not a default."""
    auth = _demo_auth()
    alice_ctx = auth.resolve_principal("alice")
    bob_ctx = auth.resolve_principal("bob")
    assert alice_ctx is not None
    assert bob_ctx is not None
    assert alice_ctx.is_operator is False
    assert bob_ctx.is_operator is True


def test_demo_resolve_principal_unknown_returns_none() -> None:
    """An unknown principal id resolves to None — never raise, never invent."""
    auth = _demo_auth()
    assert auth.resolve_principal("eve") is None


def test_demo_resolve_principal_empty_returns_none() -> None:
    """An empty string is not a valid principal id — returns None."""
    auth = _demo_auth()
    assert auth.resolve_principal("") is None


# ---------------------------------------------------------------------------
# 2. StaticTokenAuthenticator — never reversible, always None
# ---------------------------------------------------------------------------


def test_static_token_resolve_principal_always_none() -> None:
    """Static-token auth derives principal_id from token bytes → not reversible.

    The token-download path is unavailable under legacy static-token auth;
    the bearer path is unaffected.  ``resolve_principal`` must return None
    for every input so S04's token path silently no-ops.
    """
    auth = StaticTokenAuthenticator.from_config(
        static_tokens={"fin-token": ["finance"], "op-token": ["all-staff"]},
        operator_tokens=["op-token"],
    )
    assert auth.resolve_principal("anyone") is None
    assert auth.resolve_principal("fin-token") is None
    assert auth.resolve_principal("") is None
    assert auth.resolve_principal("00000000") is None  # looks like a token[:8]
