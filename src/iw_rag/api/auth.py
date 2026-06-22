"""Authentication → AccessContext (the ACL source of truth).

``principal_groups`` derive ONLY from verified auth, never from the request body
or query text. Phase 0 uses ``static_token`` (a configured token→groups map);
OIDC is the production-shaped swap. The authenticator is injectable so the API
and tests share one contract.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from iw_rag.domain.acl import AccessContext

# Synthetic ACL groups every authenticated principal implicitly carries
# (WI-DOCMGMT): ``public`` (entitles ``common`` documents, visible to all) and
# ``user:<principal_id>`` (entitles the principal's own ``specific`` documents).
PUBLIC_GROUP = "public"


def user_group(principal_id: str) -> str:
    """The per-user ACL group tag for a principal (``user:<principal_id>``)."""
    return f"user:{principal_id}"


def augment_groups(principal_id: str, base_groups: tuple[str, ...]) -> tuple[str, ...]:
    """Append the implicit ``public`` + ``user:<id>`` groups (deduped, order-stable).

    Args:
        principal_id: The authenticated principal id (the document owner key).
        base_groups: The principal's configured entitlement groups.

    Returns:
        ``base_groups`` followed by ``public`` and ``user:<principal_id>`` with
        duplicates removed, preserving first-seen order.
    """
    seen: dict[str, None] = {}
    for g in (*base_groups, PUBLIC_GROUP, user_group(principal_id)):
        seen.setdefault(g, None)
    return tuple(seen)


@runtime_checkable
class Authenticator(Protocol):
    """Resolves a bearer token to an AccessContext (or None if unauthenticated)."""

    def authenticate(self, token: str | None, request_id: str = "") -> AccessContext | None:
        """Return the AccessContext for ``token``, or None to reject."""
        ...

    def resolve_principal(self, principal_id: str) -> AccessContext | None:
        """Return the current AccessContext for ``principal_id`` (or None).

        Used by the S04 (CR-00098) download-by-token path to re-resolve
        the principal's **current** groups from the auth source so a
        revoked group takes effect immediately.  ``principal_groups``
        flow only from auth, never from the token's payload (the token
        is a capability, never an ACL assertion).

        The Protocol default is ``None``-returning so an authenticator
        that has no ``by_principal`` index (e.g. one without a stable
        principal id) silently no-ops the token path — the bearer
        header path is unaffected.
        """
        ...


@dataclass(frozen=True, slots=True)
class StaticTokenAuthenticator:
    """Maps configured tokens to groups (the local-demo auth mode)."""

    token_groups: dict[str, tuple[str, ...]]
    operator_tokens: frozenset[str]

    @classmethod
    def from_config(
        cls, static_tokens: dict[str, list[str]], operator_tokens: list[str]
    ) -> StaticTokenAuthenticator:
        """Build from settings (token→groups map + operator token list)."""
        return cls(
            token_groups={tok: tuple(groups) for tok, groups in static_tokens.items()},
            operator_tokens=frozenset(operator_tokens),
        )

    def authenticate(self, token: str | None, request_id: str = "") -> AccessContext | None:
        """Resolve a token to an AccessContext, or None if unknown/missing."""
        if token is None or token not in self.token_groups:
            return None
        return AccessContext(
            principal_id=token[:8],
            principal_groups=self.token_groups[token],
            is_operator=token in self.operator_tokens,
            request_id=request_id,
        )

    def resolve_principal(self, principal_id: str) -> AccessContext | None:
        """Return None — the legacy static-token auth is not reversible.

        The token-download path (CR-00098 S04) is unavailable under
        legacy static-token auth: ``principal_id`` is derived from the
        bearer token's first 8 bytes (``token[:8]``), which is a one-way
        derivation, so no ``principal_id → AccessContext`` index exists.
        Silently returning ``None`` lets S04's token path no-op; the
        bearer header path is unaffected.
        """
        return None


@dataclass(frozen=True, slots=True)
class _PrincipalRecord:
    """An internal demo-auth record (resolved from a token or a login)."""

    principal_id: str
    base_groups: tuple[str, ...]
    is_operator: bool


@runtime_checkable
class _DemoUserLike(Protocol):
    """Structural shape of a demo-user config (settings.DemoUserCfg)."""

    password: str
    token: str
    groups: list[str]
    operator: bool


@dataclass(frozen=True, slots=True)
class DemoUserAuthenticator:
    """Login-capable demo auth with stable per-user identity (WI-DOCMGMT).

    Unlike :class:`StaticTokenAuthenticator` (token→groups, principal_id derived
    from the token bytes), this authenticator resolves a token to a stable
    ``principal_id`` (the username) and always augments the principal's groups
    with the implicit ``public`` + ``user:<username>`` tags.  That stable id is
    the document-ownership key, and the synthetic groups make the per-user ACL
    (``specific`` → ``user:<owner>``, ``common`` → ``public``) work end to end.

    It additionally exposes :meth:`login` so a username/password can be exchanged
    for a bearer token (the document-management page + the frontend use it).
    """

    by_token: dict[str, _PrincipalRecord]
    by_principal: dict[str, _PrincipalRecord]
    credentials: dict[str, tuple[str, str]]  # username -> (password, token)

    @classmethod
    def from_demo_users(cls, demo_users: dict[str, _DemoUserLike]) -> DemoUserAuthenticator:
        """Build from the ``api.demo_users`` settings map (username → DemoUserCfg).

        Args:
            demo_users: Mapping of username to a ``DemoUserCfg``-shaped object
                (``.password``, ``.token``, ``.groups``, ``.operator``).

        Returns:
            A configured :class:`DemoUserAuthenticator`.
        """
        by_token: dict[str, _PrincipalRecord] = {}
        by_principal: dict[str, _PrincipalRecord] = {}
        credentials: dict[str, tuple[str, str]] = {}
        for username, cfg in demo_users.items():
            rec = _PrincipalRecord(username, tuple(cfg.groups), cfg.operator)
            by_token[cfg.token] = rec
            by_principal[username] = rec
            credentials[username] = (cfg.password, cfg.token)
        return cls(by_token=by_token, by_principal=by_principal, credentials=credentials)

    def authenticate(self, token: str | None, request_id: str = "") -> AccessContext | None:
        """Resolve a token to an AccessContext, or None if unknown/missing."""
        if token is None or token not in self.by_token:
            return None
        rec = self.by_token[token]
        return AccessContext(
            principal_id=rec.principal_id,
            principal_groups=augment_groups(rec.principal_id, rec.base_groups),
            is_operator=rec.is_operator,
            request_id=request_id,
        )

    def resolve_principal(self, principal_id: str) -> AccessContext | None:
        """Re-resolve a principal's **current** groups from the auth source.

        Used by the S04 (CR-00098) download-by-token path: a verified
        token carries a ``principal_id`` and the server re-resolves the
        principal's groups from this index — never from the token
        payload — so a revoked group takes effect immediately.  Groups
        are augmented the same way :meth:`authenticate` does, so the
        per-user ACL (``specific`` → ``user:<owner>``, ``common`` →
        ``public``) keeps working for download authorization.

        Args:
            principal_id: The principal id carried in the verified
                token's payload.

        Returns:
            The principal's current :class:`AccessContext` (with
            ``public`` and ``user:<id>`` groups), or ``None`` for an
            unknown / empty principal id.
        """
        if not principal_id:
            return None
        rec = self.by_principal.get(principal_id)
        if rec is None:
            return None
        return AccessContext(
            principal_id=rec.principal_id,
            principal_groups=augment_groups(rec.principal_id, rec.base_groups),
            is_operator=rec.is_operator,
        )

    def login(self, username: str, password: str) -> str | None:
        """Validate credentials and return the bearer token, or None on failure.

        Uses a length-stable comparison guard to avoid trivially leaking which
        of username/password was wrong.

        Args:
            username: The submitted username.
            password: The submitted password.

        Returns:
            The user's bearer token on success, else ``None``.
        """
        cred = self.credentials.get(username)
        if cred is None:
            return None
        expected_password, token = cred
        # Constant-time comparison so a wrong password cannot be distinguished by
        # response timing (matches the docstring's length-stable-guard contract).
        if not hmac.compare_digest(password.encode("utf-8"), expected_password.encode("utf-8")):
            return None
        return token


def bearer_token(authorization: str | None) -> str | None:
    """Extract the bearer token from an ``Authorization`` header value."""
    if not authorization:
        return None
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix) :].strip()
    return None
