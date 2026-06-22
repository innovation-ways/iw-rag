"""Demo login → bearer token (which establishes principal_groups server-side).

The username/password → token map is configured (never inferred). For the local
demo it comes from ``IW_RAG_DEMO_USERS`` (JSON: ``{username: {password, token}}``);
the same tokens must appear in the API's ``IW_RAG__API__STATIC_TOKENS`` so the
server can resolve their groups. Login does NOT carry groups — only a token.
"""

from __future__ import annotations

import json
import os

_DEFAULT_USERS = {
    "analyst": {"password": "analyst", "token": "demo-finance"},
    "hr": {"password": "hr", "token": "demo-hr"},
}


def _users() -> dict[str, dict[str, str]]:
    raw = os.environ.get("IW_RAG_DEMO_USERS")
    if not raw:
        return _DEFAULT_USERS
    try:
        parsed: dict[str, dict[str, str]] = json.loads(raw)
        return parsed
    except (ValueError, TypeError):
        return _DEFAULT_USERS


def token_for(username: str, password: str) -> str | None:
    """Return the bearer token for valid credentials, else None."""
    user = _users().get(username)
    if user is None or user.get("password") != password:
        return None
    return user.get("token")


def token_for_username(username: str) -> str | None:
    """Return the bearer token for an already-authenticated username (no password).

    Used by the ``/iw/session-token`` SSO endpoint: the Chainlit auth cookie has
    already proven the user's identity, so we map their username to the same token
    the chat uses — letting the Document Manager reuse the session without a second
    login.
    """
    user = _users().get(username)
    return user.get("token") if user else None
