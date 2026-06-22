"""Download-token primitive — HMAC-SHA256 capability tokens (CR-00098, S02).

Citations in a grounded answer are inert text today; a clickable
``download_url`` needs to authorize the file download without dragging the
caller's bearer token through a plain browser navigation.  The chainlit
frontend authenticates with a bearer header, so an unauthenticated browser
GET would 401.

This module provides the signed-URL primitive that bridges that gap.  A
short-lived, HMAC-SHA256 capability token is bound to a
``(document_id, principal_id, exp[, session_id])`` tuple and minted per
citation.  When the browser navigates to ``/documents/{id}/raw?token=…``,
S04 verifies the token, re-resolves the principal's current groups from
the auth source, and re-runs the ACL overlap check — the token is a
capability, not an ACL assertion (defense in depth, CR-00098 design
§"Defense in depth").

Security properties (all tested in ``tests/unit/test_download_tokens.py``):

- **Binding**: the token's payload carries an explicit ``document_id`` and
  ``principal_id``; the S04 download path checks the URL ``{id}`` matches
  ``document_id`` before serving bytes.
- **Expiry**: tokens carry a hard ``exp`` epoch; ``verify`` rejects when
  ``now >= exp`` (the boundary itself is reject — there is no grace).
- **Constant-time compare**: signature verification uses
  :func:`hmac.compare_digest` (CWE-208 — timing-attack safe).
- **Fail-safe on missing secret**: ``verify`` rejects every token when the
  secret is empty/blank (treats the feature as off); ``mint`` raises so
  the caller can surface a plain URL or omit it.
- **No raise on attacker-controlled input**: ``verify`` returns ``None``
  for any malformed, tampered, expired, or unsigned input.

Stdlib only: no new dependency, no hosted call (honors the local-only
rule).  Token form: ``<payload_b64url>.<sig_b64url>`` where ``sig`` is
HMAC-SHA256 over the raw (decoded) payload bytes, base64url-encoded
without padding.
"""

from __future__ import annotations

import base64
import dataclasses
import hmac
import json
import time
from hashlib import sha256
from typing import Final

# JSON-compact key aliases — short, single-letter to keep the on-wire
# payload tiny.  Documented in the module docstring above; the S04
# download path reuses the same aliases when decoding.
_KEY_D: Final = "d"
_KEY_P: Final = "p"
_KEY_EXP: Final = "exp"
_KEY_S: Final = "s"


@dataclasses.dataclass(frozen=True, slots=True)
class VerifiedDownloadToken:
    """A capability token whose signature, format, and expiry all check out.

    Returned by :func:`verify` on success.  Frozen so downstream code
    cannot widen entitlements by mutating ``principal_id`` or
    ``session_id`` post-hoc.  ``session_id`` is ``None`` for tokens minted
    for the regular corpus path (it is only set for chat-attachment
    downloads, where it is later re-validated by the S04
    ``_scope_to_session`` check).

    Attributes:
        document_id: The ``documents.id`` the token authorizes a download
            for.  The S04 path checks the URL ``{id}`` matches this
            before serving bytes.
        principal_id: The authenticated principal the token is bound to.
            S04 re-resolves the principal's current groups from this id
            (the token is never the source of groups).
        session_id: The validated chat session id when the token was
            minted for a session-scoped attachment; ``None`` otherwise.
    """

    document_id: str
    principal_id: str
    session_id: str | None


def _b64url_encode(raw: bytes) -> str:
    """Base64url-encode ``raw`` with no ``=`` padding (RFC 7515 §2 style)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Base64url-decode ``s``; tolerates missing padding.

    Args:
        s: A base64url string (with or without padding).

    Returns:
        The decoded raw bytes.

    Raises:
        ValueError: If the input is not valid base64url (caller maps to None).
    """
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode((s + pad).encode("ascii"))


def _sign(payload_bytes: bytes, secret: str) -> bytes:
    """Return the HMAC-SHA256 tag of ``payload_bytes`` keyed by ``secret``."""
    return hmac.new(secret.encode("utf-8"), payload_bytes, sha256).digest()


def mint(
    *,
    document_id: str,
    principal_id: str,
    secret: str,
    ttl_seconds: int,
    session_id: str | None = None,
    now: float | None = None,
) -> str:
    """Mint a short-lived signed capability token for a document download.

    Produces the token form ``<payload_b64url>.<sig_b64url>`` where the
    payload is a compact JSON object ``{d, p, exp[, s]}`` and ``sig`` is
    HMAC-SHA256 over the decoded payload bytes keyed by ``secret``.

    Security:
        The token is a **capability** bound to ``principal_id`` — it is
        not an ACL assertion.  The S04 download path always re-resolves
        the principal's current groups from auth and re-runs the ACL
        overlap check.  A leaked token without the corresponding
        principal id resolving to overlapping groups is therefore
        harmless.

    Args:
        document_id: The ``documents.id`` the token authorizes.
        principal_id: The authenticated principal the token is bound to.
        secret: Server secret used to compute the HMAC.  MUST be
            non-empty; ``ValueError`` is raised otherwise so the caller
            can fail loud (a missing secret must never produce a working
            token).
        ttl_seconds: Lifetime in seconds.  Short values are the point —
            the default in :class:`~iw_rag.settings.ApiSettings` is
            300 s; longer TTLs widen the exposure window.
        session_id: Optional chat session id (set for session-scoped
            attachments; ``None`` for the regular corpus path).
        now: Optional injection point for the current epoch in seconds.
            Tests pass a fixed ``now`` for determinism; production
            callers omit it and ``time.time()`` is used.

    Returns:
        The signed token string ``<payload_b64url>.<sig_b64url>``.

    Raises:
        ValueError: If ``secret`` is empty/blank, ``ttl_seconds`` is
            non-positive, or any of ``document_id``/``principal_id`` are
            empty.  These are programmer errors and surface at the call
            site rather than producing a working but mis-issued token.
    """
    if not secret:
        raise ValueError(
            "download_tokens.mint: secret is empty; refusing to mint a working token. "
            "Configure IW_RAG__API__DOWNLOAD_TOKEN_SECRET (env-only)."
        )
    if not document_id:
        raise ValueError("download_tokens.mint: document_id must be a non-empty string.")
    if not principal_id:
        raise ValueError("download_tokens.mint: principal_id must be a non-empty string.")
    if ttl_seconds <= 0:
        raise ValueError("download_tokens.mint: ttl_seconds must be positive.")

    exp = int(now if now is not None else time.time()) + int(ttl_seconds)
    payload: dict[str, str | int] = {
        _KEY_D: document_id,
        _KEY_P: principal_id,
        _KEY_EXP: exp,
    }
    if session_id is not None:
        payload[_KEY_S] = session_id

    # Compact JSON — no spaces, sorted keys for stable HMAC bytes.
    payload_bytes = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    sig = _sign(payload_bytes, secret)
    return f"{_b64url_encode(payload_bytes)}.{_b64url_encode(sig)}"


def verify(token: str, *, secret: str, now: float | None = None) -> VerifiedDownloadToken | None:
    """Verify a signed download token and return its bound fields.

    The check order is:

    1. Structural sanity (one ``.`` separator, both halves base64url
       decodable, payload JSON-parseable into the documented shape).
    2. Constant-time HMAC comparison against the recomputed tag.
    3. Expiry (``now < exp``).

    Any failure returns ``None`` — this function MUST NOT raise on
    attacker-controlled input (it is on the unauthenticated browser path
    and an exception would leak stack traces / information).

    Security:
        The signature is compared with :func:`hmac.compare_digest`
        (constant-time) so a token forgery cannot be accelerated by a
        timing side-channel (CWE-208).  An empty/blank ``secret`` is the
        fail-safe: every token is rejected, so a misconfigured deployment
        never accidentally serves downloads via the token path.

    Args:
        token: The token string from the URL ``?token=`` query
            parameter.  Treated as attacker-controlled.
        secret: Server secret to verify the HMAC against.  An empty
            value is treated as "feature disabled" — every token is
            rejected.
        now: Optional injection point for the current epoch in seconds.
            Tests pass a fixed ``now``; production callers omit it and
            ``time.time()`` is used.

    Returns:
        A :class:`VerifiedDownloadToken` on success; ``None`` on any
        failure (malformed, tampered, expired, wrong secret, or
        empty/blank secret).
    """
    if not token or not secret:
        return None

    parts = token.split(".")
    if len(parts) != 2:
        return None
    payload_b64, sig_b64 = parts
    if not payload_b64 or not sig_b64:
        return None

    try:
        payload_bytes = _b64url_decode(payload_b64)
        sig_bytes = _b64url_decode(sig_b64)
    except (ValueError, TypeError):
        return None

    expected_sig = _sign(payload_bytes, secret)
    if not hmac.compare_digest(sig_bytes, expected_sig):
        return None

    try:
        payload = json.loads(payload_bytes)
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(payload, dict):
        return None

    document_id = payload.get(_KEY_D)
    principal_id = payload.get(_KEY_P)
    exp = payload.get(_KEY_EXP)
    session_id = payload.get(_KEY_S)

    if not isinstance(document_id, str) or not document_id:
        return None
    if not isinstance(principal_id, str) or not principal_id:
        return None
    if not isinstance(exp, int):
        return None

    current = now if now is not None else time.time()
    if current >= exp:
        return None

    if session_id is not None and not isinstance(session_id, str):
        return None

    return VerifiedDownloadToken(
        document_id=document_id,
        principal_id=principal_id,
        session_id=session_id,
    )
