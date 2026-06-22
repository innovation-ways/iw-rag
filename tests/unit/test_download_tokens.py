"""Unit tests for the download-token primitive (CR-00098, S02).

The signed-token primitive (``iw_rag.api.download_tokens``) mints and verifies
short-lived HMAC-SHA256 capability tokens bound to ``(document_id,
principal_id, exp, [session_id])``.  These tests pin the security contract:

- mint→verify round-trip recovers the bound fields
- tampered payload / signature is rejected
- expired tokens (``now >= exp``) are rejected
- the wrong secret (or empty) is rejected
- ``hmac.compare_digest`` is the comparison primitive (constant-time)
- the bare structural shape of the token is ``<payload_b64>.<sig_b64>``
- both halves are base64url with no padding

The S04 download path depends on these guarantees; the keystone is that
``verify`` never raises on attacker-controlled input — it returns ``None``.
"""

from __future__ import annotations

import base64
import hmac
from typing import Final

import pytest

from iw_rag.api.download_tokens import (
    VerifiedDownloadToken,
    mint,
    verify,
)

_SECRET: Final = "test-secret-must-be-long-enough-for-hmac-sha256"
_DOC_ID: Final = "doc-abc-123"
_PRINCIPAL_ID: Final = "alice"
_SESSION_ID: Final = "sess-xyz"
_NOW: Final = 1_700_000_000


def _b64url_decode(s: str) -> bytes:
    """Decode a base64url string (with optional missing padding) to bytes."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# ---------------------------------------------------------------------------
# 1. mint→verify round-trip — every bound field is recovered
# ---------------------------------------------------------------------------


def test_mint_verify_round_trip_recovers_all_fields() -> None:
    """A freshly minted token verifies and yields the bound fields verbatim."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        session_id=_SESSION_ID,
        now=_NOW,
    )
    verified = verify(token, secret=_SECRET, now=_NOW + 10)
    assert verified is not None
    assert verified.document_id == _DOC_ID
    assert verified.principal_id == _PRINCIPAL_ID
    assert verified.session_id == _SESSION_ID


def test_mint_verify_round_trip_without_session_id() -> None:
    """session_id is optional; a None round-trips back as None."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    verified = verify(token, secret=_SECRET, now=_NOW + 10)
    assert verified is not None
    assert verified.session_id is None
    assert verified.document_id == _DOC_ID
    assert verified.principal_id == _PRINCIPAL_ID


# ---------------------------------------------------------------------------
# 2. Token structure — `<payload_b64>.<sig_b64>`, base64url, no padding
# ---------------------------------------------------------------------------


def test_token_form_is_payload_dot_signature() -> None:
    """The token is two base64url segments joined by a single '.' separator."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    parts = token.split(".")
    assert len(parts) == 2
    payload_b64, sig_b64 = parts
    # Both halves must be valid base64url (decodable, no padding required).
    payload_bytes = _b64url_decode(payload_b64)
    sig_bytes = _b64url_decode(sig_b64)
    # HMAC-SHA256 produces a 32-byte tag.
    assert len(sig_bytes) == 32
    # The encoded payload must be valid JSON containing the bound fields.
    import json

    payload = json.loads(payload_bytes)
    assert payload["d"] == _DOC_ID
    assert payload["p"] == _PRINCIPAL_ID
    assert payload["exp"] == _NOW + 300
    # No padding characters in either segment.
    assert "=" not in payload_b64
    assert "=" not in sig_b64


# ---------------------------------------------------------------------------
# 3. Tampered payload is rejected
# ---------------------------------------------------------------------------


def test_verify_rejects_tampered_payload() -> None:
    """Flipping a byte in the payload base64 fails verification (sig mismatch)."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, sig_b64 = token.split(".")
    # Tamper with the payload: substitute a forged document_id; the signature
    # still covers the original bytes → mismatch on verify.
    tampered_payload_bytes = b'{"d":"doc-OTHER","p":"alice","exp":1700000300}'
    tampered_payload_b64 = base64.urlsafe_b64encode(tampered_payload_bytes).rstrip(b"=").decode()
    tampered = f"{tampered_payload_b64}.{sig_b64}"
    assert verify(tampered, secret=_SECRET, now=_NOW + 1) is None


def test_verify_rejects_tampered_signature() -> None:
    """Modifying the signature segment breaks verification."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, sig_b64 = token.split(".")
    # Flip the first char of the sig segment.
    flipped = "A" if sig_b64[0] != "A" else "B"
    tampered = f"{payload_b64}.{flipped}{sig_b64[1:]}"
    assert verify(tampered, secret=_SECRET, now=_NOW + 1) is None


# ---------------------------------------------------------------------------
# 4. Expired token — `now >= exp` is rejected
# ---------------------------------------------------------------------------


def test_verify_rejects_expired_token() -> None:
    """A token whose exp <= the caller's `now` is rejected."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=60,
        now=_NOW,
    )
    # Exactly at the expiry boundary → must reject (now >= exp).
    assert verify(token, secret=_SECRET, now=_NOW + 60) is None
    # Well past expiry → must reject.
    assert verify(token, secret=_SECRET, now=_NOW + 120) is None


def test_verify_accepts_token_within_ttl() -> None:
    """A token whose exp > caller's `now` is accepted (boundary: now < exp)."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    # One second before expiry → still valid; must recover exact bound fields.
    verified = verify(token, secret=_SECRET, now=_NOW + 299)
    assert verified is not None
    assert verified.document_id == _DOC_ID
    assert verified.principal_id == _PRINCIPAL_ID
    assert isinstance(verified, VerifiedDownloadToken)


# ---------------------------------------------------------------------------
# 5. Wrong secret (incl. empty) is rejected
# ---------------------------------------------------------------------------


def test_verify_rejects_wrong_secret() -> None:
    """A token signed with one secret must not verify under a different secret."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    assert verify(token, secret="a-totally-different-secret", now=_NOW + 1) is None


def test_verify_rejects_empty_secret() -> None:
    """An empty secret never accepts a token (fail-safe; reject all)."""
    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    assert verify(token, secret="", now=_NOW + 1) is None


def test_mint_with_empty_secret_raises() -> None:
    """Minting with an empty/blank secret raises — caller must provide a real secret."""
    with pytest.raises(ValueError, match="secret is empty"):
        mint(
            document_id=_DOC_ID,
            principal_id=_PRINCIPAL_ID,
            secret="",
            ttl_seconds=300,
            now=_NOW,
        )


# ---------------------------------------------------------------------------
# 6. Malformed tokens — never raise, always return None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_token",
    [
        "",  # empty
        "no-dot-here",  # missing separator
        ".",  # empty halves
        "abc.",  # empty sig
        ".xyz",  # empty payload
        "a.b.c",  # too many segments
        "!!.!!",  # non-base64url chars
    ],
)
def test_verify_returns_none_for_malformed_tokens(bad_token: str) -> None:
    """``verify`` never raises on attacker-controlled input — it returns None."""
    assert verify(bad_token, secret=_SECRET, now=_NOW) is None


# ---------------------------------------------------------------------------
# 7. Constant-time comparison — uses hmac.compare_digest
# ---------------------------------------------------------------------------


def test_signature_comparison_uses_hmac_compare_digest() -> None:
    """The HMAC verification path delegates to ``hmac.compare_digest``.

    We assert by patching the module-local ``hmac.compare_digest`` to a
    sentinel; if ``verify`` does not call it the sentinel is never hit and
    we fail.  This is the structural guard that the constant-time compare
    primitive is the comparison (CWE-208 — timing-attack safe).
    """
    import iw_rag.api.download_tokens as mod

    called: list[tuple[bytes, bytes]] = []
    real = hmac.compare_digest

    def spy(a: bytes, b: bytes) -> bool:  # type: ignore[override]
        called.append((bytes(a), bytes(b)))
        return real(a, b)

    token = mint(
        document_id=_DOC_ID,
        principal_id=_PRINCIPAL_ID,
        secret=_SECRET,
        ttl_seconds=300,
        now=_NOW,
    )
    try:
        mod.hmac.compare_digest = spy  # type: ignore[assignment]
        verify(token, secret=_SECRET, now=_NOW + 1)
    finally:
        mod.hmac.compare_digest = real  # type: ignore[assignment]
    assert len(called) == 1, "verify() must call hmac.compare_digest exactly once per verification"
    sig_bytes_arg, expected_sig_arg = called[0]
    # HMAC-SHA256 produces a 32-byte tag; both arguments must be 32-byte digests.
    assert len(sig_bytes_arg) == 32
    assert len(expected_sig_arg) == 32


# ---------------------------------------------------------------------------
# 8. VerifiedDownloadToken is a frozen dataclass with the documented shape
# ---------------------------------------------------------------------------


def test_verified_download_token_is_frozen() -> None:
    """The result dataclass is immutable — assignment raises FrozenInstanceError."""
    import dataclasses

    v = VerifiedDownloadToken(document_id="d", principal_id="p", session_id=None)
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        v.document_id = "other"  # type: ignore[misc]
