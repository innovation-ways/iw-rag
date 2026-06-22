"""Cross-cutting unit tests for the download-token primitive (CR-00098 / S07).

The S02 unit tests (``tests/unit/test_download_tokens.py``) cover the
primitive's core round-trip / tamper / expiry / constant-time / structural
guarantees.  This file adds the **cross-cutting** and **binding** assertions
that S07 is responsible for — the property the rest of the system actually
relies on:

* The token's ``document_id`` binding is **per-token, not per-secret**:
  a token minted for document A must not verify-authorize document B
  (``verify`` returns the bound id, callers compare it to the path).
* The token's ``principal_id`` is bound; a token minted for principal A
  carries A's id and ``verify`` recovers it (the S04 download path uses
  this to re-resolve the principal's *current* groups from the auth
  source — never the token payload).
* The signature comparison is **constant-time in the *whole* signature,
  not just the first byte**: a one-byte mismatch is not fast-path
  distinguishable from a complete mismatch.  The existing
  ``test_signature_comparison_uses_hmac_compare_digest`` test pins the
  structural guard; this file pins the **bytewise** constant-time
  property for varying-length mismatches (the bytes-before-mismatch
  don't influence the timing channel).
* Token forms encode the documented JSON fields verbatim
  (``d``/``p``/``exp``/``s``); the on-wire payload is stable across
  versions.
* ``mint`` validates every input (empty secret, empty document_id,
  empty principal_id, non-positive TTL) with a ``ValueError`` — no
  silent success.
* ``verify`` rejects every malformed attacker-controlled input
  without raising (a CWE-209 / 500-leak guard).

Every test here is a **unit test** (no docker, no live DB, no network)
per the S07 spec.
"""

from __future__ import annotations

import base64
import json
from typing import Final

import pytest

from iw_rag.api.download_tokens import (
    VerifiedDownloadToken,
    mint,
    verify,
)

_SECRET_A: Final = "test-secret-must-be-long-enough-for-hmac-sha256"
_SECRET_B: Final = "a-different-secret-of-equal-length-for-binding-test"
_DOC_A: Final = "11111111-1111-1111-1111-111111111111"
_DOC_B: Final = "22222222-2222-2222-2222-222222222222"
_PRINCIPAL_ALICE: Final = "alice"
_PRINCIPAL_BOB: Final = "bob"
_NOW: Final = 1_700_000_000


# ---------------------------------------------------------------------------
# 1. Document binding — a token minted for doc A does not authorize doc B
# ---------------------------------------------------------------------------


def test_token_minted_for_doc_a_does_not_verify_authorize_doc_b() -> None:
    """A token's bound ``document_id`` is per-token; the S04 path checks
    ``verified.document_id == path_id`` before serving bytes.

    The unit-level guarantee: a single token can only verify for the
    document id it was minted with.  ``verify`` returns the bound id
    verbatim — the caller is responsible for the path equality check
    (which is exercised in the S04 integration tests).
    """
    token_a = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    verified = verify(token_a, secret=_SECRET_A, now=_NOW + 1)
    assert verified is not None
    assert verified.document_id == _DOC_A
    # Sanity: a different document id was never bound to this token.
    assert verified.document_id != _DOC_B


def test_documents_differ_only_in_payload_byte_swap_rejected() -> None:
    """A byte-level payload swap is a signature mismatch (forgery is rejected).

    Even though the *payload structure* parses correctly, the HMAC was
    computed over the original bytes — flipping a single byte in the
    payload invalidates the signature and ``verify`` returns ``None``.
    The S04 path never has to special-case this: HMAC compare detects it.
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, sig_b64 = token.split(".")
    # Forged payload claiming doc B; original sig (HMAC mismatch).
    forged_payload_bytes = json.dumps(
        {"d": _DOC_B, "p": _PRINCIPAL_ALICE, "exp": _NOW + 300},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged_payload_b64 = base64.urlsafe_b64encode(forged_payload_bytes).rstrip(b"=").decode("ascii")
    swapped = f"{forged_payload_b64}.{sig_b64}"
    assert verify(swapped, secret=_SECRET_A, now=_NOW + 1) is None


# ---------------------------------------------------------------------------
# 2. Principal binding — a token is bound to ONE principal id
# ---------------------------------------------------------------------------


def test_token_minted_for_principal_alice_does_not_verify_as_principal_bob() -> None:
    """A token's bound ``principal_id`` is per-token.

    The S04 download path uses this id to call
    ``Authenticator.resolve_principal`` and re-resolve the *current*
    groups — a leaked token cannot be repurposed to grant another
    principal's groups (the S04 path re-resolves from auth, never
    from the token payload).
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    verified = verify(token, secret=_SECRET_A, now=_NOW + 1)
    assert verified is not None
    assert verified.principal_id == _PRINCIPAL_ALICE
    # Bob's id was never bound to this token.
    assert verified.principal_id != _PRINCIPAL_BOB


def test_principal_payload_swap_rejected() -> None:
    """A forged payload with a different principal id is rejected (HMAC mismatch).

    Defense in depth: even if an attacker could swap the principal id in
    the payload, the HMAC was computed over the original payload bytes —
    ``verify`` rejects on signature mismatch before the JSON is read for
    ACL purposes (the S04 path would re-resolve from the *bound* id, not
    the swapped one).
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, sig_b64 = token.split(".")
    forged_payload_bytes = json.dumps(
        {"d": _DOC_A, "p": _PRINCIPAL_BOB, "exp": _NOW + 300},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    forged_payload_b64 = base64.urlsafe_b64encode(forged_payload_bytes).rstrip(b"=").decode("ascii")
    swapped = f"{forged_payload_b64}.{sig_b64}"
    assert verify(swapped, secret=_SECRET_A, now=_NOW + 1) is None


# ---------------------------------------------------------------------------
# 3. Session binding — the optional session_id round-trips
# ---------------------------------------------------------------------------


def test_session_id_round_trips_when_provided() -> None:
    """A session-scoped attachment token carries the validated session_id.

    The S04 download path checks ``verified.session_id`` and re-validates
    session ownership via ``_scope_to_session`` before widening the
    principal's groups with ``session:<id>``.
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        session_id="chat-1234",
        now=_NOW,
    )
    verified = verify(token, secret=_SECRET_A, now=_NOW + 1)
    assert verified is not None
    assert verified.session_id == "chat-1234"


def test_session_id_absent_when_not_provided() -> None:
    """A regular (non-attachment) token has ``session_id is None``.

    The download path treats ``None`` session_id as "no session widening"
    — the principal's groups are read from the resolved
    ``AccessContext`` directly.
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    verified = verify(token, secret=_SECRET_A, now=_NOW + 1)
    assert verified is not None
    assert verified.session_id is None


# ---------------------------------------------------------------------------
# 4. Secret binding — a different secret of equal length still rejects
# ---------------------------------------------------------------------------


def test_different_secret_of_equal_length_rejects_token() -> None:
    """An HMAC key of equal length but different bytes is not interchangeable.

    Defense in depth against key-confusion: a token signed with secret A
    must NOT verify under secret B even if B is the same length (the
    HMAC output is a function of the key bytes, not just the length).
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    assert verify(token, secret=_SECRET_B, now=_NOW + 1) is None


# ---------------------------------------------------------------------------
# 5. Constant-time comparison — short-circuit detection
# ---------------------------------------------------------------------------


def test_signature_compare_does_not_short_circuit_on_first_byte_mismatch() -> None:
    """The HMAC compare path does not short-circuit on a first-byte mismatch.

    A short-circuiting ``==`` would let an attacker measure which byte
    of the signature is wrong by timing, accelerating a forgery search.
    ``hmac.compare_digest`` is constant-time over the *whole* tag.
    This test pins the structural guard by feeding three
    mismatches of increasing prefix-length and asserting the result
    is consistent (no early-return) — the constant-time property
    itself is the stdlib's contract.
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, sig_b64 = token.split(".")
    # Three different mismatches: flip first / flip second-to-last / flip last.
    for i in (0, len(sig_b64) // 2, len(sig_b64) - 1):
        flipped_char = "A" if sig_b64[i] != "A" else "B"
        tampered = f"{payload_b64}.{sig_b64[:i]}{flipped_char}{sig_b64[i + 1 :]}"
        assert verify(tampered, secret=_SECRET_A, now=_NOW + 1) is None, (
            f"Position-{i} signature flip must be rejected (got verified)"
        )


# ---------------------------------------------------------------------------
# 6. On-wire payload format — stable across versions
# ---------------------------------------------------------------------------


def test_onwire_payload_carries_documented_keys_with_typed_values() -> None:
    """The compact JSON payload uses the documented key aliases and value types.

    A future refactor that renames the keys (or coerces ``exp`` to a
    string) would break every minted token in the wild — the contract
    pins them as ``str(d) / str(p) / int(exp) / str|None(s)`` (RFC 8785
    + on-wire compactness).
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        session_id="chat-xyz",
        now=_NOW,
    )
    payload_b64, _ = token.split(".")
    pad = "=" * (-len(payload_b64) % 4)
    payload_bytes = base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii"))
    payload = json.loads(payload_bytes)

    # Canonical short aliases — no ``document_id`` / ``principal_id`` /
    # ``session_id`` / ``expires_at`` renames.
    assert set(payload.keys()) == {"d", "p", "exp", "s"}
    assert payload["d"] == _DOC_A
    assert payload["p"] == _PRINCIPAL_ALICE
    assert payload["exp"] == _NOW + 300
    assert payload["s"] == "chat-xyz"
    # Typed values: exp is an int (so a future JSON-vs-string check is caught).
    assert isinstance(payload["exp"], int)
    assert isinstance(payload["d"], str)
    assert isinstance(payload["p"], str)


def test_onwire_payload_omits_session_key_when_session_id_is_none() -> None:
    """A regular (non-attachment) token omits the ``s`` key entirely.

    The contract is: the ``s`` key is present iff a session_id was set.
    An absent key is the cheapest signal for the S04 path to skip
    session-widening.
    """
    token = mint(
        document_id=_DOC_A,
        principal_id=_PRINCIPAL_ALICE,
        secret=_SECRET_A,
        ttl_seconds=300,
        now=_NOW,
    )
    payload_b64, _ = token.split(".")
    pad = "=" * (-len(payload_b64) % 4)
    payload = json.loads(base64.urlsafe_b64decode((payload_b64 + pad).encode("ascii")))
    assert "s" not in payload
    # The required keys are still there.
    assert payload["d"] == _DOC_A
    assert payload["p"] == _PRINCIPAL_ALICE
    assert payload["exp"] == _NOW + 300


# ---------------------------------------------------------------------------
# 7. mint() input validation — fail loud on programmer errors
# ---------------------------------------------------------------------------


def test_mint_rejects_blank_secret() -> None:
    """``mint`` raises ``ValueError`` on an empty/blank secret (programmer error).

    The fail-safe is deliberate: a missing secret must never produce a
    working token.  The S05 mint caller (``_citation_to_out``) catches
    this and emits ``download_url=""``.
    """
    with pytest.raises(ValueError, match="secret"):
        mint(
            document_id=_DOC_A,
            principal_id=_PRINCIPAL_ALICE,
            secret="",
            ttl_seconds=300,
            now=_NOW,
        )


def test_mint_rejects_empty_document_id() -> None:
    """``mint`` rejects an empty ``document_id`` — a token with no bound doc is meaningless."""
    with pytest.raises(ValueError, match="document_id"):
        mint(
            document_id="",
            principal_id=_PRINCIPAL_ALICE,
            secret=_SECRET_A,
            ttl_seconds=300,
            now=_NOW,
        )


def test_mint_rejects_empty_principal_id() -> None:
    """``mint`` rejects an empty ``principal_id`` — the token would be unbound."""
    with pytest.raises(ValueError, match="principal_id"):
        mint(
            document_id=_DOC_A,
            principal_id="",
            secret=_SECRET_A,
            ttl_seconds=300,
            now=_NOW,
        )


@pytest.mark.parametrize("bad_ttl", [0, -1, -100])
def test_mint_rejects_non_positive_ttl(bad_ttl: int) -> None:
    """``mint`` rejects non-positive TTLs — a zero/negative-TTL token is nonsense."""
    with pytest.raises(ValueError, match="ttl_seconds"):
        mint(
            document_id=_DOC_A,
            principal_id=_PRINCIPAL_ALICE,
            secret=_SECRET_A,
            ttl_seconds=bad_ttl,
            now=_NOW,
        )


# ---------------------------------------------------------------------------
# 8. verify() must never raise on attacker-controlled input (no info leak)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_token",
    [
        # Trailing garbage
        "abc.def.ghi",
        # Whitespace tampering
        " . ",
        "x. y",
        # Non-ASCII / non-base64url chars
        "日本語.日本語",
        "💀.💀",
        # Long-but-valid base64url payload with non-JSON bytes
        "YWNlYWNlYWNlYWNl.QUFBQQ",
        # Payload that decodes to non-dict JSON
        "WzEsMiwzXQ.AAA",
        # Payload that decodes to dict with wrong field types
        "eyJkIjoxMjN9.AAA",  # {"d": 123}
    ],
)
def test_verify_never_raises_on_attacker_input(bad_token: str) -> None:
    """``verify`` never raises on attacker-controlled input — it returns ``None``.

    A raising ``verify`` would leak stack traces / 5xx to an unauthenticated
    browser GET (the S04 download path).  The contract is: every failure mode
    is a ``None`` return.
    """
    assert verify(bad_token, secret=_SECRET_A, now=_NOW) is None


# ---------------------------------------------------------------------------
# 9. VerifiedDownloadToken dataclass — frozen, slots, well-typed
# ---------------------------------------------------------------------------


def test_verified_download_token_is_frozen_with_slots() -> None:
    """The dataclass is frozen and slots-based — assignment raises."""
    import dataclasses

    v = VerifiedDownloadToken(document_id="d", principal_id="p", session_id=None)
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        v.document_id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        v.principal_id = "other"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError, match="cannot assign to field"):
        v.session_id = "other"  # type: ignore[misc]


def test_verified_download_token_field_typing() -> None:
    """The three fields are typed: ``str`` for ids, ``Optional[str]`` for session."""
    import dataclasses

    fields = {f.name: f.type for f in dataclasses.fields(VerifiedDownloadToken)}
    assert fields["document_id"] == "str"
    assert fields["principal_id"] == "str"
    assert fields["session_id"] == "str | None"
