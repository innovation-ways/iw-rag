"""Unit tests for guardrails/acl.py — AclProvenance and assert_hotpath_acl.

TDD RED-first: these tests were written before the implementation existed.
They cover:
- ``acl_provenance`` hashes groups (no raw groups present in the returned object).
- ``assert_hotpath_acl`` raises AuthorizationError for a non-long-context result
  with ``acl_applied=False``.
- ``assert_hotpath_acl`` passes silently for the long-context carve-out.
- ``assert_hotpath_acl`` passes silently for any result with ``acl_applied=True``.
"""

from __future__ import annotations

import pytest

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError
from iw_rag.domain.models import RetrievalResult
from iw_rag.guardrails.acl import (
    AclProvenance,
    acl_provenance,
    assert_hotpath_acl,
)

# ---------------------------------------------------------------------------
# acl_provenance — provenance struct helpers
# ---------------------------------------------------------------------------


def test_acl_provenance_has_no_raw_groups() -> None:
    """Raw principal groups must NOT appear in the returned AclProvenance."""
    ctx = AccessContext(principal_id="u1", principal_groups=("finance", "all-staff"))
    prov = acl_provenance(ctx)
    assert isinstance(prov, AclProvenance)
    # The hash field must not contain the raw group strings.
    assert "finance" not in prov.principal_groups_hash
    assert "all-staff" not in prov.principal_groups_hash


def test_acl_provenance_hash_is_stable() -> None:
    """Same groups → same hash (deterministic)."""
    ctx = AccessContext(principal_id="u1", principal_groups=("finance", "hr"))
    h1 = acl_provenance(ctx).principal_groups_hash
    h2 = acl_provenance(ctx).principal_groups_hash
    assert h1 == h2


def test_acl_provenance_different_groups_different_hash() -> None:
    """Different groups → different hash."""
    ctx_a = AccessContext(principal_id="u", principal_groups=("finance",))
    ctx_b = AccessContext(principal_id="u", principal_groups=("hr",))
    hash_a = acl_provenance(ctx_a).principal_groups_hash
    hash_b = acl_provenance(ctx_b).principal_groups_hash
    assert hash_a != hash_b


def test_acl_provenance_acl_applied_and_mechanism() -> None:
    """Default mechanism is 'pre_filter'; acl_applied reflects the flag."""
    ctx = AccessContext(principal_id="u", principal_groups=("finance",))
    prov = acl_provenance(ctx)
    assert prov.acl_applied is True
    assert prov.mechanism == "pre_filter"


def test_acl_provenance_rls_mechanism() -> None:
    """Caller can pass mechanism='rls' explicitly."""
    ctx = AccessContext(principal_id="u", principal_groups=("finance",))
    prov = acl_provenance(ctx, mechanism="rls")
    assert prov.mechanism == "rls"


def test_acl_provenance_empty_groups_yields_hash() -> None:
    """Even an empty groups tuple must produce a (consistent, non-raw) hash."""
    ctx = AccessContext(principal_id="u", principal_groups=())
    prov = acl_provenance(ctx)
    # sha256("") truncated to 16 hex chars — pinned to the exact stable value.
    assert prov.principal_groups_hash == "e3b0c44298fc1c14"


# ---------------------------------------------------------------------------
# assert_hotpath_acl — mode-ACL tension guard
# ---------------------------------------------------------------------------


def _make_result(acl_applied: bool, retriever: str = "spine") -> RetrievalResult:
    return RetrievalResult(
        chunks=[],
        retriever=retriever,
        acl_applied=acl_applied,
    )


def test_assert_hotpath_acl_passes_for_true() -> None:
    """acl_applied=True always passes regardless of route."""
    result = _make_result(acl_applied=True)
    ret = assert_hotpath_acl(result, route="retrieve")
    assert ret is None  # function returns None (no exception raised)


def test_assert_hotpath_acl_raises_for_false_non_long_context() -> None:
    """acl_applied=False on a non-long-context route raises AuthorizationError."""
    result = _make_result(acl_applied=False)
    with pytest.raises(AuthorizationError, match="acl_applied"):
        assert_hotpath_acl(result, route="retrieve")


def test_assert_hotpath_acl_raises_for_false_graph_route() -> None:
    """Graph route with acl_applied=False (before entities have ACL tags) must raise."""
    result = _make_result(acl_applied=False, retriever="graphrag")
    with pytest.raises(AuthorizationError, match="acl_applied"):
        assert_hotpath_acl(result, route="graph")


def test_assert_hotpath_acl_passes_for_long_context_carve_out() -> None:
    """Long-context is the ONE sanctioned acl_applied=False carve-out (CONVENTIONS §6)."""
    result = _make_result(acl_applied=False, retriever="long_context")
    # Must NOT raise — long_context is the explicit carve-out.
    ret = assert_hotpath_acl(result, route="long_context")
    assert ret is None  # function returns None (no exception raised)


def test_assert_hotpath_acl_raises_for_false_skip_route() -> None:
    """Skip route with acl_applied=False should also raise (only long_context is exempt)."""
    result = _make_result(acl_applied=False)
    with pytest.raises(AuthorizationError, match="acl_applied"):
        assert_hotpath_acl(result, route="skip")
