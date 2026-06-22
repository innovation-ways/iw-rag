"""Unit tests for the ACL pre-filter predicate (security keystone)."""

from __future__ import annotations

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.acl_filter import acl_predicate


def test_predicate_binds_only_principal_groups() -> None:
    ctx = AccessContext(principal_id="u", principal_groups=("finance", "all-staff"))
    pred = acl_predicate(ctx)
    assert pred.sql_clause() == "acl_tags && :principal_groups"
    assert pred.params() == {"principal_groups": ["finance", "all-staff"]}


def test_predicate_groups_come_from_ctx_not_text() -> None:
    # The predicate is built ONLY from the AccessContext; there is no parameter
    # by which query text or model output could reach it.
    ctx = AccessContext(principal_id="u", principal_groups=("hr",))
    pred = acl_predicate(ctx)
    assert pred.principal_groups == ("hr",)


def test_empty_groups_is_empty_predicate() -> None:
    pred = acl_predicate(AccessContext(principal_id="u", principal_groups=()))
    assert pred.is_empty is True


def test_nonempty_groups_not_empty() -> None:
    pred = acl_predicate(AccessContext(principal_id="u", principal_groups=("finance",)))
    assert pred.is_empty is False
