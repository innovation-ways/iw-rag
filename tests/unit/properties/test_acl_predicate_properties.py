"""Hypothesis property-based tests for ``iw_rag.retrieval.acl_filter.acl_predicate``.

Invariants verified (all are security invariants, not convenience ones):
- **Params completeness**: ``params()`` contains EXACTLY the principal's groups —
  never more, never fewer, never groups from a different source.
- **Empty groups → is_empty True**: a principal with no groups produces an empty
  predicate (fail-closed behaviour).
- **No group interpolation**: no group string literal ever appears in the SQL
  clause text returned by ``sql_clause()`` — the clause is always parameterised.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from iw_rag.domain.acl import AccessContext
from iw_rag.retrieval.acl_filter import acl_predicate

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Groups are non-empty strings (a real group name is at least 1 char)
_GROUP = st.text(min_size=1, max_size=40)

_GROUPS = st.lists(_GROUP, min_size=0, max_size=20, unique=True).map(tuple)  # type: ignore[arg-type]

_PRINCIPAL_ID = st.text(min_size=1, max_size=40)


# ---------------------------------------------------------------------------
# Property 1 — params() contains EXACTLY the principal groups
# ---------------------------------------------------------------------------


@given(principal_id=_PRINCIPAL_ID, groups=_GROUPS)
def test_params_contains_exactly_principal_groups(
    principal_id: str,
    groups: tuple[str, ...],
) -> None:
    """``params()`` must contain exactly ``principal_groups``, sorted stably.

    The security invariant: the bind parameters for the SQL query must come
    exclusively from the authenticated ``AccessContext.principal_groups`` —
    never more, never fewer, never injected from elsewhere.
    """
    ctx = AccessContext(principal_id=principal_id, principal_groups=groups)
    pred = acl_predicate(ctx)
    result = pred.params()

    assert set(result.keys()) == {"principal_groups"}, (
        f"params() must have exactly one key 'principal_groups', got {set(result.keys())!r}"
    )
    assert sorted(result["principal_groups"]) == sorted(list(groups)), (
        f"params()['principal_groups'] {sorted(result['principal_groups'])} "
        f"!= sorted principal groups {sorted(list(groups))}"
    )


# ---------------------------------------------------------------------------
# Property 2 — empty groups → is_empty True
# ---------------------------------------------------------------------------


@given(principal_id=_PRINCIPAL_ID)
def test_empty_groups_yields_empty_predicate(principal_id: str) -> None:
    """A principal with no groups produces a predicate where ``is_empty`` is True.

    This encodes the fail-closed behaviour: a principal with no group entitlements
    can see nothing.  The property holds for any principal id.
    """
    ctx = AccessContext(principal_id=principal_id, principal_groups=())
    pred = acl_predicate(ctx)
    assert pred.is_empty is True, (
        f"Empty groups must produce is_empty=True, got is_empty={pred.is_empty}"
    )


@given(principal_id=_PRINCIPAL_ID, groups=_GROUPS)
def test_nonempty_groups_yields_nonempty_predicate(
    principal_id: str,
    groups: tuple[str, ...],
) -> None:
    """A principal with at least one group produces a non-empty predicate.

    Contrapositive of the fail-closed rule: entitlements must never be silently
    discarded.
    """
    from hypothesis import assume

    assume(len(groups) >= 1)
    ctx = AccessContext(principal_id=principal_id, principal_groups=groups)
    pred = acl_predicate(ctx)
    assert pred.is_empty is False, (
        f"Non-empty groups must produce is_empty=False, got is_empty={pred.is_empty}"
    )


# ---------------------------------------------------------------------------
# Property 3 — SQL clause is parameterised (no group literal interpolated)
# ---------------------------------------------------------------------------


@given(principal_id=_PRINCIPAL_ID, groups=_GROUPS)
def test_sql_clause_never_interpolates_group_strings(
    principal_id: str,
    groups: tuple[str, ...],
) -> None:
    """No group string ever appears literally inside the SQL clause text.

    The clause must use bind parameters (``:``) — never f-string/%-format
    interpolation.  This guards against SQL injection: any group name crafted by
    an attacker must not alter the query structure.
    """
    from hypothesis import assume

    assume(len(groups) >= 1)
    ctx = AccessContext(principal_id=principal_id, principal_groups=groups)
    pred = acl_predicate(ctx)
    clause = pred.sql_clause()

    # A group string MUST never appear as a SQL string literal in the
    # clause — that would be an interpolation vulnerability.  A
    # substring match is too literal (a group of ``"_"`` would falsely
    # match the underscore in the bind-placeholder name
    # ``:principal_groups``), so we look only for the group as a
    # quoted SQL literal (single-quoted, per ANSI SQL).
    import re

    literals = re.findall(r"'((?:[^']|'')*)'", clause)
    for group in groups:
        assert group not in literals, (
            f"Group string {group!r} appears as a SQL literal in sql_clause "
            f"{clause!r} — this is an interpolation vulnerability"
        )

    # The clause must reference the bind parameter placeholder, not a literal
    assert ":principal_groups" in clause, (
        f"sql_clause {clause!r} does not contain ':principal_groups' bind placeholder"
    )
