"""The ACL pre-filter predicate — the retrieval security keystone.

Both retrieval arms bind this predicate so a chunk the principal cannot see never
enters either ranked list (pre-filter, never post-filter — R-00108, lld-data-model
§5). ``principal_groups`` come ONLY from the authenticated ``AccessContext`` —
never from query text or model output.
"""

from __future__ import annotations

from dataclasses import dataclass

from iw_rag.domain.acl import AccessContext


@dataclass(frozen=True, slots=True)
class AclPredicate:
    """A bound ACL filter: the SQL clause + its parameters.

    Attributes:
        principal_groups: The caller's groups (the only ACL input).
    """

    principal_groups: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        """True when the principal has no groups (→ sees nothing; fail-closed)."""
        return len(self.principal_groups) == 0

    def sql_clause(self) -> str:
        """The array-overlap WHERE clause binding ``:principal_groups``."""
        return "acl_tags && :principal_groups"

    def params(self) -> dict[str, list[str]]:
        """The bind parameters (only the principal's groups)."""
        return {"principal_groups": list(self.principal_groups)}


def acl_predicate(ctx: AccessContext) -> AclPredicate:
    """Build the ACL predicate from an AccessContext.

    The input MUST be a typed :class:`AccessContext` — never a raw string, dict,
    or any other type that could carry attacker-controlled text (lld-guardrails
    §3.1; CONVENTIONS.md §2).  ``principal_groups`` come ONLY from the
    authenticated session; this function is the enforcement boundary.

    Args:
        ctx: The authenticated access context (the sole ACL source).

    Returns:
        An :class:`AclPredicate` bound to ``ctx.principal_groups``.

    Raises:
        TypeError: If ``ctx`` is not an :class:`AccessContext`.  This is an
            invariant violation — not a user error — so it surfaces as a hard
            failure rather than an empty result.
    """
    if not isinstance(ctx, AccessContext):
        raise TypeError(
            "ACL predicate requires an authenticated AccessContext, "
            "never free-text or model output (lld-guardrails §3.1). "
            f"Got: {type(ctx).__name__!r}"
        )
    return AclPredicate(principal_groups=ctx.principal_groups)
