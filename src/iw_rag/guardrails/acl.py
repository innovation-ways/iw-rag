"""ACL guardrail — provenance struct + mode-ACL tension guard.

Two responsibilities:

1. **AclProvenance** — a small, audit-ready struct capturing the ACL filter
   that was applied to a retrieval request.  Groups are hashed (never stored
   raw — privacy/observability boundary) via
   :func:`~iw_rag.observability.tracing.hash_groups`.  The struct is threaded
   into ``state["diagnostics"]["acl"]`` so Task 3.4 (audit log) can persist
   it without recomputing ACL.

2. **assert_hotpath_acl** — the named, centralized mode-ACL tension guard
   (CONVENTIONS.md §6/§7).  Hot-path retrieval MUST have ``acl_applied=True``;
   the long-context route is the *only* sanctioned ``acl_applied=False``
   carve-out (its entry predicate admits only non-ACL, attribution-free,
   static whole-doc queries, so there is nothing to pre-filter).  Every other
   route with ``acl_applied=False`` is a bug — the guard raises loudly rather
   than silently serve un-ACL'd chunks.

Design rule: this module **reads** the ACL provenance that the spine computed;
it does NOT recompute or widen the ACL.  The enforcement point remains the
pre-filter SQL in ``retrieval/acl_filter.py``.  See lld-guardrails §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.errors import AuthorizationError
from iw_rag.domain.models import RetrievalResult
from iw_rag.observability.tracing import hash_groups

# The single sanctioned route that may carry acl_applied=False.
_LONG_CONTEXT_ROUTE = "long_context"


@dataclass(frozen=True, slots=True)
class AclProvenance:
    """Audit-ready ACL provenance for one retrieval request.

    Carries *what* was enforced without exposing *who* (groups are hashed).
    Thread this into ``state["diagnostics"]["acl"]`` after each retrieve call
    so Task 3.4 can persist it in the audit log.

    Attributes:
        principal_groups_hash: A short, non-reversible sha-256 digest of the
            principal's groups — never the raw group names.  Computed by
            :func:`~iw_rag.observability.tracing.hash_groups`.
        acl_applied: Whether the ACL pre-filter was active for the request.
        mechanism: The enforcement mechanism; ``"pre_filter"`` is the default
            (application-layer WHERE clause); ``"rls"`` indicates Postgres
            Row-Level Security was also in effect.
    """

    principal_groups_hash: str
    acl_applied: bool
    mechanism: Literal["pre_filter", "rls"]


def acl_provenance(
    ctx: AccessContext,
    *,
    mechanism: Literal["pre_filter", "rls"] = "pre_filter",
) -> AclProvenance:
    """Build an :class:`AclProvenance` for an authenticated request.

    Groups are hashed via
    :func:`~iw_rag.observability.tracing.hash_groups` — raw group names are
    never stored.  The returned struct is immutable and safe to log / persist.

    Args:
        ctx: The authenticated access context.
        mechanism: The ACL enforcement mechanism in use.  Defaults to
            ``"pre_filter"`` (application-layer WHERE clause).  Pass
            ``"rls"`` when the Postgres Row-Level Security layer was also
            active for the request.

    Returns:
        An :class:`AclProvenance` with a hashed group digest.
    """
    return AclProvenance(
        principal_groups_hash=hash_groups(ctx.principal_groups),
        acl_applied=True,
        mechanism=mechanism,
    )


def assert_hotpath_acl(result: RetrievalResult, *, route: str) -> None:
    """Assert the ACL invariant for a hot-path retrieval result.

    Centralises and names the mode-ACL tension guard (CONVENTIONS.md §6/§7):
    - ``acl_applied=True`` → always OK.
    - ``acl_applied=False`` + ``route="long_context"`` → OK (the one sanctioned
      carve-out; long-context admits only non-ACL, attribution-free, whole-doc
      queries — there is nothing to pre-filter).
    - ``acl_applied=False`` + any other route → **bug** → raises
      :class:`~iw_rag.domain.errors.AuthorizationError` rather than silently
      serving un-ACL'd chunks.

    **This function replaces the inline assertion in ``agent/tools.py``** so the
    guard has a name, a single definition, and a precise error message.  It does
    NOT recompute ACL — it reads what the retriever already set.

    Args:
        result: The retrieval result returned by the spine or a mode retriever.
        route: The LangGraph route that triggered this retrieval (e.g.
            ``"retrieve"``, ``"graph"``, ``"long_context"``, ``"skip"``).

    Raises:
        AuthorizationError: If ``result.acl_applied is not True`` and
            ``route`` is not ``"long_context"``.  This is a security invariant
            violation — the caller must propagate, not swallow, this error.
    """
    if result.acl_applied is True:
        return
    if route == _LONG_CONTEXT_ROUTE:
        # Sanctioned carve-out: long-context route may omit ACL pre-filter
        # because its entry predicate restricts to non-ACL, attribution-free,
        # static whole-doc queries (CONVENTIONS.md §6).
        return
    raise AuthorizationError(
        f"ACL invariant violated: acl_applied={result.acl_applied!r} "
        f"on route={route!r} (retriever={result.retriever!r}). "
        "Hot-path results MUST have acl_applied=True; "
        "the long-context route is the only sanctioned carve-out "
        "(CONVENTIONS.md §6 / lld-guardrails §3.1). "
        "Never serve un-ACL'd chunks — fail loudly."
    )
