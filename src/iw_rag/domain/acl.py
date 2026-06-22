"""Access-control identity — the security keystone type.

``AccessContext`` carries the authenticated principal and its groups into the
retrieval ACL pre-filter. ``principal_groups`` is populated **from auth only**,
never from the LLM or user-supplied text, and is an immutable tuple so no
downstream caller can widen entitlements by appending to it. The enforcement
predicate is ``acl_tags && principal_groups`` (pre-filter, never post-filter).

See CONVENTIONS.md §2 and lld-api §5.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AccessContext:
    """An authenticated principal and its entitlements.

    Attributes:
        principal_id: Stable identifier for the caller (from auth).
        principal_groups: The caller's groups, used as the ACL pre-filter set.
            Immutable tuple — must never be a list a caller can mutate.
        is_operator: Whether the caller may invoke operator-only routes
            (e.g. ``/ingest``).
        request_id: Correlation id for tracing/audit.
    """

    principal_id: str
    principal_groups: tuple[str, ...]
    is_operator: bool = False
    request_id: str = ""
