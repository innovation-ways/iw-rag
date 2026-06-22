"""Append-only audit log for the defensible decision trail (Task 3.4).

Every request — answered, refused (fail-closed), or rejected (input-guard-blocked)
— writes exactly one :class:`AuditRecord` to ``iw_rag.audit_log`` in the same
Postgres instance as the data model (single-auditable-store, architecture §2 p7).

Key invariants:
- **Append-only:** no UPDATE/DELETE exposed here; enforced at the DB layer by a
  ``BEFORE UPDATE OR DELETE`` trigger (``audit_no_mutate()``) and by a REVOKE
  on the application role.
- **PII discipline:** ``redactions`` store entity types/counts only — never raw
  PII values. The audit log must not become a PII sink.
- **Audit-write failure = fail-closed:** if ``write_audit`` raises, the caller
  (``audit_node``) must refuse the request (no unaudited answer emitted).
- **Tamper-evidence (OPEN-QUESTIONS upgrade):** the Phase-3 baseline is DB-level
  append-only (trigger + revoke). An optional hash-chain (each row stores
  ``prev_hash`` = hash of the previous row's canonical serialisation) is deferred
  to an OPEN-QUESTIONS item — it would make silent deletion/reordering detectable
  without requiring external WORM storage.

References:
- task-3.4-audit-log §2 (AuditRecord shape), §3 (write_audit), §4 (assemble).
- lld-guardrails §2 (public interface), §3.5 (audit algorithm), §5 (error handling).
- architecture §2 principle 7 (one auditable store).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from iw_rag.domain.models import GuardVerdict

if TYPE_CHECKING:
    from iw_rag.agent.state import GraphState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AuditRecord
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditRecord:
    """Immutable record of the full decision trail for one request.

    Written once per request to ``iw_rag.audit_log``.  Frozen (immutable after
    creation) — it is a fact about the past and must not be mutated.

    Attributes:
        request_id: Unique request identifier (from ``AccessContext.request_id``
            or thread id).  Used to correlate the audit row with Phoenix spans.
        principal: The authenticated principal id (``AccessContext.principal_id``).
        query: The user's **original** question phrasing (``original_question``
            from state) — NEVER a rewritten version.
        acl_filter: The ``principal_groups`` applied at retrieval time (Task 3.0
            provenance). Stored verbatim as the authoritative ACL applied.
        retrieved_chunk_ids: All chunk ids considered by this request — includes
            graded survivors **and** chunks dropped by the sanitizer.
        guard_verdicts: The full ``verdicts`` trail mapped to ``GuardVerdict``
            objects (one per node invocation).  Captures each layer's decision,
            score, and reason.
        redactions: Redaction metadata only — list of
            ``{"layer": …, "entity_types": […], "count": N}`` dicts.
            **Never** the raw redacted values (PII discipline).
        disposition: Terminal outcome — one of ``"answered"``, ``"refused"``,
            or ``"rejected"``.
        timestamp: ISO-8601 UTC timestamp of record assembly.
    """

    request_id: str
    principal: str
    query: str
    acl_filter: list[str]
    retrieved_chunk_ids: list[str]
    guard_verdicts: list[GuardVerdict]
    redactions: list[dict[str, Any]]
    disposition: Literal["answered", "refused", "rejected"]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _serialize_guard_verdicts(verdicts: list[GuardVerdict]) -> list[dict[str, Any]]:
    """Serialise a list of ``GuardVerdict`` objects to a JSON-safe list of dicts.

    Each dict carries ``layer``, ``decision``, ``score``, ``reason``, and
    ``detail`` — the full fields of ``GuardVerdict``.  The result is suitable
    for storing in a Postgres ``jsonb`` column.

    Args:
        verdicts: The guard verdicts to serialise.

    Returns:
        A list of plain dicts, one per verdict, JSON-serialisable.
    """
    return [
        {
            "layer": v.layer,
            "decision": v.decision,
            "score": v.score,
            "reason": v.reason,
            "detail": v.detail,
        }
        for v in verdicts
    ]


def _verdicts_to_guard_verdicts(
    loop_verdicts: list[Any],
) -> list[GuardVerdict]:
    """Map loop ``Verdict`` trail items to ``GuardVerdict`` objects.

    The loop trail is ``list[Verdict]`` (with ``kind``, ``value``, ``score``,
    ``reason``, ``attempt``).  This maps each ``Verdict`` to a ``GuardVerdict``
    by: ``layer=kind``, ``decision=value``, ``score=score or 0.0``,
    ``reason=reason``.  This preserves the full trail in a uniform type.

    Args:
        loop_verdicts: The ``state["verdicts"]`` list (``list[Verdict]``).

    Returns:
        A list of ``GuardVerdict`` objects, one per loop verdict.
    """
    guard_verdicts: list[GuardVerdict] = []
    for v in loop_verdicts:
        # v is a Verdict dataclass or dict.
        if hasattr(v, "kind"):
            layer = v.kind
            decision = v.value
            score = v.score if v.score is not None else 0.0
            reason = v.reason or ""
        elif isinstance(v, dict):
            layer = v.get("kind", "unknown")
            decision = v.get("value", "unknown")
            score = float(v.get("score") or 0.0)
            reason = v.get("reason", "")
        else:
            continue
        # GuardVerdict.decision is typed as Decision = Literal["allow","block","redact","flag"]
        # Loop verdicts use different values (e.g. "retrieve", "grounded") — we store them
        # as-is under the "decision" field using a cast (the DB stores text, not a Literal).
        guard_verdicts.append(
            GuardVerdict(
                layer=layer,
                decision=decision,
                score=score,
                reason=reason,
            )
        )
    return guard_verdicts


# ---------------------------------------------------------------------------
# Assembly helper
# ---------------------------------------------------------------------------


def assemble_audit_record(
    state: GraphState,
    *,
    disposition: Literal["answered", "refused", "rejected"],
) -> AuditRecord:
    """Assemble an :class:`AuditRecord` from the terminal ``GraphState``.

    Reads all required provenance from state:
    - ``original_question`` (NOT the rewritten ``question``).
    - ``access_context`` for ``principal_id``, ``principal_groups``,
      ``request_id``.
    - ``graded`` chunks + ``diagnostics["sanitize_dropped_ids"]`` for the
      full set of chunk ids considered.
    - ``verdicts`` trail for the full decision history.
    - Redaction metadata from the verdicts trail (entity types/counts, never
      raw values).

    Args:
        state: The terminal ``GraphState`` returned by the graph.
        disposition: The final disposition — ``"answered"``, ``"refused"``, or
            ``"rejected"``.

    Returns:
        A frozen :class:`AuditRecord` ready for ``write_audit``.
    """
    ctx = state.get("access_context")
    original_question: str = state.get("original_question") or state.get("question") or ""

    # Principal provenance
    principal: str = ctx.principal_id if ctx is not None else "unknown"
    request_id: str = (ctx.request_id if ctx is not None else None) or "unknown"

    # ACL filter: the groups actually applied at retrieval (Task 3.0 provenance)
    acl_filter: list[str] = list(ctx.principal_groups) if ctx is not None else []

    # Retrieved chunk ids = graded survivors + sanitize-dropped ids
    graded = state.get("graded") or []
    graded_ids: list[str] = [c.chunk_id for c in graded]
    diagnostics: dict[str, Any] = dict(state.get("diagnostics") or {})
    dropped_ids: list[str] = list(diagnostics.get("sanitize_dropped_ids") or [])
    # Deduplicate while preserving order (graded first, then dropped)
    seen: set[str] = set()
    retrieved_chunk_ids: list[str] = []
    for cid in graded_ids + dropped_ids:
        if cid not in seen:
            seen.add(cid)
            retrieved_chunk_ids.append(cid)

    # Map the full verdicts trail to GuardVerdict objects
    loop_verdicts = state.get("verdicts") or []
    guard_verdicts = _verdicts_to_guard_verdicts(loop_verdicts)

    # Redaction metadata: extract entity_types/counts from diagnostics (never raw values)
    # The sanitize node records dropped_chunk_ids; actual PII entity types come from
    # the redact verdicts if present in diagnostics.
    redactions: list[dict[str, Any]] = _extract_redaction_metadata(diagnostics, loop_verdicts)

    return AuditRecord(
        request_id=request_id,
        principal=principal,
        query=original_question,
        acl_filter=acl_filter,
        retrieved_chunk_ids=retrieved_chunk_ids,
        guard_verdicts=guard_verdicts,
        redactions=redactions,
        disposition=disposition,
    )


def _extract_redaction_metadata(
    diagnostics: dict[str, Any],
    loop_verdicts: list[Any],
) -> list[dict[str, Any]]:
    """Extract redaction metadata (entity types/counts) from diagnostics and verdicts.

    PII discipline: only entity types and counts are recorded — never the raw
    redacted values.  The metadata is assembled from:
    - ``diagnostics["redaction_*"]`` keys if written by the sanitize/output
      guard nodes.
    - Verdict ``reason`` strings that contain ``pii_redacted`` markers.

    Args:
        diagnostics: The merged ``state["diagnostics"]`` dict.
        loop_verdicts: The ``state["verdicts"]`` list for supplemental signals.

    Returns:
        A list of redaction metadata dicts, each with ``layer``, ``entity_types``,
        and ``count`` — never raw values.
    """
    redactions: list[dict[str, Any]] = []

    # Look for explicit redaction metadata written by the guard nodes.
    # The sanitize/output guard nodes may write ``diagnostics["redaction_<layer>"]``
    # as a list of {entity_type, count} pairs.
    for layer in ("sanitize", "output"):
        key = f"redaction_{layer}"
        if key in diagnostics:
            entry = diagnostics[key]
            if isinstance(entry, dict):
                redactions.append(
                    {
                        "layer": layer,
                        "entity_types": entry.get("entity_types", []),
                        "count": entry.get("count", 0),
                    }
                )

    # Check if PII was flagged via verdict reasons (e.g. "pii_redacted:EMAIL_ADDRESS,2")
    for v in loop_verdicts:
        if hasattr(v, "reason"):
            reason = v.reason
        elif isinstance(v, dict):
            reason = v.get("reason", "")
        else:
            reason = ""
        if reason and "pii_redacted:" in reason:
            # Parse "pii_redacted:EMAIL_ADDRESS,2" style annotations.
            try:
                _, payload = reason.split("pii_redacted:", 1)
                parts = payload.split(",")
                entity_type = parts[0].strip()
                count = int(parts[1]) if len(parts) > 1 else 1
                if hasattr(v, "kind"):
                    layer = v.kind
                elif isinstance(v, dict):
                    layer = v.get("kind", "unknown")
                else:
                    layer = "unknown"
                redactions.append({"layer": layer, "entity_types": [entity_type], "count": count})
            except (ValueError, IndexError):
                pass

    # Also expose the output_guard_pii_redacted diagnostic as metadata
    if diagnostics.get("output_guard_pii_redacted"):
        redactions.append(
            {
                "layer": "output",
                "entity_types": [],  # entity types not tracked at this level yet
                "count": 1,
            }
        )

    return redactions


# ---------------------------------------------------------------------------
# write_audit — insert-only via AuditRepository
# ---------------------------------------------------------------------------


async def write_audit(
    record: AuditRecord,
    repo: Any,
) -> None:
    """Persist an :class:`AuditRecord` to ``iw_rag.audit_log`` via the repository.

    Insert-only — this function never issues an UPDATE or DELETE.  If the insert
    raises, the exception propagates to the caller (``audit_node``), which must
    then fail the request closed.

    Args:
        record: The frozen audit record to persist.
        repo: An :class:`~iw_rag.storage.repositories.AuditRepository` instance
            bound to an open async session.

    Raises:
        Exception: Any DB or serialisation error propagates up so the caller can
            handle it fail-closed (lld-guardrails §5).
    """
    await repo.append(record)
