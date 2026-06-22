"""Long-context route configuration and data types (WI-032 / Task 4c).

Provides the canonical type vocabulary for the long-context mode:
- :class:`LongContextConfig` — config for the gated 4th router branch.
- :class:`EntryDecision` — result of the entry-predicate evaluation.

**YaRN invariant:** ``LongContextConfig.enable_yarn`` MUST remain ``False``
for this route. A ``@field_validator`` rejects ``enable_yarn=True`` at
startup so the route never taxes short RAG traffic (~15–20% degradation,
R-00109/F3). This is enforced here as the single source of truth.

References:
- lld-long-context.md §2 (public interface + type signatures)
- task-4c-long-context-route §1 (config + invariants)
- CONVENTIONS.md §6 (acl_applied=False carve-out)
- R-00109/F3 (YaRN tax on short RAG traffic)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from pydantic import BaseModel, field_validator

# ---------------------------------------------------------------------------
# LongContextConfig
# ---------------------------------------------------------------------------


class LongContextConfig(BaseModel):
    """Configuration for the gated long-context route (lld-long-context §6).

    All fields match the ``modes.long_context.*`` YAML block loaded by
    ``Settings.modes.long_context`` (``IW_RAG__MODES__LONG_CONTEXT__*`` env
    mapping, ``config/default.yaml``).

    **YaRN invariant:** ``enable_yarn`` MUST be ``False`` — validated at
    instantiation time.  Any attempt to set it ``True`` raises ``ValueError``,
    surfacing the misconfiguration at startup before any traffic arrives.

    Attributes:
        lc_max_input: Target content cap (well under native 32K; default 24 000).
            Controlled via experiment #8 NoLiMa/RULER probes — retuned if the
            effective-length cliff moves (lld-long-context §9 [ASSUMPTION]).
        hard_ceiling: Absolute maximum tokens; native 32K window only (default
            30 000).  Post-assembly token count exceeding this raises
            :class:`~iw_rag.modes.long_context.route.LongContextOverBudget`.
        enable_yarn: INVARIANT — must remain ``False``.  Setting ``True`` is
            a ``ValueError`` (startup guard, R-00109/F3).
        enabled: Master switch.  Off by default until experiment #8 proves the
            whole-doc win within the VRAM budget (task-4c acceptance criteria).
        require_document_level_attribution_only: When ``True`` (default), only
            queries that need ≤ document-level attribution are eligible.
            Per-claim attribution queries are forced to RAG (T4).
    """

    lc_max_input: int = 24_000
    hard_ceiling: int = 30_000
    enable_yarn: bool = False
    enabled: bool = False
    require_document_level_attribution_only: bool = True

    @field_validator("enable_yarn")
    @classmethod
    def _reject_yarn(cls, v: bool) -> bool:
        """Reject enable_yarn=True at instantiation (YaRN invariant).

        Args:
            v: The value of enable_yarn.

        Returns:
            ``False`` — the only accepted value.

        Raises:
            ValueError: If ``v`` is ``True`` (the route must NEVER enable YaRN;
                R-00109/F3 — static YaRN degrades all short RAG prompts ~15–20%).
        """
        if v:
            raise ValueError(
                "enable_yarn=True is forbidden for the long-context route. "
                "The route operates strictly inside the native 32K window. "
                "YaRN degrades all short RAG traffic ~15–20%% (R-00109/F3). "
                "Set IW_RAG__MODES__LONG_CONTEXT__ENABLE_YARN=false."
            )
        return v


# ---------------------------------------------------------------------------
# EntryDecision
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryDecision:
    """Result of the entry-predicate evaluation for one query.

    Returned by :meth:`~iw_rag.modes.long_context.route.LongContextRoute.evaluate_entry`.
    Immutable (frozen dataclass) so it can be passed between async boundaries
    without accidental mutation.

    Attributes:
        eligible: Whether all four entry predicates pass.  When ``False`` the
            router falls through to ``retrieve`` (or ``graph`` for multi-hop).
        reason: Human-readable tag explaining the decision.  Canonical values:
            ``"eligible"`` · ``"no_bounded_corpus"`` · ``"not_wholedoc"`` ·
            ``"acl_required"`` · ``"attribution_strict"`` · ``"over_budget"``.
            Recorded in the verdict trail for audit and router-accuracy metrics.
        candidate_doc_ids: The doc IDs under evaluation.  Empty when the entry
            predicate failed before token estimation (e.g. ``no_bounded_corpus``
            or ``acl_required``).
        estimated_input_tokens: Token estimate from
            :meth:`~iw_rag.modes.long_context.assembler.WholeDocAssembler.estimate_tokens`.
            Zero when estimation was skipped (entry failed early).
    """

    eligible: bool
    reason: str
    candidate_doc_ids: list[uuid.UUID]
    estimated_input_tokens: int = field(default=0)
