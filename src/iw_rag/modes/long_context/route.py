"""Long-context route — entry predicate, assembly, and exit fallback (WI-032 / Task 4c).

The gated 4th branch of the Adaptive-RAG router. Unlike the graph and visual
modes (which are retrievers), this route **bypasses retrieval** and stuffs
whole document(s) into the native 32K window — but still emits a
``RetrievalResult``-shaped object (the stuffed docs as document-level
``RetrievedChunk``s with document-level citations) so the generator/eval stay
mode-agnostic.

**Two structural exclusions (T4, R-00109/F5):**
1. Never serves ACL-scoped queries — forced to RAG by the entry predicate.
2. Never serves per-claim-attribution queries — document-level citations only.

``acl_applied=False`` on the emitted ``RetrievalResult`` is correct *because*
the entry predicate admits only non-ACL, attribution-free, static whole-doc
queries — nothing to pre-filter (the SINGLE documented carve-out to the spine's
``acl_applied=True`` invariant, CONVENTIONS.md §6).

References:
- lld-long-context.md §2 (public interface), §3 (algorithms), §5 (error handling)
- task-4c-long-context-route §2–§5 (step-by-step implementation)
- CONVENTIONS.md §6 (acl_applied=False carve-out)
- R-00109/F3–F6 (YaRN, routing economics, prefix cache, exit predicates)
- T4 (RAG stays default; long-context never serves ACL/attribution-strict)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from typing import Any

from iw_rag.domain.acl import AccessContext
from iw_rag.domain.models import RetrievalResult
from iw_rag.modes.long_context.assembler import WholeDocAssembler
from iw_rag.modes.long_context.types import EntryDecision, LongContextConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LongContextOverBudget — the single structured error from this route
# ---------------------------------------------------------------------------


class LongContextOverBudget(Exception):
    """Raised when post-assembly token count exceeds the hard ceiling.

    The caller (the router node) catches this and demotes to ``retrieve``.
    Never silently enable YaRN on overflow — abort and fall back (lld-
    long-context §5).
    """


# ---------------------------------------------------------------------------
# Canonical fallback signals — signals that trigger should_fallback=True
# ---------------------------------------------------------------------------

#: Generation signals that mandate a RAG fallback.
_FALLBACK_SIGNALS: frozenset[str] = frozenset(
    {
        "unsupported",  # not grounded by evidence → recycle to RAG
        "not_answering",  # grounded but off-target → rewrite + RAG
        "summarize_not_answer",  # Databricks failure mode (R-00109/F6)
        "refusal",  # model refused to answer
        "over_budget",  # budget overrun detected post-generation
    }
)

#: Only this signal keeps the long-context route in control.
_PASS_SIGNAL: str = "grounded"


# ---------------------------------------------------------------------------
# LongContextRoute
# ---------------------------------------------------------------------------


class LongContextRoute:
    """The gated 4th branch of the Adaptive-RAG router.

    Owns the entry predicate, whole-document assembly (prefix-cache-friendly),
    and the exit/fallback predicate.  Does NOT own the classifier that labels
    a query as ``wholedoc`` — that lives in ``agent/`` (the router).

    **Invariants (lld-long-context §1, R-00109/F3/F5):**
    - Never enables the 131K YaRN window (``cfg.enable_yarn=False``).
    - Never serves ACL-scoped queries (``acl_required``).
    - Never serves per-claim-attribution queries (``attribution_strict``).
    - ``acl_applied=False`` on the emitted ``RetrievalResult`` is the SINGLE
      sanctioned carve-out (CONVENTIONS.md §6), correct because the entry
      predicate ensures nothing to ACL-filter.

    Attributes:
        assembler: The :class:`~iw_rag.modes.long_context.assembler.WholeDocAssembler`
            used for token estimation and prompt assembly.
        cfg: The :class:`~iw_rag.modes.long_context.types.LongContextConfig`
            (validated at instantiation — YaRN guard has already fired).
    """

    def __init__(
        self,
        assembler: WholeDocAssembler,
        cfg: LongContextConfig,
    ) -> None:
        """Initialise the route with an assembler and validated config.

        Args:
            assembler: The document assembler (token estimation + assembly).
            cfg: The validated long-context config (``enable_yarn=False``
                already enforced by the pydantic validator).
        """
        self.assembler = assembler
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Entry predicate (§3 algorithm, task-4c §2)
    # ------------------------------------------------------------------

    async def evaluate_entry(
        self,
        query: str,
        access: AccessContext,
        *,
        candidate_doc_ids: Sequence[uuid.UUID] | None = None,
        query_type: str | None = None,
        require_per_claim_attribution: bool = False,
    ) -> EntryDecision:
        """Evaluate whether the query is eligible for the long-context route.

        All four conditions must pass; any failure → ``eligible=False`` with
        the failing condition's ``reason`` tag (lld-long-context §3).

        Predicate order (short-circuit for structural failures first):
        1. Candidate doc set present (``no_bounded_corpus`` if not).
        2. Whole-doc / global intent (``not_wholedoc`` if not; router supplied
           ``query_type`` is the signal).
        3. No per-result ACL required (``acl_required`` if ACL-scoped; T4).
        4. Attribution ≤ document-level (``attribution_strict`` if per-claim).
        5. Token estimate ≤ ``lc_max_input`` (``over_budget`` if over).

        Args:
            query: The user's question text.
            access: The authenticated :class:`~iw_rag.domain.acl.AccessContext`
                (from auth, never from the LLM).
            candidate_doc_ids: The pre-identified whole-doc candidates.  When
                ``None`` (router could not identify a bounded corpus) →
                ``eligible=False, reason="no_bounded_corpus"``.
            query_type: The router classifier's label.  ``"wholedoc"`` is the
                only label that passes predicate #2.  When ``None``, the method
                falls back to heuristic intent detection from the query text.
            require_per_claim_attribution: When ``True``, the caller signals
                that per-claim attribution is required → ``attribution_strict``.

        Returns:
            An :class:`~iw_rag.modes.long_context.types.EntryDecision`.
        """
        # ------------------------------------------------------------------
        # Predicate 1: bounded corpus required
        # ------------------------------------------------------------------
        if not candidate_doc_ids:
            return EntryDecision(
                eligible=False,
                reason="no_bounded_corpus",
                candidate_doc_ids=[],
                estimated_input_tokens=0,
            )

        doc_ids: list[uuid.UUID] = list(candidate_doc_ids)

        # ------------------------------------------------------------------
        # Predicate 2: whole-doc / global intent
        # (router-supplied query_type OR heuristic fallback)
        # ------------------------------------------------------------------
        if not self._intent_is_wholedoc(query, query_type):
            return EntryDecision(
                eligible=False,
                reason="not_wholedoc",
                candidate_doc_ids=[],
                estimated_input_tokens=0,
            )

        # ------------------------------------------------------------------
        # Predicate 3: no per-result ACL required (structural, T4)
        # Check BEFORE token estimation to short-circuit early.
        # ------------------------------------------------------------------
        if self._requires_per_result_acl(access):
            return EntryDecision(
                eligible=False,
                reason="acl_required",
                candidate_doc_ids=[],
                estimated_input_tokens=0,
            )

        # ------------------------------------------------------------------
        # Predicate 4: attribution ≤ document-level (structural, T4)
        # ------------------------------------------------------------------
        if require_per_claim_attribution and self.cfg.require_document_level_attribution_only:
            return EntryDecision(
                eligible=False,
                reason="attribution_strict",
                candidate_doc_ids=[],
                estimated_input_tokens=0,
            )

        # ------------------------------------------------------------------
        # Predicate 5: token estimate within lc_max_input
        # ------------------------------------------------------------------
        est = await self.assembler.estimate_tokens(doc_ids)
        if est > self.cfg.lc_max_input:
            return EntryDecision(
                eligible=False,
                reason="over_budget",
                candidate_doc_ids=doc_ids,
                estimated_input_tokens=est,
            )

        logger.debug(
            "LongContextRoute.evaluate_entry: eligible query=%r doc_count=%d est=%d",
            query[:80],
            len(doc_ids),
            est,
        )
        return EntryDecision(
            eligible=True,
            reason="eligible",
            candidate_doc_ids=doc_ids,
            estimated_input_tokens=est,
        )

    # ------------------------------------------------------------------
    # build_context — assemble and emit RetrievalResult (§4, task-4c §4)
    # ------------------------------------------------------------------

    async def build_context(
        self,
        query: str,
        decision: EntryDecision,
        access: AccessContext,
    ) -> RetrievalResult:
        """Assemble whole-doc(s) and return a RetrievalResult.

        Assembles the document(s) identified in ``decision.candidate_doc_ids``
        in prefix-cache-friendly order, verifies the token count is within
        ``hard_ceiling``, and wraps the result in a :class:`RetrievalResult`
        with ``retriever="long_context"`` and ``acl_applied=False`` (the
        single sanctioned carve-out — CONVENTIONS.md §6).

        **YaRN never enabled** — ``cfg.enable_yarn`` is already validated
        ``False`` at config instantiation.

        Args:
            query: The user's original question (used for diagnostics only;
                the query is NOT included in the assembled context so the
                stable prefix spans the maximum APC-eligible prefix).
            decision: The :class:`~iw_rag.modes.long_context.types.EntryDecision`
                from a successful ``evaluate_entry`` call (``eligible=True``).
            access: The authenticated :class:`~iw_rag.domain.acl.AccessContext`.

        Returns:
            A :class:`~iw_rag.domain.models.RetrievalResult` with:
            - ``chunks``: document-level ``RetrievedChunk``s (one per doc).
            - ``retriever="long_context"``.
            - ``acl_applied=False`` (the sanctioned carve-out).
            - ``diagnostics``: ``route``, ``input_tokens``,
              ``prefix_cache_friendly``.

        Raises:
            LongContextOverBudget: If post-assembly token count exceeds
                ``cfg.hard_ceiling``.  Caller demotes to ``retrieve``.
        """
        assembled = await self.assembler.assemble(decision.candidate_doc_ids)

        # Post-assembly hard ceiling check — NEVER silently enable YaRN
        if assembled.token_count > self.cfg.hard_ceiling:
            logger.warning(
                "LongContextRoute.build_context: post-assembly token count %d > "
                "hard_ceiling %d — aborting (falling back to retrieve). "
                "YaRN is NOT enabled.",
                assembled.token_count,
                self.cfg.hard_ceiling,
            )
            raise LongContextOverBudget(
                f"Assembled context ({assembled.token_count} tokens) exceeds "
                f"hard_ceiling ({self.cfg.hard_ceiling} tokens). "
                "Falling back to retrieve. YaRN is NOT enabled."
            )

        diagnostics: dict[str, Any] = {
            "route": "long_context",
            "input_tokens": assembled.token_count,
            "prefix_cache_friendly": True,
            "doc_count": len(decision.candidate_doc_ids),
            "yarn_enabled": False,  # INVARIANT: always False
        }

        logger.debug(
            "LongContextRoute.build_context: assembled %d doc(s), %d tokens "
            "(within hard_ceiling=%d, yarn=False)",
            len(decision.candidate_doc_ids),
            assembled.token_count,
            self.cfg.hard_ceiling,
        )

        return RetrievalResult(
            query=query,
            chunks=assembled.as_doc_level_chunks(),
            expansion_mode="window",
            candidate_count=len(decision.candidate_doc_ids),
            retriever="long_context",
            acl_applied=False,  # sanctioned carve-out (CONVENTIONS §6)
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------
    # should_fallback — exit predicate (§5, task-4c §5)
    # ------------------------------------------------------------------

    def should_fallback(self, gen_signal: Any) -> bool:
        """Decide whether to demote to RAG based on the generation signal.

        Reads the loop's :class:`~iw_rag.agent.state.GenVerdict` (or any
        string signal) and returns ``True`` if the long-context route should
        give up and let the RAG spine re-answer.  Long-context never *blocks*
        an answer — RAG is the floor (lld-long-context §3).

        Fallback triggers (any of):
        - ``gen_signal`` is not ``"grounded"`` (unsupported / not_answering /
          over_budget / summarize_not_answer / refusal).
        - ``gen_signal`` is ``None`` or an unexpected value (fail-closed).

        No-fallback signal: only ``"grounded"`` keeps the route in control.

        Args:
            gen_signal: The loop's generation-grade signal (``GenVerdict``
                string, or ``None``).

        Returns:
            ``True`` when the route should demote to ``retrieve``; ``False``
            only when ``gen_signal == "grounded"``.
        """
        if gen_signal is None:
            logger.debug("LongContextRoute.should_fallback: signal=None → fallback=True")
            return True
        if gen_signal == _PASS_SIGNAL:
            return False
        if gen_signal in _FALLBACK_SIGNALS:
            logger.debug(
                "LongContextRoute.should_fallback: signal=%r in fallback_set → fallback=True",
                gen_signal,
            )
            return True
        # Unknown signal — fail-closed: demote to RAG
        logger.warning(
            "LongContextRoute.should_fallback: unexpected signal=%r → fallback=True (fail-closed)",
            gen_signal,
        )
        return True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _intent_is_wholedoc(query: str, query_type: str | None) -> bool:
        """Return True if the query intent is whole-document or global.

        The router classifier's ``query_type`` label is the primary signal.
        When ``None`` (degraded path), a simple keyword heuristic is applied.

        Args:
            query: The user question text.
            query_type: The router classifier's label (``"wholedoc"`` passes;
                all others fail; ``None`` triggers heuristic fallback).

        Returns:
            ``True`` if the intent is whole-doc / global.
        """
        if query_type is not None:
            return query_type == "wholedoc"
        # Heuristic fallback when no classifier label available.
        # Keywords associated with whole-doc / global intent:
        _WHOLEDOC_HINTS = frozenset(
            {
                "summarise",
                "summarize",
                "summary",
                "compare",
                "across",
                "entire",
                "whole",
                "all sections",
                "throughout",
                "overview",
                "synthesize",
                "synthesise",
            }
        )
        lower = query.lower()
        return any(hint in lower for hint in _WHOLEDOC_HINTS)

    @staticmethod
    def _requires_per_result_acl(access: AccessContext) -> bool:
        """Return True if the access context requires per-result ACL filtering.

        When ``principal_groups`` is non-empty, the query is ACL-scoped and
        the long-context route cannot safely serve it (T4 structural rule):
        there is no per-document ACL enforcement when a whole blob is stuffed.

        Args:
            access: The authenticated :class:`~iw_rag.domain.acl.AccessContext`.

        Returns:
            ``True`` when the principal has non-empty ``principal_groups``
            (ACL-scoped query → forced to RAG).
        """
        return bool(access.principal_groups)
