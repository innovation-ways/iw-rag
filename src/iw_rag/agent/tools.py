"""Thin async node factory that wraps the RetrievalSpine as a LangGraph node.

**Design deviation from LLD §2.2 / §6.2:**
The LLD sketches ``retrieve = ToolNode([spine_tool])`` using LangGraph's
prebuilt ``ToolNode``.  That prebuilt form expects the *generator* to emit a
tool-call message (i.e. the LLM decides to call the spine as a tool).  In the
Phase-2 loop the retrieval step is **deterministically triggered by the graph
edge** ``route_after_router → retrieve``, not by an LLM tool-call.  Using the
prebuilt ``ToolNode`` would require a tool-call round-trip with no benefit and
break the typed contract.

The recommended approach (task-2.2-retrieve-toolnode §1 [ASSUMPTION]) is a
**thin async node** returned by ``make_retrieve_node``:
- Calls the spine directly with the typed ``(query, ctx)`` interface.
- Maps ``RetrievalResult.chunks → state["chunks"]``.
- Asserts the ACL invariant (``acl_applied is True``).
- Records a retrieval diagnostic in ``verdicts``.
- Propagates ``RetrievalError`` without swallowing it.

References:
- task-2.2-retrieve-toolnode §1, §2, §5 (error propagation)
- lld-agentic-loop §6.2 (spine as opaque callable), §7 (error handling)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate), §6 (acl_applied)
- ADR-002 (spine as opaque callable returning RetrievalResult)
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Coroutine
from typing import Any

from iw_rag.agent.state import GraphState, Verdict
from iw_rag.domain.errors import AuthorizationError, RetrievalError
from iw_rag.guardrails.acl import acl_provenance, assert_hotpath_acl
from iw_rag.observability.tracing import span

logger = logging.getLogger(__name__)


def make_retrieve_node(
    spine: Any,
) -> Callable[[GraphState], Coroutine[Any, Any, dict[str, Any]]]:
    """Create a thin async node that retrieves via the spine and enforces ACL.

    The spine is treated as an opaque callable (ADR-002) — only the typed
    interface ``async retrieve(query: str, ctx: AccessContext) -> RetrievalResult``
    is assumed.

    The ``access_context`` is threaded from ``state["access_context"]`` into the
    spine call **unchanged** — no code path may derive the ACL predicate from the
    question text or any model output (CONVENTIONS.md §6, R-00108).

    **ACL invariant (non-negotiable):** if the returned ``RetrievalResult`` has
    ``acl_applied is not True`` this node raises ``RetrievalError`` immediately.
    An empty result (``chunks=[]``) with ``acl_applied=True`` is legitimate —
    ACL filtered everything out — and the node returns ``chunks=[]`` without
    error (downstream fails closed, never widens the ACL).

    **Design note:** this is a thin async node, not ``ToolNode([spine_tool])``.
    The prebuilt ``ToolNode`` assumes the LLM emits a tool-call message, which
    does not apply when retrieval is triggered deterministically by the graph
    edge.  The ``@tool`` / ``ToolNode`` form is reserved for if/when the
    generator is given retrieval as a callable tool (R-00103 F4).  This
    deviation from the LLD's literal wording is intentional and documented here
    per task-2.2-retrieve-toolnode §1 [ASSUMPTION].

    Args:
        spine: The retrieval spine — any duck-type with
            ``async retrieve(query: str, ctx: AccessContext) -> RetrievalResult``.

    Returns:
        An async node function with signature
        ``(state: GraphState) -> dict[str, Any]``, suitable for
        ``StateGraph.add_node("retrieve", ...)``.
    """

    async def retrieve(state: GraphState) -> dict[str, Any]:
        """Retrieve ACL-filtered chunks from the spine for the current question.

        Reads ``state["question"]`` and ``state["access_context"]``; writes
        ``state["chunks"]`` and appends one retrieval entry to ``state["verdicts"]``.

        Args:
            state: Current graph state.

        Returns:
            Partial state dict with ``chunks`` (``list[RetrievedChunk]``) and
            ``verdicts`` (one retrieval diagnostic entry appended).

        Raises:
            RetrievalError: If ``spine.retrieve`` raises, or if the returned
                ``RetrievalResult.acl_applied is not True`` (ACL invariant).
        """
        question: str = state.get("question", "")
        ctx = state["access_context"]  # ACL source — passed through UNCHANGED

        # One wrapping span so retrieval shows as a single step in the per-request
        # timing breakdown (its fine-grained spine sub-spans nest inside it).
        with span("iw_rag.agent.retrieve", {"node": "retrieve"}):
            result = await spine.retrieve(question, ctx)

            # -------------------------------------------------------------------
            # ACL invariant: acl_applied MUST be True on the hot path.
            # assert_hotpath_acl centralises the named security guard (guardrails/acl.py).
            # The only sanctioned carve-out (long-context whole-doc route) does not
            # flow through this node (CONVENTIONS.md §6).  Any other False signals
            # a bug — assert_hotpath_acl raises AuthorizationError rather than
            # silently serving un-ACL'd chunks.
            # -------------------------------------------------------------------
            route: str = state.get("route") or "retrieve"
            try:
                assert_hotpath_acl(result, route=route)
            except AuthorizationError as exc:
                # Re-raise as RetrievalError so the agent's error-handling path
                # (which catches RetrievalError) stays unchanged.
                raise RetrievalError(str(exc)) from exc

            # -------------------------------------------------------------------
            # Thread ACL provenance into diagnostics so Task 3.4 (audit log) can
            # persist the entitlement set actually applied, without recomputing ACL.
            # -------------------------------------------------------------------
            prov = acl_provenance(ctx)
            diagnostics: dict[str, object] = dict(result.diagnostics)
            diagnostics["acl"] = {
                "principal_groups_hash": prov.principal_groups_hash,
                "acl_applied": prov.acl_applied,
                "mechanism": prov.mechanism,
            }

            logger.debug(
                "retrieve: question=%r → %d chunks (retriever=%r, candidate_count=%d, acl_hash=%r)",
                question[:80],
                len(result.chunks),
                result.retriever,
                result.candidate_count,
                prov.principal_groups_hash,
            )

            # ------------------------------------------------------------------
            # Chat memory (conversation-as-source): prepend the prior-turn
            # synthetic chunks so the agent can ground a follow-up in what was
            # said earlier and CITE the originating turn.  They are placed FIRST
            # so the grader's token-budget fit (which keeps highest-ranked head,
            # drops the tail) never evicts them, and they carry a high score for
            # the same reason.  These chunks are principal-scoped by construction
            # (loaded only for the owning conversation), so the ACL keystone is
            # untouched — they never flow through the spine's ACL pre-filter.
            # ``sanitize_node`` exempts them (kind=="conversation") so the user's
            # own words survive PII redaction / datamarking verbatim.
            # ------------------------------------------------------------------
            history_chunks = state.get("history_chunks") or []
            merged_chunks = [*history_chunks, *result.chunks]

            # Record a retrieval diagnostic in the append-only verdict trail.
            verdict = Verdict(
                kind="retrieve",
                value="ok",
                reason=(
                    f"retriever={result.retriever}; "
                    f"returned={len(result.chunks)}; "
                    f"candidates={result.candidate_count}; "
                    f"history={len(history_chunks)}"
                ),
            )

            return {
                "chunks": merged_chunks,  # list[RetrievedChunk] — NEVER Candidate
                "verdicts": [verdict],
                "diagnostics": diagnostics,
            }

    return retrieve
