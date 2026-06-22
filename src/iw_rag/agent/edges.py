"""Conditional edge predicates for the Phase-2/3/4 agentic loop.

All predicates are **pure, synchronous** functions of ``GraphState`` — they
read state and return the name of the next node.  Being pure makes them trivially
unit-testable without a running graph.

Phase-2 status:
- ``route_after_router`` — implemented (Task 2.1; updated Task 2.2 to route to
  ``"retrieve"`` instead of the legacy monolithic ``"answer"`` node).
- ``route_after_grade_documents`` — implemented (Task 2.3; budget-aware CRAG routing).
- ``route_after_grade_generation`` — implemented (Task 2.4/2.5; budget-aware Self-RAG routing).
  Budget-exhausted branches now route to ``"fail_closed"`` (Task 2.5 change from ``"done"``).

Phase-3 additions:
- ``route_after_input_guard`` — implemented (Task 3.1): ``block`` → ``"reject"`` terminal;
  ``allow`` → ``"router"`` (or ``"answer"`` on the llm=None degraded path).
- ``route_after_output_guard`` — implemented (Task 3.3): ``block`` → ``"fail_closed"``
  (ungrounded or invalid citation); ``allow`` → ``"done"`` (verified answer, PII-redacted).

Phase-4 addition (WI-031):
- ``route_after_router`` updated to dispatch ``"visual"`` → ``"visual"`` node when
  ``modes.visual.enabled=True``, otherwise collapse to ``"retrieve"`` (spine).
  Same collapse-pattern as graph/long_context; the intended route is preserved in
  verdicts for router-accuracy metrics.

References:
- lld-agentic-loop §2.3, §3.1, §3.2, §3.5, §3.6
- task-2.1-router-node §4 (route_after_router)
- task-2.2-retrieve-toolnode §4 (graph wiring: router → retrieve)
- task-2.3-grade-documents-and-rewrite §5 (route_after_grade_documents)
- task-2.4-generate-and-grade-generation §5 (route_after_grade_generation)
- task-2.5-budgets-and-fail-closed §2, §3 (budget-exhausted → fail_closed)
- task-3.1-input-guard §5 (route_after_input_guard)
- task-3.3-output-guard-grounding §5 (route_after_output_guard)
- task-4b-visual-rag-route §10 (visual router branch wiring)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk], never Candidate)
"""

from __future__ import annotations

from typing import Literal

from iw_rag.agent.budgets import budget_left
from iw_rag.agent.state import GraphState


def route_after_input_guard(state: GraphState) -> Literal["reject", "router"]:
    """Return the next node name after input_guard_node runs.

    Pure and synchronous (no I/O) — suitable for LangGraph conditional edges.

    Routing table (Task 3.1 / lld-guardrails §3.2):
    - ``input_decision == "block"`` → ``"reject"`` (terminal refusal; no spine
      call, no LLM call — the turn is dead before routing).
    - any other value (``"allow"`` or missing/``None``) → ``"router"`` (the
      normal Adaptive-RAG entry node).

    The predicate is intentionally conservative: any non-``"allow"`` decision
    routes to reject (fail-closed on an unexpected value).

    Args:
        state: The current ``GraphState``; reads ``state["input_decision"]``.

    Returns:
        ``"reject"`` or ``"router"``.
    """
    decision = state.get("input_decision")
    if decision == "allow":
        return "router"
    # block, redact, flag, None, or any unexpected value → reject (fail-closed)
    return "reject"


def route_after_input_guard_degraded(state: GraphState) -> Literal["reject", "answer"]:
    """Degraded-path variant for the ``llm=None`` Phase-2.0 graph.

    When ``llm=None`` the graph has an ``"answer"`` node instead of ``"router"``;
    this predicate mirrors ``route_after_input_guard`` but routes to ``"answer"``
    on allow so the degraded path remains valid.

    Args:
        state: The current ``GraphState``; reads ``state["input_decision"]``.

    Returns:
        ``"reject"`` or ``"answer"``.
    """
    decision = state.get("input_decision")
    if decision == "allow":
        return "answer"
    return "reject"


def route_after_router(
    state: GraphState,
) -> Literal["answer_direct", "retrieve", "visual", "graph", "long_context"]:
    """Return the next node name after the router classifies the question.

    Pure and synchronous (no I/O) — suitable for LangGraph conditional edges.

    Routing table (Phase 2.2+ / Phase 4 WI-031 / WI-029 / WI-032):
    - ``skip``         → ``"answer_direct"`` (parametric answer, no retrieval)
    - ``retrieve``     → ``"retrieve"`` (deterministic spine call → grade_documents → generate)
    - ``graph``        → ``"graph"`` when ``modes.graphrag.enabled=True``; otherwise
                         collapses to ``"retrieve"`` (spine fallback — same pattern
                         as visual/long_context).  The intended route is preserved
                         in verdicts for router-accuracy metrics.
    - ``long_context`` → ``"long_context"`` when ``modes.long_context.enabled=True``
                         AND the node is wired; otherwise collapses to ``"retrieve"``
                         (spine fallback — same collapse pattern as graph/visual).
                         The intended route is preserved in verdicts for metrics.
    - ``visual``       → ``"visual"`` when ``modes.visual.enabled=True``; otherwise
                         collapses to ``"retrieve"`` (spine fallback — same pattern
                         as graph/long_context).  The intended route is preserved
                         in verdicts for router-accuracy metrics.
    - ``None``/missing → ``"retrieve"`` (safe retrieve fallback)

    D18 [DECISION — visual route]: The ``visual`` route is added to the router
    in WI-031.  When ``modes.visual.enabled=false`` (default), the edge collapses
    to ``"retrieve"`` (unchanged default path).  When enabled, the visual node
    runs ``VisualRetriever.retrieve``; on encoder error (VisualEncodeUnavailable),
    the node itself falls back to the spine.

    D22 [DECISION — graph route (WI-029)]: The ``graph`` route dispatches to the
    ``"graph"`` node when ``modes.graphrag.enabled=True`` AND the node is wired
    in the compiled graph.  When disabled (default), the edge collapses to
    ``"retrieve"`` preserving the default path unchanged.  On any error
    (``GraphRouteTimeout`` / ``GraphStoreUnavailable``), the graph node itself
    falls back to the spine so the edge always has a safe path.

    D23 [DECISION — long_context route (WI-032)]: The ``long_context`` route
    dispatches to the ``"long_context"`` node when ``modes.long_context.enabled=True``
    AND the node is wired in the compiled graph.  When disabled (default),
    the edge collapses to ``"retrieve"`` preserving the default path unchanged.
    On any entry failure (ineligible / over-budget / ACL-scoped / attribution-strict),
    the long_context node itself falls back to the spine (entry failure → demote to
    retrieve immediately; generation failure → should_fallback → demote to retrieve).

    Args:
        state: The current ``GraphState``; reads ``state["route"]``.

    Returns:
        The name of the next node: ``"answer_direct"``, ``"retrieve"``,
        ``"visual"`` (only when modes.visual.enabled=True),
        ``"graph"`` (only when modes.graphrag.enabled=True), or
        ``"long_context"`` (only when modes.long_context.enabled=True).
    """
    route = state.get("route")
    if route == "skip":
        return "answer_direct"
    if route == "visual":
        # Check if the visual mode is enabled (config read at edge time).
        # Lazy import to keep this module lightweight (the settings object is
        # cached; this is a very cheap call on the hot path).
        try:
            from iw_rag.settings import get_settings

            cfg = get_settings()
            if cfg.modes.visual.enabled:
                return "visual"
        except Exception:
            pass
        # modes.visual.enabled=false OR config unavailable → collapse to spine.
        return "retrieve"
    if route == "graph":
        # Check if the graphrag mode is enabled (config read at edge time).
        # Same collapse pattern as the visual route: disabled → spine fallback.
        try:
            from iw_rag.settings import get_settings

            cfg = get_settings()
            if cfg.modes.graphrag.enabled:
                return "graph"
        except Exception:
            pass
        # modes.graphrag.enabled=false OR config unavailable → collapse to spine.
        return "retrieve"
    if route == "long_context":
        # Check if the long-context mode is enabled (config read at edge time).
        # Same collapse pattern as visual/graph: disabled → spine fallback.
        # D23 [DECISION]: Long-context mode is off by default (enabled=false)
        # until experiment #8 proves the whole-doc win within the VRAM budget.
        try:
            from iw_rag.settings import get_settings

            cfg = get_settings()
            if cfg.modes.long_context.enabled:
                return "long_context"
        except Exception:
            pass
        # modes.long_context.enabled=false OR config unavailable → collapse to spine.
        return "retrieve"
    # retrieve | None — all enter the retrieve node.
    return "retrieve"


def route_after_grade_documents(
    state: GraphState,
) -> Literal["generate", "consume_retry"]:
    """Return the next node name after grade_documents evaluates retrieved chunks.

    Pure and synchronous (no I/O) — the loop's correctness depends on this
    predicate being total over all possible state combinations.

    Routing table (Task 2.3 / lld-agentic-loop §3.2):
    - ``doc_verdict in ("correct", "ambiguous")`` → ``"generate"``
      (generate from the relevant/mixed subset; ambiguity is recorded in
      ``verdicts`` and forwarded to grade_generation / fail-closed, Tasks 2.4/2.5).
    - ``doc_verdict == "incorrect"`` AND ``budget_left(state)`` → ``"consume_retry"``
      (increment retries → rewrite_query → retrieve — corrective cycle).
    - ``doc_verdict == "incorrect"`` AND budget exhausted → ``"generate"``
      (fail-closed seam: let generate + grade_generation/fail-closed handle it
      in Tasks 2.4/2.5; the graded list will be empty, so the generator/grader
      will refuse to produce a grounded answer).

    Budget boundary semantics:
    - ``retries == max_retries - 1`` → ``budget_left`` is ``True`` → ``"consume_retry"``.
    - ``retries == max_retries``     → ``budget_left`` is ``False`` → ``"generate"``.
    - ``retries > max_retries``      → ``budget_left`` is ``False`` → ``"generate"``.

    Args:
        state: The current ``GraphState``; reads ``doc_verdict``, ``retries``,
            and ``max_retries``.

    Returns:
        The name of the next node: ``"generate"`` or ``"consume_retry"``.
    """
    doc_verdict = state.get("doc_verdict")

    # correct or ambiguous → proceed to generate
    if doc_verdict in ("correct", "ambiguous"):
        return "generate"

    # incorrect (or None / unknown verdict treated as incorrect).
    # Assist mode (allow_ungrounded): the documents don't cover this — don't burn
    # corrective retries chasing grounding that isn't there. Go straight to
    # generate, which falls back to a marked general-knowledge answer. This keeps
    # assist mode responsive (single-pass) for off-corpus questions.
    if state.get("allow_ungrounded"):
        return "generate"

    # Strict mode: spend the correction budget re-retrieving, else fail-close.
    if budget_left(state):
        return "consume_retry"

    # Budget exhausted: route to generate so Task 2.4/2.5 can fail-close.
    # An empty graded list will cause grade_generation to produce 'unsupported'.
    return "generate"


def route_after_output_guard(
    state: GraphState,
) -> Literal["done", "fail_closed"]:
    """Return the next node name after output_guard_node runs.

    Pure and synchronous (no I/O) — suitable for LangGraph conditional edges.

    Routing table (Task 3.3 / lld-guardrails §3.4):
    - ``output_guard_decision == "allow"`` → ``"done"``
      (grounded + valid citations + PII-redacted; proceed to terminal emit).
    - ``output_guard_decision == "block"`` OR any unexpected value → ``"fail_closed"``
      (ungrounded or invalid citation; refuse, never emit).

    Citation failure is a hard deterministic block — it cannot be overridden by
    the groundedness check.  The predicate is intentionally conservative: any
    non-``"allow"`` value routes to fail_closed (fail-closed on unexpected values).

    Args:
        state: The current ``GraphState``; reads ``state["output_guard_decision"]``.

    Returns:
        ``"done"`` or ``"fail_closed"``.
    """
    decision = state.get("output_guard_decision")
    if decision == "allow":
        return "done"
    # block, None, or any unexpected value → fail_closed (fail-closed invariant)
    return "fail_closed"


def route_after_grade_generation(
    state: GraphState,
) -> Literal["output_guard", "consume_retry", "fail_closed"]:
    """Return the next node name after grade_generation grades the draft.

    Pure and synchronous (no I/O) — the loop's correctness depends on this
    predicate being total over all possible state combinations.

    Routing table (Task 2.4/2.5/3.3 / lld-agentic-loop §3.5):
    - ``gen_verdict == "grounded"`` → ``"output_guard"``
      (both ISSUP and ISUSE passed → output guard checks citation + PII).
      Task 3.3: the grounded path now flows through output_guard before done.
      The output_guard reads ``state["gen_verdict"]`` for groundedness (ONE
      authority, no double model call) and routes to ``done`` or ``fail_closed``.
    - ``gen_verdict in ("unsupported", "not_answering")`` AND ``budget_left``
      → ``"consume_retry"`` (recycle: consume_retry → rewrite_query → retrieve).
    - ``gen_verdict in ("unsupported", "not_answering")`` AND budget exhausted
      → ``"fail_closed"`` (Task 2.5: dedicated terminal refusal node; NEVER an
      ungrounded answer).
    - ``gen_verdict is None`` → treated as non-grounded; budget checked.

    Budget boundary semantics (mirrors route_after_grade_documents):
    - ``retries == max_retries - 1`` → ``budget_left`` is ``True`` → ``"consume_retry"``.
    - ``retries == max_retries``     → ``budget_left`` is ``False`` → ``"fail_closed"``.
    - ``retries > max_retries``      → ``budget_left`` is ``False`` → ``"fail_closed"``.

    Args:
        state: The current ``GraphState``; reads ``gen_verdict``, ``retries``,
            and ``max_retries``.

    Returns:
        The name of the next node: ``"output_guard"``, ``"consume_retry"``, or
        ``"fail_closed"``.
    """
    gen_verdict = state.get("gen_verdict")

    # Grounded → proceed to output_guard (Task 3.3: inserted between grade_generation
    # and done).  output_guard reads state["gen_verdict"] for groundedness (ONE authority)
    # and routes to done (allow) or fail_closed (block) via route_after_output_guard.
    if gen_verdict == "grounded":
        return "output_guard"

    # Not grounded (unsupported / not_answering / None).
    # Assist mode (allow_ungrounded): the generate node has produced a marked
    # general-knowledge answer — emit it via output_guard (which treats
    # groundedness as informational in assist mode) without spending retries.
    # This keeps assist mode single-pass and responsive.
    if state.get("allow_ungrounded"):
        return "output_guard"

    # Strict mode: if budget remains, recycle through
    # consume_retry → rewrite_query → retrieve to try to ground the answer.
    if budget_left(state):
        return "consume_retry"

    # Budget exhausted and still ungrounded → fail_closed terminal: enforce the
    # no-ungrounded-answer invariant (lld-agentic-loop §7).
    return "fail_closed"
