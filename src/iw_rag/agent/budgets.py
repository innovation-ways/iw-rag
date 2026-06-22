"""Budget constants and guard functions for the agentic loop.

Provides the two budget-enforcement primitives (``budget_left``,
``consume_retry``) that keep the CRAG/Self-RAG corrective cycle bounded.

The state counter (``retries < max_retries``) is the **primary** brake;
LangGraph's ``recursion_limit=25`` is a secondary safety net that trips if
the counter is ever bypassed.  ``retries`` is incremented in exactly one
place — the ``consume_retry`` node (lld-agentic-loop §2.4, §3.6).

References:
- lld-agentic-loop §2.4 (budget_left / consume_retry signatures)
- task-2.3-grade-documents-and-rewrite §6 (consume_retry as a node)
- R-00105 F5 (demo: 1–2 correction passes)
"""

from __future__ import annotations

from dataclasses import dataclass

from iw_rag.agent.state import GraphState

# ---------------------------------------------------------------------------
# Module-level constants — used in tests and as defaults for AgentBudgets
# ---------------------------------------------------------------------------

DEFAULT_MAX_RETRIES: int = 2
"""Default correction-pass limit (demo: 1–2 per R-00105 F5)."""

RECURSION_LIMIT: int = 25
"""LangGraph super-step safety net; must trip after the state counter."""


# ---------------------------------------------------------------------------
# Budget guard — pure predicate (sync, no I/O)
# ---------------------------------------------------------------------------


def budget_left(state: GraphState) -> bool:
    """Return True when at least one correction pass remains.

    Checks ``state["retries"] < state["max_retries"]``.  Both values must be
    present in state (they are initialised by ``answer_question`` in
    ``graph.py``).

    This is the **primary** budget brake checked on every corrective edge
    (``grade_documents → consume_retry``, ``grade_generation → rewrite``).
    It is a pure predicate — it never mutates state.

    Args:
        state: The current ``GraphState``; reads ``retries`` and ``max_retries``.

    Returns:
        ``True`` if ``retries < max_retries``, ``False`` when the budget is
        exhausted (``retries >= max_retries``).
    """
    retries: int = state.get("retries", 0)
    max_retries: int = state.get("max_retries", DEFAULT_MAX_RETRIES)
    return retries < max_retries


# ---------------------------------------------------------------------------
# AgentBudgets — frozen dataclass for injection / testing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentBudgets:
    """Immutable budget configuration for one graph invocation.

    Attributes:
        max_retries: Maximum correction passes (retrieve+generate share one
            budget counter — incremented in ``consume_retry``, checked in
            ``budget_left``).  Default 2 (R-00105 F5).
        recursion_limit: LangGraph super-step safety net; passed as
            ``config["recursion_limit"]`` to ``graph.ainvoke``.  Default 25.
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    recursion_limit: int = RECURSION_LIMIT
