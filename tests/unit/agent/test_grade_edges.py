"""Unit tests for route_after_grade_documents edge predicate.

Table-driven coverage over every doc_verdict × budget combination, including
the boundary cases at ``retries == max_retries - 1`` (budget remains) and
``retries == max_retries`` (budget exhausted).

Being pure and synchronous, these are the highest-value tests — the loop's
correctness *is* its edge logic.

References:
- task-2.3-grade-documents-and-rewrite §5, Tests to write
- lld-agentic-loop §2.3, §3.6 (budget-aware routing)
- CONVENTIONS.md §1 (chunks: list[RetrievedChunk])
"""

from __future__ import annotations

import asyncio

import pytest

from iw_rag.agent.state import GraphState
from iw_rag.domain.acl import AccessContext

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("hr",),
    is_operator=False,
    request_id="req-grade-edges",
)


def _state(
    doc_verdict: str | None = None,
    retries: int = 0,
    max_retries: int = 2,
) -> GraphState:
    """Build a minimal GraphState for edge predicate tests.

    Args:
        doc_verdict: The CRAG aggregate verdict to inject into state.
        retries: Number of correction passes consumed so far.
        max_retries: Maximum allowed correction passes.

    Returns:
        A ``GraphState`` dict with the specified budget and verdict.
    """
    s: GraphState = {
        "question": "test?",
        "original_question": "test?",
        "access_context": _CTX,
        "retries": retries,
        "max_retries": max_retries,
        "verdicts": [],
        "chunks": [],
        "graded": [],
    }
    if doc_verdict is not None:
        s["doc_verdict"] = doc_verdict  # type: ignore[typeddict-item]
    return s


# ---------------------------------------------------------------------------
# Table-driven: full routing table for route_after_grade_documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "doc_verdict, retries, max_retries, expected_node",
    [
        # correct/ambiguous → always generate (regardless of budget)
        ("correct", 0, 2, "generate"),
        ("correct", 2, 2, "generate"),  # budget exhausted but still → generate
        ("correct", 3, 2, "generate"),
        ("ambiguous", 0, 2, "generate"),
        ("ambiguous", 2, 2, "generate"),
        # incorrect with budget remaining → consume_retry
        ("incorrect", 0, 2, "consume_retry"),
        # Boundary: retries == max_retries - 1 → still has budget → consume_retry
        ("incorrect", 1, 2, "consume_retry"),
        # Boundary: retries == max_retries → budget exhausted → generate (fail-closed seam)
        ("incorrect", 2, 2, "generate"),
        # Boundary: retries > max_retries → budget exhausted → generate
        ("incorrect", 3, 2, "generate"),
        # max_retries=1: boundary tighter
        ("incorrect", 0, 1, "consume_retry"),  # retries < max → still go
        ("incorrect", 1, 1, "generate"),  # retries == max → exhausted
        # max_retries=0: no budget at all → generate immediately
        ("incorrect", 0, 0, "generate"),
    ],
)
def test_route_after_grade_documents_table(
    doc_verdict: str,
    retries: int,
    max_retries: int,
    expected_node: str,
) -> None:
    """Verifies route_after_grade_documents returns the correct next node for every case.

    Args:
        doc_verdict: The CRAG aggregate verdict (correct/ambiguous/incorrect).
        retries: Correction passes consumed so far.
        max_retries: Budget limit.
        expected_node: The expected next node name.
    """
    from iw_rag.agent.edges import route_after_grade_documents

    state = _state(doc_verdict=doc_verdict, retries=retries, max_retries=max_retries)
    result = route_after_grade_documents(state)

    assert result == expected_node, (
        f"doc_verdict={doc_verdict!r}, retries={retries}, max_retries={max_retries}: "
        f"expected={expected_node!r}, got={result!r}"
    )


def test_route_after_grade_documents_is_sync() -> None:
    """Verifies route_after_grade_documents is synchronous (not a coroutine)."""
    from iw_rag.agent.edges import route_after_grade_documents

    state = _state(doc_verdict="correct", retries=0, max_retries=2)
    result = route_after_grade_documents(state)

    assert not asyncio.iscoroutine(result), "route_after_grade_documents must be sync, not async"
    assert isinstance(result, str)


def test_route_after_grade_documents_is_total() -> None:
    """Verifies route_after_grade_documents handles None/missing doc_verdict safely.

    A None verdict (e.g. empty chunks before grading) must not raise — it should
    route to consume_retry (budget permitting) or generate.
    """
    from iw_rag.agent.edges import route_after_grade_documents

    # None doc_verdict should not crash — treat as incorrect
    state = _state(doc_verdict=None, retries=0, max_retries=2)
    result = route_after_grade_documents(state)

    # Must return a valid next-node string
    assert result in ("generate", "consume_retry"), (
        f"Expected 'generate' or 'consume_retry' for None verdict, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Boundary: retries == max_retries - 1 vs max_retries
# ---------------------------------------------------------------------------


def test_boundary_retries_equals_max_minus_one_routes_consume_retry() -> None:
    """At retries == max_retries - 1 the budget is not yet exhausted → consume_retry."""
    from iw_rag.agent.edges import route_after_grade_documents

    max_retries = 2
    state = _state(doc_verdict="incorrect", retries=max_retries - 1, max_retries=max_retries)
    result = route_after_grade_documents(state)

    assert result == "consume_retry", (
        f"Budget boundary: retries={max_retries - 1} < max={max_retries} → "
        f"expected 'consume_retry', got {result!r}"
    )


def test_boundary_retries_equals_max_routes_generate() -> None:
    """At retries == max_retries the budget is exhausted → route to generate (fail-closed seam).

    The spec says: incorrect AND budget exhausted → 'generate' so that the
    generate + grade_generation / fail-closed path (Task 2.4/2.5) handles it.
    """
    from iw_rag.agent.edges import route_after_grade_documents

    max_retries = 2
    state = _state(doc_verdict="incorrect", retries=max_retries, max_retries=max_retries)
    result = route_after_grade_documents(state)

    assert result == "generate", (
        f"Budget boundary: retries={max_retries} == max={max_retries} → "
        f"expected 'generate' (fail-closed seam), got {result!r}"
    )
