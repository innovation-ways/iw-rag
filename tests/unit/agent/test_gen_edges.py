"""Unit tests for route_after_grade_generation edge predicate.

Table-driven coverage over every GenVerdict × budget combination, including
the boundary cases at ``retries == max_retries - 1`` (budget remains) and
``retries == max_retries`` (budget exhausted).

Being pure and synchronous, these are the highest-value tests — the loop's
correctness *is* its edge logic.

Task 2.5 change: budget-exhausted branches now route to ``"fail_closed"``
instead of the ``"done"`` seam.  All table entries updated accordingly.

Task 3.3 change: grounded path now routes to ``"output_guard"`` (not ``"done"``)
so the output guard can run citation validation + PII redaction before done.
The output_guard reads state["gen_verdict"] (ONE authority) and routes to done
or fail_closed via route_after_output_guard.

References:
- task-2.4-generate-and-grade-generation §5
- task-2.5-budgets-and-fail-closed §2 (fail_closed replaces done seam)
- task-3.3-output-guard-grounding §5 (output_guard inserted before done)
- lld-agentic-loop §2.3, §3.5, §3.6
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
    request_id="req-gen-edges",
)


def _state(
    gen_verdict: str | None = None,
    retries: int = 0,
    max_retries: int = 2,
) -> GraphState:
    """Build a minimal GraphState for edge predicate tests.

    Args:
        gen_verdict: The GenVerdict to inject into state.
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
    if gen_verdict is not None:
        s["gen_verdict"] = gen_verdict  # type: ignore[typeddict-item]
    return s


# ---------------------------------------------------------------------------
# Table-driven: full routing table for route_after_grade_generation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gen_verdict, retries, max_retries, expected_node",
    [
        # grounded → output_guard (Task 3.3: output_guard inserted before done)
        ("grounded", 0, 2, "output_guard"),
        ("grounded", 2, 2, "output_guard"),  # budget exhausted but still → output_guard
        ("grounded", 3, 2, "output_guard"),
        # unsupported with budget remaining → consume_retry
        ("unsupported", 0, 2, "consume_retry"),
        # Boundary: retries == max_retries - 1 → still has budget → consume_retry
        ("unsupported", 1, 2, "consume_retry"),
        # Boundary: retries == max_retries → budget exhausted → fail_closed (Task 2.5)
        ("unsupported", 2, 2, "fail_closed"),
        # Boundary: retries > max_retries → budget exhausted → fail_closed
        ("unsupported", 3, 2, "fail_closed"),
        # not_answering with budget remaining → consume_retry
        ("not_answering", 0, 2, "consume_retry"),
        ("not_answering", 1, 2, "consume_retry"),
        # not_answering at budget exhaustion → fail_closed (Task 2.5)
        ("not_answering", 2, 2, "fail_closed"),
        ("not_answering", 3, 2, "fail_closed"),
        # max_retries=1: tighter boundary
        ("unsupported", 0, 1, "consume_retry"),  # retries < max → recycle
        ("unsupported", 1, 1, "fail_closed"),  # retries == max → exhausted → fail_closed
        # max_retries=0: no budget at all → terminal immediately → fail_closed
        ("unsupported", 0, 0, "fail_closed"),
        ("not_answering", 0, 0, "fail_closed"),
    ],
)
def test_route_after_grade_generation_table(
    gen_verdict: str,
    retries: int,
    max_retries: int,
    expected_node: str,
) -> None:
    """Verifies route_after_grade_generation returns the correct next node for every case.

    Args:
        gen_verdict: The GenVerdict (grounded/unsupported/not_answering).
        retries: Correction passes consumed so far.
        max_retries: Budget limit.
        expected_node: The expected next node name.
    """
    from iw_rag.agent.edges import route_after_grade_generation

    state = _state(gen_verdict=gen_verdict, retries=retries, max_retries=max_retries)
    result = route_after_grade_generation(state)

    assert result == expected_node, (
        f"gen_verdict={gen_verdict!r}, retries={retries}, max_retries={max_retries}: "
        f"expected={expected_node!r}, got={result!r}"
    )


def test_route_after_grade_generation_is_sync() -> None:
    """Verifies route_after_grade_generation is synchronous (not a coroutine)."""
    from iw_rag.agent.edges import route_after_grade_generation

    state = _state(gen_verdict="grounded", retries=0, max_retries=2)
    result = route_after_grade_generation(state)

    assert not asyncio.iscoroutine(result), "route_after_grade_generation must be sync, not async"
    assert isinstance(result, str)


def test_route_after_grade_generation_is_total() -> None:
    """Verifies route_after_grade_generation handles None/missing gen_verdict safely.

    A None verdict must not raise — treat as non-grounded and check budget.
    """
    from iw_rag.agent.edges import route_after_grade_generation

    state = _state(gen_verdict=None, retries=0, max_retries=2)
    result = route_after_grade_generation(state)

    # Must return a valid next-node string — not raise
    assert result in ("output_guard", "consume_retry", "fail_closed"), (
        f"Expected a valid next node for a None verdict, got {result!r}"
    )


def test_route_after_grade_generation_grounded_always_output_guard() -> None:
    """gen_verdict='grounded' → 'output_guard' regardless of budget (Task 3.3).

    Task 3.3 inserts output_guard between grade_generation and done so the output
    guard can run citation validation + PII redaction before the final emit.
    """
    from iw_rag.agent.edges import route_after_grade_generation

    # Even at retries=0 (budget plenty) grounded → output_guard (Task 3.3)
    for retries in (0, 1, 2, 3):
        state = _state(gen_verdict="grounded", retries=retries, max_retries=2)
        result = route_after_grade_generation(state)
        assert result == "output_guard", (
            f"grounded must always route to 'output_guard' (Task 3.3), "
            f"got {result!r} (retries={retries})"
        )


def test_boundary_gen_retries_equals_max_minus_one_routes_consume_retry() -> None:
    """At retries == max_retries - 1 budget is not exhausted → consume_retry."""
    from iw_rag.agent.edges import route_after_grade_generation

    max_retries = 2
    state = _state(gen_verdict="unsupported", retries=max_retries - 1, max_retries=max_retries)
    result = route_after_grade_generation(state)

    assert result == "consume_retry", (
        f"Budget boundary: retries={max_retries - 1} < max={max_retries} → "
        f"expected 'consume_retry', got {result!r}"
    )


def test_boundary_gen_retries_equals_max_routes_fail_closed() -> None:
    """At retries == max_retries budget is exhausted → 'fail_closed' (Task 2.5).

    Task 2.4 used 'done' as a seam; Task 2.5 repoints to 'fail_closed' — the
    dedicated terminal refusal node.
    """
    from iw_rag.agent.edges import route_after_grade_generation

    max_retries = 2
    for verdict in ("unsupported", "not_answering"):
        state = _state(gen_verdict=verdict, retries=max_retries, max_retries=max_retries)
        result = route_after_grade_generation(state)

        assert result == "fail_closed", (
            f"Budget exhausted: gen_verdict={verdict!r}, retries={max_retries} → "
            f"expected 'fail_closed', got {result!r}"
        )
