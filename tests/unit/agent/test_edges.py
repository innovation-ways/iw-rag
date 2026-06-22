"""Unit tests for the route_after_router edge predicate.

Table-driven coverage over every Route value plus the missing-route fallback.
These are the highest-value tests: the loop's correctness *is* its edge logic.

References:
- lld-agentic-loop §2.3, §3.1
- task-2.1-router-node §4 (route_after_router)
"""

from __future__ import annotations

import pytest

from iw_rag.domain.acl import AccessContext

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("finance",),
    is_operator=False,
    request_id="req-edges",
)


@pytest.mark.parametrize(
    "route, expected_node",
    [
        # skip → answer_direct (the skip branch)
        ("skip", "answer_direct"),
        # retrieve → "retrieve" (the deterministic spine node, Task 2.2)
        ("retrieve", "retrieve"),
        # graph and long_context collapse to "retrieve" in Phase 2
        ("graph", "retrieve"),
        ("long_context", "retrieve"),
        # None / missing → falls back to retrieve branch
        (None, "retrieve"),
    ],
)
def test_route_after_router_total_function(route: str | None, expected_node: str) -> None:
    """Verifies route_after_router maps every Route value to the correct next node.

    This is the total-coverage test for the edge predicate — every branch,
    including the Phase-2 collapse of graph/long_context to retrieve.

    Args:
        route: The ``state["route"]`` value to inject.
        expected_node: The expected node name returned by the predicate.
    """
    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    state: GraphState = {
        "question": "test?",
        "original_question": "test?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    if route is not None:
        state["route"] = route  # type: ignore[assignment]

    result = route_after_router(state)
    assert result == expected_node, (
        f"route={route!r}: expected next_node={expected_node!r}, got {result!r}"
    )


def test_route_after_router_is_sync() -> None:
    """Verifies route_after_router is synchronous (not a coroutine)."""
    import asyncio

    from iw_rag.agent.edges import route_after_router
    from iw_rag.agent.state import GraphState

    state: GraphState = {
        "question": "q",
        "original_question": "q",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
        "route": "retrieve",  # type: ignore[assignment]
    }
    result = route_after_router(state)
    # Must NOT be a coroutine
    assert not asyncio.iscoroutine(result), "route_after_router must be sync, not async"
    assert isinstance(result, str)
