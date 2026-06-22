"""Unit tests for the three guard-routing edge predicates.

Parametrized table tests with complete branch coverage for:

- ``route_after_input_guard`` — any non-``"allow"`` must route to ``"reject"``
- ``route_after_input_guard_degraded`` — degraded path, allow → ``"answer"``
- ``route_after_output_guard`` — any non-``"allow"`` must route to ``"fail_closed"``

The security invariant is fail-closed: EVERY non-``"allow"`` value (block, None,
redact, flag, any garbage string) must reach the reject / fail_closed terminal.
Mutating ``if decision == "allow"`` to ``!=`` or changing a target string will
cause every table entry to fail.

References:
- task-3.1-input-guard §5 (route_after_input_guard)
- task-3.3-output-guard-grounding §5 (route_after_output_guard)
- lld-guardrails §3.2, §3.4
"""

from __future__ import annotations

import asyncio

import pytest

from iw_rag.agent.state import GraphState
from iw_rag.domain.acl import AccessContext

_CTX = AccessContext(
    principal_id="u1",
    principal_groups=("security",),
    is_operator=False,
    request_id="req-guard-edges",
)


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------


def _state(
    input_decision: str | None = None,
    output_guard_decision: str | None = None,
) -> GraphState:
    """Build a minimal GraphState for guard-routing edge predicate tests.

    Args:
        input_decision: The guard decision from ``input_guard_node``; ``None``
            means the key is absent from state (tests the missing-key path).
        output_guard_decision: The guard decision from ``output_guard_node``;
            ``None`` means the key is absent from state.

    Returns:
        A ``GraphState`` dict with the relevant guard decision field injected.
    """
    s: GraphState = {
        "question": "Is this safe?",
        "original_question": "Is this safe?",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    if input_decision is not None:
        s["input_decision"] = input_decision  # type: ignore[typeddict-item]
    if output_guard_decision is not None:
        s["output_guard_decision"] = output_guard_decision  # type: ignore[typeddict-item]
    return s


# ---------------------------------------------------------------------------
# route_after_input_guard — table-driven
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_decision, expected_node",
    [
        # allow → normal Adaptive-RAG entrypoint
        ("allow", "router"),
        # block → reject terminal (explicit block decision)
        ("block", "reject"),
        # redact → reject (fail-closed; any non-allow → reject)
        ("redact", "reject"),
        # flag → reject (fail-closed; any non-allow → reject)
        ("flag", "reject"),
        # None / missing key → reject (fail-closed on absent guard decision)
        (None, "reject"),
        # unexpected garbage value → reject (fail-closed on any unknown value)
        ("UNEXPECTED_GARBAGE_VALUE", "reject"),
    ],
)
def test_route_after_input_guard_table(
    input_decision: str | None,
    expected_node: str,
) -> None:
    """Verifies route_after_input_guard maps every decision to the correct next node.

    Args:
        input_decision: The ``state["input_decision"]`` value to inject (or
            absent when ``None``).
        expected_node: The exact routing target string the predicate must return.
    """
    from iw_rag.agent.edges import route_after_input_guard

    state = _state(input_decision=input_decision)
    result = route_after_input_guard(state)

    assert result == expected_node, (
        f"input_decision={input_decision!r}: expected next_node={expected_node!r}, got {result!r}"
    )


def test_route_after_input_guard_allow_returns_exactly_router() -> None:
    """allow decision returns exactly the string 'router', nothing else."""
    from iw_rag.agent.edges import route_after_input_guard

    state = _state(input_decision="allow")
    result = route_after_input_guard(state)

    assert result == "router", f"expected 'router', got {result!r}"


def test_route_after_input_guard_block_returns_exactly_reject() -> None:
    """block decision returns exactly the string 'reject', nothing else."""
    from iw_rag.agent.edges import route_after_input_guard

    state = _state(input_decision="block")
    result = route_after_input_guard(state)

    assert result == "reject", f"expected 'reject', got {result!r}"


def test_route_after_input_guard_missing_key_returns_reject() -> None:
    """Missing input_decision key (state.get returns None) routes to 'reject'."""
    from iw_rag.agent.edges import route_after_input_guard

    # Build state without setting input_decision at all
    state: GraphState = {
        "question": "q",
        "original_question": "q",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    result = route_after_input_guard(state)

    assert result == "reject", (
        f"Missing input_decision must fail-closed to 'reject', got {result!r}"
    )


def test_route_after_input_guard_is_sync() -> None:
    """Verifies route_after_input_guard is synchronous (not a coroutine)."""
    from iw_rag.agent.edges import route_after_input_guard

    state = _state(input_decision="allow")
    result = route_after_input_guard(state)

    assert not asyncio.iscoroutine(result), "route_after_input_guard must be sync, not async"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# route_after_input_guard_degraded — table-driven
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "input_decision, expected_node",
    [
        # allow → degraded answer node (llm=None path uses "answer" not "router")
        ("allow", "answer"),
        # block → reject terminal
        ("block", "reject"),
        # redact → reject (fail-closed)
        ("redact", "reject"),
        # flag → reject (fail-closed)
        ("flag", "reject"),
        # None / missing key → reject (fail-closed on absent guard decision)
        (None, "reject"),
        # unexpected garbage value → reject (fail-closed on any unknown value)
        ("UNEXPECTED_GARBAGE_VALUE", "reject"),
    ],
)
def test_route_after_input_guard_degraded_table(
    input_decision: str | None,
    expected_node: str,
) -> None:
    """Verifies route_after_input_guard_degraded maps every decision to the correct next node.

    Args:
        input_decision: The ``state["input_decision"]`` value to inject (or
            absent when ``None``).
        expected_node: The exact routing target string the predicate must return.
    """
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state = _state(input_decision=input_decision)
    result = route_after_input_guard_degraded(state)

    assert result == expected_node, (
        f"input_decision={input_decision!r}: expected next_node={expected_node!r}, got {result!r}"
    )


def test_route_after_input_guard_degraded_allow_returns_exactly_answer() -> None:
    """allow decision on degraded path returns exactly the string 'answer'."""
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state = _state(input_decision="allow")
    result = route_after_input_guard_degraded(state)

    assert result == "answer", f"expected 'answer' (degraded path), got {result!r}"


def test_route_after_input_guard_degraded_block_returns_exactly_reject() -> None:
    """block decision on degraded path returns exactly the string 'reject'."""
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state = _state(input_decision="block")
    result = route_after_input_guard_degraded(state)

    assert result == "reject", f"expected 'reject', got {result!r}"


def test_route_after_input_guard_degraded_differs_from_normal_on_allow() -> None:
    """Degraded path routes 'allow' to 'answer', not 'router' — the paths differ.

    This is the key distinction between the two predicates: same fail-closed
    invariant, different allow target.
    """
    from iw_rag.agent.edges import (
        route_after_input_guard,
        route_after_input_guard_degraded,
    )

    state = _state(input_decision="allow")

    normal_result = route_after_input_guard(state)
    degraded_result = route_after_input_guard_degraded(state)

    assert normal_result == "router", f"Normal path: allow → 'router', got {normal_result!r}"
    assert degraded_result == "answer", f"Degraded path: allow → 'answer', got {degraded_result!r}"
    assert normal_result != degraded_result, (
        "Normal and degraded predicates must route allow to different targets"
    )


def test_route_after_input_guard_degraded_missing_key_returns_reject() -> None:
    """Missing input_decision key on degraded path routes to 'reject'."""
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state: GraphState = {
        "question": "q",
        "original_question": "q",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    result = route_after_input_guard_degraded(state)

    assert result == "reject", (
        f"Missing input_decision on degraded path must fail-closed to 'reject', got {result!r}"
    )


def test_route_after_input_guard_degraded_is_sync() -> None:
    """Verifies route_after_input_guard_degraded is synchronous (not a coroutine)."""
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state = _state(input_decision="allow")
    result = route_after_input_guard_degraded(state)

    assert not asyncio.iscoroutine(result), (
        "route_after_input_guard_degraded must be sync, not async"
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# route_after_output_guard — table-driven
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output_guard_decision, expected_node",
    [
        # allow → done terminal (grounded + valid citations + PII-redacted)
        ("allow", "done"),
        # block → fail_closed terminal (ungrounded or invalid citation)
        ("block", "fail_closed"),
        # None / missing key → fail_closed (fail-closed on absent guard decision)
        (None, "fail_closed"),
        # redact → fail_closed (fail-closed; any non-allow → fail_closed)
        ("redact", "fail_closed"),
        # flag → fail_closed (fail-closed; any non-allow → fail_closed)
        ("flag", "fail_closed"),
        # unexpected garbage value → fail_closed (fail-closed on any unknown value)
        ("UNEXPECTED_GARBAGE_VALUE", "fail_closed"),
    ],
)
def test_route_after_output_guard_table(
    output_guard_decision: str | None,
    expected_node: str,
) -> None:
    """Verifies route_after_output_guard maps every decision to the correct next node.

    Args:
        output_guard_decision: The ``state["output_guard_decision"]`` value to
            inject (or absent when ``None``).
        expected_node: The exact routing target string the predicate must return.
    """
    from iw_rag.agent.edges import route_after_output_guard

    state = _state(output_guard_decision=output_guard_decision)
    result = route_after_output_guard(state)

    assert result == expected_node, (
        f"output_guard_decision={output_guard_decision!r}: "
        f"expected next_node={expected_node!r}, got {result!r}"
    )


def test_route_after_output_guard_allow_returns_exactly_done() -> None:
    """allow decision returns exactly the string 'done', nothing else."""
    from iw_rag.agent.edges import route_after_output_guard

    state = _state(output_guard_decision="allow")
    result = route_after_output_guard(state)

    assert result == "done", f"expected 'done', got {result!r}"


def test_route_after_output_guard_block_returns_exactly_fail_closed() -> None:
    """block decision returns exactly the string 'fail_closed', nothing else."""
    from iw_rag.agent.edges import route_after_output_guard

    state = _state(output_guard_decision="block")
    result = route_after_output_guard(state)

    assert result == "fail_closed", f"expected 'fail_closed', got {result!r}"


def test_route_after_output_guard_missing_key_returns_fail_closed() -> None:
    """Missing output_guard_decision key routes to 'fail_closed'."""
    from iw_rag.agent.edges import route_after_output_guard

    state: GraphState = {
        "question": "q",
        "original_question": "q",
        "access_context": _CTX,
        "retries": 0,
        "max_retries": 2,
        "verdicts": [],
    }
    result = route_after_output_guard(state)

    assert result == "fail_closed", (
        f"Missing output_guard_decision must fail-closed to 'fail_closed', got {result!r}"
    )


def test_route_after_output_guard_is_sync() -> None:
    """Verifies route_after_output_guard is synchronous (not a coroutine)."""
    from iw_rag.agent.edges import route_after_output_guard

    state = _state(output_guard_decision="allow")
    result = route_after_output_guard(state)

    assert not asyncio.iscoroutine(result), "route_after_output_guard must be sync, not async"
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Cross-predicate: non-allow fail-closed invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision",
    ["block", "redact", "flag", None, "GARBAGE", "", "  ", "ALLOW", "Allow"],
)
def test_input_guard_fail_closed_invariant(decision: str | None) -> None:
    """Every non-allow value routes to 'reject' for route_after_input_guard.

    Verifies the fail-closed invariant holds for every plausible non-allow
    value, including case-variants (ALLOW, Allow) and whitespace strings.
    """
    from iw_rag.agent.edges import route_after_input_guard

    state = _state(input_decision=decision)
    result = route_after_input_guard(state)

    assert result == "reject", (
        f"Fail-closed invariant violated: input_decision={decision!r} → "
        f"expected 'reject', got {result!r}"
    )


@pytest.mark.parametrize(
    "decision",
    ["block", "redact", "flag", None, "GARBAGE", "", "  ", "ALLOW", "Allow"],
)
def test_input_guard_degraded_fail_closed_invariant(decision: str | None) -> None:
    """Every non-allow value routes to 'reject' for route_after_input_guard_degraded.

    Verifies the fail-closed invariant holds on the degraded path for every
    plausible non-allow value.
    """
    from iw_rag.agent.edges import route_after_input_guard_degraded

    state = _state(input_decision=decision)
    result = route_after_input_guard_degraded(state)

    assert result == "reject", (
        f"Fail-closed invariant violated (degraded): input_decision={decision!r} → "
        f"expected 'reject', got {result!r}"
    )


@pytest.mark.parametrize(
    "decision",
    ["block", "redact", "flag", None, "GARBAGE", "", "  ", "ALLOW", "Allow"],
)
def test_output_guard_fail_closed_invariant(decision: str | None) -> None:
    """Every non-allow value routes to 'fail_closed' for route_after_output_guard.

    Verifies the fail-closed invariant holds for every plausible non-allow
    value, including case-variants and whitespace strings.
    """
    from iw_rag.agent.edges import route_after_output_guard

    state = _state(output_guard_decision=decision)
    result = route_after_output_guard(state)

    assert result == "fail_closed", (
        f"Fail-closed invariant violated: output_guard_decision={decision!r} → "
        f"expected 'fail_closed', got {result!r}"
    )
