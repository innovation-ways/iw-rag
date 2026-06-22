"""Trajectory scoring for the agentic loop — structural invariants + labels.

Scores the agent's *decision path* (the append-only verdict trail + route +
retry budget), not just its final answer. The four structural invariants
(I1-I4) need no gold annotation; route accuracy and expected-disposition are
scored only when the golden query supplies a label. Pure / sync / CPU — mirrors
:mod:`iw_rag.eval.retrieval_metrics`. No I/O, no model calls.

The invariant definitions are pinned to the *live* edge wiring (agent/edges.py,
agent/graph.py), which differs from the original design note in two ways:

- The ``skip`` route's ``answer_direct`` node flows into ``grade_generation``
  (``answer_direct → grade_generation``), so a ``skip`` trajectory legitimately
  carries a generation grade. I4 therefore forbids only ``grade_documents`` on
  the skip path (no retrieval ran), not ``grade_generation``.
- The ``long_context`` route bypasses ``grade_documents`` when its mode is
  enabled (``long_context → generate``) but collapses to the spine — *with*
  ``grade_documents`` — when disabled. I4 therefore only asserts the
  gd-before-gg ordering for it, never ``grade_documents`` presence.
- Only the grader nodes (``grade_documents``/``grade_generation``) stamp
  ``Verdict.attempt`` with the current retry pass (they read ``state["retries"]``);
  ``router``/``retrieve``/``rewrite_query``/``generate`` leave it at the default
  ``0``. A monotonicity check over *every* verdict's ``attempt`` therefore
  false-positives on those interleaved zeros, so I3 evaluates monotonicity over
  the **grader** attempts (the ones that carry the real pass index) plus the
  budget bound ``max(grader attempt) <= retries <= max_retries``. (Verified live
  on the demo corpus, 2026-06-14.)

See: docs/build/03-tasks/trajectory-eval/ (rationale + invariant definitions),
lld-eval-observability §6 (experiment #9), lld-agentic-loop §3 (the edge grammar).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

# Verdict ``value`` strings that explain a refusal (I2).
_NEGATIVE_VALUES = frozenset({"unsupported", "not_answering", "incorrect", "block", "blocked"})
# Terminal verdict ``kind`` strings that are refusals by construction (I2).
_TERMINAL_REFUSAL_KINDS = frozenset({"fail_closed", "reject"})
# Routes whose retrieval family always flows through ``grade_documents`` (I4):
# the enabled mode node → sanitize → grade_documents, or the collapsed spine path.
_GRADED_RETRIEVE_FAMILY = frozenset({"retrieve", "graph", "visual"})
# Verdict kinds that reliably stamp ``attempt`` with the current retry pass (I3):
# they read ``state["retries"]`` at grade time; other nodes leave it at 0.
_RETRY_PASS_KINDS = frozenset({"grade_documents", "grade_generation"})


@dataclass(frozen=True, slots=True)
class TrajectoryScore:
    """Per-query trajectory score.

    Attributes:
        grounded_implies_grade: I1 — answered+grounded (non-skip) was preceded by
            a ``grade_generation == "grounded"`` verdict.
        refuse_implies_cause: I2 — a refusal has an explicit negative cause.
        retry_budget_respected: I3 — ``attempt`` non-decreasing and
            ``0 <= retries <= max_retries``.
        route_consistent: I4 — route/verdict-kind ordering holds.
        trajectory_valid: AND of I1-I4 — the binary headline signal.
        route_correct: Whether the router matched the gold expectation, or None
            when no route label is available.
        disposition_correct: Whether the terminal disposition matched the gold
            expectation, or None when unlabeled.
        retries: Correction passes consumed (carried through for aggregation).
    """

    grounded_implies_grade: bool
    refuse_implies_cause: bool
    retry_budget_respected: bool
    route_consistent: bool
    trajectory_valid: bool
    route_correct: bool | None
    disposition_correct: bool | None
    retries: int


def _kind_indices(trajectory: Sequence[dict], kind: str) -> list[int]:  # type: ignore[type-arg]
    """Indices at which ``kind`` appears in the trajectory (in order)."""
    return [i for i, v in enumerate(trajectory) if v.get("kind") == kind]


def _expected_route(query_type: str | None, expected_route: str | None) -> str | None:
    """Resolve the gold-expected route.

    An explicit ``expected_route`` label wins; otherwise fall back to the coarse
    rule shared with ``experiments._compute_router_accuracy`` (``trivial`` →
    ``skip``, else ``retrieve``). Returns None when neither a label nor a
    query_type is available.

    Args:
        query_type: The golden query's type, if any.
        expected_route: An explicit route label on the golden query, if any.

    Returns:
        The expected route string, or None when unlabeled.
    """
    if expected_route is not None:
        return expected_route
    if query_type is None:
        return None
    return "skip" if query_type == "trivial" else "retrieve"


def _route_consistent(route: str | None, gd_idx: list[int], gg_idx: list[int]) -> bool:
    """Evaluate I4 (route/verdict-kind ordering) for one trajectory.

    Args:
        route: The router's intended branch.
        gd_idx: Ordered indices of ``grade_documents`` verdicts.
        gg_idx: Ordered indices of ``grade_generation`` verdicts.

    Returns:
        Whether the route/verdict ordering is consistent with the live grammar.
    """
    if route == "skip":
        # skip is node-resolved (skip_enabled is applied before state is set), so
        # the path is answer_direct → grade_generation with no retrieval: a
        # grade_documents verdict here is a hard inconsistency.
        return not gd_idx
    if route in _GRADED_RETRIEVE_FAMILY:
        # These always pass through grade_documents (enabled mode node or the
        # collapsed spine), so a generation grade must be preceded by a doc grade.
        if gg_idx:
            return bool(gd_idx) and gd_idx[0] < gg_idx[0]
        return True
    if route == "long_context":
        # Bypasses grade_documents when enabled, collapses to the spine (with it)
        # when disabled — so only assert ordering when both kinds appear.
        return (not gd_idx) or (not gg_idx) or (gd_idx[0] < gg_idx[0])
    # Unknown / None route — no ordering claim.
    return True


def score_trajectory(
    trajectory: Sequence[dict],  # type: ignore[type-arg]
    *,
    route: str | None,
    retries: int,
    max_retries: int,
    disposition: str | None,
    grounded: bool,
    query_type: str | None = None,
    expected_route: str | None = None,
    expected_disposition: str | None = None,
) -> TrajectoryScore:
    """Score one agent trajectory against structural invariants and optional labels.

    Args:
        trajectory: Ordered verdict dicts (``kind``/``value``/``attempt``) from
            ``QueryOutcome.diagnostics["trajectory"]``.
        route: The router's intended branch (``diagnostics["route"]``).
        retries: Correction passes consumed (``diagnostics["retries"]``).
        max_retries: The retry budget (``diagnostics["max_retries"]``).
        disposition: Terminal ``"answered"`` / ``"refused"``.
        grounded: True groundedness (``QueryOutcome.answer.grounded``; folds in
            the output-guard verdict).
        query_type: Gold query type, for the coarse route expectation (optional).
        expected_route: Explicit gold route label (optional; wins over query_type).
        expected_disposition: Explicit gold terminal disposition (optional).

    Returns:
        A :class:`TrajectoryScore`.
    """
    gen_grade_values = [v.get("value") for v in trajectory if v.get("kind") == "grade_generation"]
    # Only the grader nodes stamp the real retry pass into ``attempt``; other
    # nodes leave it at 0, so monotonicity is evaluated over the grader subsequence.
    grader_attempts = [
        int(v.get("attempt", 0)) for v in trajectory if v.get("kind") in _RETRY_PASS_KINDS
    ]

    # I1 — answered+grounded (non-skip) ⇒ a grounded generation grade exists.
    if disposition == "answered" and grounded and route != "skip":
        grounded_implies_grade = "grounded" in gen_grade_values
    else:
        grounded_implies_grade = True

    # I2 — refusal ⇒ an explicit negative cause in the trail.
    if disposition == "refused":
        has_negative_value = any(v.get("value") in _NEGATIVE_VALUES for v in trajectory)
        has_terminal = any(v.get("kind") in _TERMINAL_REFUSAL_KINDS for v in trajectory)
        refuse_implies_cause = has_negative_value or has_terminal
    else:
        refuse_implies_cause = True

    # I3 — grader attempts non-decreasing, and no grader claims a pass beyond the
    # consumed budget; the budget itself is bounded by max_retries.
    monotonic_grades = grader_attempts == sorted(grader_attempts)
    max_grader_attempt = max(grader_attempts) if grader_attempts else 0
    retry_budget_respected = (
        monotonic_grades and 0 <= retries <= max_retries and max_grader_attempt <= retries
    )

    # I4 — route/verdict-kind ordering.
    route_consistent = _route_consistent(
        route,
        _kind_indices(trajectory, "grade_documents"),
        _kind_indices(trajectory, "grade_generation"),
    )

    trajectory_valid = (
        grounded_implies_grade
        and refuse_implies_cause
        and retry_budget_respected
        and route_consistent
    )

    exp_route = _expected_route(query_type, expected_route)
    route_correct = None if exp_route is None or route is None else (route == exp_route)
    disposition_correct = (
        None if expected_disposition is None else (disposition == expected_disposition)
    )

    return TrajectoryScore(
        grounded_implies_grade=grounded_implies_grade,
        refuse_implies_cause=refuse_implies_cause,
        retry_budget_respected=retry_budget_respected,
        route_consistent=route_consistent,
        trajectory_valid=trajectory_valid,
        route_correct=route_correct,
        disposition_correct=disposition_correct,
        retries=retries,
    )
