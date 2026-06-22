"""Unit tests for the trajectory scorer (src/iw_rag/eval/trajectory_metrics.py).

TDD RED-first: these cover the four structural invariants (I1-I4), their
truth-table edges, and the optional label-based checks (route / disposition).
The scorer is pure / sync / CPU — every test is a synthetic trajectory dict; no
DB, no GPU, no model calls.

Invariant edge cases verified against the *live* edge wiring (agent/edges.py,
agent/graph.py), which deviates from the original design note in two ways the
scorer must honour:

- The ``skip`` route's ``answer_direct`` node DOES flow into ``grade_generation``
  (graph.py: ``answer_direct → grade_generation``), so I4-skip asserts only the
  absence of ``grade_documents`` — a ``grade_generation`` verdict is legal.
- The ``long_context`` route BYPASSES ``grade_documents`` when its mode is
  enabled (graph.py: ``long_context → generate``) but collapses to the spine
  (with ``grade_documents``) when disabled, so I4 only asserts the gd-before-gg
  ordering for it, never gd *presence*.

References:
- docs/build/03-tasks/trajectory-eval/03-design-and-implementation.md §2
- src/iw_rag/agent/edges.py (the legal-trajectory grammar)
"""

from __future__ import annotations

from iw_rag.eval.trajectory_metrics import TrajectoryScore, score_trajectory


def _v(kind: str, value: str = "", attempt: int = 0) -> dict:  # type: ignore[type-arg]
    """Build one trajectory verdict dict (``kind``/``value``/``attempt``)."""
    return {"kind": kind, "value": value, "score": None, "attempt": attempt}


# ---------------------------------------------------------------------------
# I1 — grounded_implies_grade
# ---------------------------------------------------------------------------


def test_i1_pass_grounded_answer_has_generation_grade() -> None:
    """answered+grounded on the retrieve route with a grounded gen grade → I1 passes."""
    traj = [
        _v("router", "retrieve"),
        _v("grade_documents", "correct"),
        _v("grade_generation", "grounded"),
        _v("output_guard", "allow"),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.grounded_implies_grade is True
    assert s.trajectory_valid is True


def test_i1_fail_grounded_answer_without_generation_grade() -> None:
    """answered+grounded but no grade_generation=grounded verdict → I1 fails."""
    traj = [
        _v("router", "retrieve"),
        _v("grade_documents", "correct"),
        _v("output_guard", "allow"),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.grounded_implies_grade is False
    assert s.trajectory_valid is False


def test_i1_exempt_skip_route() -> None:
    """answered+grounded but route=skip (parametric) is exempt from I1."""
    traj = [_v("router", "skip"), _v("answer_direct", "parametric")]
    s = score_trajectory(
        traj, route="skip", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.grounded_implies_grade is True
    assert s.trajectory_valid is True


def test_i1_vacuous_when_not_grounded() -> None:
    """answered but grounded=False (assist mode) satisfies I1 vacuously."""
    traj = [
        _v("router", "retrieve"),
        _v("grade_documents", "correct"),
        _v("grade_generation", "unsupported"),
        _v("output_guard", "allow"),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=False
    )
    assert s.grounded_implies_grade is True


# ---------------------------------------------------------------------------
# I2 — refuse_implies_cause
# ---------------------------------------------------------------------------


def test_i2_pass_fail_closed_with_unsupported() -> None:
    """refused with an unsupported gen grade + fail_closed terminal → I2 passes."""
    traj = [
        _v("router", "retrieve"),
        _v("grade_documents", "correct"),
        _v("grade_generation", "unsupported"),
        _v("fail_closed", "refused"),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=2, max_retries=2, disposition="refused", grounded=False
    )
    assert s.refuse_implies_cause is True
    assert s.trajectory_valid is True


def test_i2_pass_reject_on_input_block() -> None:
    """refused via input_guard=block + reject terminal → I2 passes."""
    traj = [_v("input_guard", "block"), _v("reject", "refused")]
    s = score_trajectory(
        traj, route=None, retries=0, max_retries=2, disposition="refused", grounded=False
    )
    assert s.refuse_implies_cause is True


def test_i2_fail_silent_refusal() -> None:
    """refused with no negative value and no terminal refusal kind → I2 fails."""
    traj = [_v("router", "retrieve"), _v("grade_documents", "correct")]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="refused", grounded=False
    )
    assert s.refuse_implies_cause is False
    assert s.trajectory_valid is False


# ---------------------------------------------------------------------------
# I3 — retry_budget_respected
# ---------------------------------------------------------------------------


def test_i3_pass_monotonic_attempts_within_budget() -> None:
    """Non-decreasing attempts [0,0,1,1] with retries 1 <= max 2 → I3 passes."""
    traj = [
        _v("router", "retrieve", attempt=0),
        _v("grade_documents", "incorrect", attempt=0),
        _v("rewrite_query", "rewrite", attempt=1),
        _v("grade_documents", "correct", attempt=1),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=1, max_retries=2, disposition="answered", grounded=True
    )
    assert s.retry_budget_respected is True


def test_i3_fail_non_monotonic_attempts() -> None:
    """A non-monotonic attempt sequence [0,1,0] → I3 fails."""
    traj = [
        _v("router", "retrieve", attempt=0),
        _v("grade_documents", "incorrect", attempt=1),
        _v("grade_documents", "correct", attempt=0),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=1, max_retries=2, disposition="answered", grounded=True
    )
    assert s.retry_budget_respected is False
    assert s.trajectory_valid is False


def test_i3_fail_over_budget() -> None:
    """retries 3 > max_retries 2 → I3 fails."""
    traj = [_v("router", "retrieve", attempt=0)]
    s = score_trajectory(
        traj, route="retrieve", retries=3, max_retries=2, disposition="refused", grounded=False
    )
    assert s.retry_budget_respected is False


def test_i3_pass_with_interleaved_non_grader_attempt_resets() -> None:
    """A real two-retry fail-closed trajectory passes I3 despite attempt-0 resets.

    Live wiring: only grade_documents/grade_generation stamp ``attempt`` with the
    retry pass (they read state["retries"]); rewrite_query/retrieve/generate/router
    leave it at 0. The grader attempts (0,1,2,2) are non-decreasing and the budget
    holds (retries=2<=max=2), so the interleaved 0s are NOT a violation.
    """
    traj = [
        _v("input_guard", "allow", attempt=0),
        _v("router", "graph", attempt=0),
        _v("retrieve", "ok", attempt=0),
        _v("grade_documents", "incorrect", attempt=0),
        _v("rewrite_query", "rewritten", attempt=0),
        _v("retrieve", "ok", attempt=0),
        _v("grade_documents", "incorrect", attempt=1),
        _v("rewrite_query", "rewritten", attempt=0),
        _v("retrieve", "ok", attempt=0),
        _v("grade_documents", "incorrect", attempt=2),
        _v("generate", "generated", attempt=0),
        _v("grade_generation", "unsupported", attempt=2),
        _v("fail_closed", "refused", attempt=2),
    ]
    s = score_trajectory(
        traj, route="graph", retries=2, max_retries=2, disposition="refused", grounded=False
    )
    assert s.retry_budget_respected is True
    assert s.trajectory_valid is True


def test_i3_fail_grader_attempt_exceeds_consumed_budget() -> None:
    """A grader stamped beyond the consumed retry count (attempt 3 > retries 1) → I3 fails."""
    traj = [
        _v("router", "retrieve", attempt=0),
        _v("grade_documents", "correct", attempt=3),
        _v("grade_generation", "grounded", attempt=3),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=1, max_retries=2, disposition="answered", grounded=True
    )
    assert s.retry_budget_respected is False


# ---------------------------------------------------------------------------
# I4 — route_consistent
# ---------------------------------------------------------------------------


def test_i4_pass_retrieve_ordered() -> None:
    """route=retrieve with grade_documents before grade_generation → I4 passes."""
    traj = [
        _v("router", "retrieve"),
        _v("grade_documents", "correct"),
        _v("grade_generation", "grounded"),
    ]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_consistent is True


def test_i4_fail_generation_grade_without_document_grade() -> None:
    """route=retrieve with grade_generation but no grade_documents → I4 fails."""
    traj = [_v("router", "retrieve"), _v("grade_generation", "grounded")]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_consistent is False
    assert s.trajectory_valid is False


def test_i4_skip_with_grade_documents_violates() -> None:
    """route=skip but a grade_documents verdict present → I4 fails (no retrieval on skip)."""
    traj = [_v("router", "skip"), _v("grade_documents", "correct")]
    s = score_trajectory(
        traj, route="skip", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_consistent is False


def test_i4_skip_with_only_generation_grade_is_legal() -> None:
    """route=skip with a grade_generation but no grade_documents → I4 passes.

    The live ``answer_direct → grade_generation`` wiring means the skip path is
    grounding-graded; only ``grade_documents`` is forbidden on skip.
    """
    traj = [
        _v("router", "skip"),
        _v("answer_direct", "parametric"),
        _v("grade_generation", "grounded"),
    ]
    s = score_trajectory(
        traj, route="skip", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_consistent is True


def test_i4_long_context_without_document_grade_is_legal() -> None:
    """route=long_context with grade_generation but no grade_documents → I4 passes.

    The enabled long-context node bypasses grade_documents (whole-doc stuffing).
    """
    traj = [
        _v("router", "long_context"),
        _v("long_context_retrieve", "stuffed"),
        _v("grade_generation", "grounded"),
    ]
    s = score_trajectory(
        traj,
        route="long_context",
        retries=0,
        max_retries=2,
        disposition="answered",
        grounded=True,
    )
    assert s.route_consistent is True


def test_i4_unknown_route_is_vacuously_true() -> None:
    """An unknown/None route makes no ordering claim → I4 passes."""
    traj = [_v("grade_generation", "grounded")]
    s = score_trajectory(
        traj, route=None, retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_consistent is True


# ---------------------------------------------------------------------------
# Label-based checks — route_correct / disposition_correct
# ---------------------------------------------------------------------------


def test_route_correct_is_none_without_label() -> None:
    """No query_type and no expected_route → route_correct is None (unlabeled)."""
    traj = [_v("router", "retrieve"), _v("grade_documents", "correct")]
    s = score_trajectory(
        traj, route="retrieve", retries=0, max_retries=2, disposition="answered", grounded=True
    )
    assert s.route_correct is None


def test_route_correct_from_query_type_trivial() -> None:
    """query_type=trivial expects route=skip → route_correct reflects the match."""
    traj = [_v("router", "skip")]
    hit = score_trajectory(
        traj,
        route="skip",
        retries=0,
        max_retries=2,
        disposition="answered",
        grounded=True,
        query_type="trivial",
    )
    miss = score_trajectory(
        traj,
        route="retrieve",
        retries=0,
        max_retries=2,
        disposition="answered",
        grounded=True,
        query_type="trivial",
    )
    assert hit.route_correct is True
    assert miss.route_correct is False


def test_route_correct_explicit_expected_route_wins() -> None:
    """An explicit expected_route overrides the coarse query_type rule."""
    traj = [_v("router", "graph")]
    s = score_trajectory(
        traj,
        route="graph",
        retries=0,
        max_retries=2,
        disposition="answered",
        grounded=True,
        query_type="factual",  # coarse rule would say "retrieve"
        expected_route="graph",
    )
    assert s.route_correct is True


def test_disposition_correct_mirrors_expected_disposition() -> None:
    """disposition_correct is None without a label, bool with expected_disposition."""
    traj = [_v("router", "retrieve"), _v("grade_documents", "incorrect"), _v("fail_closed", "x")]
    none = score_trajectory(
        traj, route="retrieve", retries=2, max_retries=2, disposition="refused", grounded=False
    )
    hit = score_trajectory(
        traj,
        route="retrieve",
        retries=2,
        max_retries=2,
        disposition="refused",
        grounded=False,
        expected_disposition="refused",
    )
    miss = score_trajectory(
        traj,
        route="retrieve",
        retries=2,
        max_retries=2,
        disposition="refused",
        grounded=False,
        expected_disposition="answered",
    )
    assert none.disposition_correct is None
    assert hit.disposition_correct is True
    assert miss.disposition_correct is False


# ---------------------------------------------------------------------------
# Return contract
# ---------------------------------------------------------------------------


def test_score_trajectory_returns_frozen_dataclass_with_retries() -> None:
    """score_trajectory returns a TrajectoryScore carrying the retry count."""
    traj = [_v("router", "retrieve"), _v("grade_documents", "correct")]
    s = score_trajectory(
        traj, route="retrieve", retries=1, max_retries=2, disposition="answered", grounded=True
    )
    assert isinstance(s, TrajectoryScore)
    assert s.retries == 1
