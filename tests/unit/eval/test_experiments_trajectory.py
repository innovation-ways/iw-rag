"""Unit tests for headline experiment #9 (agentic-loop trajectory) — TDD RED-first.

Tests cover:
- TrajectoryReport dataclass fields, defaults, and immutability.
- run_experiment_9_trajectory aggregates per-query modal samples and persists
  exactly one eval_runs + one eval_results row (append-only).
- trajectory_validity_rate / invariant_pass_rates reflect the structural scorer.
- route_accuracy from explicit expected_route labels (preferred) and from the
  coarse query_type rule (fallback).
- disposition_accuracy is None without labels, a fraction with labels.
- disposition_stability / route_stability reflect multi-sample agreement.
- samples_per_query is recorded in the run config (no silent caps).

All tests use a fake EvalRepository — no live Postgres.  Mirrors exp #6/#7/#8
harness-gate tests (tests/unit/test_experiment_6_graphrag.py).

References:
- docs/build/03-tasks/trajectory-eval/03-design-and-implementation.md §3.1
- src/iw_rag/eval/experiments.py run_experiment_9_trajectory
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.eval.experiments import TrajectoryReport, run_experiment_9_trajectory
from iw_rag.eval.repository import GoldenQuery

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _v(kind: str, value: str = "", attempt: int = 0) -> dict[str, Any]:
    """Build one trajectory verdict dict."""
    return {"kind": kind, "value": value, "score": None, "attempt": attempt}


def _valid_answered_sample(route: str = "retrieve") -> dict[str, Any]:
    """A structurally valid answered+grounded sample on the retrieve route."""
    return {
        "trajectory": [
            _v("input_guard", "allow"),
            _v("router", route),
            _v("grade_documents", "correct"),
            _v("grade_generation", "grounded"),
            _v("output_guard", "allow"),
        ],
        "route": route,
        "retries": 0,
        "max_retries": 2,
        "disposition": "answered",
        "grounded": True,
    }


def _valid_refused_sample() -> dict[str, Any]:
    """A structurally valid refused sample with an explicit fail-closed cause."""
    return {
        "trajectory": [
            _v("input_guard", "allow"),
            _v("router", "retrieve"),
            _v("grade_documents", "incorrect", attempt=0),
            _v("grade_generation", "unsupported", attempt=2),
            _v("fail_closed", "refused", attempt=2),
        ],
        "route": "retrieve",
        "retries": 2,
        "max_retries": 2,
        "disposition": "refused",
        "grounded": False,
    }


def _make_fake_repo(*, run_id: str = "run-exp9") -> Any:
    """Build a fake EvalRepository with async create_run / add_results."""
    call_count = [0]

    async def _create_run(**kwargs: Any) -> str:
        call_count[0] += 1
        return f"{run_id}-{call_count[0]}"

    async def _add_results(run_id: str, rows: Any) -> None:
        return None

    repo = MagicMock()
    repo.create_run = AsyncMock(side_effect=_create_run)
    repo.add_results = AsyncMock(side_effect=_add_results)
    return repo


# ---------------------------------------------------------------------------
# TrajectoryReport dataclass
# ---------------------------------------------------------------------------


def test_trajectory_report_is_frozen_dataclass() -> None:
    """TrajectoryReport is a frozen dataclass with the expected fields."""
    report = TrajectoryReport(
        run_id="r1",
        trajectory_validity_rate=1.0,
        route_accuracy=0.9,
        disposition_accuracy=None,
        mean_retries=0.5,
        disposition_stability=1.0,
        route_stability=1.0,
        invariant_pass_rates={"I1_grounded_implies_grade": 1.0},
        n_queries=3,
        samples_per_query=3,
    )
    assert report.trajectory_validity_rate == pytest.approx(1.0)
    assert report.disposition_accuracy is None
    assert report.notes == ""
    with pytest.raises((AttributeError, TypeError), match="cannot assign to field"):
        report.run_id = "r2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# run_experiment_9_trajectory — structure + persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_9_persists_one_run_and_one_result() -> None:
    """run_experiment_9_trajectory persists exactly one eval_runs + one eval_results row."""
    repo = _make_fake_repo()
    queries = [GoldenQuery(id="q1", question="What is the leave policy?", query_type="factual")]
    samples = {"What is the leave policy?": [_valid_answered_sample() for _ in range(3)]}

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3, git_sha="abc"
    )

    assert isinstance(report, TrajectoryReport)
    assert repo.create_run.call_count == 1
    assert repo.add_results.call_count == 1
    # The single create_run is the trajectory experiment.
    _, kwargs = repo.create_run.call_args
    assert kwargs["experiment"] == "agent_loop_trajectory"
    # samples_per_query is recorded in the config (no silent caps).
    assert kwargs["config"]["samples_per_query"] == 3
    assert report.run_id != ""


@pytest.mark.asyncio
async def test_run_experiment_9_all_valid_gives_full_validity_rate() -> None:
    """All-valid samples → trajectory_validity_rate 1.0 and every invariant at 1.0."""
    repo = _make_fake_repo()
    queries = [
        GoldenQuery(id="q1", question="q1?", query_type="factual"),
        GoldenQuery(id="q2", question="q2?", query_type="factual"),
    ]
    samples = {
        "q1?": [_valid_answered_sample() for _ in range(3)],
        "q2?": [_valid_refused_sample() for _ in range(3)],
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )

    assert report.trajectory_validity_rate == pytest.approx(1.0)
    assert report.n_queries == 2
    assert all(rate == pytest.approx(1.0) for rate in report.invariant_pass_rates.values())
    # Deterministic samples → fully stable.
    assert report.disposition_stability == pytest.approx(1.0)
    assert report.route_stability == pytest.approx(1.0)
    # mean_retries: (0 + 2) / 2 = 1.0
    assert report.mean_retries == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_experiment_9_detects_invalid_trajectory() -> None:
    """A structurally invalid modal sample drops the validity rate below 1.0."""
    repo = _make_fake_repo()
    queries = [
        GoldenQuery(id="q1", question="ok?", query_type="factual"),
        GoldenQuery(id="q2", question="bad?", query_type="factual"),
    ]
    # bad? — refused with no negative cause and no terminal refusal kind → I2 fails.
    bad_sample = {
        "trajectory": [
            _v("input_guard", "allow"),
            _v("router", "retrieve"),
            _v("grade_documents", "correct"),
        ],
        "route": "retrieve",
        "retries": 0,
        "max_retries": 2,
        "disposition": "refused",
        "grounded": False,
    }
    samples = {
        "ok?": [_valid_answered_sample() for _ in range(3)],
        "bad?": [bad_sample for _ in range(3)],
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )

    # 1 of 2 queries valid.
    assert report.trajectory_validity_rate == pytest.approx(0.5)
    assert report.invariant_pass_rates["I2_refuse_implies_cause"] == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_experiment_9_route_accuracy_from_explicit_labels() -> None:
    """Explicit expected_route labels drive route_accuracy (preferred over query_type)."""
    repo = _make_fake_repo()
    queries = [
        GoldenQuery(id="q1", question="graphy?", query_type="factual"),
        GoldenQuery(id="q2", question="planey?", query_type="factual"),
    ]
    samples = {
        "graphy?": [_valid_answered_sample(route="retrieve") for _ in range(3)],
        "planey?": [_valid_answered_sample(route="retrieve") for _ in range(3)],
    }
    # q1 expects retrieve (match); q2 expects graph (miss, actual is retrieve).
    expected_routes = {"graphy?": "retrieve", "planey?": "graph"}

    report = await run_experiment_9_trajectory(
        repo,
        queries=queries,
        samples=samples,
        samples_per_query=3,
        expected_routes=expected_routes,
    )

    assert report.route_accuracy == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_experiment_9_route_accuracy_from_query_type_fallback() -> None:
    """Without explicit labels, route_accuracy uses the coarse query_type rule."""
    repo = _make_fake_repo()
    queries = [
        GoldenQuery(id="q1", question="trivial-q?", query_type="trivial"),
        GoldenQuery(id="q2", question="factual-q?", query_type="factual"),
    ]
    samples = {
        # trivial → expected skip; actual skip → correct
        "trivial-q?": [
            {**_valid_answered_sample(route="skip"), "trajectory": [_v("router", "skip")]}
            for _ in range(3)
        ],
        # factual → expected retrieve; actual retrieve → correct
        "factual-q?": [_valid_answered_sample(route="retrieve") for _ in range(3)],
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )

    assert report.route_accuracy == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_experiment_9_disposition_accuracy_none_without_labels() -> None:
    """disposition_accuracy is None when no query carries an expected_disposition."""
    repo = _make_fake_repo()
    queries = [GoldenQuery(id="q1", question="q?", query_type="factual")]
    samples = {"q?": [_valid_answered_sample() for _ in range(3)]}

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )
    assert report.disposition_accuracy is None


@pytest.mark.asyncio
async def test_run_experiment_9_disposition_accuracy_with_labels() -> None:
    """disposition_accuracy is a fraction over label-bearing queries."""
    repo = _make_fake_repo()
    queries = [
        GoldenQuery(id="q1", question="ans?", query_type="factual"),
        GoldenQuery(id="q2", question="ref?", query_type="factual"),
    ]
    samples = {
        "ans?": [_valid_answered_sample() for _ in range(3)],
        "ref?": [_valid_refused_sample() for _ in range(3)],
    }
    # ans? expects answered (match); ref? expects answered (miss — it refuses).
    expected_dispositions = {"ans?": "answered", "ref?": "answered"}

    report = await run_experiment_9_trajectory(
        repo,
        queries=queries,
        samples=samples,
        samples_per_query=3,
        expected_dispositions=expected_dispositions,
    )
    assert report.disposition_accuracy == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_run_experiment_9_stability_reflects_sample_disagreement() -> None:
    """Mixed dispositions across samples lower disposition_stability below 1.0."""
    repo = _make_fake_repo()
    queries = [GoldenQuery(id="q1", question="flaky?", query_type="factual")]
    # 2 answered + 1 refused over k=3 → modal=answered, stability = 2/3.
    samples = {
        "flaky?": [
            _valid_answered_sample(),
            _valid_answered_sample(),
            _valid_refused_sample(),
        ]
    }

    report = await run_experiment_9_trajectory(
        repo, queries=queries, samples=samples, samples_per_query=3
    )
    assert report.disposition_stability == pytest.approx(2 / 3)


@pytest.mark.asyncio
async def test_run_experiment_9_override_samples_hook() -> None:
    """The _override_samples test hook bypasses the samples arg for the harness gate."""
    repo = _make_fake_repo()
    queries = [GoldenQuery(id="q1", question="q?", query_type="factual")]

    report = await run_experiment_9_trajectory(
        repo,
        queries=queries,
        samples={},  # empty — override supplies the data
        samples_per_query=2,
        _override_samples={"q?": [_valid_answered_sample() for _ in range(2)]},
    )
    assert report.n_queries == 1
    assert report.trajectory_validity_rate == pytest.approx(1.0)
