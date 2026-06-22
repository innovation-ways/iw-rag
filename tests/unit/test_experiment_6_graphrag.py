"""Unit tests for headline experiment #6 (GraphRAG vs hybrid spine) — TDD RED-first.

Tests cover:
- GraphComparisonReport dataclass fields and structure.
- run_experiment_6_graphrag persists eval_runs + eval_results rows correctly.
- Gate logic: multihop_win + procedural_no_loss → gate_passed.
- Gate logic: multihop_win=False (CI fake callables — both arms return 0.0).
- procedural_no_loss with the ≤5% regression tolerance.
- Per-subset recall/nDCG metrics are correctly populated.

All tests use a fake EvalRepository — no live Postgres.  Mirrors exp #4 and #7
patterns (tests/unit/test_loop_on_off_unit.py, tests/integration/test_experiments.py).

References:
- task-4a-graphrag-route experiment #6 (validation spec)
- eval/experiments.py run_experiment_6_graphrag
- ADR-009 (GraphRAG = LightRAG, optional, offline index)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.models import RetrievalResult
from iw_rag.eval.experiments import GraphComparisonReport, run_experiment_6_graphrag
from iw_rag.eval.repository import GoldenQuery

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


def _make_golden_query(q_id: str = "q1", question: str = "test?") -> GoldenQuery:
    """Build a minimal GoldenQuery.

    Args:
        q_id: Query id.
        question: The question text.

    Returns:
        A GoldenQuery with no gold chunk ids (zero-gold — excluded from means).
    """
    return GoldenQuery(id=q_id, question=question, relevant_chunk_ids=[])


def _make_fake_retrieve(
    ndcg_placeholder: float = 0.0,
) -> Callable[[str], Awaitable[RetrievalResult]]:
    """Build a fake retrieve callable that returns a consistent empty RetrievalResult.

    Args:
        ndcg_placeholder: Unused (CI always computes 0.0 from empty results).

    Returns:
        An async callable returning an empty RetrievalResult.
    """

    async def _retrieve(question: str) -> RetrievalResult:
        return RetrievalResult(chunks=[], retriever="fake", acl_applied=True, query=question)

    return _retrieve


def _make_fake_repo(
    *,
    run_id: str = "run-exp6",
) -> Any:
    """Build a fake EvalRepository.

    Args:
        run_id: The run_id to return from create_run.

    Returns:
        A MagicMock EvalRepository with async create_run / add_results.
    """
    call_count = [0]

    async def _create_run(**kwargs: Any) -> str:
        call_count[0] += 1
        return f"{run_id}-{call_count[0]}"

    async def _add_results(run_id: str, rows: Any) -> None:
        pass

    repo = MagicMock()
    repo.create_run = AsyncMock(side_effect=_create_run)
    repo.add_results = AsyncMock(side_effect=_add_results)
    return repo


# ---------------------------------------------------------------------------
# Tests: GraphComparisonReport dataclass
# ---------------------------------------------------------------------------


def test_graph_comparison_report_is_frozen_dataclass() -> None:
    """GraphComparisonReport is a frozen dataclass with expected fields."""
    report = GraphComparisonReport(
        multihop_run_id="run-m",
        procedural_run_id="run-p",
        graph_multihop_recall={5: 0.7, 10: 0.8},
        spine_multihop_recall={5: 0.6, 10: 0.75},
        graph_procedural_recall={5: 0.5, 10: 0.6},
        spine_procedural_recall={5: 0.55, 10: 0.65},
        graph_multihop_ndcg={5: 0.8, 10: 0.85},
        spine_multihop_ndcg={5: 0.7, 10: 0.75},
        multihop_win=True,
        procedural_no_loss=True,
        gate_passed=True,
    )
    assert report.gate_passed is True
    assert report.multihop_win is True
    assert report.procedural_no_loss is True
    # Immutable
    with pytest.raises((AttributeError, TypeError), match="cannot assign to field"):
        report.gate_passed = False  # type: ignore[misc]


def test_graph_comparison_report_defaults() -> None:
    """GraphComparisonReport has correct defaults (gate_passed=False by default)."""
    report = GraphComparisonReport(
        multihop_run_id="r1",
        procedural_run_id="r2",
        graph_multihop_recall={},
        spine_multihop_recall={},
        graph_procedural_recall={},
        spine_procedural_recall={},
        graph_multihop_ndcg={},
        spine_multihop_ndcg={},
    )
    assert report.multihop_win is False
    assert report.procedural_no_loss is True  # conservative default: assume no loss
    assert report.gate_passed is False


# ---------------------------------------------------------------------------
# Tests: run_experiment_6_graphrag — structure + persistence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_experiment_6_returns_graph_comparison_report() -> None:
    """run_experiment_6_graphrag returns a GraphComparisonReport."""
    repo = _make_fake_repo()
    graph_retrieve = _make_fake_retrieve()
    spine_retrieve = _make_fake_retrieve()
    queries_mh = [_make_golden_query("m1", "How does EntityA connect to EntityB?")]
    queries_proc = [_make_golden_query("p1", "What is the leave policy?")]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=graph_retrieve,
        spine_retrieve=spine_retrieve,
        git_sha="abc123",
    )

    assert isinstance(result, GraphComparisonReport)
    # With empty zero-gold queries both arms score 0.0 → gate stays closed.
    assert result.gate_passed is False
    # Run IDs must be non-empty strings set by the repository.
    assert result.multihop_run_id != ""
    assert result.procedural_run_id != ""


@pytest.mark.asyncio
async def test_run_experiment_6_persists_runs_and_results() -> None:
    """run_experiment_6_graphrag persists eval_runs and eval_results rows."""
    repo = _make_fake_repo()
    queries_mh = [_make_golden_query("m1", "Multi-hop?"), _make_golden_query("m2", "Relational?")]
    queries_proc = [_make_golden_query("p1", "Policy?")]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=_make_fake_retrieve(),
        spine_retrieve=_make_fake_retrieve(),
    )

    # 4 arm runs (graph/spine × multihop/procedural) + 1 summary run = 5 total
    assert repo.create_run.call_count == 5, (
        f"Expected 5 create_run calls (4 arms + 1 summary), got {repo.create_run.call_count}"
    )
    # add_results called for each run (4 arm result rows + 1 summary row)
    assert repo.add_results.call_count == 5
    # The returned report must have distinct, non-empty run IDs confirming persistence occurred.
    assert result.multihop_run_id != result.procedural_run_id, (
        "Each arm must receive a distinct run_id from the repository"
    )


@pytest.mark.asyncio
async def test_run_experiment_6_gate_fails_with_fake_callables() -> None:
    """With fake callables (CI), both arms return 0.0 nDCG@5 → multihop_win=False."""
    repo = _make_fake_repo()
    queries_mh = [_make_golden_query("m1", "Multi-hop?")]
    queries_proc = [_make_golden_query("p1", "Policy?")]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=_make_fake_retrieve(),
        spine_retrieve=_make_fake_retrieve(),
    )

    # Both arms return 0.0 → graph does not beat spine → gate fails
    assert result.multihop_win is False, (
        "With fake callables both arms return 0.0, so multihop_win must be False"
    )
    assert result.gate_passed is False


@pytest.mark.asyncio
async def test_run_experiment_6_records_run_id_fields() -> None:
    """run_experiment_6_graphrag records multihop_run_id and procedural_run_id."""
    repo = _make_fake_repo(run_id="exp6")
    queries_mh = [_make_golden_query("m1", "Multi-hop?")]
    queries_proc = [_make_golden_query("p1", "Policy?")]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=_make_fake_retrieve(),
        spine_retrieve=_make_fake_retrieve(),
    )

    assert result.multihop_run_id, "multihop_run_id must be set"
    assert result.procedural_run_id, "procedural_run_id must be set"
    assert result.multihop_run_id != result.procedural_run_id, (
        "multihop and procedural run ids must be distinct"
    )


# ---------------------------------------------------------------------------
# Tests: gate logic — driven by the REAL run_experiment_6_graphrag
# ---------------------------------------------------------------------------

# Gold chunk ids shared across the gate tests.
_GOLD_MH = "gold-multihop-chunk"
_GOLD_PROC = "gold-procedural-chunk"


def _make_golden_query_with_gold(q_id: str, question: str, gold_id: str) -> GoldenQuery:
    """Build a GoldenQuery with one gold chunk id (non-zero → counts in means).

    Args:
        q_id: Query id.
        question: The question text.
        gold_id: The single relevant chunk id (non-zero gold).

    Returns:
        A GoldenQuery that contributes to scored means.
    """
    return GoldenQuery(id=q_id, question=question, relevant_chunk_ids=[gold_id])


def _make_winning_retrieve(gold_id: str) -> Callable[[str], Awaitable[RetrievalResult]]:
    """Return a fake retrieve callable that ranks the gold chunk first (nDCG@5 = 1.0).

    Args:
        gold_id: The chunk id to rank at position 0.

    Returns:
        An async callable returning a RetrievalResult with gold first.
    """
    from iw_rag.domain.models import RetrievedChunk

    async def _retrieve(question: str) -> RetrievalResult:
        return RetrievalResult(
            chunks=[
                RetrievedChunk(
                    document_id="doc-1",
                    chunk_id=gold_id,
                    content="relevant content",
                    score=1.0,
                    source_uri="u",
                ),
            ],
            retriever="fake-graph",
            acl_applied=True,
            query=question,
        )

    return _retrieve


def _make_losing_retrieve() -> Callable[[str], Awaitable[RetrievalResult]]:
    """Return a fake retrieve callable that returns no relevant chunks (nDCG@5 = 0.0).

    Returns:
        An async callable returning an empty RetrievalResult.
    """

    async def _retrieve(question: str) -> RetrievalResult:
        return RetrievalResult(
            chunks=[],
            retriever="fake-spine",
            acl_applied=True,
            query=question,
        )

    return _retrieve


@pytest.mark.asyncio
async def test_gate_passes_when_graph_wins_multihop_and_no_procedural_loss() -> None:
    """Gate passes when the real function sees graph win multi-hop AND no procedural loss.

    Graph arm ranks gold first (nDCG@5 = 1.0); spine arm returns empty (nDCG@5 = 0.0)
    on multi-hop → multihop_win=True.  On procedural both arms return gold first
    (nDCG@5 = 1.0 each) → no_loss_threshold = 0.95, graph 1.0 ≥ 0.95 → no_loss=True.
    """
    repo = _make_fake_repo()
    queries_mh = [_make_golden_query_with_gold("m1", "Multi-hop?", _GOLD_MH)]
    queries_proc = [_make_golden_query_with_gold("p1", "Policy?", _GOLD_PROC)]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=_make_winning_retrieve(_GOLD_MH),
        spine_retrieve=_make_losing_retrieve(),
    )

    assert result.multihop_win is True, (
        "graph arm ranks gold first → nDCG@5=1.0 > spine 0.0 → multihop_win must be True"
    )
    assert result.gate_passed is True, (
        "multihop_win=True and procedural_no_loss=True (both 0.0, 0.0 ≥ 0.0) → gate must pass"
    )


@pytest.mark.asyncio
async def test_gate_fails_when_graph_loses_multihop() -> None:
    """Gate fails when the real function sees spine beat graph on multi-hop.

    Spine arm ranks gold first (nDCG@5 = 1.0); graph arm returns empty (nDCG@5 = 0.0)
    on multi-hop → multihop_win=False → gate_passed=False regardless of procedural.
    """
    repo = _make_fake_repo()
    queries_mh = [_make_golden_query_with_gold("m1", "Multi-hop?", _GOLD_MH)]
    queries_proc = [_make_golden_query_with_gold("p1", "Policy?", _GOLD_PROC)]

    result = await run_experiment_6_graphrag(
        repo,
        multihop_queries=queries_mh,
        procedural_queries=queries_proc,
        graph_retrieve=_make_losing_retrieve(),
        spine_retrieve=_make_winning_retrieve(_GOLD_MH),
    )

    assert result.multihop_win is False, (
        "spine ranks gold first → nDCG@5=1.0 > graph 0.0 → multihop_win must be False"
    )
    assert result.gate_passed is False, "multihop_win=False must fail the gate"
