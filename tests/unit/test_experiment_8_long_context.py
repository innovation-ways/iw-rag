"""Unit tests for headline experiment #8 (long-context vs RAG) — harness gate.

Tests cover:
- LongContextComparisonReport dataclass fields + immutability.
- run_experiment_8_long_context persists two eval_runs (whole-doc + large-corpus)
  and returns a populated report.
- Gate logic: wholedoc_win + largecorpus_no_loss → gate_passed.

All tests use a fake EvalRepository — no live Postgres, no live models. Mirrors
the exp #6 / #7 harness-gate pattern (real accuracy × latency × VRAM numbers are
a deferred live run, per task-4c experiment #8).

References:
- task-4c-long-context-route experiment #8 (validation spec)
- eval/experiments.py run_experiment_8_long_context
- ADR-012 (RAG default; long-context = narrow gated 4th route, enabled only if #8 passes)
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from iw_rag.domain.models import RetrievalResult
from iw_rag.eval.experiments import (
    LongContextComparisonReport,
    run_experiment_8_long_context,
)
from iw_rag.eval.repository import GoldenQuery


def _make_golden_query(q_id: str = "q1", question: str = "summarise this doc?") -> GoldenQuery:
    """Build a minimal zero-gold GoldenQuery (excluded from means)."""
    return GoldenQuery(id=q_id, question=question, relevant_chunk_ids=[])


def _make_fake_retrieve() -> Callable[[str], Awaitable[RetrievalResult]]:
    """Build a fake retrieve callable returning an empty RetrievalResult."""

    async def _retrieve(question: str) -> RetrievalResult:
        return RetrievalResult(chunks=[], retriever="fake", acl_applied=True, query=question)

    return _retrieve


def _make_fake_repo(*, run_id: str = "run-exp8") -> Any:
    """Build a fake EvalRepository with async create_run / add_results."""
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


def test_long_context_comparison_report_is_frozen() -> None:
    """LongContextComparisonReport is a frozen dataclass with expected fields."""
    report = LongContextComparisonReport(
        wholedoc_run_id="run-w",
        largecorpus_run_id="run-l",
        lc_wholedoc_recall={5: 0.8},
        spine_wholedoc_recall={5: 0.7},
        lc_largecorpus_recall={5: 0.6},
        spine_largecorpus_recall={5: 0.65},
        lc_wholedoc_ndcg={5: 0.85},
        spine_wholedoc_ndcg={5: 0.7},
        wholedoc_win=True,
        largecorpus_no_loss=True,
        gate_passed=True,
    )
    assert report.gate_passed is True
    assert report.wholedoc_win is True
    with pytest.raises((AttributeError, TypeError), match="cannot assign to field"):
        report.gate_passed = False  # type: ignore[misc]


def test_long_context_comparison_report_defaults() -> None:
    """Defaults: gate_passed=False, largecorpus_no_loss=True (conservative)."""
    report = LongContextComparisonReport(
        wholedoc_run_id="r1",
        largecorpus_run_id="r2",
        lc_wholedoc_recall={},
        spine_wholedoc_recall={},
        lc_largecorpus_recall={},
        spine_largecorpus_recall={},
        lc_wholedoc_ndcg={},
        spine_wholedoc_ndcg={},
    )
    assert report.wholedoc_win is False
    assert report.largecorpus_no_loss is True
    assert report.gate_passed is False


@pytest.mark.asyncio
async def test_run_experiment_8_persists_two_runs() -> None:
    """run_experiment_8_long_context persists a whole-doc + large-corpus run."""
    repo = _make_fake_repo()
    report = await run_experiment_8_long_context(
        repo,
        wholedoc_queries=[_make_golden_query("wd1")],
        largecorpus_queries=[_make_golden_query("lc1")],
        lc_retrieve=_make_fake_retrieve(),
        spine_retrieve=_make_fake_retrieve(),
        git_sha="abc123",
    )
    assert isinstance(report, LongContextComparisonReport)
    assert report.wholedoc_run_id != ""
    assert report.largecorpus_run_id != ""
    assert report.wholedoc_run_id != report.largecorpus_run_id
    # 4 arm runs (lc/spine × wholedoc/largecorpus) + 1 summary run = 5 total;
    # mirrors run_experiment_6_graphrag's exact-count contract.
    assert repo.create_run.await_count == 5, (
        f"Expected 5 create_run calls (4 arms + 1 summary), got {repo.create_run.await_count}"
    )
    # CI fake callables return empty results → no win → gate stays closed.
    assert report.gate_passed is False
