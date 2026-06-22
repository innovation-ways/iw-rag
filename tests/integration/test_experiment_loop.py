"""Integration tests for the exp #4 loop ON vs OFF experiment.

Uses testcontainers Postgres (via the ``clean_pg`` fixture). No live models;
a fake FaithfulnessJudge is injected to verify the judge path.

Verifies:
- run_loop_on_off persists TWO eval_runs (one per arm).
- Returns a LoopComparison with per-arm recall.
- Zero-gold queries excluded from means (counted separately).
- When a fake FaithfulnessJudge is supplied, per-arm faithfulness is populated.
- When no judge, faithfulness fields are None.
- Budget exhaustion rate is computed.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, replace
from typing import Any

import pytest

from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.eval.experiments import LoopComparison, run_loop_on_off
from iw_rag.eval.judges import JudgeVerdict
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_GOLD = str(uuid.uuid4())
_X = str(uuid.uuid4())
_Y = str(uuid.uuid4())


def _result(ids: list[str]) -> RetrievalResult:
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                document_id="doc-chunk",
                chunk_id=i,
                content="text",
                score=1.0 - n * 0.1,
                source_uri="u",
            )
            for n, i in enumerate(ids)
        ],
        retriever="spine",
        acl_applied=True,
    )


# Fake on/off retrieve callables.
async def _on_retrieve(question: str) -> RetrievalResult:
    """Loop ON arm ranks gold first (better retrieval)."""
    return _result([_GOLD, _X, _Y]) if question != "unanswerable" else _result([_X])


async def _off_retrieve(question: str) -> RetrievalResult:
    """Loop OFF arm ranks gold last (worse retrieval)."""
    return _result([_X, _Y, _GOLD]) if question != "unanswerable" else _result([_X])


# Fake GraphState results for each arm.
def _make_on_state(question: str) -> dict[str, Any]:
    return {
        "disposition": "answered",
        "answer": "grounded answer",
        "chunks": _result([_GOLD, _X, _Y]).chunks,
        "graded": _result([_GOLD]).chunks,
        "retries": 1,
        "route": "retrieve",
    }


def _make_off_state(question: str) -> dict[str, Any]:
    return {
        "disposition": "refused",
        "refusal": "not enough evidence",
        "chunks": _result([_X, _Y, _GOLD]).chunks,
        "graded": [],
        "retries": 0,
        "route": "retrieve",
    }


# ---------------------------------------------------------------------------
# Fake FaithfulnessJudge
# ---------------------------------------------------------------------------


@dataclass
class _FakeFaithfulnessJudge:
    """Returns FAITHFUL for on-arm, UNFAITHFUL for off-arm based on answer."""

    async def judge(self, question: str, context: str, answer: str) -> JudgeVerdict:
        """Return FAITHFUL for non-empty answers."""
        passed = bool(answer) and answer != "not enough evidence"
        return JudgeVerdict(passed=passed, score=1.0 if passed else 0.0, reason="fake")


# ---------------------------------------------------------------------------
# Integration test helpers
# ---------------------------------------------------------------------------


async def _run_loop_experiment(dsn: str, *, with_judge: bool) -> LoopComparison:
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            repo = EvalRepository(session)
            ds = await repo.create_dataset("golden-loop", "v1")
            queries = [
                GoldenQuery(question="q-gold", relevant_chunk_ids=[_GOLD], query_type="factual"),
                GoldenQuery(
                    question="q-zero", relevant_chunk_ids=[], query_type="factual"
                ),  # zero-gold
            ]
            query_ids = await repo.add_queries(ds, queries)
            golden = [replace(q, id=qid) for q, qid in zip(queries, query_ids, strict=True)]

            judge = _FakeFaithfulnessJudge() if with_judge else None

            comp = await run_loop_on_off(
                repo,
                queries=golden,
                on_retrieve=_on_retrieve,
                off_retrieve=_off_retrieve,
                on_state_fn=_make_on_state,
                off_state_fn=_make_off_state,
                faithfulness=judge,  # type: ignore[arg-type]
                dataset_id=ds,
                git_sha="abc123",
            )
            await session.commit()
            return comp
    finally:
        await db.dispose()


@pytest.mark.integration
def test_loop_on_off_persists_two_runs(clean_pg: str) -> None:
    """Two eval_runs (one per arm) are persisted."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=False))
    assert comp.on_run_id != comp.off_run_id
    assert comp.on_run_id != ""
    assert comp.off_run_id != ""


@pytest.mark.integration
def test_loop_on_off_zero_gold_excluded(clean_pg: str) -> None:
    """Zero-gold queries are excluded from means but counted."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=False))
    assert comp.n_zero_gold == 1
    assert comp.n_scored == 1  # only q-gold has relevant ids


@pytest.mark.integration
def test_loop_on_off_on_arm_better_mrr(clean_pg: str) -> None:
    """The ON arm (gold ranked first) has higher MRR than OFF arm (gold ranked last)."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=False))
    assert comp.on_mean_mrr > comp.off_mean_mrr
    assert comp.on_mean_mrr == pytest.approx(1.0)  # gold at rank 1
    assert comp.off_mean_mrr == pytest.approx(1 / 3)  # gold at rank 3


@pytest.mark.integration
def test_loop_on_off_faithfulness_when_judge_supplied(clean_pg: str) -> None:
    """When a fake judge is supplied, per-arm faithfulness is populated."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=True))
    # on arm always returns "grounded answer" → judge scores 1.0 for all queries
    assert comp.on_faithfulness == pytest.approx(1.0), (
        "on arm answers are all grounded → mean faithfulness must be 1.0"
    )
    # off arm always returns "not enough evidence" → judge scores 0.0 for all queries
    assert comp.off_faithfulness == pytest.approx(0.0), (
        "off arm refusals are all unfaithful → mean faithfulness must be 0.0"
    )


@pytest.mark.integration
def test_loop_on_off_no_faithfulness_when_no_judge(clean_pg: str) -> None:
    """When no judge is supplied, faithfulness fields are None."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=False))
    assert comp.on_faithfulness is None
    assert comp.off_faithfulness is None


@pytest.mark.integration
def test_loop_on_off_exhaustion_rate(clean_pg: str) -> None:
    """Budget exhaustion rate is computed per arm (refused dispositions)."""
    comp = asyncio.run(_run_loop_experiment(clean_pg, with_judge=False))
    # OFF arm always refuses → 100% exhaustion
    assert comp.off_exhaustion_rate == pytest.approx(1.0)
    # ON arm always answers → 0% exhaustion
    assert comp.on_exhaustion_rate == pytest.approx(0.0)
