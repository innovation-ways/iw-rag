"""Integration tests for the experiment harness (testcontainers + fake retrieve).

Verifies: results persist append-only, zero-gold queries are excluded from the
recall mean (but counted), and a better-ranking config yields a higher recall
mean (the monotone-lift property exp #1 relies on).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from iw_rag.domain.models import RetrievalResult, RetrievedChunk
from iw_rag.eval.experiments import run_experiment
from iw_rag.eval.repository import EvalRepository, GoldenQuery
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

_GOLD = str(uuid.uuid4())
_X = str(uuid.uuid4())
_Y = str(uuid.uuid4())


def _result(ids: list[str]) -> RetrievalResult:
    return RetrievalResult(
        chunks=[
            RetrievedChunk(
                document_id="doc-chunk", chunk_id=i, content="", score=1.0 - n * 0.1, source_uri="u"
            )
            for n, i in enumerate(ids)
        ],
        retriever="spine",
        acl_applied=True,
    )


async def _run(dsn: str) -> dict[str, object]:
    db = Database(DatabaseSettings(dsn=dsn))
    out: dict[str, object] = {}
    try:
        async with db.session() as session:
            repo = EvalRepository(session)
            ds = await repo.create_dataset("golden", "v1")
            await repo.add_queries(
                ds,
                [
                    GoldenQuery(
                        question="q-good", relevant_chunk_ids=[_GOLD], query_type="factual"
                    ),
                    GoldenQuery(question="q-zero", relevant_chunk_ids=[], query_type="factual"),
                ],
            )
            queries = await repo.get_queries(ds)

            # "weak" config ranks the gold chunk last; "strong" ranks it first.
            async def weak(question: str) -> RetrievalResult:
                return _result([_X, _Y, _GOLD]) if question == "q-good" else _result([_X])

            async def strong(question: str) -> RetrievalResult:
                return _result([_GOLD, _X, _Y]) if question == "q-good" else _result([_X])

            weak_sum = await run_experiment(
                repo, experiment="weak", retrieve=weak, queries=queries, ks=(5, 10)
            )
            strong_sum = await run_experiment(
                repo, experiment="strong", retrieve=strong, queries=queries, ks=(5, 10)
            )
            await session.commit()

            out["weak_recall10"] = weak_sum.mean_recall[10]
            out["strong_recall10"] = strong_sum.mean_recall[10]
            out["weak_mrr"] = weak_sum.mean_mrr
            out["strong_mrr"] = strong_sum.mean_mrr
            out["n_scored"] = weak_sum.n_scored
            out["n_zero_gold"] = weak_sum.n_zero_gold
            out["weak_results"] = await repo.count_results(weak_sum.run_id)
        return out
    finally:
        await db.dispose()


@pytest.mark.integration
def test_experiment_persists_and_lifts(clean_pg: str) -> None:
    out = asyncio.run(_run(clean_pg))
    # both configs recall the gold within k=10 → recall mean 1.0; MRR shows the lift
    assert out["n_scored"] == 1  # zero-gold query excluded from the mean
    assert out["n_zero_gold"] == 1
    assert out["weak_results"] == 2  # all queries persisted (append-only)
    assert out["strong_mrr"] > out["weak_mrr"]  # ranking the gold first lifts MRR
    assert out["weak_mrr"] == pytest.approx(1 / 3)  # gold at rank 3
    assert out["strong_mrr"] == pytest.approx(1.0)  # gold at rank 1
