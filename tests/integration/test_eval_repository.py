"""Integration tests for EvalRepository (testcontainers ParadeDB).

Verifies the eval persistence round-trip: a dataset + golden queries (with human
relevant_chunk_ids) and an append-only run + results.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from iw_rag.eval.repository import EvalRepository, EvalResultRow, GoldenQuery
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database


async def _run(dsn: str) -> tuple[list[GoldenQuery], int]:
    db = Database(DatabaseSettings(dsn=dsn))
    try:
        async with db.session() as session:
            repo = EvalRepository(session)
            dataset_id = await repo.create_dataset("golden", "v1")
            gold_chunk = str(uuid.uuid4())
            await repo.add_queries(
                dataset_id,
                [
                    GoldenQuery(
                        question="What was the Q3 invoice total?",
                        relevant_chunk_ids=[gold_chunk],
                        query_type="factual",
                        reference_answer="15000 USD",
                    )
                ],
            )
            queries = await repo.get_queries(dataset_id)

            run_id = await repo.create_run(
                dataset_id=dataset_id, experiment="exp1_spine_ablation", git_sha="abc123"
            )
            await repo.add_results(
                run_id,
                [
                    EvalResultRow(
                        query_id=queries[0].id,
                        retrieval_metrics={"recall": {"10": 0.9}, "mrr": 1.0},
                        trace_ref={"trace_id": "t", "span_id": "s"},
                    )
                ],
            )
            count = await repo.count_results(run_id)
            return queries, count
    finally:
        await db.dispose()


@pytest.mark.integration
def test_eval_round_trip(clean_pg: str) -> None:
    queries, count = asyncio.run(_run(clean_pg))
    assert len(queries) == 1
    assert queries[0].query_type == "factual"
    assert len(queries[0].relevant_chunk_ids) == 1  # human gold preserved
    assert queries[0].id is not None
    assert count == 1  # append-only result persisted
