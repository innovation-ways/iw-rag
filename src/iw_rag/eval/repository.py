"""Eval persistence — datasets, golden queries, runs, and append-only results.

The single store for the trust layer (lld-data-model §3, eval tables). Golden
queries carry human ``relevant_chunk_ids`` (the contamination-proof gold);
``eval_runs``/``eval_results`` are append-only regression history. ids are ``str``
at the boundary (DB columns are uuid).
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import func, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.storage import schema


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    """A curated evaluation query with human gold labels.

    Attributes:
        question: The query text.
        relevant_chunk_ids: Human-curated gold chunk ids (str).
        query_type: ``factual|multihop|wholedoc|relational`` (optional).
        reference_answer: Optional reference answer.
        id: The persisted id (set after insert).
    """

    question: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    query_type: str | None = None
    reference_answer: str | None = None
    id: str | None = None


@dataclass(frozen=True, slots=True)
class EvalResultRow:
    """One per-query scored result within a run.

    Attributes:
        query_id: The scored query id (str), or None for a synthetic result.
        retrieval_metrics: recall@k / nDCG@k / mrr.
        generation_metrics: faithfulness / context precision-recall (optional).
        trace_ref: Phoenix ``{trace_id, span_id}`` (optional).
    """

    query_id: str | None
    retrieval_metrics: dict[str, Any]
    generation_metrics: dict[str, Any] | None = None
    trace_ref: dict[str, Any] | None = None


class EvalRepository:
    """CRUD over the eval_* tables (datasets/queries/runs/results)."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind to an open session."""
        self._session = session

    async def create_dataset(self, name: str, version: str) -> str:
        """Insert an eval dataset; return its id."""
        stmt = (
            insert(schema.eval_datasets)
            .values(name=name, version=version)
            .returning(schema.eval_datasets.c.id)
        )
        return str(cast(uuid.UUID, (await self._session.execute(stmt)).scalar_one()))

    async def add_queries(self, dataset_id: str, queries: Sequence[GoldenQuery]) -> list[str]:
        """Insert golden queries for a dataset; return their ids (input order)."""
        if not queries:
            return []
        values = [
            {
                "dataset_id": uuid.UUID(dataset_id),
                "question": q.question,
                "reference_answer": q.reference_answer,
                "relevant_chunk_ids": [uuid.UUID(c) for c in q.relevant_chunk_ids],
                "query_type": q.query_type,
            }
            for q in queries
        ]
        stmt = insert(schema.eval_queries).returning(schema.eval_queries.c.id)
        result = await self._session.execute(stmt, values)
        return [str(row[0]) for row in result]

    async def get_queries(self, dataset_id: str) -> list[GoldenQuery]:
        """Fetch all golden queries for a dataset."""
        stmt = select(schema.eval_queries).where(
            schema.eval_queries.c.dataset_id == uuid.UUID(dataset_id)
        )
        rows = (await self._session.execute(stmt)).mappings()
        return [
            GoldenQuery(
                id=str(r["id"]),
                question=r["question"],
                reference_answer=r["reference_answer"],
                relevant_chunk_ids=[str(c) for c in (r["relevant_chunk_ids"] or [])],
                query_type=r["query_type"],
            )
            for r in rows
        ]

    async def create_run(
        self,
        *,
        dataset_id: str | None,
        experiment: str,
        git_sha: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> str:
        """Insert an eval run; return its id."""
        stmt = (
            insert(schema.eval_runs)
            .values(
                dataset_id=uuid.UUID(dataset_id) if dataset_id else None,
                experiment=experiment,
                git_sha=git_sha,
                config=config or {},
            )
            .returning(schema.eval_runs.c.id)
        )
        return str(cast(uuid.UUID, (await self._session.execute(stmt)).scalar_one()))

    async def add_results(self, run_id: str, results: Sequence[EvalResultRow]) -> int:
        """Append per-query results to a run; return the count inserted."""
        if not results:
            return 0
        values = [
            {
                "run_id": uuid.UUID(run_id),
                "query_id": uuid.UUID(r.query_id) if r.query_id else None,
                "retrieval_metrics": r.retrieval_metrics,
                "generation_metrics": r.generation_metrics,
                "trace_ref": r.trace_ref,
            }
            for r in results
        ]
        await self._session.execute(insert(schema.eval_results), values)
        return len(values)

    async def count_results(self, run_id: str) -> int:
        """Count results stored for a run."""
        stmt = (
            select(func.count())
            .select_from(schema.eval_results)
            .where(schema.eval_results.c.run_id == uuid.UUID(run_id))
        )
        return int((await self._session.execute(stmt)).scalar_one())
