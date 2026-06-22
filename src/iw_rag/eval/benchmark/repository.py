"""Persistence for Benchmark Studio reports (F-00093, S06).

The :class:`BenchmarkReportRepository` is the typed async insert / read
surface over the ``iw_rag.benchmark_reports`` table — one row per eval
run, holding the rolled-up RAG quality, latency, and resource
measurements.  It mirrors the existing :class:`iw_rag.eval.repository.
EvalRepository` and :class:`iw_rag.storage.repositories.
DocumentRepository` patterns:

* **Async, session-injected.**  The constructor takes an
  :class:`sqlalchemy.ext.asyncio.AsyncSession`; transaction scope is
  owned by the caller.
* **str ↔ uuid at the boundary.**  All public methods accept and return
  ``str`` ids; the SQL boundary converts to ``uuid.UUID`` for the
  Postgres ``uuid`` columns.
* **``.mappings()`` for read-back.**  ``JSONB`` columns round-trip
  through the driver as Python dicts/lists, so the read methods return
  plain dicts and JSONB fields are visible as nested dicts without
  extra parsing.

**Append-only surface — no update / delete methods (Invariant 2).**
The repository exposes only ``create_report`` + ``get_report`` +
``list_reports``.  A persisted ``benchmark_reports`` row is never
updated or deleted by this feature, and the FKs to ``eval_runs`` /
``eval_datasets`` do not cascade-delete reports.

**``created_by`` is the principal id from auth (Invariant 3).**  The
repository stores ``created_by`` exactly as passed — never derived
from the LLM, never from a request body field, never defaulted.  A
``None`` value means the offline batch path has no authenticated
principal (the orchestrator is responsible for that, not the
repository).

This module is **offline-only** (CONVENTIONS §10 / lld-eval-observability
§5 — two-clocks rule) and MUST NOT be imported by any hot-path module
(``api/``, ``agent/``, ``retrieval/``, ``generation/``).  The
``tests/unit/test_offline_only_import_guard.py`` static check enforces
this.

Integration tests (testcontainers Postgres, S09) exercise the DB
round-trip end-to-end.  The unit tests for this repository's
*assembly contract* (the mapping from a ``BenchmarkMetrics`` + perf
results to a row payload) live in
``tests/unit/eval/benchmark/test_assembly.py`` and use an in-memory
fake — that fake implements the same ``create_report``-shaped
protocol the assembly module consumes, so the test stays decoupled
from the production signature.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from sqlalchemy import insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from iw_rag.storage import schema


class BenchmarkReportRepository:
    """Async insert + read surface over ``iw_rag.benchmark_reports``.

    Append-only: the only mutation is :meth:`create_report`.  No
    update / delete method is exposed (Invariant 2 — reports are
    append-only history).

    Attributes:
        _session: The bound async session.
    """

    def __init__(self, session: AsyncSession) -> None:
        """Bind the repository to an open async session.

        Args:
            session: The active async session (transaction scope owned
                by the caller — the repository does not commit).
        """
        self._session = session

    async def create_report(
        self,
        *,
        run_id: str,
        dataset_id: str,
        report_label: str,
        git_sha: str | None,
        created_by: str | None,
        model_config: dict[str, Any],
        recall_at_5: float | None,
        recall_at_10: float | None,
        ndcg_at_5: float | None,
        ndcg_at_10: float | None,
        mrr: float | None,
        faithfulness_mean: float | None,
        faithfulness_kappa: float | None,
        context_precision: float | None,
        context_recall: float | None,
        factual_correctness: float | None,
        metrics_row: dict[str, Any],
        latency: dict[str, Any] | None,
        resource: dict[str, Any] | None,
        ci_95: dict[str, Any] | None,
        n_queries: int,
        n_judged: int | None,
    ) -> str:
        """Insert one ``benchmark_reports`` row; return its id.

        The argument list mirrors the S06 spec — explicit keyword-only
        parameters, one per logical input.  The T1 retrieval means
        (``recall_at_5/10``, ``ndcg_at_5/10``, ``mrr``), T2 reference-
        free judged means (``faithfulness_mean``, ``faithfulness_kappa``,
        ``context_precision``), and T3 reference-required means
        (``context_recall``, ``factual_correctness``) are promoted to
        the matching float columns (AC1) — the ``metrics_row`` blob
        (the full per-tier superset, paired by query index) is passed
        through to the ``metrics`` JSONB column verbatim.

        Args:
            run_id: The :class:`eval_runs` id (str at the boundary;
                converted to ``uuid`` for the SQL column).
            dataset_id: The :class:`eval_datasets` id.
            report_label: Human-readable report label (NOT NULL).
            git_sha: Git SHA of the configuration snapshot, or
                ``None``.
            created_by: The authenticated principal id from auth
                (Invariant 3 — never derived from the LLM, never
                from a request body field).  Stored verbatim.
            model_config: The static model configuration snapshot
                (JSONB).
            recall_at_5: T1 mean recall@5, or ``None`` when the run
                produced no T1 metrics.
            recall_at_10: T1 mean recall@10.
            ndcg_at_5: T1 mean nDCG@5.
            ndcg_at_10: T1 mean nDCG@10.
            mrr: T1 mean reciprocal rank.
            faithfulness_mean: T2 mean faithfulness (Lynx-8B), or
                ``None`` when the judge errored for every sample.
            faithfulness_kappa: Optional judge↔human kappa (from the
                existing kappa harness), or ``None`` when not
                measured for the run.
            context_precision: T2 mean context precision.
            context_recall: T3 mean context recall.  ``None`` when
                the dataset has no ``reference_answer`` (AC4 /
                Invariant 5) or when the judge errored.
            factual_correctness: T3 mean factual correctness.  Same
                nullability rule as ``context_recall``.
            metrics_row: The full per-tier blob (T1 + T2 + T3 means
                + per-query vectors paired by index), passed through
                to the ``metrics`` JSONB column.  The means are
                duplicated here so a read of just the blob still
                recovers the headline numbers (no join to the
                promoted columns required).
            latency: The latency JSONB (per-stage ms, TTFT/TPOT/E2E
                percentiles, ``model_load_ms``), or ``None`` for an
                unmeasured run.
            resource: The resource JSONB (GPU + vLLM server), or
                ``None`` for an unmeasured run.
            ci_95: The per-metric bootstrap 95% CI dict, or
                ``None`` when the S07 stats module is unavailable.
            n_queries: The number of golden queries in the run (NOT
                NULL).  Caller (assembly) is responsible for
                rejecting ``n_queries == 0``; the repository trusts
                the caller's invariant assertion.
            n_judged: The number of unique queries that were sent
                to a judge, or ``None`` when no judge was injected.

        Returns:
            The new report's id, as a str (uuid-stringified).

        Raises:
            sqlalchemy.exc.IntegrityError: If ``run_id`` or
                ``dataset_id`` does not exist in the parent tables
                (FK violation).  The repository does not catch it;
                the caller's session scope is responsible.
        """
        stmt = (
            insert(schema.benchmark_reports)
            .values(
                run_id=uuid.UUID(run_id),
                dataset_id=uuid.UUID(dataset_id),
                report_label=report_label,
                git_sha=git_sha,
                created_by=created_by,
                model_config=model_config,
                recall_at_5=recall_at_5,
                recall_at_10=recall_at_10,
                ndcg_at_5=ndcg_at_5,
                ndcg_at_10=ndcg_at_10,
                mrr=mrr,
                faithfulness_mean=faithfulness_mean,
                faithfulness_kappa=faithfulness_kappa,
                context_precision=context_precision,
                context_recall=context_recall,
                factual_correctness=factual_correctness,
                metrics=metrics_row,
                latency=latency,
                resource=resource,
                ci_95=ci_95,
                n_queries=n_queries,
                n_judged=n_judged,
            )
            .returning(schema.benchmark_reports.c.id)
        )
        return str(cast(uuid.UUID, (await self._session.execute(stmt)).scalar_one()))

    async def get_report(self, report_id: str) -> dict[str, Any] | None:
        """Read one row by id; return ``None`` when absent.

        JSONB columns round-trip as native Python dicts/lists — the
        caller does not need to re-parse.  All 24 columns are
        present in the returned dict (``n_queries`` always; the
        rest are nullable per the schema).

        Args:
            report_id: The report's id (str at the boundary).

        Returns:
            The row as a dict, or ``None`` when no row matches.
        """
        stmt = select(schema.benchmark_reports).where(
            schema.benchmark_reports.c.id == uuid.UUID(report_id)
        )
        row = (await self._session.execute(stmt)).mappings().first()
        return None if row is None else dict(row)

    async def list_reports(self, dataset_id: str) -> list[dict[str, Any]]:
        """List every report for a dataset, newest first.

        Backed by the ``benchmark_reports_dataset_idx`` B-tree index
        declared in the 0008 migration.  Empty list when the dataset
        has no reports.

        Args:
            dataset_id: The dataset id (str at the boundary).

        Returns:
            A list of row dicts, ordered by ``created_at DESC``.
        """
        stmt = (
            select(schema.benchmark_reports)
            .where(schema.benchmark_reports.c.dataset_id == uuid.UUID(dataset_id))
            .order_by(schema.benchmark_reports.c.created_at.desc())
        )
        return [dict(row) for row in (await self._session.execute(stmt)).mappings()]


__all__ = ["BenchmarkReportRepository"]
