"""iw_rag.benchmark_reports — Benchmark Studio measurement-core summary table (F-00093).

Creates the ``iw_rag.benchmark_reports`` table: one row per eval run holding
the rolled-up RAG quality (T1 retrieval, T2 reference-free judged, T3
reference-required judged), latency, and resource measurements so two runs
of the same golden question set can be lined up side by side.

The shape is the base schema from R-00121 §5.2 extended with the latency /
resource JSONB fields from R-00122 §"Recommendation".  T1 retrieval columns
(``recall_at_5/10``, ``ndcg_at_5/10``, ``mrr``) are promoted floats; the
T2 reference-free judged columns (``faithfulness_mean``,
``faithfulness_kappa``, ``context_precision``) and the T3 reference-required
judged columns (``context_recall``, ``factual_correctness``) live alongside.
The ``latency`` JSONB holds per-stage percentiles (retrieve / encode / rerank
/ generate), TTFT / TPOT / E2E percentiles, throughput, and
``model_load_ms``.  The ``resource`` JSONB holds the GPU baseline / peak /
incremental VRAM and mean / p95 utilisation plus the vLLM ``/metrics``
snapshot (TTFT/TPOT/queue/kv-cache percentiles and prefix-cache hit rate).
``ci_95`` is the per-metric 95% bootstrap CI; ``metrics`` is the full
per-tier blob (a superset of the promoted columns); ``model_config`` is the
static configuration snapshot so two reports are comparable only when this
column matches.

Column shapes mirror ``storage/schema.py`` (``benchmark_reports``) exactly.
Indexes (B-tree on ``dataset_id`` to back the list-by-dataset query path)
are created here, not declared in ``schema.py``, per that module's
"indexes live in the migration" convention.

Invariants (Feature_Design §Invariants 2, 3, 5, 7):
- Reports are append-only history.  FKs to ``eval_runs.id`` and
  ``eval_datasets.id`` are NOT cascade-delete, and no UPDATE/DELETE surface
  exists in :class:`BenchmarkReportRepository` (S06).
- ``created_by`` is the authenticated principal id supplied to the
  report-assembly call by the auth layer — NEVER derived from the LLM
  or from a request body field.
- Reference-required metrics (``context_recall``, ``factual_correctness``)
  are NULL when the dataset supplies no ``reference_answer``; the engine
  must NOT call a judge for them in that case.
- The migration is PENDING-rebased (``down_revision = "PENDING"``,
  the IW daemon rebases the real chain at merge time per
  ``script.py.mako``) and ``downgrade()`` fully drops the table + index,
  so a testcontainer round-trip (upgrade → downgrade → upgrade) holds.

Revision ID: 0008_benchmark_reports
Revises: PENDING
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0008_benchmark_reports"
down_revision: str | None = "0007_ephemeral_sessions"  # chains from ephemeral-sessions
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.benchmark_reports + the ``(dataset_id)`` B-tree index.

    Columns mirror ``storage/schema.py::benchmark_reports`` exactly:

    - id, run_id (FK eval_runs), dataset_id (FK eval_datasets), created_at
    - report_label (NOT NULL), git_sha, created_by
    - model_config (jsonb NOT NULL, default '{}'::jsonb)
    - T1: recall_at_5/10, ndcg_at_5/10, mrr (double precision, all nullable)
    - T2: faithfulness_mean, faithfulness_kappa, context_precision
    - T3: context_recall, factual_correctness (nullable; NULL when dataset
      has no reference_answer; engine MUST NOT call a judge for them)
    - latency, resource, ci_95 (jsonb, all nullable)
    - n_queries (NOT NULL), n_judged (nullable)
    - metrics (jsonb NOT NULL, default '{}'::jsonb) — the full per-tier blob

    FKs to ``iw_rag.eval_runs(id)`` and ``iw_rag.eval_datasets(id)`` with NO
    ``ON DELETE CASCADE`` (reports are append-only history — Invariant 2).
    """
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS iw_rag.benchmark_reports (
            id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            run_id                uuid NOT NULL
                                       REFERENCES iw_rag.eval_runs(id),
            dataset_id            uuid NOT NULL
                                       REFERENCES iw_rag.eval_datasets(id),
            created_at            timestamptz NOT NULL DEFAULT now(),
            report_label          text NOT NULL,
            git_sha               text,
            created_by            text,
            model_config          jsonb NOT NULL DEFAULT '{}'::jsonb,
            recall_at_5           double precision,
            recall_at_10          double precision,
            ndcg_at_5             double precision,
            ndcg_at_10            double precision,
            mrr                   double precision,
            faithfulness_mean     double precision,
            faithfulness_kappa    double precision,
            context_precision     double precision,
            context_recall        double precision,
            factual_correctness   double precision,
            latency               jsonb,
            resource              jsonb,
            ci_95                 jsonb,
            n_queries             integer NOT NULL,
            n_judged              integer,
            metrics               jsonb NOT NULL DEFAULT '{}'::jsonb
        )
        """
    )
    # B-tree on dataset_id — backs the list-by-dataset query path
    # (BenchmarkReportRepository.list_by_dataset in S06).
    op.execute(
        "CREATE INDEX IF NOT EXISTS benchmark_reports_dataset_idx "
        "ON iw_rag.benchmark_reports (dataset_id)"
    )


def downgrade() -> None:
    """Drop the dataset_id index then the benchmark_reports table (full reversal).

    Drops in reverse creation order: the index, then the table (CASCADE drops
    anything else depending on it).  No data migration was required, so the
    downgrade is lossless and the upgrade → downgrade → upgrade round-trip
    holds (Invariant 7).
    """
    op.execute("DROP INDEX IF EXISTS iw_rag.benchmark_reports_dataset_idx")
    op.execute("DROP TABLE IF EXISTS iw_rag.benchmark_reports CASCADE")
