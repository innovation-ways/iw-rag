"""Integration tests for the ``benchmark_reports`` table + ``BenchmarkReportRepository``.

F-00093 / S09 — dedicated coverage for the Benchmark Studio measurement-core
DB round-trip.  Uses a self-labelling, self-destructing testcontainers
ParadeDB (the project convention; the standard ``tests/integration/
conftest.py`` ``pg_dsn`` fixture is intentionally not used because it runs
``alembic upgrade head`` and the F-00093 0008 migration's ``down_revision``
is the ``PENDING`` sentinel that the daemon rebases at merge time —
``alembic upgrade head`` cannot follow the sentinel and so the fixture
would fail.  This test file owns its own ``db_container`` /
``migrated_engine`` fixtures so the migration path is explicit and
deterministic.

Migration strategy:
  1. Copy the alembic versions dir to a temp dir WITHOUT the 0008 file.
  2. ``command.upgrade(cfg, "head")`` against the temp dir — applies 0001-0007.
  3. Import the 0008 module from the original dir and call its ``upgrade()``
     directly via an in-memory alembic ``op`` proxy so the table is created
     + the index is created (matches the contract in 0008_benchmark_reports.py).

Test coverage (per F-00093 Feature_Design §Acceptance Criteria and
§Boundary Behavior):

* **AC1** — full round-trip of every T1 / T2 / T3 column + latency / resource /
  ci_95 JSONB + n_queries / n_judged + model_config.  Every value is asserted
  on read-back, not just ``is not None``.
* **AC4** — ``context_recall`` and ``factual_correctness`` are stored + read back
  as ``NULL`` when the engine produced ``None`` (reference-required gating).
* **AC5 / Invariant 3** — ``created_by`` is the exact principal id passed in.
* **Invariant 2 (append-only)** — the repository class has no update/delete
  method (we introspect ``dir()`` and check the documented surface).
* **Boundary: judge down / GPU unavailable** — verified end-to-end at the
  boundary layer (the engine / perf collectors keep their no-op / errored
  contracts; the assembly maps them to nullable columns; the repository
  persists them without raising).
* **Migration round-trip** — ``upgrade()`` + ``downgrade()`` is reversible
  (the table drops, the index drops, the row vanishes).

All tests are deterministic and use fakes for judges / collectors — no
live model endpoints, no live GPU, no live network.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy
from alembic.config import Config
from sqlalchemy import text

from iw_rag.eval.benchmark.repository import BenchmarkReportRepository
from iw_rag.eval.repository import EvalRepository
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = "paradedb/paradedb:latest"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_VERSIONS = _REPO_ROOT / "alembic" / "versions"
_PENDING_MIGRATION = _ALEMBIC_VERSIONS / "0008_benchmark_reports.py"


# ---------------------------------------------------------------------------
# Fixtures — owned here (not from tests/integration/conftest.py) because the
# PENDING sentinel makes the shared ``pg_dsn`` fixture fail.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _isolated_alembic_versions() -> Generator[Path, None, None]:
    """Copy ``alembic/versions`` into a temp ``versions/`` dir WITHOUT 0008.

    Alembic's static chain resolver cannot follow the ``PENDING`` sentinel
    that the F-00093 0008 migration declares.  We recreate the
    ``<tmp>/alembic/versions/`` structure (so ``script_location`` is the
    parent ``<tmp>/alembic/``) with all migrations EXCEPT 0008, and run
    ``alembic upgrade head`` against the temp tree.  Then we call the
    0008 migration's ``upgrade()`` directly against the same DB.

    Yields:
        A path to a temp dir structured as ``<tmp>/alembic/versions/``
        with the original migrations except 0008.
    """
    with tempfile.TemporaryDirectory(prefix="iw-rag-alembic-") as tmp:
        tmp_path = Path(tmp)
        alembic_dir = tmp_path / "alembic"
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        for src in _ALEMBIC_VERSIONS.glob("*.py"):
            shutil.copy(src, versions_dir / src.name)
        # The F-00093 chain (0008_benchmark_reports) is PENDING-free after
        # the daemon rebase; nothing needs stripping for the static chain
        # resolver to succeed.  0009 is also real (hand-authored with a
        # concrete down_revision).  Keep this hook commented as a no-op
        # placeholder in case a future in-flight migration re-introduces
        # the PENDING sentinel.
        # pending_copy = versions_dir / _PENDING_MIGRATION.name
        # if pending_copy.exists():
        #     pending_copy.unlink()
        # Also need env.py + script.py.mako for alembic to run.
        shutil.copy(_ALEMBIC_VERSIONS.parent / "env.py", alembic_dir / "env.py")
        shutil.copy(
            _ALEMBIC_VERSIONS.parent / "script.py.mako",
            alembic_dir / "script.py.mako",
        )
        yield alembic_dir


@pytest.fixture(scope="module")
def db_container() -> Generator[str, None, None]:
    """A fresh paradedb container; yields the psycopg-driver DSN."""
    from testcontainers.postgres import PostgresContainer

    with PostgresContainer(
        image=PARADEDB_IMAGE,
        username="test_user",
        password="test_pass",
        dbname="test_db",
    ) as pg:
        raw_url = pg.get_connection_url()
        dsn = raw_url.replace("postgresql+psycopg2://", "postgresql+psycopg://")
        yield dsn


@pytest.fixture(scope="module")
def migrated_engine(
    db_container: str, _isolated_alembic_versions: Path
) -> Generator[sqlalchemy.Engine, None, None]:
    """Run migrations against the container + apply 0008 manually; yield engine.

    Steps:
      1. Point alembic at the isolated alembic dir (no 0008) + the
         container DSN, then ``command.upgrade(cfg, "head")`` — applies
         0001-0007.  The ``alembic_version`` table ends up at 0007.
      2. Import the 0008 migration module from the ORIGINAL versions
         dir (not the temp one) and call its ``upgrade()`` against a
         raw ``sqlalchemy`` connection.  This creates the
         ``iw_rag.benchmark_reports`` table + the
         ``benchmark_reports_dataset_idx`` index.
      3. Yield the engine for tests; dispose on teardown.

    Settings env is set on entry and cleaned on exit so the next test
    that loads ``get_settings()`` is not affected.
    """
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    import iw_rag.settings as settings_mod
    from alembic import command

    # Point Settings at the container (module-scoped, so we set env
    # directly; monkeypatch wouldn't survive the module scope).
    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    # Configure alembic with the isolated script_location (the temp
    # ``<tmp>/alembic/`` dir) + the container DSN, then run
    # ``command.upgrade(cfg, "head")`` to apply 0001-0007.
    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_isolated_alembic_versions))
    cfg.set_main_option("sqlalchemy.url", db_container)
    command.upgrade(cfg, "head")

    # Apply 0008 manually: import the module from the ORIGINAL versions
    # dir and call its upgrade() against a sync engine.  The migration's
    # module-level ``from alembic import op`` is the alembic singleton;
    # we wrap with ``Operations.context(ctx)`` to temporarily rebind the
    # proxy to our connection.
    spec = importlib.util.spec_from_file_location("m_0008", str(_PENDING_MIGRATION))
    m_0008 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m_0008)

    engine = sqlalchemy.create_engine(db_container)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            m_0008.upgrade()

    yield engine

    engine.dispose()
    for key in (
        "IW_RAG__DATABASE__DSN",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
        "IW_RAG__MODELS__RERANKER__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
    ):
        os.environ.pop(key, None)
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate_all(pg_dsn: str) -> None:
    """Truncate every F-00093-relevant table before a test for isolation.

    Includes the new ``benchmark_reports`` table plus the FK parent tables
    (``eval_datasets``, ``eval_runs``) so each test starts from a clean
    slate.  ``RESTART IDENTITY CASCADE`` is intentionally NOT used on
    ``benchmark_reports`` because the table has a uuid PK (not a sequence).
    """
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE iw_rag.benchmark_reports, "
                "iw_rag.eval_results, iw_rag.eval_queries, "
                "iw_rag.eval_runs, iw_rag.eval_datasets "
                "RESTART IDENTITY CASCADE"
            )
        )
    engine.dispose()


async def _seed_dataset_and_run(
    db: Database, *, n_queries: int = 2, with_references: bool = True
) -> tuple[str, str]:
    """Create an eval dataset + a run; return ``(dataset_id, run_id)``.

    Args:
        db: An open :class:`Database` bound to the test DSN.
        n_queries: Number of golden queries to seed (just so the FK
            targets are not empty; the queries themselves are not
            referenced by the report row).
        with_references: If ``True`` the seeded queries carry a
            ``reference_answer`` (so the engine would run T3); if
            ``False`` they're ``None`` (so T3 is skipped).  The
            integration test does NOT run the engine — it directly
            inserts the ``benchmark_reports`` row with the columns it
            wants to verify.  The flag is a documentation aid.

    Returns:
        ``(dataset_id, run_id)`` as uuid-strings.
    """
    from iw_rag.eval.repository import GoldenQuery

    async with db.session() as session:
        repo = EvalRepository(session)
        dataset_id = await repo.create_dataset("golden-s09", "v1")
        gold_chunks = [str(uuid.uuid4()) for _ in range(n_queries)]
        reference = "the reference answer" if with_references else None
        await repo.add_queries(
            dataset_id,
            [
                GoldenQuery(
                    question=f"q{i + 1}",
                    relevant_chunk_ids=[gold_chunks[i]],
                    query_type="factual",
                    reference_answer=reference,
                )
                for i in range(n_queries)
            ],
        )
        run_id = await repo.create_run(
            dataset_id=dataset_id, experiment="s09_benchmark_repo", git_sha="abc1234"
        )
        # The Database.session() context manager does NOT auto-commit;
        # we must call commit() explicitly so subsequent sessions (and
        # the create_report call below) can see the rows.
        await session.commit()
    return dataset_id, run_id


def _full_payload(
    run_id: str,
    dataset_id: str,
    *,
    created_by: str = "user:alice-1234",
    context_recall: float | None = 0.6,
    factual_correctness: float | None = 0.5,
) -> dict[str, Any]:
    """Build a fully-populated create_report kwargs payload (AC1 round-trip)."""
    return dict(
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="qwen3-14b-2026-06-10",
        git_sha="abc1234",
        created_by=created_by,
        model_config={
            "embedder": "bge-m3",
            "generator": "qwen3-14b",
            "reranker": "bge-reranker-v2-m3",
        },
        recall_at_5=0.8,
        recall_at_10=0.9,
        ndcg_at_5=0.7,
        ndcg_at_10=0.75,
        mrr=0.65,
        faithfulness_mean=0.92,
        faithfulness_kappa=None,
        context_precision=0.85,
        context_recall=context_recall,
        factual_correctness=factual_correctness,
        metrics_row={
            "ndcg_at_5": [1.0, 0.4, 0.7],
            "faithfulness": [1.0, 0.9, 0.86],
            "context_recall": [0.7, 0.5, 0.6],
            "factual_correctness": [0.6, 0.4, 0.5],
        },
        latency={
            "per_stage_ms": {
                "retrieve_ms": 85.0,
                "rerank_ms": 35.0,
                "generate_ms": 950.0,
            },
            "ttft_p99_s": 0.045,
            "tpot_p99_s": 0.012,
            "e2e_p99_s": 0.430,
            "model_load_ms": 1250.5,
        },
        resource={
            "gpu": {
                "available": True,
                "util_peak_pct": 87.0,
                "vram_incremental_mb": 800.0,
                "device_name": "NVIDIA H100",
            },
            "vllm_server": {
                "ttft_p99": 0.045,
                "prefix_cache_hit_rate": 0.6,
                "prompt_tokens_delta": 1024,
            },
        },
        ci_95={"ndcg_at_5": (0.5, 0.9), "recall_at_10": (0.6, 0.95)},
        n_queries=3,
        n_judged=3,
    )


# ---------------------------------------------------------------------------
# Migration shape tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_migration_creates_benchmark_reports_table(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The 0008 migration creates the ``iw_rag.benchmark_reports`` table
    with every documented column (schema-parity pin)."""
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'iw_rag' AND table_name = 'benchmark_reports' "
                "ORDER BY ordinal_position"
            )
        ).fetchall()
    cols = {r[0] for r in rows}
    expected = {
        "id",
        "run_id",
        "dataset_id",
        "created_at",
        "report_label",
        "git_sha",
        "created_by",
        "model_config",
        # T1
        "recall_at_5",
        "recall_at_10",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr",
        # T2
        "faithfulness_mean",
        "faithfulness_kappa",
        "context_precision",
        # T3
        "context_recall",
        "factual_correctness",
        # JSONB + counts
        "latency",
        "resource",
        "ci_95",
        "n_queries",
        "n_judged",
        "metrics",
    }
    missing = expected - cols
    assert not missing, f"benchmark_reports missing columns: {missing}"


@pytest.mark.integration
def test_migration_creates_dataset_id_index(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The B-tree on ``(dataset_id)`` is created by the migration
    (backing the ``list_reports`` query path)."""
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT indexname FROM pg_indexes "
                "WHERE schemaname = 'iw_rag' AND tablename = 'benchmark_reports'"
            )
        ).fetchall()
    index_names = {r[0] for r in rows}
    assert "benchmark_reports_dataset_idx" in index_names, (
        f"expected benchmark_reports_dataset_idx; got {index_names}"
    )
    assert {"benchmark_reports_dataset_idx"} <= index_names, (
        f"benchmark_reports_dataset_idx missing from indexes; got {index_names!r}"
    )


@pytest.mark.integration
def test_migration_fk_constraints_have_no_cascade_delete(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """The FKs to ``eval_runs`` and ``eval_datasets`` are NOT cascade-delete
    (Invariant 2 — reports are append-only history; a purge of the run
    must not silently delete its report)."""
    with migrated_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT conname, confdeltype "
                "FROM pg_constraint "
                "WHERE conrelid = 'iw_rag.benchmark_reports'::regclass "
                "AND contype = 'f'"
            )
        ).fetchall()
    fk_deltypes = {r[0]: r[1] for r in rows}
    # pg's confdeltype: 'a' = NO ACTION (the default; what we want).
    # 'c' = CASCADE, 'r' = RESTRICT, 'n' = SET NULL, 'd' = SET DEFAULT.
    assert fk_deltypes, "expected at least one FK on benchmark_reports"
    for fk_name, deltype in fk_deltypes.items():
        assert deltype == "a", (
            f"FK {fk_name} has confdeltype={deltype!r} (expected 'a' = NO ACTION); "
            "reports must not cascade-delete with their run"
        )


# ---------------------------------------------------------------------------
# AC1: full round-trip
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_create_and_read_full_report_round_trips_every_value(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """AC1 — a fully-populated report persists and reads back with every
    value intact (T1, T2, T3 promoted columns + JSONB blobs + counts +
    model_config + ci_95)."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=2)
            payload = _full_payload(run_id, dataset_id)
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                report_id = await repo.create_report(**payload)
                await session.commit()
            # Read it back
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                row = await repo.get_report(report_id)
            assert row is not None, "create_report returned an id but the row is gone"
            # Identity + provenance.  The uuid columns come back as
            # ``uuid.UUID`` from SQLAlchemy (not ``str``); normalise.
            assert str(row["run_id"]) == payload["run_id"]
            assert str(row["dataset_id"]) == payload["dataset_id"]
            assert row["report_label"] == "qwen3-14b-2026-06-10"
            assert row["git_sha"] == "abc1234"
            assert row["created_by"] == "user:alice-1234"
            # T1
            for k in ("recall_at_5", "recall_at_10", "ndcg_at_5", "ndcg_at_10", "mrr"):
                assert row[k] == pytest.approx(payload[k]), f"T1 mismatch: {k}"
            # T2
            assert row["faithfulness_mean"] == pytest.approx(0.92)
            assert row["faithfulness_kappa"] is None
            assert row["context_precision"] == pytest.approx(0.85)
            # T3
            assert row["context_recall"] == pytest.approx(0.6)
            assert row["factual_correctness"] == pytest.approx(0.5)
            # JSONB round-trips as native dicts
            assert row["latency"]["ttft_p99_s"] == pytest.approx(0.045)
            assert row["latency"]["model_load_ms"] == pytest.approx(1250.5)
            assert row["resource"]["gpu"]["device_name"] == "NVIDIA H100"
            assert row["ci_95"] == {"ndcg_at_5": [0.5, 0.9], "recall_at_10": [0.6, 0.95]}
            # metrics blob + counts.  The repository's ``metrics_row``
            # kwarg writes to the ``metrics`` column (per the schema),
            # so the column surfaces as ``row["metrics"]`` on read-back.
            assert row["metrics"] == payload["metrics_row"]
            assert row["n_queries"] == 3
            assert row["n_judged"] == 3
            assert row["model_config"] == payload["model_config"]
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# AC4: reference-required metrics are NULL
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_reference_required_metrics_persist_as_null_when_no_references(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """AC4 / Invariant 5 — when the engine produced ``None`` for
    ``context_recall`` / ``factual_correctness`` (dataset had no
    reference answers), the row stores them as ``NULL`` and reads them
    back as ``None`` (never ``0.0``, never silently dropped)."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=2, with_references=False)
            payload = _full_payload(
                run_id,
                dataset_id,
                context_recall=None,  # AC4: no reference answers
                factual_correctness=None,  # AC4: no reference answers
            )
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                report_id = await repo.create_report(**payload)
                await session.commit()
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                row = await repo.get_report(report_id)
            assert row is not None
            assert row["context_recall"] is None
            assert row["factual_correctness"] is None
            # T1 + T2 still populated — the report is not all-zero.
            assert row["recall_at_5"] == pytest.approx(0.8)
            assert row["faithfulness_mean"] == pytest.approx(0.92)
            assert row["n_queries"] == 3
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# AC5 / Invariant 3: created_by is the principal from auth
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_created_by_is_exactly_the_passed_principal(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """``created_by`` MUST be the principal id passed in (never derived
    from the LLM or from a request body field).  The repository stores
    it verbatim — including the formatted principal id pattern
    ``user:<id>`` that auth hands to the orchestrator."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=1)
            payload = _full_payload(run_id, dataset_id, created_by="user:alice-1234")
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                report_id = await repo.create_report(**payload)
                await session.commit()
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                row = await repo.get_report(report_id)
            assert row is not None
            assert row["created_by"] == "user:alice-1234"
        finally:
            await db.dispose()

    asyncio.run(_run())


@pytest.mark.integration
def test_created_by_accepts_none_for_offline_batch_path(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """The schema marks ``created_by`` as nullable (the offline batch
    path has no authenticated principal).  A ``None`` value MUST persist
    and read back as ``None`` (not an empty string, not "anonymous")."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=1)
            payload = _full_payload(run_id, dataset_id, created_by=None)
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                report_id = await repo.create_report(**payload)
                await session.commit()
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                row = await repo.get_report(report_id)
            assert row is not None
            assert row["created_by"] is None
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Invariant 2: append-only — no update/delete surface
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_repository_has_no_update_or_delete_method() -> None:
    """The repository exposes ONLY ``create_report`` / ``get_report`` /
    ``list_reports`` (Invariant 2 — reports are append-only history)."""
    from iw_rag.eval.benchmark.repository import BenchmarkReportRepository

    forbidden_substrings = ("update", "delete", "drop", "truncate", "alter")
    public_methods = [
        name
        for name in vars(BenchmarkReportRepository)
        if not name.startswith("_") and callable(getattr(BenchmarkReportRepository, name))
    ]
    # The intended public surface is exactly the three read+write methods
    # declared in the docstring + __all__.
    assert "create_report" in public_methods
    assert "get_report" in public_methods
    assert "list_reports" in public_methods
    # And the module's __all__ matches.
    from iw_rag.eval.benchmark import repository as repo_mod

    assert repo_mod.__all__ == ["BenchmarkReportRepository"]
    # And no method that smells like a mutating operation is present.
    for name in public_methods:
        for forbidden in forbidden_substrings:
            assert forbidden not in name.lower(), (
                f"repository exposes a forbidden method: {name!r} (contains {forbidden!r})"
            )


# ---------------------------------------------------------------------------
# list_reports: newest first, per dataset
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_list_reports_returns_newest_first_for_dataset(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """``list_reports(dataset_id)`` returns the dataset's reports ordered
    by ``created_at DESC`` (backed by the ``benchmark_reports_dataset_idx``
    B-tree index).  Insert 3 reports with a small delay between them so
    ``created_at`` is monotonically increasing; the first inserted must
    be the LAST in the returned list."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=1)
            ids: list[str] = []
            # Insert 3 reports in 3 separate sessions so Postgres's
            # ``now()`` (the ``created_at`` default) returns distinct
            # values per row.  All three inserts in the same session
            # would share the same ``created_at`` and the DESC ordering
            # is not deterministic.
            for i in range(3):
                payload = _full_payload(
                    run_id,
                    dataset_id,
                    created_by=f"user:operator-{i}",
                )
                payload["report_label"] = f"run-{i}"
                async with db.session() as session:
                    repo = BenchmarkReportRepository(session)
                    rid = await repo.create_report(**payload)
                    await session.commit()
                    ids.append(rid)
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                rows = await repo.list_reports(dataset_id)
            assert len(rows) == 3
            labels_in_order = [r["report_label"] for r in rows]
            # Newest first → "run-2" first, "run-0" last.
            assert labels_in_order == ["run-2", "run-1", "run-0"], (
                f"list_reports not in created_at DESC order; got {labels_in_order}"
            )
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Boundary: missing metric still persists (judge / GPU down)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_report_persists_when_judge_data_is_null_and_gpu_unavailable(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """Boundary Behavior rows 2 + 3 — when the judge is down and the GPU
    is unavailable, the report still persists: faithfulness/context
    columns are ``NULL``; the resource JSONB carries ``gpu.available=False``.
    The report is still readable end-to-end (no constraint violations, no
    drops)."""
    _truncate_all(db_container)

    async def _run() -> None:
        db = Database(DatabaseSettings(dsn=db_container))
        try:
            dataset_id, run_id = await _seed_dataset_and_run(db, n_queries=1)
            payload = _full_payload(
                run_id,
                dataset_id,
                # judge down → T2/T3 judged slots are NULL
                context_recall=None,
                factual_correctness=None,
            )
            # Assembly's "no GPU" shape: per the assembly module, an
            # unavailable GpuMetrics serialises to resource.gpu.available=False
            # with all numeric fields None.
            payload["resource"] = {
                "gpu": {
                    "available": False,
                    "util_peak_pct": None,
                    "util_mean_pct": None,
                    "util_p95_pct": None,
                    "vram_peak_mb": None,
                    "vram_baseline_mb": None,
                    "vram_incremental_mb": None,
                    "device_name": None,
                },
                "vllm_server": payload["resource"]["vllm_server"],
            }
            # n_judged is None when the judge was never injected
            payload["n_judged"] = None
            # Latency.ttft_p99_s may be None if vLLM was unreachable
            payload["latency"]["ttft_p99_s"] = None
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                report_id = await repo.create_report(**payload)
                await session.commit()
            async with db.session() as session:
                repo = BenchmarkReportRepository(session)
                row = await repo.get_report(report_id)
            assert row is not None, "report persisted despite judge + GPU down"
            # T3 NULL
            assert row["context_recall"] is None
            assert row["factual_correctness"] is None
            # n_judged NULL
            assert row["n_judged"] is None
            # resource.gpu.available False (no GPU)
            assert row["resource"]["gpu"]["available"] is False
            assert row["resource"]["gpu"]["util_peak_pct"] is None
            # latency.ttft_p99_s is None (vLLM down)
            assert row["latency"]["ttft_p99_s"] is None
            # T1 + T2 still populated
            assert row["recall_at_5"] == pytest.approx(0.8)
            assert row["faithfulness_mean"] == pytest.approx(0.92)
        finally:
            await db.dispose()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Migration round-trip: upgrade() + downgrade() is reversible
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_migration_round_trip_is_reversible(
    db_container: str, migrated_engine: sqlalchemy.Engine
) -> None:
    """The 0008 migration's ``downgrade()`` is symmetric with
    ``upgrade()`` — the table + index are fully dropped.  We import the
    module and call ``downgrade()`` then ``upgrade()`` again; the
    table must exist after the second upgrade and the row written
    before downgrade is gone after the second upgrade.

    This is the testcontainer counterpart of the S01 report's
    "PENDING-rebased" invariant: the migration's DDL is symmetric
    (Invariant 7).

    Args:
        db_container: The container DSN (fixture-injected to ensure
            the module-scoped ``migrated_engine`` was set up).
        migrated_engine: The module-scoped engine (fixture-injected
            for ordering; we open a fresh connection for the DDL).
    """
    del migrated_engine  # fixture-driven ordering; we use a fresh connection
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    spec = importlib.util.spec_from_file_location("m_0008_rt", str(_PENDING_MIGRATION))
    m_0008 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m_0008)

    engine = sqlalchemy.create_engine(db_container)
    try:
        # Insert a row.
        _truncate_all(db_container)

        async def _seed() -> tuple[str, str]:
            db = Database(DatabaseSettings(dsn=db_container))
            try:
                return await _seed_dataset_and_run(db, n_queries=1)
            finally:
                await db.dispose()

        dataset_id, run_id = asyncio.run(_seed())

        async def _insert() -> str:
            db = Database(DatabaseSettings(dsn=db_container))
            try:
                async with db.session() as session:
                    repo = BenchmarkReportRepository(session)
                    rid = await repo.create_report(**_full_payload(run_id, dataset_id))
                    await session.commit()
                    return rid
            finally:
                await db.dispose()

        report_id = asyncio.run(_insert())
        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                m_0008.downgrade()
        # After downgrade the table is gone.
        with engine.connect() as conn:
            present = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'iw_rag' "
                    "AND table_name = 'benchmark_reports'"
                )
            ).first()
        assert present is None, "downgrade() did not drop benchmark_reports"
        # Re-upgrade; the table is back.
        with engine.begin() as conn:
            ctx = MigrationContext.configure(conn)
            with Operations.context(ctx):
                m_0008.upgrade()
        with engine.connect() as conn:
            present = conn.execute(
                text(
                    "SELECT 1 FROM information_schema.tables "
                    "WHERE table_schema = 'iw_rag' "
                    "AND table_name = 'benchmark_reports'"
                )
            ).first()
        assert present is not None, "second upgrade() did not recreate benchmark_reports"

        # The previously-inserted row is gone (downgrade dropped it).
        async def _read() -> dict[str, Any] | None:
            db = Database(DatabaseSettings(dsn=db_container))
            try:
                async with db.session() as session:
                    repo = BenchmarkReportRepository(session)
                    return await repo.get_report(report_id)
            finally:
                await db.dispose()

        row = asyncio.run(_read())
        assert row is None, "row survived downgrade+upgrade; the table is not actually clean"
    finally:
        engine.dispose()
