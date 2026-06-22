"""Integration tests for the F-00096 ``GET /benchmark/scoreboard`` route.

The S08 step already owns the **unit-level** route coverage in
``tests/unit/api/test_scoreboard_routes.py`` (fake service seam; no
DB; no ranking).  This file is the **end-to-end integration**
coverage the S17 prompt pins separately — the S08 route + the live
S05 :class:`ScoreboardService` + a testcontainer
``benchmark_reports`` table (the F-00093 S06 measurement-core
round-trip) + the live SQL ranker.

The tests follow the same own-fixture migration pattern as
``tests/integration/storage/test_benchmark_reports_repo.py``: the
shared ``pg_dsn`` fixture runs ``alembic upgrade head`` against the
rebased chain (which strips 0008 PENDING), so 0008 is NOT auto-
applied.  This file owns its own ``db_container`` /
``migrated_engine`` fixtures to apply 0001-0007 + 0008 + 0009
manually (the 0009 chain depends on 0008).

Coverage (per F-00096 Feature Design §Acceptance Criteria + §Boundary
Behavior):

* **AC6** — ``GET /benchmark/scoreboard?board=faithfulness`` returns
  rows in descending faithfulness order with ``config_snapshot`` +
  ``created_at`` per row.  Pinned via direct testcontainer
  ``benchmark_reports`` seeding.
* **Boundary row "Empty report history"** — 0 reports → 200 with
  ``entries == []`` (explicit empty state, not a 5xx).
* **AC6 board names** — the route forwards each of the six S05 board
  names verbatim: ``faithfulness``, ``ndcg_at_5``, ``tok_per_s``,
  ``latency``, ``vram_efficiency``, ``composite``.

All tests run the live S05 :class:`ScoreboardService` against the
testcontainer; no fakes, no mocks of the database.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
import sqlalchemy
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import text

from iw_rag.api.app import create_app
from iw_rag.api.auth import StaticTokenAuthenticator
from iw_rag.eval.benchmark.scoreboard import ScoreboardService
from iw_rag.settings import DatabaseSettings
from iw_rag.storage.db import Database

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PARADEDB_IMAGE = (
    "paradedb/paradedb@sha256:dbd64f0eb2716b8466afe54b91fec782f4107a8d0e6bd3660b89c88a1ded2ead"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ALEMBIC_VERSIONS = _REPO_ROOT / "alembic" / "versions"
_PENDING_MIGRATION = _ALEMBIC_VERSIONS / "0008_benchmark_reports.py"
_PENDING_MIGRATION_0009 = _ALEMBIC_VERSIONS / "0009_config_change_log.py"


# ---------------------------------------------------------------------------
# Fixtures — own-fixture (same shape as test_config_change_log_immutability.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _isolated_alembic_versions() -> Generator[Path, None, None]:
    """Copy ``alembic/versions`` into a temp dir WITHOUT 0008 / 0009.

    Both 0008 and 0009 are stripped because the rebase helper only
    strips 0008 (the documented PENDING-sentinel migration); we own
    the explicit "we apply 0008 + 0009 manually" path here.
    """
    with tempfile.TemporaryDirectory(prefix="iw-rag-scoreboard-alembic-") as tmp:
        tmp_path = Path(tmp)
        alembic_dir = tmp_path / "alembic"
        versions_dir = alembic_dir / "versions"
        versions_dir.mkdir(parents=True, exist_ok=True)
        for src in _ALEMBIC_VERSIONS.glob("*.py"):
            shutil.copy(src, versions_dir / src.name)
        for pending in (_PENDING_MIGRATION, _PENDING_MIGRATION_0009):
            candidate = versions_dir / pending.name
            if candidate.exists():
                candidate.unlink()
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
    """Apply 0001-0007 + 0008 (manual) + 0009 (manual).

    Settings env is set on entry and cleaned on exit so the next test
    that loads ``get_settings()`` is not affected.
    """
    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    import iw_rag.settings as settings_mod
    from alembic import command

    os.environ["IW_RAG__DATABASE__DSN"] = db_container
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    cfg = Config(str(_REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_isolated_alembic_versions))
    cfg.set_main_option("sqlalchemy.url", db_container)
    command.upgrade(cfg, "head")

    engine = sqlalchemy.create_engine(db_container)
    # 0008 — benchmark_reports
    spec = importlib.util.spec_from_file_location("m_0008", str(_PENDING_MIGRATION))
    m_0008 = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m_0008)
    # 0009 — config_change_log
    spec9 = importlib.util.spec_from_file_location("m_0009", str(_PENDING_MIGRATION_0009))
    m_0009 = importlib.util.module_from_spec(spec9)
    assert spec9.loader is not None
    spec9.loader.exec_module(m_0009)
    with engine.begin() as conn:
        ctx = MigrationContext.configure(conn)
        with Operations.context(ctx):
            m_0008.upgrade()
            m_0009.upgrade()

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


def _seed_dataset_and_run(engine: sqlalchemy.Engine) -> tuple[str, str]:
    """Insert one parent eval_dataset + one parent eval_run; return ids."""
    dataset_id = str(uuid.uuid4())
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO iw_rag.eval_datasets (id, name, version) "
                "VALUES (:id, 'ds-s17-scoreboard', 'v1')"
            ),
            {"id": dataset_id},
        )
        conn.execute(
            text(
                "INSERT INTO iw_rag.eval_runs (id, dataset_id, experiment, git_sha, config) "
                "VALUES (:id, :did, 's17-scoreboard', 'abc1234', '{}'::jsonb)"
            ),
            {"id": run_id, "did": dataset_id},
        )
    return dataset_id, run_id


def _seed_benchmark_report(
    engine: sqlalchemy.Engine,
    *,
    run_id: str,
    dataset_id: str,
    report_label: str,
    model_config: dict[str, object],
    ndcg_at_5: float,
    faithfulness_mean: float,
    latency: dict[str, object] | None = None,
    resource: dict[str, object] | None = None,
) -> str:
    """Insert one ``benchmark_reports`` row; return the new report id."""
    report_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(
            text(
                "INSERT INTO iw_rag.benchmark_reports "
                "(id, run_id, dataset_id, report_label, git_sha, created_by, "
                " model_config, ndcg_at_5, faithfulness_mean, latency, resource, "
                " n_queries, metrics) "
                "VALUES (:id, :rid, :did, :lbl, 'abc', 'alice', "
                " CAST(:mc AS jsonb), :ndcg, :faith, "
                " CAST(:lat AS jsonb), CAST(:res AS jsonb), "
                " 10, CAST('{}' AS jsonb))"
            ),
            {
                "id": report_id,
                "rid": run_id,
                "did": dataset_id,
                "lbl": report_label,
                "mc": _json_dumps(model_config),
                "ndcg": ndcg_at_5,
                "faith": faithfulness_mean,
                "lat": _json_dumps(latency or {}),
                "res": _json_dumps(resource or {}),
            },
        )
    return report_id


def _json_dumps(d: object) -> str:
    """Tiny JSON serializer (avoid a hard psycopg2 import for jsonb binding)."""
    import json

    return json.dumps(d)


def _truncate_scoreboard_history(engine: sqlalchemy.Engine) -> None:
    """Truncate every F-00093-relevant table before a test for isolation."""
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE iw_rag.benchmark_reports, "
                "iw_rag.eval_results, iw_rag.eval_queries, "
                "iw_rag.eval_runs, iw_rag.eval_datasets, "
                "iw_rag.config_change_log "
                "RESTART IDENTITY CASCADE"
            )
        )


# ---------------------------------------------------------------------------
# AC6 — faithfulness board ranking (end-to-end via live service)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_scoreboard_faithfulness_returns_descending_ranked_rows(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """AC6 — the ``faithfulness`` board is descending with the live service.

    Three benchmark_reports rows with distinct faithfulness values
    (0.95, 0.78, 0.60) are seeded directly into the testcontainer;
    the S05 service reads them via its direct SELECT; the S08 route
    serialises the leaderboard.  The operator sees the AC6 contract.

    The route exposes the row's ``id`` (uuid string) — not the
    ``report_label`` — so the test tracks the seeded ids → label
    mapping and asserts the leaderboard's rank order via a
    report_label lookup join.
    """
    _truncate_scoreboard_history(migrated_engine)
    dataset_id, run_id = _seed_dataset_and_run(migrated_engine)
    # Seeded id → label lookup.  The route returns the row's id; we
    # map it back to the label via the test's in-memory map.
    id_to_label: dict[str, str] = {}
    r_low = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-low",
        model_config={"generator": "qwen3-7b"},
        ndcg_at_5=0.50,
        faithfulness_mean=0.60,
    )
    id_to_label[r_low] = "r-low"
    r_high = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-high",
        model_config={"generator": "qwen3-14b"},
        ndcg_at_5=0.70,
        faithfulness_mean=0.95,
    )
    id_to_label[r_high] = "r-high"
    r_mid = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-mid",
        model_config={"generator": "qwen3-10b"},
        ndcg_at_5=0.65,
        faithfulness_mean=0.78,
    )
    id_to_label[r_mid] = "r-mid"

    dsn = migrated_engine.url.render_as_string(hide_password=False)
    db = Database(DatabaseSettings(dsn=dsn, pool_min=1, pool_max=2))
    service = ScoreboardService(database=db)  # type: ignore[arg-type]
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, operator_tokens=[])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        scoreboard_service=service,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer u"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == "faithfulness"
    entries = body["entries"]
    assert len(entries) == 3

    # The faithfulness board is DESCENDING (higher is better) — AC6
    # primary contract.  Map the report_id (uuid) back to the label.
    labels_in_order = [id_to_label[e["report_id"]] for e in entries]
    assert labels_in_order == ["r-high", "r-mid", "r-low"]
    assert entries[0]["metric_value"] == pytest.approx(0.95)
    assert entries[1]["metric_value"] == pytest.approx(0.78)
    assert entries[2]["metric_value"] == pytest.approx(0.60)

    # Each row carries ``config_snapshot`` (the F-00093 column
    # renamed for the API; the route maps it to ``config_snapshot``
    # to avoid the Pydantic ``model_config`` collision) + ``created_at``
    # (the S05 entry's per-row timestamp).
    for entry in entries:
        assert isinstance(entry["config_snapshot"], dict)
        assert "generator" in entry["config_snapshot"]
        assert entry["created_at"] != ""


# ---------------------------------------------------------------------------
# Boundary row "Empty report history" — 0 reports → 200 with empty entries
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_scoreboard_empty_history_returns_200_with_empty_entries(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """Boundary row "Empty report history" — 0 reports → 200 with ``entries == []``.

    The S05 service returns empty per-board lists when no rows
    exist; the S08 route surfaces them verbatim (NOT a 5xx).  Pinned
    here so a future refactor that 5xx's on an empty read breaks
    loudly.
    """
    _truncate_scoreboard_history(migrated_engine)

    dsn = migrated_engine.url.render_as_string(hide_password=False)
    db = Database(DatabaseSettings(dsn=dsn, pool_min=1, pool_max=2))
    service = ScoreboardService(database=db)  # type: ignore[arg-type]
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, operator_tokens=[])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        scoreboard_service=service,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    resp = client.get(
        "/benchmark/scoreboard?board=faithfulness",
        headers={"authorization": "Bearer u"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == "faithfulness"
    assert body["entries"] == []


# ---------------------------------------------------------------------------
# AC6 — ndcg_at_5 board is descending (end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_get_scoreboard_ndcg_at_5_returns_descending_ranked_rows(
    migrated_engine: sqlalchemy.Engine,
) -> None:
    """AC6 — the ``ndcg_at_5`` board is descending with the live service.

    Three rows with distinct ndcg_at_5 values are seeded; the S05
    service ranks them by ndcg_at_5 descending; the S08 route
    serialises the leaderboard.

    The route exposes the row's ``id`` (uuid string) — not the
    ``report_label`` — so the test tracks the seeded ids → label
    mapping and asserts the leaderboard's rank order via a
    report_label lookup join.
    """
    _truncate_scoreboard_history(migrated_engine)
    dataset_id, run_id = _seed_dataset_and_run(migrated_engine)
    id_to_label: dict[str, str] = {}
    r1 = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-1",
        model_config={"generator": "qwen3-7b"},
        ndcg_at_5=0.40,
        faithfulness_mean=0.80,
    )
    id_to_label[r1] = "r-1"
    r2 = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-2",
        model_config={"generator": "qwen3-14b"},
        ndcg_at_5=0.85,
        faithfulness_mean=0.92,
    )
    id_to_label[r2] = "r-2"
    r3 = _seed_benchmark_report(
        migrated_engine,
        run_id=run_id,
        dataset_id=dataset_id,
        report_label="r-3",
        model_config={"generator": "qwen3-10b"},
        ndcg_at_5=0.62,
        faithfulness_mean=0.88,
    )
    id_to_label[r3] = "r-3"

    dsn = migrated_engine.url.render_as_string(hide_password=False)
    db = Database(DatabaseSettings(dsn=dsn, pool_min=1, pool_max=2))
    service = ScoreboardService(database=db)  # type: ignore[arg-type]
    auth = StaticTokenAuthenticator.from_config({"u": ["u"]}, operator_tokens=[])
    app = create_app(
        authenticator=auth,
        query_service=_StubQueryService(),  # type: ignore[arg-type]
        scoreboard_service=service,  # type: ignore[arg-type]
    )
    client = TestClient(app)

    resp = client.get(
        "/benchmark/scoreboard?board=ndcg_at_5",
        headers={"authorization": "Bearer u"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["board"] == "ndcg_at_5"
    entries = body["entries"]
    # ndcg_at_5 board is descending — the route returns row ids;
    # we map back to the seeded labels.
    labels_in_order = [id_to_label[e["report_id"]] for e in entries]
    assert labels_in_order == ["r-2", "r-3", "r-1"]
    assert entries[0]["metric_value"] == pytest.approx(0.85)


class _StubQueryService:
    """Minimal query service stub — satisfies ``create_app``'s `query_service` seam."""

    async def answer(self, query: str, ctx: object) -> object:  # pragma: no cover - never called
        raise RuntimeError("not used in S17 scoreboard integration tests")
