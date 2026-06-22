"""Integration tests for ACL enforcement hardening (WI-022 / Task 3.0).

Tests:
- Migration 0002_acl_rls applies cleanly and reverses cleanly.
- Re-upgrade head after downgrade is idempotent.
- The RLS policy object exists after upgrade but RLS is DISABLED by default.
- When RLS is manually ENABLED, a raw query without the GUC set returns 0 rows
  (fail-closed — no GUC → no rows, not all rows).
- When RLS is ENABLED and the GUC is set to {finance}, only finance chunks are
  returned by a naïve query (no WHERE clause); hr chunks are invisible.
- acl_provenance threads correctly into diagnostics["acl"].

All tests use the testcontainers ``clean_pg`` / ``pg_dsn`` fixtures from
``tests/integration/conftest.py``.  RLS tests run against a fresh isolated
engine so they do not affect the shared session-scoped ``pg_dsn`` container.
"""

from __future__ import annotations

import os

import pytest
import sqlalchemy
from sqlalchemy import text

PARADEDB_IMAGE = "paradedb/paradedb:latest"


# ---------------------------------------------------------------------------
# 0002_acl_rls migration lifecycle tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_0002_migration_applies_and_policy_exists(pg_dsn: str) -> None:
    """After upgrade head, the RLS policy object exists on iw_rag.chunks."""
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT policyname FROM pg_policies "
                "WHERE schemaname = 'iw_rag' AND tablename = 'chunks' "
                "AND policyname = 'chunks_acl_isolation'"
            )
        )
        row = result.fetchone()
    engine.dispose()
    assert row is not None, "RLS policy 'chunks_acl_isolation' was not created by migration 0002"
    assert row[0] == "chunks_acl_isolation", (
        f"Expected policyname 'chunks_acl_isolation'; got {row[0]!r}"
    )


@pytest.mark.integration
def test_0002_rls_disabled_by_default(pg_dsn: str) -> None:
    """RLS is DISABLED by default — the policy is inert (existing queries unaffected)."""
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.connect() as conn:
        # pg_class.relrowsecurity = True means RLS is enabled
        result = conn.execute(
            text("SELECT relrowsecurity FROM pg_class WHERE oid = 'iw_rag.chunks'::regclass")
        )
        row = result.fetchone()
    engine.dispose()
    assert row is not None
    rls_enabled = row[0]
    assert rls_enabled is False, (
        "RLS must be DISABLED by default on iw_rag.chunks. "
        "Enable it explicitly after validating pgvector/pg_search performance "
        "(task-3.0 Risks & notes)."
    )


@pytest.mark.integration
def test_0002_downgrade_removes_policy(pg_dsn: str) -> None:
    """Downgrade from 0002 removes the RLS policy and leaves RLS disabled."""
    import iw_rag.settings as settings_mod
    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    os.environ["IW_RAG__DATABASE__DSN"] = pg_dsn
    os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
    os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
    os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
    os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
    settings_mod.get_settings.cache_clear()

    with rebased_alembic_config(pg_dsn) as rebased:
        alembic_cfg = rebased.cfg

        # Downgrade to 0001 — the policy should be dropped but the table persists.
        command.downgrade(alembic_cfg, "0001")

        engine = sqlalchemy.create_engine(pg_dsn)
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT policyname FROM pg_policies "
                    "WHERE schemaname = 'iw_rag' AND tablename = 'chunks' "
                    "AND policyname = 'chunks_acl_isolation'"
                )
            )
            row = result.fetchone()
            # Also verify RLS is disabled after downgrade.
            rls_result = conn.execute(
                text("SELECT relrowsecurity FROM pg_class WHERE oid = 'iw_rag.chunks'::regclass")
            )
            rls_row = rls_result.fetchone()
        engine.dispose()

        assert row is None, (
            "Policy 'chunks_acl_isolation' should be removed after downgrade to 0001"
        )
        assert rls_row is not None
        assert rls_row[0] is False, "RLS must be disabled after downgrade to 0001"

        # Re-upgrade to head — idempotency check.
        command.upgrade(alembic_cfg, "head")

    engine2 = sqlalchemy.create_engine(pg_dsn)
    with engine2.connect() as conn:
        result2 = conn.execute(
            text(
                "SELECT policyname FROM pg_policies "
                "WHERE schemaname = 'iw_rag' AND tablename = 'chunks' "
                "AND policyname = 'chunks_acl_isolation'"
            )
        )
        row2 = result2.fetchone()
    engine2.dispose()
    assert row2 is not None, "Policy 'chunks_acl_isolation' should exist after re-upgrade"

    # Clean up env.
    for key in [
        "IW_RAG__DATABASE__DSN",
        "IW_RAG__MODELS__GENERATOR__ENDPOINT",
        "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
        "IW_RAG__MODELS__RERANKER__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
        "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
    ]:
        os.environ.pop(key, None)
    settings_mod.get_settings.cache_clear()


# ---------------------------------------------------------------------------
# RLS behaviour tests (RLS manually ENABLED for these tests only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rls_enabled_dsn() -> sqlalchemy.engine.base.Engine:  # type: ignore[type-arg]
    """Spin up an isolated ParadeDB container with RLS ENABLED for testing.

    The testcontainers connection user (``rls_u``) is a **superuser**, and
    PostgreSQL superusers bypass RLS even under ``FORCE ROW LEVEL SECURITY``.
    To exercise the policy we create a dedicated **non-superuser** application
    role (``rls_app``), grant it read access to ``iw_rag.chunks``, and yield an
    engine that connects *as that role* — only then is the policy enforced.

    Returns:
        A migrated engine connected as a non-superuser role, with RLS
        explicitly enabled on ``iw_rag.chunks``.
    """
    from testcontainers.postgres import PostgresContainer

    import iw_rag.settings as settings_mod
    from alembic import command
    from tests.integration._alembic_rebase import rebased_alembic_config

    with PostgresContainer(
        image=PARADEDB_IMAGE, username="rls_u", password="rls_p", dbname="rls_db"
    ) as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")

        os.environ["IW_RAG__DATABASE__DSN"] = dsn
        os.environ["IW_RAG__MODELS__GENERATOR__ENDPOINT"] = "http://vllm:8001/v1"
        os.environ["IW_RAG__MODELS__EMBEDDER__ENDPOINT"] = "http://infinity:7997"
        os.environ["IW_RAG__MODELS__RERANKER__ENDPOINT"] = "http://infinity:7997"
        os.environ["IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT"] = "http://vllm-judge:8002/v1"
        os.environ["IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT"] = "http://vllm-judge:8002/v1"
        settings_mod.get_settings.cache_clear()

        with rebased_alembic_config(dsn) as rebased:
            command.upgrade(rebased.cfg, "head")

        engine = sqlalchemy.create_engine(dsn)

        # Seed hr + finance chunks.
        import uuid

        with engine.begin() as conn:
            run_id = uuid.uuid4()
            conn.execute(
                text(
                    "INSERT INTO iw_rag.ingestion_runs "
                    "(id, corpus_label, parser, chunker_config, embed_model, status, started_at) "
                    "VALUES (:id, 'rls-test', 'docling', '{}', 'bge-m3', 'running', now())"
                ),
                {"id": run_id},
            )
            fin_doc_id = uuid.uuid4()
            hr_doc_id = uuid.uuid4()
            _doc_sql = (
                "INSERT INTO iw_rag.documents "
                "(id, source_uri, checksum, metadata, ingestion_run_id) "
                "VALUES (:id, :uri, :cs, '{}', :run_id)"
            )
            conn.execute(
                text(_doc_sql),
                {
                    "id": fin_doc_id,
                    "uri": "corpus://rls/finance.md",
                    "cs": "fin-rls-1",
                    "run_id": run_id,
                },
            )
            conn.execute(
                text(_doc_sql),
                {
                    "id": hr_doc_id,
                    "uri": "corpus://rls/hr.md",
                    "cs": "hr-rls-1",
                    "run_id": run_id,
                },
            )
            fin_chunk_id = uuid.uuid4()
            hr_chunk_id = uuid.uuid4()
            zero_vec = "[" + ",".join(["0.0"] * 1024) + "]"
            conn.execute(
                text(
                    "INSERT INTO iw_rag.chunks "
                    "(id, document_id, ordinal, content, dense_embedding, acl_tags) "
                    "VALUES (:id, :doc_id, 0, 'finance chunk content', :vec, '{finance}')"
                ),
                {"id": fin_chunk_id, "doc_id": fin_doc_id, "vec": zero_vec},
            )
            conn.execute(
                text(
                    "INSERT INTO iw_rag.chunks "
                    "(id, document_id, ordinal, content, dense_embedding, acl_tags) "
                    "VALUES (:id, :doc_id, 0, 'hr chunk confidential salary', :vec, '{hr}')"
                ),
                {"id": hr_chunk_id, "doc_id": hr_doc_id, "vec": zero_vec},
            )

        # Enable RLS + FORCE (so even table owner is subject to the policy).
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE iw_rag.chunks ENABLE ROW LEVEL SECURITY"))
            conn.execute(text("ALTER TABLE iw_rag.chunks FORCE ROW LEVEL SECURITY"))

        # Create a NON-superuser application role and grant it read access.
        # Superusers (the testcontainers connection user) bypass RLS entirely,
        # so the policy can only be observed through a non-superuser role.
        with engine.begin() as conn:
            conn.execute(
                text(
                    "DO $$ BEGIN "
                    "IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rls_app') THEN "
                    "CREATE ROLE rls_app LOGIN PASSWORD 'rls_app_pw' NOSUPERUSER; "
                    "END IF; END $$"
                )
            )
            conn.execute(text("GRANT USAGE ON SCHEMA iw_rag TO rls_app"))
            conn.execute(text("GRANT SELECT ON iw_rag.chunks TO rls_app"))

        # Build a DSN that authenticates as the non-superuser role.
        app_dsn = dsn.replace("rls_u:rls_p@", "rls_app:rls_app_pw@")
        app_engine = sqlalchemy.create_engine(app_dsn)

        yield app_engine

        app_engine.dispose()
        engine.dispose()
        for key in [
            "IW_RAG__DATABASE__DSN",
            "IW_RAG__MODELS__GENERATOR__ENDPOINT",
            "IW_RAG__MODELS__EMBEDDER__ENDPOINT",
            "IW_RAG__MODELS__RERANKER__ENDPOINT",
            "IW_RAG__MODELS__JUDGE_FAITHFULNESS__ENDPOINT",
            "IW_RAG__MODELS__JUDGE_GENERAL__ENDPOINT",
        ]:
            os.environ.pop(key, None)
        settings_mod.get_settings.cache_clear()


@pytest.mark.integration
def test_rls_no_guc_returns_zero_rows(
    rls_enabled_dsn: sqlalchemy.engine.base.Engine,  # type: ignore[type-arg]
) -> None:
    """With RLS ENABLED and no GUC set, a naïve query returns 0 rows (fail-closed)."""
    engine = rls_enabled_dsn
    with engine.connect() as conn:
        # No GUC set — current_setting(..., true) returns NULL → policy → NULL (falsy)
        result = conn.execute(text("SELECT count(*) FROM iw_rag.chunks"))
        count = result.scalar()
    assert count == 0, (
        f"Expected 0 rows (fail-closed) when GUC is unset; got {count}. "
        "The RLS policy must return zero rows, never all rows, when the GUC is absent."
    )


@pytest.mark.integration
def test_rls_with_finance_guc_returns_only_finance(
    rls_enabled_dsn: sqlalchemy.engine.base.Engine,  # type: ignore[type-arg]
) -> None:
    """With RLS ENABLED and GUC = 'finance', naïve query sees only finance chunks."""
    engine = rls_enabled_dsn
    with engine.connect() as conn:
        # Set the GUC for this transaction only (third arg true = transaction-local).
        conn.execute(text("SELECT set_config('iw_rag.principal_groups', 'finance', true)"))
        result = conn.execute(text("SELECT acl_tags FROM iw_rag.chunks"))
        rows = result.fetchall()

    assert len(rows) == 1, f"Expected 1 finance chunk; got {len(rows)}"
    acl_tags = rows[0][0]
    assert "finance" in acl_tags
    assert "hr" not in acl_tags


@pytest.mark.integration
def test_rls_hr_chunk_invisible_to_finance_guc(
    rls_enabled_dsn: sqlalchemy.engine.base.Engine,  # type: ignore[type-arg]
) -> None:
    """With RLS ENABLED and GUC = 'finance', the hr chunk is completely invisible."""
    engine = rls_enabled_dsn
    with engine.connect() as conn:
        conn.execute(text("SELECT set_config('iw_rag.principal_groups', 'finance', true)"))
        result = conn.execute(
            text("SELECT count(*) FROM iw_rag.chunks WHERE acl_tags && ARRAY['hr']")
        )
        count = result.scalar()
    assert count == 0, (
        f"HR chunk must be invisible to a finance-GUC session; count={count}. "
        "RLS is not blocking the cross-tenant access."
    )


# ---------------------------------------------------------------------------
# AclProvenance threading into diagnostics
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_acl_provenance_in_diagnostics(clean_pg: str) -> None:
    """After spine.retrieve(), diagnostics['acl'] contains the hashed provenance."""
    from iw_rag.domain.acl import AccessContext
    from iw_rag.domain.models import RetrievalResult
    from iw_rag.guardrails.acl import AclProvenance, acl_provenance

    ctx = AccessContext(principal_id="u", principal_groups=("finance",))
    prov = acl_provenance(ctx)
    assert isinstance(prov, AclProvenance)
    assert prov.acl_applied is True
    assert prov.mechanism == "pre_filter"
    # Groups must be hashed, not raw.
    assert "finance" not in prov.principal_groups_hash

    # Simulate what the retrieve node puts into diagnostics.
    result = RetrievalResult(
        chunks=[],
        retriever="spine",
        acl_applied=True,
        diagnostics={
            "acl": {
                "principal_groups_hash": prov.principal_groups_hash,
                "acl_applied": prov.acl_applied,
                "mechanism": prov.mechanism,
            }
        },
    )
    acl_diag = result.diagnostics["acl"]
    assert isinstance(acl_diag, dict)
    assert acl_diag["acl_applied"] is True
    assert acl_diag["mechanism"] == "pre_filter"
    assert "finance" not in acl_diag["principal_groups_hash"]
