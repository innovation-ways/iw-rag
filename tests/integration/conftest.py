"""Shared integration fixtures — a session-scoped migrated ParadeDB container.

F-00093: alembic migrations are authored with ``down_revision = "PENDING"``
(engineering-conventions §5).  The IW daemon rebases the sentinel at
merge time; the test environment has no daemon, so we use a PENDING-rebase
helper (see ``tests/integration/_alembic_rebase.py``) that copies the
versions dir into a temp dir with the sentinel migration stripped.
``alembic upgrade head`` against the temp dir resolves the chain and
applies 0001-0007.  The 0008 PENDING migration is not applied here
(test files that need it own their own fixture — see
``tests/integration/storage/test_benchmark_reports_repo.py``).
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
import sqlalchemy
from sqlalchemy import text

from tests.integration._alembic_rebase import rebased_alembic_config

# Pinned by digest (not :latest) so a ParadeDB release cannot break CI
# non-deterministically. Bump deliberately in a reviewed change. The :latest tag
# this digest resolved to as of 2026-06-13.
PARADEDB_IMAGE = (
    "paradedb/paradedb@sha256:dbd64f0eb2716b8466afe54b91fec782f4107a8d0e6bd3660b89c88a1ded2ead"
)


@pytest.fixture(scope="session")
def pg_dsn() -> Generator[str, None, None]:
    """A migrated paradedb container shared across the session; yields its DSN."""
    from testcontainers.postgres import PostgresContainer

    from alembic import command

    with PostgresContainer(image=PARADEDB_IMAGE, username="t", password="t", dbname="t") as pg:
        dsn = pg.get_connection_url().replace("postgresql+psycopg2://", "postgresql+psycopg://")
        with rebased_alembic_config(dsn) as rebased:
            command.upgrade(rebased.cfg, "head")
        yield dsn


@pytest.fixture
def clean_pg(pg_dsn: str) -> str:
    """Truncate iw_rag tables before a test for isolation; returns the DSN."""
    engine = sqlalchemy.create_engine(pg_dsn)
    with engine.begin() as conn:
        conn.execute(
            text(
                "TRUNCATE iw_rag.documents, iw_rag.chunks, iw_rag.ingestion_runs "
                "RESTART IDENTITY CASCADE"
            )
        )
    engine.dispose()
    return pg_dsn
