"""Alembic environment — resolves the DSN and runs migrations (sync engine).

URL precedence: an explicit ``sqlalchemy.url`` set on the Alembic config (e.g.
``alembic_cfg.set_main_option("sqlalchemy.url", ...)`` in tests) wins; otherwise
the DSN comes from the typed ``Settings``. ``target_metadata`` is the Core schema
so future ``--autogenerate`` (via ``make migration-pending``) can diff against it.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import create_engine, pool

from alembic import context
from iw_rag.storage.schema import metadata as target_metadata

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False: fileConfig defaults to True, which would
    # disable the iw_rag.* loggers for the rest of the process (breaking app
    # logging + tests that assert on log records after a migration runs).
    fileConfig(config.config_file_name, disable_existing_loggers=False)

_PLACEHOLDER_URL = "postgresql+psycopg://localhost/iw_rag"


def _resolve_url() -> str:
    """Resolve the DB URL: explicit config url wins, else Settings DSN."""
    url = config.get_main_option("sqlalchemy.url")
    if url and url != _PLACEHOLDER_URL:
        return url
    from iw_rag.settings import get_settings

    return get_settings().database.dsn


def run_migrations_offline() -> None:
    """Emit SQL to stdout (``alembic upgrade --sql``) without a DB connection."""
    context.configure(
        url=_resolve_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table_schema="public",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live connection (sync psycopg engine)."""
    connectable = create_engine(_resolve_url(), poolclass=pool.NullPool)
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            version_table_schema="public",
        )
        with context.begin_transaction():
            context.run_migrations()
    connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
