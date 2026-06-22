"""Alembic helpers — build a configured ``Config`` from ``Settings``.

Keeps the DSN resolution in one place so ``make db-migrate`` and programmatic
callers (tests) share the same wiring. The ``PENDING`` down-revision convention
is implemented in ``alembic/script.py.mako`` (engineering-conventions §5).
"""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config

from iw_rag.settings import get_settings

# Repo-root-relative alembic.ini (the migrations live under ./alembic).
_ALEMBIC_INI = Path("alembic.ini")


def make_alembic_config(dsn: str | None = None) -> Config:
    """Build an Alembic ``Config`` with the SQLAlchemy URL injected.

    Args:
        dsn: Explicit database DSN; if ``None``, taken from ``Settings``.

    Returns:
        A ready-to-use Alembic ``Config`` (``sqlalchemy.url`` set).
    """
    config = Config(str(_ALEMBIC_INI))
    url = dsn if dsn is not None else get_settings().database.dsn
    config.set_main_option("sqlalchemy.url", url)
    return config
