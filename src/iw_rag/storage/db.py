"""Async database engine, session factory, and a deep health check.

The only module that owns the Postgres connection (repo-structure §2). Uses the
async psycopg 3 driver. ``health()`` runs ``SELECT 1`` and asserts both required
extensions (``vector``, ``pg_search``) are installed — the deep readiness signal
``/health?deep=true`` reuses (lld-api §3).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from iw_rag.domain.errors import ConfigError
from iw_rag.settings import DatabaseSettings

REQUIRED_EXTENSIONS = ("vector", "pg_search")

# psycopg 3 async driver prefix; we normalise any psycopg2/plain DSN to it.
_ASYNC_DRIVER = "postgresql+psycopg://"


def _to_async_dsn(dsn: str) -> str:
    """Normalise a DSN to the async psycopg 3 driver form.

    Args:
        dsn: A SQLAlchemy/psycopg DSN (any ``postgresql[+driver]://…`` form).

    Returns:
        The DSN using the ``postgresql+psycopg://`` (async-capable) driver.
    """
    for prefix in ("postgresql+psycopg2://", "postgresql+asyncpg://", "postgresql://"):
        if dsn.startswith(prefix):
            return _ASYNC_DRIVER + dsn[len(prefix) :]
    return dsn


@dataclass(frozen=True)
class HealthStatus:
    """Result of a deep database health check.

    Attributes:
        ok: True only if the DB is reachable AND both extensions are present.
        reachable: Whether ``SELECT 1`` succeeded.
        extensions: Mapping of required extension name → present?.
        detail: Human-readable summary (names the failing dependency).
    """

    ok: bool
    reachable: bool
    extensions: dict[str, bool]
    detail: str


class Database:
    """Owns the async engine + session factory for the iw_rag store.

    Construct once at startup from ``DatabaseSettings`` and inject the
    ``session()`` context manager into repositories.
    """

    def __init__(self, settings: DatabaseSettings) -> None:
        """Create the async engine with pool sizing from settings.

        Args:
            settings: Database connection + pool configuration.

        Raises:
            ConfigError: If the DSN is empty.
        """
        if not settings.dsn:
            raise ConfigError("DatabaseSettings.dsn is empty (set IW_RAG__DATABASE__DSN).")
        self._engine: AsyncEngine = create_async_engine(
            _to_async_dsn(settings.dsn),
            pool_size=settings.pool_min,
            max_overflow=max(0, settings.pool_max - settings.pool_min),
            pool_pre_ping=True,
        )
        self._sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    @property
    def engine(self) -> AsyncEngine:
        """The underlying async engine."""
        return self._engine

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Yield an ``AsyncSession`` inside a transaction-scoped context.

        Yields:
            An open ``AsyncSession``; commits on clean exit, rolls back on error.
        """
        async with self._sessionmaker() as session:
            yield session

    async def health(self) -> HealthStatus:
        """Run ``SELECT 1`` and assert both required extensions are installed.

        Returns:
            A :class:`HealthStatus`; ``ok`` is True only when reachable and both
            ``vector`` and ``pg_search`` extensions are present.
        """
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                rows = await conn.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = ANY(:names)"),
                    {"names": list(REQUIRED_EXTENSIONS)},
                )
                present = {r[0] for r in rows}
        except Exception as exc:  # noqa: BLE001 — health must never raise
            return HealthStatus(
                ok=False,
                reachable=False,
                extensions={ext: False for ext in REQUIRED_EXTENSIONS},
                detail=f"database unreachable: {exc}",
            )
        extensions = {ext: ext in present for ext in REQUIRED_EXTENSIONS}
        missing = [ext for ext, ok in extensions.items() if not ok]
        ok = not missing
        detail = "ok" if ok else f"missing extensions: {', '.join(missing)}"
        return HealthStatus(ok=ok, reachable=True, extensions=extensions, detail=detail)

    async def dispose(self) -> None:
        """Dispose the engine's connection pool."""
        await self._engine.dispose()
