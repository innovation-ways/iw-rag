"""Checkpointer factory for the LangGraph agentic loop.

Implements the CONVENTIONS §8 canonical rule:

- **Dev / demo (default):** ``InMemorySaver`` — lives only in-process; requires no
  external service.  Used in all tests and dev runs.
- **Durable:** ``AsyncPostgresSaver`` (``langgraph-checkpoint-postgres``) connecting
  to the **same Postgres instance** as the data model (lld-data-model) for the
  one-auditable-store principle (architecture §2, principle 7).

``SqliteSaver`` is **not** the dev default — it is an optional local-persistence
alternative that is NOT used here.  Do not write "sqlite dev".

Selection rule (CONVENTIONS §8):
    1. If ``durable`` is passed explicitly as ``True`` → Postgres.
    2. If ``durable`` is passed explicitly as ``False`` → InMemory.
    3. Otherwise, ``settings.checkpointer == "postgres"`` → Postgres;
       any other value (e.g. ``"in_memory"``) → InMemory.

Usage::

    # Dev / in-process (no Postgres needed):
    saver = make_checkpointer(settings)

    # Durable (Postgres):
    saver = make_checkpointer(settings, durable=True, dsn=settings.database.dsn)
    await setup_checkpointer(saver)   # idempotent table creation

    # Build graph with saver:
    graph = build_graph(spine, assembler, llm, checkpointer=saver)

References:
- CONVENTIONS.md §8 (checkpointer canon)
- task-2.6-checkpointing-and-hitl §1 (saver factory step)
- lld-agentic-loop §5 (checkpointing/resume/HITL)
- ADR-008 (checkpointer for resume/time-travel/HITL)
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from iw_rag.settings import AgentSettings

logger = logging.getLogger(__name__)


def make_checkpointer(
    settings: AgentSettings,
    *,
    durable: bool | None = None,
    dsn: str | None = None,
) -> Any:
    """Build and return a checkpointer conforming to CONVENTIONS §8.

    The returned object is suitable for passing directly to
    ``build_graph(..., checkpointer=saver)``.  Call :func:`setup_checkpointer`
    on the result before the first graph invocation when using the Postgres
    saver (idempotent; creates the checkpointer tables).

    Selection rule:

    1. ``durable=True``  → ``AsyncPostgresSaver`` (Postgres, same instance as
       data model).
    2. ``durable=False`` → ``InMemorySaver`` (dev / in-process).
    3. ``durable=None``  → inferred from ``settings.checkpointer``:
       ``"postgres"`` → Postgres; anything else → InMemory.

    The Postgres saver is **lazy-imported** inside this function so importing
    ``agent.checkpoint`` does not require the ``langgraph-checkpoint-postgres``
    package to be installed on the dev (InMemory) path — the import only
    executes when a Postgres saver is actually requested.

    Args:
        settings: The ``AgentSettings`` instance supplying the
            ``checkpointer`` field (``"in_memory"`` or ``"postgres"``).
        durable: Explicit durability override.  ``True`` forces Postgres;
            ``False`` forces InMemory; ``None`` defers to ``settings``.
        dsn: PostgreSQL DSN string (required when the Postgres saver is
            selected).  Should be ``settings.database.dsn`` in production.

    Returns:
        An ``InMemorySaver`` (dev) or an ``AsyncPostgresSaver`` (durable)
        instance.  Both implement the LangGraph ``BaseCheckpointSaver``
        interface and can be passed directly to ``build_graph``.

    Raises:
        ValueError: If the Postgres path is selected but ``dsn`` is ``None``.
    """
    want_postgres: bool
    if durable is True:
        want_postgres = True
    elif durable is False:
        want_postgres = False
    else:
        # Infer from settings.
        want_postgres = settings.checkpointer == "postgres"

    if not want_postgres:
        logger.debug("Checkpointer: InMemorySaver (dev/demo)")
        return InMemorySaver()

    # Durable path — lazy import so the dev path stays light.
    if dsn is None:
        raise ValueError(
            "dsn must be provided when selecting the Postgres checkpointer "
            "(settings.checkpointer='postgres' or durable=True). "
            "Pass dsn=settings.database.dsn or set the DSN explicitly."
        )

    # Lazy import — only executed on the durable path.
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool
    except ImportError as exc:
        raise ImportError(
            "langgraph-checkpoint-postgres is required for the durable Postgres "
            "checkpointer.  Add it to [project.dependencies] via "
            "`uv add langgraph-checkpoint-postgres`."
        ) from exc

    logger.info("Checkpointer: AsyncPostgresSaver (durable, dsn=%r)", dsn)
    # Build an AsyncConnectionPool in the closed state — `.open()` / `async with`
    # must be called before the first checkpoint read/write.  We create it here
    # (construction is cheap and synchronous); the app lifespan coroutine owns
    # the open/close lifecycle. Typed Any: psycopg row-type generics don't line
    # up with AsyncPostgresSaver's expected dict-row connection under mypy --strict.
    pool: Any = AsyncConnectionPool(conninfo=dsn, open=False)
    return AsyncPostgresSaver(pool)


async def setup_checkpointer(saver: Any) -> None:
    """Run any one-time setup required by the given checkpointer.

    For ``AsyncPostgresSaver`` this calls ``.setup()`` which creates the
    checkpoint tables (``checkpoints``, ``checkpoint_blobs``,
    ``checkpoint_writes``) idempotently — safe to call on every startup.

    For ``InMemorySaver`` this is a no-op: no tables or external resources are
    needed.

    Args:
        saver: The checkpointer returned by :func:`make_checkpointer`.
    """
    if isinstance(saver, InMemorySaver):
        # No setup needed for the in-memory saver.
        return

    # For any other saver (e.g. AsyncPostgresSaver), call .setup() if available.
    setup_fn = getattr(saver, "setup", None)
    if callable(setup_fn):
        logger.info("Running checkpointer setup (idempotent table creation)…")
        await setup_fn()
        logger.info("Checkpointer setup complete.")
    else:
        logger.debug("Checkpointer %r has no .setup() method; skipping.", type(saver).__name__)
