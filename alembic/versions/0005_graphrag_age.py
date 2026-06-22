"""iw_rag.graphrag_age — AGE extension assertion + graph back-ref table (WI-028).

Creates the ``iw_rag.graph_chunk_backrefs`` table that maps graph entity/relation
names to their source ``chunks.id`` values.  This is the load-bearing citation
and ACL post-filter path (lld-graphrag §3, task-4a §1).

AGE extension guard (best-effort, never breaks upgrade head):
The migration attempts ``CREATE EXTENSION IF NOT EXISTS age`` inside a suppressed
block.  If the ``age`` extension is **not** available in the Postgres instance
(e.g. the plain paradedb image without the AGE build) the attempt is silently
skipped so ``alembic upgrade head`` always succeeds — the graphrag mode is OFF
by default and the spine/other modes must not be blocked.  When the mode is
enabled against an AGE-less instance, the runtime store fail-fasts (below).
When the mode is enabled, use the AGE-enabled Postgres image (see
``docker/postgres/init/01_age_bootstrap.sql``).

Fail-fast behaviour at runtime (not migration-time):
:class:`~iw_rag.modes.graphrag.store.AgeGraphStore` raises
:class:`~iw_rag.modes.graphrag.store.GraphStoreUnavailable` when ``age`` is absent.
The spine and other modes are unaffected — the graphrag route stays disabled.

Mode-enabled note:
The ``graph_chunk_backrefs`` table is always created by this migration regardless
of ``modes.graphrag.enabled``.  The table existing is harmless when the graphrag
mode is off — it will remain empty until ``scripts/index_graph.py`` runs.
This avoids making the migration conditional, which would complicate the chain.

Indexes:
- B-tree on ``(entity_name, graph_name)`` — primary lookup for ``resolve_to_chunks``.
- B-tree on ``chunk_id`` — back-reference lookup from chunks.

Fully reversible: ``downgrade()`` drops the table + indexes (and attempts to
drop the AGE extension, which may fail if other objects depend on it — that
failure is suppressed to keep downgrade idempotent).

ACL caveat (lld-graphrag §6):
The graph route is disabled for ACL-scoped corpora until the pre-filter-on-
traversal open item is resolved.  The ``graph_chunk_backrefs`` table stores
``chunk_id`` back-refs so WI-029 can perform ACL post-filtering on resolved chunks.
When the open item is resolved, ``acl_tags`` should be added to entity/relation
nodes so traversal itself can be ACL-scoped (pre-filter at the AGE query).

References:
- task-4a-graphrag-route §1 (store boundary + back-refs)
- lld-graphrag §3, §6 (ACL caveat)
- lld-data-model §8/§9 (Alembic owns schema)
- CLAUDE.md (Alembic migration rules — CAST not ::, hand-authored, PENDING chain)

Revision ID: 0005_graphrag_age
Revises: 0004_visual_pages
"""

from __future__ import annotations

import contextlib

from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_graphrag_age"
down_revision: str | None = "0004_visual_pages"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Assert AGE extension + create graph_chunk_backrefs table + indexes.

    The ``CREATE EXTENSION IF NOT EXISTS age`` is wrapped to give a clear,
    catchable error when AGE is absent (not installed in the Postgres image).
    The ``graph_chunk_backrefs`` table is created unconditionally — it is
    harmless when the graphrag mode is off.
    """
    # AGE extension is **best-effort** at migration time: the graphrag mode is
    # OFF by default, so this migration MUST NOT break ``alembic upgrade head``
    # on a Postgres image without the AGE build (the default paradedb test
    # container).  When AGE is present we create the extension + default graph;
    # when it is absent we silently skip — the runtime store
    # (AgeGraphStore) fail-fasts with GraphStoreUnavailable if the graphrag mode
    # is enabled against an AGE-less instance.  The graph_chunk_backrefs table
    # below is a plain table (no AGE dependency) and is always created.
    # Pre-check availability with a safe SELECT first: issuing a failing
    # CREATE EXTENSION inside Alembic's single migration transaction would
    # poison it (InFailedSqlTransaction), breaking every later statement
    # regardless of Python-level try/except.  pg_available_extensions lists
    # extensions whose control files are installed, without attempting to
    # create them — so we only run CREATE EXTENSION when AGE is genuinely
    # installable.
    conn = op.get_bind()
    age_installable = conn.execute(
        text("SELECT 1 FROM pg_available_extensions WHERE name = 'age'")
    ).scalar()
    if age_installable:
        op.execute("CREATE EXTENSION IF NOT EXISTS age")
        op.execute("LOAD 'age'")
        op.execute("SET search_path = ag_catalog, '$user', public")
        # create_graph raises if the graph already exists — suppress it.
        with contextlib.suppress(Exception):
            op.execute("SELECT create_graph('iw_rag_graph')")

    # graph_chunk_backrefs: entity/relation name → source chunk_id mapping.
    # This table enables resolve_to_chunks without requiring a full AGE traversal.
    # Populated by AgeGraphStore.upsert_entities/upsert_relations.
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS iw_rag.graph_chunk_backrefs (
            entity_name TEXT NOT NULL,
            chunk_id    TEXT NOT NULL,
            graph_name  TEXT NOT NULL DEFAULT 'iw_rag_graph',
            PRIMARY KEY (entity_name, chunk_id, graph_name)
        )
        """
    )

    # B-tree index for resolve_to_chunks lookup (entity_name → chunk_id).
    op.execute(
        "CREATE INDEX IF NOT EXISTS graph_backrefs_entity_idx "
        "ON iw_rag.graph_chunk_backrefs (entity_name, graph_name)"
    )

    # B-tree index for back-reference lookups from chunks (chunk_id → entities).
    op.execute(
        "CREATE INDEX IF NOT EXISTS graph_backrefs_chunk_idx "
        "ON iw_rag.graph_chunk_backrefs (chunk_id)"
    )


def downgrade() -> None:
    """Drop graph_chunk_backrefs + its indexes; attempt to drop AGE extension.

    The AGE extension drop may fail if other objects depend on it; that failure
    is suppressed to keep downgrade idempotent.
    """
    op.execute("DROP INDEX IF EXISTS iw_rag.graph_backrefs_chunk_idx")
    op.execute("DROP INDEX IF EXISTS iw_rag.graph_backrefs_entity_idx")
    op.execute("DROP TABLE IF EXISTS iw_rag.graph_chunk_backrefs CASCADE")

    # Attempt to drop AGE extension — may fail if other objects depend on it.
    # Suppressed: downgrade is best-effort for extension management.
    with contextlib.suppress(Exception):
        op.execute("DROP EXTENSION IF EXISTS age CASCADE")
