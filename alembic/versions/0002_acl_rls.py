"""Optional Postgres Row-Level Security (RLS) on iw_rag.chunks — defense-in-depth.

Creates the ``chunks_acl_isolation`` RLS policy keyed off the
``iw_rag.principal_groups`` session GUC, but **leaves RLS DISABLED by default**.

Rationale for keeping RLS off by default
-----------------------------------------
The application-layer pre-filter (``WHERE acl_tags && :principal_groups``) is the
**primary enforcement point** (lld-data-model §5, CONVENTIONS.md §2).  RLS is a
belt-and-braces layer — it would catch a hand-written query that accidentally
omits the WHERE clause.  However:

* pg_search BM25 and pgvector HNSW planning interact with RLS predicates in ways
  that can degrade index use; this must be validated before enabling in any
  latency-sensitive deployment (task-3.0 Risks & notes).
* The policy is created here so the SQL is reviewed and version-controlled; an
  operator enables it with a single statement:
      ALTER TABLE iw_rag.chunks ENABLE ROW LEVEL SECURITY;
  and disables it with:
      ALTER TABLE iw_rag.chunks DISABLE ROW LEVEL SECURITY;

Policy behaviour when enabled
-------------------------------
* The policy uses ``current_setting('iw_rag.principal_groups', true)`` — the
  second argument ``true`` makes it return NULL (not raise) if the GUC is unset.
  ``string_to_array(NULL, ',')`` → NULL; ``acl_tags && NULL`` → NULL (falsy), so
  a session without the GUC returns **zero rows** (fail-closed).
* The application sets the GUC per transaction from the same ``AccessContext``
  (see ``storage/db.py`` — ``set_config('iw_rag.principal_groups', ..., true)``).
* ``FORCE ROW LEVEL SECURITY`` applies the policy even to the table owner.
* ``principal_groups`` values MUST NOT contain commas (group id constraint).
  If group ids ever need commas, switch the GUC encoding to a JSON array and
  parse accordingly.

GUC encoding assumption
------------------------
Groups are stored as comma-separated text in the GUC.  The application writes::

    SELECT set_config('iw_rag.principal_groups', 'finance,hr', true)

and the policy reads back with ``string_to_array(..., ',')``.  This is safe as
long as group identifiers contain no commas — a constraint already enforced by
the ``AccessContext`` domain type (no commas in group names per lld-api §5).

Revision ID: 0002_acl_rls
Revises: 0001
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002_acl_rls"
down_revision: str | None = "0001"  # chains from the initial schema migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add RLS policy on iw_rag.chunks (policy created, RLS left DISABLED).

    To activate the belt-and-braces layer after validating performance::

        ALTER TABLE iw_rag.chunks ENABLE ROW LEVEL SECURITY;
        ALTER TABLE iw_rag.chunks FORCE ROW LEVEL SECURITY;

    The application MUST set the GUC per transaction::

        SELECT set_config('iw_rag.principal_groups', :groups, true)

    where ``:groups`` is a comma-joined string of the principal's groups.
    """
    # Create the policy — RLS is NOT enabled here; the policy is inert until
    # an operator explicitly enables it with ALTER TABLE ... ENABLE ROW LEVEL SECURITY.
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_policies
                WHERE schemaname = 'iw_rag'
                  AND tablename  = 'chunks'
                  AND policyname = 'chunks_acl_isolation'
            ) THEN
                CREATE POLICY chunks_acl_isolation ON iw_rag.chunks
                    USING (
                        acl_tags && string_to_array(
                            current_setting('iw_rag.principal_groups', true),
                            ','
                        )
                    );
            END IF;
        END
        $$
        """
    )
    # Intentionally NOT enabling RLS here.
    # The existing application pre-filter is the primary enforcement point.
    # Enable RLS only after validating that pgvector HNSW and pg_search BM25
    # planning are not degraded by the policy predicate (task-3.0 Risks & notes).
    #
    # To enable (run as superuser or table owner after perf validation):
    #   ALTER TABLE iw_rag.chunks ENABLE ROW LEVEL SECURITY;
    #   ALTER TABLE iw_rag.chunks FORCE ROW LEVEL SECURITY;


def downgrade() -> None:
    """Drop the RLS policy and ensure RLS is disabled on iw_rag.chunks."""
    # Disable RLS first (idempotent if already disabled).
    op.execute("ALTER TABLE iw_rag.chunks DISABLE ROW LEVEL SECURITY")
    # Drop the policy (idempotent via IF EXISTS).
    op.execute("DROP POLICY IF EXISTS chunks_acl_isolation ON iw_rag.chunks")
