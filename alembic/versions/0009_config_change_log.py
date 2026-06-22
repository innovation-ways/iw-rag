"""iw_rag.config_change_log — Operator Console live-config audit trail (F-00096).

Creates ``iw_rag.config_change_log``: one immutable row per applied config
edit surfaced by the Operator Console.  Append-only is enforced two ways
(defense-in-depth per F-00096 Invariant 2, mirroring the
``audit_log`` precedent from Task 3.4):

1. ``REVOKE UPDATE, DELETE ON iw_rag.config_change_log FROM iw_rag_app`` —
   the application role cannot mutate rows (complements the trigger for
   roles that are NOT superusers).  Wrapped in the same
   ``DO/EXCEPTION`` guard against a missing role as ``0003_audit_log``,
   so the migration is safe to run in the testcontainers environment
   where the role is not pre-created.
2. ``BEFORE UPDATE OR DELETE`` trigger ``config_change_log_immutable``
   that calls ``iw_rag.config_change_no_mutate()`` and raises an
   exception — fires for ALL sessions including superusers (the
   testcontainers integration tests connect as superuser; the trigger
   is the observable enforcement layer there).

The ``applied_mode`` column carries a CHECK constraint pinning its
values to ``hot`` or ``pending_restart`` (mirror of
``audit_log.disposition``).  ``principal_id`` is NOT NULL because
Invariant 3 requires the value to come from the authenticated
``AccessContext`` — a NULL would let a stray anonymous insert land.

NOTE: this revision was hand-authored following the precedent set by
``0003_audit_log.py``.  The canonical workflow (``make
migration-pending``) runs ``alembic revision --autogenerate``, which
requires a live DB to diff against; that is not available in this
offline, no-Docker environment, so the DDL is written explicitly.
``down_revision`` is set to the real chain head (``0008_benchmark_reports``)
so ``walk_revisions()`` resolves the graph and the
``tests/unit/test_migration_chain.py`` guard passes.  The PENDING-rebase
flow managed by the daemon applies only to concurrent worktrees merging
simultaneously; a lone hand-authored revision with a concrete
``down_revision`` is safe and correct here.

Revision ID: 0009_config_change_log
Revises: 0008_benchmark_reports
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0009_config_change_log"
down_revision: str | None = "0008_benchmark_reports"  # chains from benchmark_reports
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.config_change_log with append-only enforcement.

    Creates:

    - The ``iw_rag.config_change_log`` table with uuid pk, the
      F-00096 §"Database Changes" columns (id, ts, principal_id,
      config_key, old_value, new_value, applied_mode, trace_id), and a
      CHECK constraint pinning ``applied_mode`` to
      ``hot`` or ``pending_restart`` (mirror of ``audit_log.disposition``).
    - The ``config_change_log_key_time`` index on
      ``(config_key, ts)`` — backs the operator review query that
      lists "all changes for a key, newest first" (mirror of
      ``audit_log_principal_time``).
    - The ``iw_rag.config_change_no_mutate()`` PL/pgSQL function that
      raises on any mutation attempt.
    - The ``config_change_log_immutable`` BEFORE UPDATE OR DELETE
      trigger that calls the function (fires for all sessions,
      including superusers).
    - REVOKE UPDATE, DELETE on the table from the application role
      ``iw_rag_app`` (defense-in-depth for non-superuser app
      connections; the trigger is the observable layer in integration
      tests where the testcontainers connection is a superuser).

    The ``DO $$ ... IF EXISTS (SELECT 1 FROM pg_roles WHERE
    rolname='iw_rag_app') ... $$`` guard wraps the REVOKE because
    ``iw_rag_app`` may not exist in the testcontainers env (same
    pattern as ``0003_audit_log``).
    """
    op.execute(
        """
        CREATE TABLE iw_rag.config_change_log (
            id            uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            ts            timestamptz NOT NULL DEFAULT now(),
            principal_id  text NOT NULL,
            config_key    text NOT NULL,
            old_value     text,
            new_value     text NOT NULL,
            applied_mode  text NOT NULL
                              CHECK (applied_mode IN ('hot','pending_restart')),
            trace_id      text
        )
        """
    )

    # Index for the operator-review query "all changes for a key, newest first".
    op.execute(
        "CREATE INDEX config_change_log_key_time ON iw_rag.config_change_log (config_key, ts)"
    )

    # Defense-in-depth layer 1: PL/pgSQL function that always raises.
    # This fires even for superusers (unlike REVOKE-only enforcement).
    op.execute(
        """
        CREATE FUNCTION iw_rag.config_change_no_mutate() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'config_change_log is append-only';
        END
        $$ LANGUAGE plpgsql
        """
    )

    # Defense-in-depth layer 1 (cont.): BEFORE UPDATE OR DELETE trigger.
    # FOR EACH ROW ensures it fires once per affected row; BEFORE means
    # the mutation is prevented before it writes to storage.
    op.execute(
        """
        CREATE TRIGGER config_change_log_immutable
            BEFORE UPDATE OR DELETE ON iw_rag.config_change_log
            FOR EACH ROW EXECUTE FUNCTION iw_rag.config_change_no_mutate()
        """
    )

    # Defense-in-depth layer 2: revoke mutation privileges from the
    # application role.  Effective for non-superuser connections (the
    # normal app role in production).  The role may not exist in the
    # testcontainers env; use the DO/IF EXISTS guard to skip gracefully.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'iw_rag_app'
            ) THEN
                REVOKE UPDATE, DELETE ON iw_rag.config_change_log FROM iw_rag_app;
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Drop the config_change_log trigger, function, and table (full reversal).

    Drops in reverse creation order:
    1. The trigger (depends on the function).
    2. The function.
    3. The table (CASCADE drops indexes).

    Grants are automatically revoked when the table is dropped.
    """
    # Drop trigger before the function (trigger references it).
    op.execute("DROP TRIGGER IF EXISTS config_change_log_immutable ON iw_rag.config_change_log")
    # Drop the function.
    op.execute("DROP FUNCTION IF EXISTS iw_rag.config_change_no_mutate()")
    # Drop the table (CASCADE drops indexes).
    op.execute("DROP TABLE IF EXISTS iw_rag.config_change_log CASCADE")
