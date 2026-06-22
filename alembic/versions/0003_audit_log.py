"""Append-only iw_rag.audit_log table + trigger-enforced immutability (Task 3.4).

Creates ``iw_rag.audit_log`` — the defensible decision trail persisted for every
request (answered, refused, or rejected).  Append-only is enforced two ways
(defense-in-depth per task-3.4-audit-log §1):

1. ``REVOKE UPDATE, DELETE ON iw_rag.audit_log FROM iw_rag_app`` — the
   application role cannot mutate rows (complements the trigger for roles that
   are NOT superusers).
2. ``BEFORE UPDATE OR DELETE`` trigger ``audit_log_immutable`` that calls
   ``iw_rag.audit_no_mutate()`` and raises an exception — fires for ALL
   sessions including superusers (testcontainers uses superuser; the trigger
   is the observable enforcement layer in integration tests).

NOTE: this revision was hand-authored following the precedent established by
``0002_acl_rls.py``.  The canonical workflow (``make migration-pending``) runs
``alembic revision --autogenerate``, which requires a live DB to diff against.
That is not available in this offline, no-Docker environment, so the DDL is
written explicitly — same rationale as 0002_acl_rls.  The PENDING-rebase flow
managed by the daemon applies only to concurrent worktrees merging simultaneously;
a lone hand-authored revision with a concrete ``down_revision`` is safe and
correct here.

Revision ID: 0003_audit_log
Revises: 0002_acl_rls
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0003_audit_log"
down_revision: str | None = "0002_acl_rls"  # chains from the ACL/RLS migration
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.audit_log with append-only enforcement.

    Creates:
    - The ``iw_rag.audit_log`` table with uuid pk, full decision-trail columns,
      and a CHECK constraint on ``disposition``.
    - Two indexes: ``audit_log_request`` on ``request_id`` and
      ``audit_log_principal_time`` on ``(principal, created_at)``.
    - The ``iw_rag.audit_no_mutate()`` PL/pgSQL function that raises on any
      mutation attempt.
    - The ``audit_log_immutable`` BEFORE UPDATE OR DELETE trigger that calls
      the function (fires for all sessions, including superusers).
    - REVOKE UPDATE, DELETE on the table from the application role
      ``iw_rag_app`` (defense-in-depth for non-superuser app connections;
      the trigger is the observable layer in integration tests where the
      testcontainers connection is a superuser).

    The ``retrieved_chunk_ids`` column uses ``text[]`` (not ``uuid[]``) because
    chunk ids flow through the state as ``str`` (``RetrievedChunk.chunk_id``),
    not parsed UUIDs — using text[] avoids cast errors when chunk ids are
    synthetic test strings.
    """
    op.execute(
        """
        CREATE TABLE iw_rag.audit_log (
            id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
            request_id          text NOT NULL,
            principal           text NOT NULL,
            query               text NOT NULL,
            acl_filter          text[] NOT NULL,
            retrieved_chunk_ids text[] NOT NULL DEFAULT '{}',
            guard_verdicts      jsonb NOT NULL DEFAULT '[]'::jsonb,
            redactions          jsonb NOT NULL DEFAULT '[]'::jsonb,
            disposition         text NOT NULL
                                    CHECK (disposition IN ('answered','refused','rejected')),
            created_at          timestamptz NOT NULL DEFAULT now()
        )
        """
    )

    # Indexes for the common query patterns (by request correlation id and
    # by principal+time for operator review).
    op.execute("CREATE INDEX audit_log_request ON iw_rag.audit_log (request_id)")
    op.execute("CREATE INDEX audit_log_principal_time ON iw_rag.audit_log (principal, created_at)")

    # Defense-in-depth layer 1: PL/pgSQL function that always raises.
    # This fires even for superusers (unlike REVOKE-only enforcement).
    op.execute(
        """
        CREATE FUNCTION iw_rag.audit_no_mutate() RETURNS trigger AS $$
        BEGIN
            RAISE EXCEPTION 'audit_log is append-only';
        END
        $$ LANGUAGE plpgsql
        """
    )

    # Defense-in-depth layer 1 (cont.): BEFORE UPDATE OR DELETE trigger.
    # FOR EACH ROW ensures it fires once per affected row; BEFORE means the
    # mutation is prevented before it writes to storage.
    op.execute(
        """
        CREATE TRIGGER audit_log_immutable
            BEFORE UPDATE OR DELETE ON iw_rag.audit_log
            FOR EACH ROW EXECUTE FUNCTION iw_rag.audit_no_mutate()
        """
    )

    # Defense-in-depth layer 2: revoke mutation privileges from the application
    # role.  This is effective for non-superuser connections (the normal app
    # role in production).  The role may not exist in the testcontainers env;
    # use DO/EXCEPTION to skip gracefully.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_roles WHERE rolname = 'iw_rag_app'
            ) THEN
                REVOKE UPDATE, DELETE ON iw_rag.audit_log FROM iw_rag_app;
            END IF;
        END
        $$
        """
    )


def downgrade() -> None:
    """Drop the audit_log trigger, function, and table (full reversal).

    Drops in reverse creation order:
    1. The trigger (depends on the function).
    2. The function.
    3. The table (cascades indexes).

    Grants are automatically revoked when the table is dropped.
    """
    # Drop trigger before the function (trigger references it).
    op.execute("DROP TRIGGER IF EXISTS audit_log_immutable ON iw_rag.audit_log")
    # Drop the function.
    op.execute("DROP FUNCTION IF EXISTS iw_rag.audit_no_mutate()")
    # Drop the table (CASCADE drops indexes).
    op.execute("DROP TABLE IF EXISTS iw_rag.audit_log CASCADE")
