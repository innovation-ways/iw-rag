"""iw_rag.ephemeral_sessions — chat-scoped attachment ownership (WI-DOCMGMT).

Creates the ``iw_rag.ephemeral_sessions`` table that binds a chat ``session_id``
to the principal that owns it, with an ``expires_at`` TTL.  It is the server-side
authority for ephemeral chat attachments: a request only gains the synthetic
``session:<id>`` ACL group after this table confirms the session exists, is owned
by the authenticated caller, and has not expired — the group is never trusted
from request input (lld; CLAUDE.md "ACL is a pre-filter, NEVER from user input").

Rows (and the attachment documents they gate) are purged on chat end
(``DocumentService.purge_session``) and by the TTL sweep
(``DocumentService.sweep_expired`` → ``EphemeralSessionRepository.expired``).

Column shapes mirror ``storage/schema.py`` (``ephemeral_sessions``) exactly:
``session_id`` PK, ``principal_id`` NOT NULL, ``created_at`` default now(),
``expires_at`` NOT NULL.  A B-tree index on ``expires_at`` backs the TTL sweep
(``EphemeralSessionRepository.expired``) — it is created in the migration (not
declared in ``schema.py``, per that module's "indexes live in the migration"
convention) and matches the index already present on the live DB.

CONTEXT: this completes WI-DOCMGMT's schema on the live DB, which was stamped at
``0006_document_ownership`` but never received this table (its migration was never
authored/applied). Until this runs, managed-doc upload/list/delete/query works but
chat *attachments* fail.

Fully reversible: ``downgrade()`` drops the table.

Revision ID: 0007_ephemeral_sessions
Revises: 0006_document_ownership
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0007_ephemeral_sessions"
down_revision: str | None = "0006_document_ownership"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.ephemeral_sessions (session_id PK, ...) + the expires_at TTL index."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS iw_rag.ephemeral_sessions (
            session_id   text        PRIMARY KEY,
            principal_id text        NOT NULL,
            created_at   timestamptz NOT NULL DEFAULT now(),
            expires_at   timestamptz NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS ephemeral_sessions_expires_idx "
        "ON iw_rag.ephemeral_sessions (expires_at)"
    )


def downgrade() -> None:
    """Drop the ephemeral_sessions index then table (full reversal)."""
    op.execute("DROP INDEX IF EXISTS iw_rag.ephemeral_sessions_expires_idx")
    op.execute("DROP TABLE IF EXISTS iw_rag.ephemeral_sessions CASCADE")
