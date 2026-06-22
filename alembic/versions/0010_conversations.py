"""iw_rag.conversations + conversation_messages — multi-turn chat memory.

Adds server-side persisted chat history so the agent can answer follow-ups that
depend on earlier turns of the SAME conversation ("what is my name?" after "my
name is Sergio").  Prior turns are replayed into the agent loop as citable,
principal-scoped synthetic chunks (conversation-as-source), so the fail-closed
grounding contract is preserved — a memory answer cites the chat turn it came
from rather than bypassing the grounding gate.

Two tables (column shapes mirror ``storage/schema.py`` exactly):

* ``conversations`` — one row per chat thread, owned by exactly one
  ``principal_id`` (sourced from the authenticated AccessContext, never request
  input).  ``last_active_at`` is touched on every new turn so the console can
  order threads most-recent-first.
* ``conversation_messages`` — one row per turn (``role`` user|assistant), with
  a FK to ``conversations`` ``ON DELETE CASCADE`` so the "Clear" button purges a
  whole thread in one statement.  Assistant turns carry their resolved
  ``citations`` (jsonb) and ``disposition`` so the history endpoint can
  re-render sources without re-running the query.

Indexes (managed here, not in ``schema.py``, per that module's convention):
* ``conversation_messages (conversation_id, created_at)`` — the history read
  ("recent N turns of this conversation, in order").
* ``conversations (principal_id, last_active_at DESC)`` — owner thread listing.

Fully reversible: ``downgrade()`` drops both tables.

Revision ID: 0010_conversations
Revises: 0009_config_change_log
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0010_conversations"
down_revision: str | None = "0009_config_change_log"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create iw_rag.conversations + conversation_messages + their indexes."""
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS iw_rag.conversations (
            id             uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
            principal_id   text        NOT NULL,
            title          text,
            created_at     timestamptz NOT NULL DEFAULT now(),
            last_active_at timestamptz NOT NULL DEFAULT now()
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS iw_rag.conversation_messages (
            id              uuid        PRIMARY KEY DEFAULT gen_random_uuid(),
            conversation_id uuid        NOT NULL
                REFERENCES iw_rag.conversations (id) ON DELETE CASCADE,
            role            text        NOT NULL,
            content         text        NOT NULL,
            citations       jsonb       NOT NULL DEFAULT '[]'::jsonb,
            disposition     text,
            -- clock_timestamp() (NOT now()): a turn's user + assistant rows are
            -- inserted in the SAME transaction; now() would stamp both with the
            -- identical transaction-start time, making created_at-ordering
            -- ambiguous within a turn. clock_timestamp() advances per statement.
            created_at      timestamptz NOT NULL DEFAULT clock_timestamp(),
            CONSTRAINT conversation_messages_role_chk
                CHECK (role IN ('user', 'assistant'))
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS conversation_messages_conv_created_idx "
        "ON iw_rag.conversation_messages (conversation_id, created_at)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS conversations_principal_active_idx "
        "ON iw_rag.conversations (principal_id, last_active_at DESC)"
    )


def downgrade() -> None:
    """Drop conversation_messages then conversations (full reversal)."""
    op.execute("DROP TABLE IF EXISTS iw_rag.conversation_messages CASCADE")
    op.execute("DROP TABLE IF EXISTS iw_rag.conversations CASCADE")
