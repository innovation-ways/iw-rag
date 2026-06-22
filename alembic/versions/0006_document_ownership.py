"""iw_rag.documents — owner_id + visibility columns for document management (WI-DOCMGMT).

Adds the two nullable metadata columns the document-management layer uses to
list + authorize per-user documents:

- ``owner_id``  — the uploading principal id (NULL for system/corpus docs
  ingested offline via ``make ingest``).
- ``visibility`` — ``common`` (surfaced to everyone) or ``specific`` (owner-only);
  NULL on both means a corpus doc managed offline, not shown in the management UI.

These columns are metadata for *listing* + *delete-authorization* only — retrieval
visibility is still driven solely by ``chunks.acl_tags`` (the ACL pre-filter),
never by these columns (lld-data-model; CLAUDE.md "ACL is a pre-filter").

Indexes (B-tree) on each column back the management-UI ``list_visible`` query
(owner + common, newest first).

RECOVERY NOTE: the original ``0006_document_ownership`` migration was authored
untracked, applied to the live DB (which is stamped at this revision), and then
lost from disk — so this file is a faithful reconstruction from the live schema +
``storage/schema.py``.  It is intentionally idempotent (``IF [NOT] EXISTS``) so it
is a no-op on the already-migrated live DB and a clean create on a fresh DB.

Fully reversible: ``downgrade()`` drops the indexes and columns.

Revision ID: 0006_document_ownership
Revises: 0005_graphrag_age
"""

from __future__ import annotations

from alembic import op

# revision identifiers, used by Alembic.
revision = "0006_document_ownership"
down_revision: str | None = "0005_graphrag_age"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add documents.owner_id + visibility (nullable) and their B-tree indexes."""
    op.execute("ALTER TABLE iw_rag.documents ADD COLUMN IF NOT EXISTS owner_id text")
    op.execute("ALTER TABLE iw_rag.documents ADD COLUMN IF NOT EXISTS visibility text")
    op.execute("CREATE INDEX IF NOT EXISTS documents_owner_idx ON iw_rag.documents (owner_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS documents_visibility_idx ON iw_rag.documents (visibility)"
    )


def downgrade() -> None:
    """Drop the indexes then the columns (full reversal)."""
    op.execute("DROP INDEX IF EXISTS iw_rag.documents_owner_idx")
    op.execute("DROP INDEX IF EXISTS iw_rag.documents_visibility_idx")
    op.execute("ALTER TABLE iw_rag.documents DROP COLUMN IF EXISTS visibility")
    op.execute("ALTER TABLE iw_rag.documents DROP COLUMN IF EXISTS owner_id")
