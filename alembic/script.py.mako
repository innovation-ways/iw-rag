"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
Create Date: ${create_date}

NOTE (engineering-conventions §5): ``down_revision`` is emitted as the literal
"PENDING" on purpose. The IW daemon's migration_rebase resolves the real chain
head at merge time, so parallel work items never collide on down_revision.
Do NOT hand-edit it and do NOT run ``alembic revision --autogenerate`` directly —
use ``make migration-pending MSG="..."``.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
${imports if imports else ""}

# revision identifiers, used by Alembic.
revision: str = ${repr(up_revision)}
down_revision: str | None = "PENDING"  # rebased by the daemon at merge time
branch_labels: str | Sequence[str] | None = ${repr(branch_labels)}
depends_on: str | Sequence[str] | None = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
