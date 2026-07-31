"""remove_session_id_and_add_message_ordering

Revision ID: a1b2c3d4e5f6
Revises: 5e131107e9bc
Create Date: 2026-07-31 11:47:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "5e131107e9bc"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index("ix_conversation_kb_session", table_name="conversations")
    op.drop_index("ix_conversations_session_id", table_name="conversations")
    op.drop_column("conversations", "session_id")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "conversations",
        sa.Column("session_id", sa.String(length=255), nullable=False),
    )
    op.create_index(
        "ix_conversations_session_id",
        "conversations",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_conversation_kb_session",
        "conversations",
        ["knowledge_base_id", "session_id"],
        unique=False,
    )
