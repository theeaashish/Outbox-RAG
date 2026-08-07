"""add_cursor_pagination_indexes

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-08-07 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2d3e4f5a6b7"
down_revision: str | Sequence[str] | None = "b1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CONVERSATIONS_INDEX = "ix_conversations_knowledge_base_created_at_id_desc"
MESSAGES_INDEX = "ix_messages_conversation_created_at_id_desc"


def upgrade() -> None:
    """Create keyset pagination indexes concurrently outside a transaction."""

    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {CONVERSATIONS_INDEX}
            ON conversations (knowledge_base_id, created_at DESC, id DESC)
            """
        )
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {MESSAGES_INDEX}
            ON messages (conversation_id, created_at DESC, id DESC)
            """
        )


def downgrade() -> None:
    """Drop keyset pagination indexes concurrently outside a transaction."""

    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {MESSAGES_INDEX}")
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {CONVERSATIONS_INDEX}")
