"""add document processing token and outbox dead letter

Revision ID: f7a8b9c0d1e2
Revises: d5e6f7a8b9c0
Create Date: 2026-09-05 18:45:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f7a8b9c0d1e2"
down_revision: str | Sequence[str] | None = "d5e6f7a8b9c0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add processing_token to documents and dead_lettered_at to outbox_events."""
    op.add_column(
        "documents",
        sa.Column("processing_token", sa.Uuid(), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("dead_lettered_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_outbox_events_dead_lettered",
        "outbox_events",
        ["dead_lettered_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove dead_lettered_at from outbox_events and processing_token from documents."""
    op.drop_index(
        "ix_outbox_events_dead_lettered",
        table_name="outbox_events",
    )
    op.drop_column("outbox_events", "dead_lettered_at")
    op.drop_column("documents", "processing_token")
