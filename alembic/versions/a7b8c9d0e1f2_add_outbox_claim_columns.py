"""add_outbox_claim_columns

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-08-17 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a7b8c9d0e1f2"
down_revision: str | Sequence[str] | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add claimed_at and claim_token columns and claimable index to outbox_events."""
    op.add_column(
        "outbox_events",
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "outbox_events",
        sa.Column("claim_token", sa.Uuid(), nullable=True),
    )
    op.create_index(
        "ix_outbox_events_claimable",
        "outbox_events",
        ["published_at", "claimed_at", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Remove claimable index and claimed_at/claim_token columns from outbox_events."""
    op.drop_index(
        "ix_outbox_events_claimable",
        table_name="outbox_events",
    )
    op.drop_column("outbox_events", "claim_token")
    op.drop_column("outbox_events", "claimed_at")
