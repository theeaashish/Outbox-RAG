"""add_outbox_events_table

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-08-16 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f6a7b8c9d0e1"
down_revision: str | Sequence[str] | None = "e5f6a7b8c9d0"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create outbox_events table and associated indexes."""
    outbox_event_type = postgresql.ENUM(
        "document.process",
        name="outbox_event_type",
        create_type=False,
    )
    # Ensure type exists if not already present
    op.execute(
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'outbox_event_type') THEN "
        "CREATE TYPE outbox_event_type AS ENUM ('document.process'); "
        "END IF; END $$;"
    )

    op.create_table(
        "outbox_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column(
            "event_type",
            outbox_event_type,
            nullable=False,
        ),
        sa.Column(
            "aggregate_type",
            sa.String(length=100),
            nullable=False,
            server_default="document",
        ),
        sa.Column("aggregate_id", sa.Uuid(), nullable=False),
        sa.Column(
            "payload",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column(
            "published_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "attempt_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_outbox_events")),
    )

    op.create_index(
        op.f("ix_outbox_events_event_type"),
        "outbox_events",
        ["event_type"],
        unique=False,
    )
    op.create_index(
        op.f("ix_outbox_events_aggregate_id"),
        "outbox_events",
        ["aggregate_id"],
        unique=False,
    )
    op.create_index(
        "ix_outbox_events_unpublished",
        "outbox_events",
        ["published_at", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Drop outbox_events table and associated indexes."""
    op.drop_index(
        "ix_outbox_events_unpublished",
        table_name="outbox_events",
    )
    op.drop_index(
        op.f("ix_outbox_events_aggregate_id"),
        table_name="outbox_events",
    )
    op.drop_index(
        op.f("ix_outbox_events_event_type"),
        table_name="outbox_events",
    )
    op.drop_table("outbox_events")

    op.execute("DROP TYPE IF EXISTS outbox_event_type;")
