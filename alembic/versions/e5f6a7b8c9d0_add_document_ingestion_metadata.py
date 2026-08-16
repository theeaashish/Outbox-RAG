"""add_document_ingestion_metadata

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-09 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: str | Sequence[str] | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add operational metadata columns used by async document ingestion."""
    op.add_column(
        "documents",
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column(
            "retry_count",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "documents",
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "documents",
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    """Remove operational metadata columns from documents."""
    op.drop_column("documents", "processed_at")
    op.drop_column("documents", "processing_started_at")
    op.drop_column("documents", "retry_count")
    op.drop_column("documents", "last_error")
