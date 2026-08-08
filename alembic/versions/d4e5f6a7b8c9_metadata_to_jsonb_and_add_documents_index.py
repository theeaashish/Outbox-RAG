"""metadata_to_jsonb_and_add_documents_index

Revision ID: d4e5f6a7b8c9
Revises: 3f8983ba4253
Create Date: 2026-08-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: str | Sequence[str] | None = "3f8983ba4253"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Switch chunk metadata to JSONB and index documents by KB + created_at."""
    op.alter_column(
        "document_chunks",
        "metadata",
        type_=JSONB(),
        postgresql_using="metadata::jsonb",
        existing_type=sa.JSON(),
        existing_nullable=True,
    )
    op.create_index(
        "ix_documents_kb_created_at_desc",
        "documents",
        ["knowledge_base_id", sa.text("created_at DESC")],
        unique=False,
    )


def downgrade() -> None:
    """Revert metadata to JSON and drop the documents index."""
    op.drop_index("ix_documents_kb_created_at_desc", table_name="documents")
    op.alter_column(
        "document_chunks",
        "metadata",
        type_=sa.JSON(),
        postgresql_using="metadata::json",
        existing_nullable=True,
    )
