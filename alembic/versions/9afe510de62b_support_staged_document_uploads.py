"""support staged document uploads

Revision ID: 9afe510de62b
Revises: b8c9d0e1f2a3
Create Date: 2026-10-11 18:39:04.181476

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9afe510de62b"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Support staged document uploads."""

    # Add the new value to PostgreSQL's native enum.
    op.execute("ALTER TYPE document_status ADD VALUE IF NOT EXISTS 'uploading'")

    # Permit a missing hash while the upload is in progress.
    op.alter_column(
        "documents",
        "sha256_hash",
        existing_type=sa.VARCHAR(length=64),
        nullable=True,
    )

    # Enforce the relationship between status and content hash.
    op.create_check_constraint(
        "ck_documents_uploading_hash_state",
        "documents",
        "(status::text = 'uploading' AND sha256_hash IS NULL) "
        "OR (status::text <> 'uploading' AND sha256_hash IS NOT NULL)",
    )


def downgrade() -> None:
    """Revert staged-upload schema changes."""

    # Do not silently discard documents that are still uploading.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM documents
                WHERE sha256_hash IS NULL
            ) THEN
                RAISE EXCEPTION
                    'Cannot downgrade: documents with NULL hashes exist. '
                    'Resolve staged uploads before downgrading.';
            END IF;
        END
        $$;
        """
    )

    op.drop_constraint(
        "ck_documents_uploading_hash_state",
        "documents",
        type_="check",
    )

    op.alter_column(
        "documents",
        "sha256_hash",
        existing_type=sa.VARCHAR(length=64),
        nullable=False,
    )

    # PostgreSQL does not directly support removing an enum value.
    # Keep 'uploading' in document_status during downgrade.
