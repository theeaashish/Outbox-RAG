"""add message sources and feedback

Revision ID: b8c9d0e1f2a3
Revises: f7a8b9c0d1e2
Create Date: 2026-09-28 19:25:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b8c9d0e1f2a3"
down_revision: str | Sequence[str] | None = "f7a8b9c0d1e2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create message_sources and message_feedbacks tables with constraints."""
    # 1. Create feedback_rating enum
    feedback_rating = postgresql.ENUM("up", "down", name="feedback_rating")
    feedback_rating.create(op.get_bind(), checkfirst=True)

    # 2. Create message_sources table
    op.create_table(
        "message_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("citation_number", sa.Integer(), nullable=False),
        sa.Column(
            "is_cited",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column("source_order", sa.Integer(), nullable=False),
        sa.Column("document_id", sa.Uuid(), nullable=True),
        sa.Column("document_chunk_id", sa.Uuid(), nullable=True),
        sa.Column("document_name_snapshot", sa.String(length=255), nullable=False),
        sa.Column("chunk_index_snapshot", sa.Integer(), nullable=False),
        sa.Column("retrieval_score", sa.Float(), nullable=False),
        sa.Column("page_start_snapshot", sa.Integer(), nullable=True),
        sa.Column("page_end_snapshot", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["document_chunk_id"],
            ["document_chunks.id"],
            name=op.f("fk_message_sources_document_chunk_id_document_chunks"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.id"],
            name=op.f("fk_message_sources_document_id_documents"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_message_sources_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_message_sources")),
        sa.UniqueConstraint(
            "message_id",
            "citation_number",
            name="uq_message_sources_message_citation_number",
        ),
        sa.UniqueConstraint(
            "message_id",
            "source_order",
            name="uq_message_sources_message_source_order",
        ),
        sa.CheckConstraint(
            "citation_number >= 1",
            name="ck_message_source_citation_number_positive",
        ),
        sa.CheckConstraint(
            "source_order >= 0",
            name="ck_message_source_order_non_negative",
        ),
    )
    op.create_index(
        op.f("ix_message_sources_document_id"),
        "message_sources",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_message_sources_document_chunk_id"),
        "message_sources",
        ["document_chunk_id"],
        unique=False,
    )

    # 3. Create message_feedbacks table
    op.create_table(
        "message_feedbacks",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("message_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "rating",
            postgresql.ENUM(
                "up",
                "down",
                name="feedback_rating",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["message_id"],
            ["messages.id"],
            name=op.f("fk_message_feedbacks_message_id_messages"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_message_feedbacks_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_message_feedbacks")),
        sa.UniqueConstraint(
            "message_id",
            "user_id",
            name="uq_message_feedbacks_message_user",
        ),
    )
    op.create_index(
        op.f("ix_message_feedbacks_user_id"),
        "message_feedbacks",
        ["user_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop message_feedbacks, message_sources, and feedback_rating enum."""
    op.drop_index(
        op.f("ix_message_feedbacks_user_id"),
        table_name="message_feedbacks",
    )
    op.drop_table("message_feedbacks")

    op.drop_index(
        op.f("ix_message_sources_document_chunk_id"),
        table_name="message_sources",
    )
    op.drop_index(
        op.f("ix_message_sources_document_id"),
        table_name="message_sources",
    )
    op.drop_table("message_sources")

    feedback_rating = postgresql.ENUM("up", "down", name="feedback_rating")
    feedback_rating.drop(op.get_bind(), checkfirst=True)
