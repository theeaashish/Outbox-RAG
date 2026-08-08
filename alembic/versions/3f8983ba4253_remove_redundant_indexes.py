"""remove_redundant_indexes

Revision ID: 3f8983ba4253
Revises: c2d3e4f5a6b7
Create Date: 2026-08-07 16:36:03.838382

"""

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f8983ba4253"
down_revision: str | Sequence[str] | None = "c2d3e4f5a6b7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


DOCUMENTS_CREATED_INDEX = "ix_documents_kb_created_at_desc"


def upgrade() -> None:
    """Drop redundant indexes, convert chunk metadata to jsonb, and add document list index."""
    op.drop_index(op.f("ix_messages_conversation_id"), table_name="messages")
    op.drop_index(op.f("ix_messages_conversation_created_at"), table_name="messages")
    op.drop_index(
        op.f("ix_conversations_knowledge_base_id"), table_name="conversations"
    )
    op.drop_index(op.f("ix_documents_knowledge_base_id"), table_name="documents")
    op.drop_index(op.f("ix_documents_sha256_hash"), table_name="documents")
    op.drop_index(op.f("ix_document_chunks_document_id"), table_name="document_chunks")

    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN metadata TYPE jsonb USING metadata::jsonb"
    )

    with op.get_context().autocommit_block():
        op.execute(
            f"""
            CREATE INDEX CONCURRENTLY IF NOT EXISTS {DOCUMENTS_CREATED_INDEX}
            ON documents (knowledge_base_id, created_at DESC)
            """
        )


def downgrade() -> None:
    """Recreate the dropped indexes and revert metadata column type."""
    with op.get_context().autocommit_block():
        op.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {DOCUMENTS_CREATED_INDEX}")

    op.execute(
        "ALTER TABLE document_chunks ALTER COLUMN metadata TYPE json USING metadata::json"
    )

    op.create_index(
        op.f("ix_document_chunks_document_id"),
        "document_chunks",
        ["document_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_sha256_hash"),
        "documents",
        ["sha256_hash"],
        unique=False,
    )
    op.create_index(
        op.f("ix_documents_knowledge_base_id"),
        "documents",
        ["knowledge_base_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_conversations_knowledge_base_id"),
        "conversations",
        ["knowledge_base_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_messages_conversation_created_at"),
        "messages",
        ["conversation_id", "created_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_messages_conversation_id"),
        "messages",
        ["conversation_id"],
        unique=False,
    )
