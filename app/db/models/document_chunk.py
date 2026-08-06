from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.constants import EMBEDDING_DIMENSION, JSONDict
from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.document import Document


class DocumentChunk(UUIDMixin, TimestampMixin, Base):
    """Document chunk model."""

    __tablename__ = "document_chunks"

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Metadata
    chunk_metadata: Mapped[JSONDict | None] = mapped_column(
        "metadata", JSON, nullable=True
    )

    # Embedding
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIMENSION), nullable=False
    )

    # Foreign Keys
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    document: Mapped[Document] = relationship(
        back_populates="chunks",
        lazy="select",
    )

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_document_chunk_index",
        ),
        Index(
            "ix_document_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_chunk_index_non_negative",
        ),
        CheckConstraint(
            "char_start IS NULL OR char_start >= 0",
            name="ck_chunk_start_non_negative",
        ),
        CheckConstraint(
            "char_end IS NULL OR char_end >= char_start",
            name="ck_chunk_end_valid",
        ),
    )
