from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin
from app.db.models.enums import DocumentStatus

if TYPE_CHECKING:
    from app.db.models.document_chunk import DocumentChunk
    from app.db.models.knowledge_base import KnowledgeBase


class Document(UUIDMixin, TimestampMixin, Base):
    """Document model."""

    __tablename__ = "documents"

    title: Mapped[str] = mapped_column(String(255), nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)
    storage_path: Mapped[str] = mapped_column(String(512), nullable=False)
    sha256_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    file_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            native_enum=True,
            values_callable=lambda enum: [e.value for e in enum],
        ),
        nullable=False,
        default=DocumentStatus.PENDING,
        server_default=DocumentStatus.PENDING.value,
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )
    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="documents", lazy="select"
    )

    chunks: Mapped[list[DocumentChunk]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="select",
    )

    __table_args__ = (
        UniqueConstraint(
            "knowledge_base_id",
            "sha256_hash",
            name="uq_document_kb_sha256",
        ),
        Index(
            "ix_documents_kb_created_at_desc",
            "knowledge_base_id",
            text("created_at DESC"),
        ),
    )
