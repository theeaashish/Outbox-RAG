from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.message import Message


class MessageSource(UUIDMixin, Base):
    """Represents a retrieved context source backing an assistant message."""

    __tablename__ = "message_sources"

    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )

    citation_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    is_cited: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    source_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
    )

    document_chunk_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("document_chunks.id", ondelete="SET NULL"),
        nullable=True,
    )

    document_name_snapshot: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    chunk_index_snapshot: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    retrieval_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    page_start_snapshot: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    page_end_snapshot: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    message: Mapped[Message] = relationship(
        back_populates="sources",
        lazy="select",
    )

    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "citation_number",
            name="uq_message_sources_message_citation_number",
        ),
        UniqueConstraint(
            "message_id",
            "source_order",
            name="uq_message_sources_message_source_order",
        ),
        Index(
            "ix_message_sources_document_id",
            "document_id",
        ),
        Index(
            "ix_message_sources_document_chunk_id",
            "document_chunk_id",
        ),
        CheckConstraint(
            "citation_number >= 1",
            name="ck_message_source_citation_number_positive",
        ),
        CheckConstraint(
            "source_order >= 0",
            name="ck_message_source_order_non_negative",
        ),
    )
