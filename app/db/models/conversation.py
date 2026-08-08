from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Index, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.knowledge_base import KnowledgeBase
    from app.db.models.message import Message


class Conversation(UUIDMixin, TimestampMixin, Base):
    """Represents a chat conversation."""

    __tablename__ = "conversations"

    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="conversations",
        lazy="select",
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="Message.created_at",
    )

    __table_args__ = (
        Index(
            "ix_conversations_knowledge_base_created_at_id_desc",
            "knowledge_base_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )
