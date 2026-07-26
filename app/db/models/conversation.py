from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.knowledge_base import KnowledgeBase
    from app.db.models.message import Message


class Conversation(UUIDMixin, TimestampMixin, Base):
    """Represents a chat conversation."""

    __tablename__ = "conversations"

    session_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        index=True,
    )

    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(
        back_populates="conversations",
        lazy="selectin",
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index(
            "ix_conversation_kb_session",
            "knowledge_base_id",
            "session_id",
        ),
    )
