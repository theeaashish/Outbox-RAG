from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Enum, ForeignKey, Index, Text, text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin
from app.db.models.enums import MessageRole

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.message_feedback import MessageFeedback
    from app.db.models.message_source import MessageSource


class Message(UUIDMixin, TimestampMixin, Base):
    """Represents a single message in a conversation."""

    __tablename__ = "messages"

    role: Mapped[MessageRole] = mapped_column(
        Enum(
            MessageRole,
            name="message_role",
            native_enum=True,
            values_callable=lambda enum: [e.value for e in enum],
        ),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    conversation: Mapped[Conversation] = relationship(
        back_populates="messages",
        lazy="select",
    )

    sources: Mapped[list[MessageSource]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
        order_by="MessageSource.source_order.asc()",
    )

    feedbacks: Mapped[list[MessageFeedback]] = relationship(
        back_populates="message",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="raise",
    )

    __table_args__ = (
        Index(
            "ix_messages_conversation_created_at_id_desc",
            "conversation_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )
