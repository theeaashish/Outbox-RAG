from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Enum, ForeignKey, Index, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin
from app.db.models.enums import FeedbackRating

if TYPE_CHECKING:
    from app.db.models.message import Message
    from app.db.models.user import User


class MessageFeedback(UUIDMixin, TimestampMixin, Base):
    """Represents user feedback rating on an assistant message."""

    __tablename__ = "message_feedbacks"

    message_id: Mapped[UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    rating: Mapped[FeedbackRating] = mapped_column(
        Enum(
            FeedbackRating,
            name="feedback_rating",
            native_enum=True,
            values_callable=lambda enum: [e.value for e in enum],
        ),
        nullable=False,
    )

    message: Mapped[Message] = relationship(
        back_populates="feedbacks",
        lazy="select",
    )

    user: Mapped[User] = relationship(
        lazy="raise",
    )

    __table_args__ = (
        UniqueConstraint(
            "message_id",
            "user_id",
            name="uq_message_feedbacks_message_user",
        ),
        Index(
            "ix_message_feedbacks_user_id",
            "user_id",
        ),
    )
