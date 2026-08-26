from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.user import User


class PasswordCredential(UUIDMixin, TimestampMixin, Base):
    """Password credential owned by an application user."""

    __tablename__ = "password_credentials"

    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    password_hash: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )

    password_changed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    user: Mapped[User] = relationship(
        back_populates="password_credential",
        lazy="raise",
    )
