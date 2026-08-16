from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import DateTime, Enum, Index, Integer, String, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.constants import JSONDict
from app.db.base import Base, TimestampMixin, UUIDMixin
from app.db.models.enums import OutboxEventType


class OutboxEvent(UUIDMixin, TimestampMixin, Base):
    """Durable event waiting to be published to the task broker."""

    __tablename__ = "outbox_events"

    event_type: Mapped[OutboxEventType] = mapped_column(
        Enum(
            OutboxEventType,
            name="outbox_event_type",
            native_enum=True,
            values_callable=lambda enum: [e.value for e in enum],
        ),
        nullable=False,
        index=True,
    )

    aggregate_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="document",
        server_default="document",
    )

    aggregate_id: Mapped[UUID] = mapped_column(
        nullable=False,
        index=True,
    )

    payload: Mapped[JSONDict] = mapped_column(
        JSONB,
        nullable=False,
    )

    published_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    attempt_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default=text("0"),
    )

    last_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    __table_args__ = (
        Index(
            "ix_outbox_events_unpublished",
            "published_at",
            "created_at",
        ),
    )
