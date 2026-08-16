from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models.outbox_event import OutboxEvent
from app.repositories.base import BaseRepository


class OutboxEventRepository(BaseRepository[OutboxEvent]):
    """Repository for durable outbox event operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(db=db, model=OutboxEvent)

    def list_unpublished(self, *, limit: int = 100) -> Sequence[OutboxEvent]:
        """Claim unpublished events for publication."""

        statement = (
            select(OutboxEvent)
            .where(OutboxEvent.published_at.is_(None))
            .order_by(
                OutboxEvent.created_at.asc(),
                OutboxEvent.id.asc(),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        return list(self.db.scalars(statement))

    def mark_published(self, *, event: OutboxEvent) -> None:
        """Mark an outbox event as successfully published."""

        event.published_at = datetime.now(UTC)

    def record_failure(self, *, event: OutboxEvent, error: str) -> None:
        """Record a failed publication attempt."""

        event.attempt_count += 1
        event.last_error = error[:4000]
