from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session

from app.db.models.outbox_event import OutboxEvent
from app.repositories.base import BaseRepository


class OutboxEventRepository(BaseRepository[OutboxEvent]):
    """Repository for durable outbox event operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(db=db, model=OutboxEvent)

    def claim_batch(
        self,
        *,
        limit: int,
        lease_seconds: int,
    ) -> tuple[UUID, list[OutboxEvent]]:
        """
        Select and claim a batch of unpublished outbox events.

        The caller must commit the transaction after this method returns
        to make the claim durable and release the row locks.
        """

        now = datetime.now(UTC)
        lease_expiry = now - timedelta(seconds=lease_seconds)
        claim_token = uuid4()

        statement = (
            select(OutboxEvent)
            .where(
                OutboxEvent.published_at.is_(None),
                (
                    OutboxEvent.claimed_at.is_(None)
                    | (OutboxEvent.claimed_at < lease_expiry)
                ),
            )
            .order_by(
                OutboxEvent.created_at.asc(),
                OutboxEvent.id.asc(),
            )
            .limit(limit)
            .with_for_update(skip_locked=True)
        )

        events = list(self.db.scalars(statement))

        if not events:
            return claim_token, []

        for event in events:
            event.claimed_at = now
            event.claim_token = claim_token
            event.attempt_count += 1

        return claim_token, events

    def mark_published(
        self,
        *,
        event_id: UUID,
        claim_token: UUID,
    ) -> bool:
        """Mark an event as published only when the claim token still matches."""

        statement = (
            update(OutboxEvent)
            .where(
                OutboxEvent.id == event_id,
                OutboxEvent.published_at.is_(None),
                OutboxEvent.claim_token == claim_token,
            )
            .values(
                published_at=datetime.now(UTC),
                claimed_at=None,
                claim_token=None,
                last_error=None,
            )
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount == 1

    def record_failure(
        self,
        *,
        event_id: UUID,
        claim_token: UUID,
        error: str,
    ) -> bool:
        """Record a publication failure while retaining the event for retry."""

        statement = (
            update(OutboxEvent)
            .where(
                OutboxEvent.id == event_id,
                OutboxEvent.published_at.is_(None),
                OutboxEvent.claim_token == claim_token,
            )
            .values(
                claimed_at=None,
                claim_token=None,
                last_error=error[:4000],
            )
        )

        result = cast(CursorResult[None], self.db.execute(statement))

        return result.rowcount == 1
