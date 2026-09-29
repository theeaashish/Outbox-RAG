from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from app.db.models import MessageFeedback
from app.db.models.enums import FeedbackRating
from app.repositories.base import BaseRepository


class MessageFeedbackRepository(BaseRepository[MessageFeedback]):
    """Repository for MessageFeedback model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=MessageFeedback,
        )

    def upsert(
        self,
        *,
        message_id: UUID,
        user_id: UUID,
        rating: FeedbackRating,
    ) -> MessageFeedback:
        """
        Concurrency-safe upsert of user feedback rating.

        Uses PostgreSQL ON CONFLICT DO UPDATE.
        """
        statement = (
            pg_insert(MessageFeedback)
            .values(
                id=uuid4(),
                message_id=message_id,
                user_id=user_id,
                rating=rating,
            )
            .on_conflict_do_update(
                constraint="uq_message_feedbacks_message_user",
                set_={
                    "rating": rating,
                    "updated_at": func.now(),
                },
            )
            .returning(MessageFeedback)
        )

        result = self.db.scalar(statement)
        assert result is not None
        return result

    def delete_by_message_and_user(
        self,
        *,
        message_id: UUID,
        user_id: UUID,
    ) -> bool:
        """Delete feedback rating for a specific message and user. Returns True if deleted."""
        statement = delete(MessageFeedback).where(
            MessageFeedback.message_id == message_id,
            MessageFeedback.user_id == user_id,
        )

        result = cast(CursorResult[None], self.db.execute(statement))
        return (result.rowcount or 0) > 0

    def get_ratings_for_messages_and_user(
        self,
        *,
        message_ids: Sequence[UUID],
        user_id: UUID,
    ) -> dict[UUID, FeedbackRating]:
        """
        Batch-load feedback ratings for a sequence of messages for a single user.

        Avoids N+1 queries and ensures no feedback from other users is leaked.
        """
        if not message_ids:
            return {}

        statement = select(MessageFeedback.message_id, MessageFeedback.rating).where(
            MessageFeedback.user_id == user_id,
            MessageFeedback.message_id.in_(message_ids),
        )

        rows = self.db.execute(statement).all()
        return {row[0]: row[1] for row in rows}
