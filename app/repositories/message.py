from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Message
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=Message,
        )

    def list_by_conversation(
        self,
        *,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """Retrieve messages for a given conversation, oldest first, up to limit."""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(statement))

    def list_recent_by_conversation(
        self,
        *,
        conversation_id: UUID,
        limit: int,
    ) -> Sequence[Message]:
        """Return the most recent messages in chronological order."""

        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )
        messages = list(self.db.scalars(statement))
        messages.reverse()
        return messages
