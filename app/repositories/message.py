from __future__ import annotations

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

    def get_by_conversation(self, *, conversation_id: UUID) -> list[Message]:
        """Retrieve all messages for a given conversation."""
        statement = (
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc())
        )
        return list(self.db.scalars(statement))
