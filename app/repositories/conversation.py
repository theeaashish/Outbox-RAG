from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Conversation
from app.repositories.base import BaseRepository


class ConversationRepository(BaseRepository[Conversation]):
    """Repository for Conversation model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=Conversation,
        )

    def list_by_knowledge_base(
        self,
        *,
        knowledge_base_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """Return conversations for a knowledge base ordered newest first."""
        statement = (
            select(Conversation)
            .where(Conversation.knowledge_base_id == knowledge_base_id)
            .order_by(
                Conversation.created_at.desc(),
                Conversation.id.desc(),
            )
            .limit(limit)
            .offset(offset)
        )
        return self.db.scalars(statement).all()
