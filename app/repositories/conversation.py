from __future__ import annotations

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

    def get_by_session_and_knowledge_base(
        self,
        *,
        session_id: str,
        knowledge_base_id: UUID,
    ) -> Conversation | None:
        """Retrieve a conversation by session ID and knowledge base ID."""
        statement = select(Conversation).where(
            Conversation.session_id == session_id,
            Conversation.knowledge_base_id == knowledge_base_id,
        )
        return self.db.scalar(statement)
