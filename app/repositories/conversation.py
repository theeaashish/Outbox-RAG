from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from app.core.pagination import CursorPage, CursorPosition
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

    def list_page_by_knowledge_base(
        self,
        *,
        knowledge_base_id: UUID,
        page_size: int,
        snapshot_timestamp: datetime,
        after: CursorPosition | None = None,
        before: CursorPosition | None = None,
    ) -> CursorPage[Conversation]:
        """Return a stable keyset page ordered by creation time descending."""

        filters = [
            Conversation.knowledge_base_id == knowledge_base_id,
            Conversation.created_at <= snapshot_timestamp,
        ]
        if after is not None:
            filters.append(
                or_(
                    Conversation.created_at < after.created_at,
                    and_(
                        Conversation.created_at == after.created_at,
                        Conversation.id < after.entity_id,
                    ),
                )
            )
        if before is not None:
            filters.append(
                or_(
                    Conversation.created_at > before.created_at,
                    and_(
                        Conversation.created_at == before.created_at,
                        Conversation.id > before.entity_id,
                    ),
                )
            )

        descending = before is None
        ordering = (
            (Conversation.created_at.desc(), Conversation.id.desc())
            if descending
            else (Conversation.created_at.asc(), Conversation.id.asc())
        )
        statement = (
            select(Conversation)
            .where(*filters)
            .order_by(*ordering)
            .limit(page_size + 1)
        )
        items = list(self.db.scalars(statement))
        has_more = len(items) > page_size
        items = items[:page_size]
        if not descending:
            items.reverse()

        return CursorPage(
            items=items,
            has_next_page=has_more if descending else before is not None,
            has_previous_page=after is not None if descending else has_more,
            snapshot_timestamp=snapshot_timestamp,
        )
