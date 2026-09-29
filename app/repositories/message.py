from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session, selectinload

from app.core.pagination import CursorPage, CursorPosition
from app.db.models import Conversation, Message
from app.repositories.base import BaseRepository


class MessageRepository(BaseRepository[Message]):
    """Repository for Message model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=Message,
        )

    def get_by_user_and_id(
        self,
        user_id: UUID,
        message_id: UUID,
    ) -> Message | None:
        """Retrieve a message by ID verifying that its parent conversation belongs to user_id."""
        statement = (
            select(Message)
            .join(Conversation, Message.conversation_id == Conversation.id)
            .where(
                Message.id == message_id,
                Conversation.user_id == user_id,
            )
        )
        return self.db.scalar(statement)

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
            .options(selectinload(Message.sources))
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at.asc(), Message.id.asc())
            .limit(limit)
            .offset(offset)
        )
        return list(self.db.scalars(statement))

    def get_by_id_with_sources(self, id: UUID) -> Message | None:
        """Fetch a message by ID eagerly loading its durable sources."""
        statement = (
            select(Message)
            .options(selectinload(Message.sources))
            .where(Message.id == id)
        )
        return self.db.scalar(statement)

    def list_recent_by_conversation(
        self,
        *,
        conversation_id: UUID,
        limit: int,
        exclude_message_id: UUID | None = None,
    ) -> Sequence[Message]:
        """
        Return the most recent messages in chronological order.

        Optionally exclude a message that has already been persisted but should
        not participate in the current prompt history.
        """

        filters = [Message.conversation_id == conversation_id]

        if exclude_message_id is not None:
            filters.append(Message.id != exclude_message_id)

        statement = (
            select(Message)
            .where(*filters)
            .order_by(Message.created_at.desc(), Message.id.desc())
            .limit(limit)
        )

        messages = list(self.db.scalars(statement))
        messages.reverse()
        return messages

    def list_page_by_conversation(
        self,
        *,
        conversation_id: UUID,
        page_size: int,
        snapshot_timestamp: datetime,
        after: CursorPosition | None = None,
        before: CursorPosition | None = None,
    ) -> CursorPage[Message]:
        """Return a stable newest-first keyset page of conversation messages."""

        filters = [
            Message.conversation_id == conversation_id,
            Message.created_at <= snapshot_timestamp,
        ]
        if after is not None:
            filters.append(
                or_(
                    Message.created_at < after.created_at,
                    and_(
                        Message.created_at == after.created_at,
                        Message.id < after.entity_id,
                    ),
                )
            )
        if before is not None:
            filters.append(
                or_(
                    Message.created_at > before.created_at,
                    and_(
                        Message.created_at == before.created_at,
                        Message.id > before.entity_id,
                    ),
                )
            )

        descending = before is None
        ordering = (
            (Message.created_at.desc(), Message.id.desc())
            if descending
            else (Message.created_at.asc(), Message.id.asc())
        )
        statement = (
            select(Message)
            .options(selectinload(Message.sources))
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
