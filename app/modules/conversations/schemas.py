from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel

from app.db.models.enums import MessageRole


class ConversationResponse(BaseModel):
    """Response schema for conversation operations."""

    id: UUID
    knowledge_base_id: UUID
    created_at: datetime
    updated_at: datetime


class ConversationListResponse(BaseModel):
    """List of conversation"""

    results: list[ConversationResponse]


class MessageResponse(BaseModel):
    """Response schema for message operations."""

    id: UUID
    role: MessageRole
    content: str
    created_at: datetime
    updated_at: datetime


class MessageListResponse(BaseModel):
    """List of messages"""

    results: list[MessageResponse]


class CursorPageInfo(BaseModel):
    next_cursor: str | None
    previous_cursor: str | None
    has_next_page: bool
    has_previous_page: bool
    page_size: int


class ConversationCursorPageResponse(BaseModel):
    items: list[ConversationResponse]
    page_info: CursorPageInfo


class MessageCursorPageResponse(BaseModel):
    items: list[MessageResponse]
    page_info: CursorPageInfo
