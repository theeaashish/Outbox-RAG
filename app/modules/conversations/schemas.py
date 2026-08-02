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
