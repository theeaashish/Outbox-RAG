from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from app.db.models.enums import FeedbackRating, MessageRole


class MessageSourceResponse(BaseModel):
    """Durable context source backing an assistant message."""

    id: UUID
    citation: int = Field(
        ...,
        description="1-based context citation number matching [n]",
    )
    is_cited: bool = Field(..., description="True if cited in the assistant response")
    source_order: int = Field(
        ..., description="0-based sequence position in assembled context"
    )
    document_id: UUID | None = Field(
        None, description="Authoritative document ID if document exists"
    )
    document_chunk_id: UUID | None = Field(
        None, description="Authoritative chunk ID if chunk exists"
    )
    document_name: str = Field(
        ..., description="Document title snapshot at citation time"
    )
    chunk_index: int = Field(..., description="0-based chunk sequence index")
    score: float = Field(..., description="Cosine similarity score (0.0 to 1.0)")
    page_start: int | None = Field(None, description="Starting page if available")
    page_end: int | None = Field(None, description="Ending page if available")


class FeedbackRequest(BaseModel):
    """Request payload to set message feedback."""

    rating: FeedbackRating


class MessageFeedbackResponse(BaseModel):
    """Response schema for feedback operations."""

    id: UUID
    message_id: UUID
    user_id: UUID = Field(..., description="User ID of the feedback author")
    rating: FeedbackRating
    created_at: datetime
    updated_at: datetime


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
    sources: list[MessageSourceResponse] = Field(default_factory=list)
    feedback: FeedbackRating | None = Field(default=None)


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
