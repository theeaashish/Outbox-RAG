from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.modules.conversations.schemas import MessageResponse


class ChatRequest(BaseModel):
    """Request payload for a synchronous chat turn."""

    content: str = Field(max_length=settings.chat_max_message_characters)

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, value: object) -> object:
        """Trim content and reject messages that contain only whitespace."""

        if not isinstance(value, str):
            return value

        content = value.strip()
        if not content:
            raise ValueError("Message content cannot be blank")

        return content


class ChatSourceResponse(BaseModel):
    """Citation metadata for a context chunk used in an answer."""

    citation: int
    document_id: UUID
    document_name: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    """Response for a completed synchronous chat turn."""

    assistant_message: MessageResponse
    sources: list[ChatSourceResponse]
