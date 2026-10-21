from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.ai.llm.models import LLMUsage
from app.core.config import settings
from app.modules.conversations.schemas import (
    MessageResponse,
    MessageSourceResponse,
)


class ChatModelInfo(BaseModel):
    """Metadata describing a chat model option for the client."""

    id: str
    name: str
    provider: str
    is_default: bool = False

    model_config = ConfigDict(from_attributes=True)


class ChatModelsResponse(BaseModel):
    """Response containing available chat models and default selection."""

    default_model: str
    models: list[ChatModelInfo]


class ChatRequest(BaseModel):
    """Request payload for a synchronous chat turn."""

    content: str = Field(max_length=settings.chat_max_message_characters)
    model: str | None = Field(
        default=None,
        description="Optional model identifier or provider name (e.g. 'openai/gpt-oss-120b', 'gemini-3.5-flash-lite', 'groq', 'gemini').",
    )

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
    sources: list[MessageSourceResponse]


class ChatStreamMetadataResponse(BaseModel):
    conversation_id: UUID
    source_count: int


class ChatStreamTokenResponse(BaseModel):
    delta: str


class ChatStreamCompleteResponse(BaseModel):
    assistant_message: MessageResponse
    model: str
    finish_reason: str
    usage: LLMUsage | None


class ChatStreamErrorResponse(BaseModel):
    code: str
    message: str
