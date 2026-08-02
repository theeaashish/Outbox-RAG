from __future__ import annotations

from dataclasses import dataclass

from app.db.models.enums import MessageRole


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Provider-agnostic representation of a chat message."""

    role: MessageRole
    content: str


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Normalized token usage reported by an LLM provider."""

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response returned by an LLM provider."""

    content: str

    model: str

    finish_reason: str

    usage: LLMUsage | None
