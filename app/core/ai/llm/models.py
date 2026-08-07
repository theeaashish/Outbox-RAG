from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

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


@dataclass(frozen=True, slots=True)
class LLMStreamDelta:
    """A normalized incremental assistant-text update."""

    content: str


@dataclass(frozen=True, slots=True)
class LLMStreamCompletion:
    """The normalized terminal metadata for an LLM stream."""

    model: str
    finish_reason: str
    usage: LLMUsage | None


type LLMStreamEvent = LLMStreamDelta | LLMStreamCompletion


class LLMStream(Protocol):
    """Closable provider-neutral stream of LLM events."""

    def __iter__(self) -> Iterator[LLMStreamEvent]: ...

    def __next__(self) -> LLMStreamEvent: ...

    def close(self) -> None: ...
