from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.ai.llm.models import ChatMessage, LLMResponse, LLMStream


class LLMProvider(ABC):
    """Abstract interface for chat language models."""

    @abstractmethod
    def generate(
        self,
        messages: list[ChatMessage],
    ) -> LLMResponse:
        """
        Generate an assistant response for the supplied chat messages.
        """
        raise NotImplementedError

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
    ) -> LLMStream:
        """Stream a provider-neutral assistant response."""
        raise NotImplementedError
