from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.ai.llm.models import ChatMessage, LLMResponse


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
