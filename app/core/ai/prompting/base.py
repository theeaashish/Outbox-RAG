from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.ai.context.models import AssembledContext
from app.core.ai.llm.models import ChatMessage


class PromptBuilder(ABC):
    """
    Abstract interface for prompt builders.
    """

    @abstractmethod
    def build(
        self,
        *,
        context: AssembledContext,
        conversation: list[ChatMessage],
        user_query: str,
    ) -> list[ChatMessage]:
        """
        Build a provider-agnostic chat prompt.
        """
        raise NotImplementedError
