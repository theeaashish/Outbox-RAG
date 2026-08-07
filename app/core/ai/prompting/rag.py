from __future__ import annotations

from app.core.ai.context.models import AssembledContext
from app.core.ai.llm.models import ChatMessage
from app.core.ai.prompting.base import PromptBuilder
from app.core.ai.prompting.templates import RAG_SYSTEM_PROMPT
from app.db.models.enums import MessageRole


class RAGPromptBuilder(PromptBuilder):
    """
    Builds prompts for retrieval augmented generation.
    """

    def build(
        self,
        *,
        context: AssembledContext,
        conversation: list[ChatMessage],
        user_query: str,
    ) -> list[ChatMessage]:

        messages: list[ChatMessage] = []

        messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=RAG_SYSTEM_PROMPT,
            )
        )

        messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self._build_context_message(context),
            )
        )

        messages.extend(conversation)

        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=user_query,
            )
        )

        return messages

    @staticmethod
    def _build_context_message(
        context: AssembledContext,
    ) -> str:
        """
        Format retrieved context for the language model.
        """

        return f"Retrieved Context\n=================\n\n{context.block}"
