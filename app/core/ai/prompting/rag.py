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

    _ALLOWED_HISTORY_ROLES: frozenset[MessageRole] = frozenset(
        {MessageRole.USER, MessageRole.ASSISTANT}
    )

    def build(
        self,
        *,
        context: AssembledContext,
        conversation: list[ChatMessage],
        user_query: str,
    ) -> list[ChatMessage]:
        """
        Build a provider-agnostic chat prompt with a single authoritative system message,
        filtered dialogue history, and a user turn bundling context data and query.
        """
        messages: list[ChatMessage] = []

        # 1. Authoritative System Message
        messages.append(
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=RAG_SYSTEM_PROMPT,
            )
        )

        # 2. Filtered Dialogue History (USER and ASSISTANT only)
        for msg in conversation:
            if msg.role in self._ALLOWED_HISTORY_ROLES and msg.content.strip():
                messages.append(msg)

        # 3. Final User Message with Encapsulated Context Data and Query
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=self._build_user_message(
                    context=context, user_query=user_query
                ),
            )
        )

        return messages

    @classmethod
    def _build_user_message(
        cls,
        *,
        context: AssembledContext,
        user_query: str,
    ) -> str:
        """
        Format retrieved context and user query inside a structured user turn.
        """
        return (
            "<retrieved_context>\n"
            f"{context.block}\n"
            "</retrieved_context>\n\n"
            f"User Question:\n{user_query}"
        )
