from __future__ import annotations

from typing import ClassVar

from app.core.ai.llm.models import ChatMessage
from app.core.ai.prompting.templates import CONVERSATIONAL_SYSTEM_PROMPT
from app.db.models.enums import MessageRole


class ConversationalPromptBuilder:
    """
    Builds conversational prompts for non-retrieval chat turns.
    """

    _ALLOWED_HISTORY_ROLES: ClassVar[frozenset[MessageRole]] = frozenset(
        {MessageRole.USER, MessageRole.ASSISTANT}
    )

    def build(
        self,
        *,
        conversation: list[ChatMessage],
        user_query: str,
    ) -> list[ChatMessage]:
        """
        Build a chat prompt with a single conversational system message,
        filtered dialogue history, and a plain user turn (without context tags).
        """
        messages: list[ChatMessage] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=CONVERSATIONAL_SYSTEM_PROMPT,
            )
        ]

        for msg in conversation:
            if msg.role in self._ALLOWED_HISTORY_ROLES and msg.content.strip():
                messages.append(msg)

        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=user_query,
            )
        )

        return messages
