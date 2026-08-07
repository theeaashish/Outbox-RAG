from __future__ import annotations

from app.core.ai.llm.models import ChatMessage
from app.db.models import Message


def to_chat_message(message: Message) -> ChatMessage:
    """Map a persisted conversation message to the provider-neutral form."""

    return ChatMessage(
        role=message.role,
        content=message.content,
    )
