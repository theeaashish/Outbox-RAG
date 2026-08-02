from __future__ import annotations

from collections.abc import Sequence

from app.db.models import Conversation, Message
from app.modules.conversations.schemas import (
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
    MessageResponse,
)


def to_conversation_response(conversation: Conversation) -> ConversationResponse:
    """Map a conversation orm model to its API response."""

    return ConversationResponse(
        id=conversation.id,
        knowledge_base_id=conversation.knowledge_base_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


def to_conversation_list_response(
    conversations: Sequence[Conversation],
) -> ConversationListResponse:
    """Map a list of conversation orm models to their API response."""

    return ConversationListResponse(
        results=[
            to_conversation_response(conversation) for conversation in conversations
        ]
    )


def to_message_response(message: Message) -> MessageResponse:
    """Map a message orm model to its API response."""

    return MessageResponse(
        id=message.id,
        role=message.role,
        content=message.content,
        created_at=message.created_at,
        updated_at=message.updated_at,
    )


def to_message_list_response(
    messages: Sequence[Message],
) -> MessageListResponse:
    """Map a list of message orm models to their API response."""

    return MessageListResponse(
        results=[to_message_response(message) for message in messages]
    )
