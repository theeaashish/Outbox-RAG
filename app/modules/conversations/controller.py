from __future__ import annotations

from uuid import UUID

from app.modules.conversations import mapper
from app.modules.conversations.schemas import (
    ConversationListResponse,
    ConversationResponse,
    MessageListResponse,
)
from app.modules.conversations.service import ConversationService


class ConversationController:
    """Thin controller responsible for conversation request orchestration."""

    def __init__(
        self,
        *,
        conversation_service: ConversationService,
    ) -> None:
        self._conversation_service = conversation_service

    def create_conversation(
        self,
        *,
        knowledge_base_id: UUID,
    ) -> ConversationResponse:
        """Create a new conversation."""

        conversation = self._conversation_service.create_conversation(
            knowledge_base_id=knowledge_base_id,
        )

        return mapper.to_conversation_response(conversation)

    def get_conversation(
        self,
        *,
        conversation_id: UUID,
    ) -> ConversationResponse:
        """Retrieve a conversation."""

        conversation = self._conversation_service.get_conversation(
            conversation_id=conversation_id,
        )

        return mapper.to_conversation_response(conversation)

    def list_conversations(
        self,
        *,
        knowledge_base_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> ConversationListResponse:
        """List conversations for a knowledge base."""

        conversations = self._conversation_service.list_conversations(
            knowledge_base_id=knowledge_base_id,
            limit=limit,
            offset=offset,
        )

        return mapper.to_conversation_list_response(conversations)

    def list_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> MessageListResponse:
        """List messages for a conversation."""

        messages = self._conversation_service.list_messages(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

        return mapper.to_message_list_response(messages)

    def delete_conversation(
        self,
        *,
        conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""

        self._conversation_service.delete_conversation(
            conversation_id=conversation_id,
        )
