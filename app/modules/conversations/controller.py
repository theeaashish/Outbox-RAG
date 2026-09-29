from __future__ import annotations

from uuid import UUID

from app.db.models.enums import FeedbackRating, MessageRole
from app.modules.conversations import mapper
from app.modules.conversations.schemas import (
    ConversationCursorPageResponse,
    ConversationListResponse,
    ConversationResponse,
    MessageCursorPageResponse,
    MessageFeedbackResponse,
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
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> ConversationResponse:
        """Create a new conversation."""

        conversation = self._conversation_service.create_conversation(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        return mapper.to_conversation_response(conversation)

    def create_project_conversation(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> ConversationResponse:
        conversation = self._conversation_service.create_project_conversation(
            user_id=user_id,
            project_id=project_id,
        )
        return mapper.to_conversation_response(conversation)

    def get_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> ConversationResponse:
        """Retrieve a conversation."""

        conversation = self._conversation_service.get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        return mapper.to_conversation_response(conversation)

    def list_conversations(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> ConversationListResponse:
        """List conversations for a knowledge base."""

        conversations = self._conversation_service.list_conversations(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            limit=limit,
            offset=offset,
        )

        return mapper.to_conversation_list_response(conversations)

    def list_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> MessageListResponse:
        """List messages for a conversation."""

        messages = self._conversation_service.list_messages(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )
        assistant_ids = [m.id for m in messages if m.role == MessageRole.ASSISTANT]
        feedback_map = (
            self._conversation_service.get_user_feedback_map(
                user_id=user_id,
                message_ids=assistant_ids,
            )
            if assistant_ids
            else {}
        )

        return mapper.to_message_list_response(messages, feedback_map=feedback_map)

    def list_conversations_cursor(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ) -> ConversationCursorPageResponse:
        page = self._conversation_service.list_conversations_cursor(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            page_size=page_size,
            after=after,
            before=before,
        )
        return mapper.to_conversation_cursor_page_response(
            page=page,
            page_size=page_size,
            knowledge_base_id=knowledge_base_id,
            cursor_codec=self._conversation_service.cursor_codec,
        )

    def list_messages_cursor(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ) -> MessageCursorPageResponse:
        page, feedback_map = self._conversation_service.list_messages_cursor(
            user_id=user_id,
            conversation_id=conversation_id,
            page_size=page_size,
            after=after,
            before=before,
        )
        return mapper.to_message_cursor_page_response(
            page=page,
            page_size=page_size,
            conversation_id=conversation_id,
            cursor_codec=self._conversation_service.cursor_codec,
            feedback_map=feedback_map,
        )

    def set_feedback(
        self,
        *,
        user_id: UUID,
        message_id: UUID,
        rating: FeedbackRating,
    ) -> MessageFeedbackResponse:
        """Upsert user feedback for an assistant message."""
        feedback = self._conversation_service.upsert_feedback(
            user_id=user_id,
            message_id=message_id,
            rating=rating,
        )
        return mapper.to_message_feedback_response(feedback)

    def delete_feedback(
        self,
        *,
        user_id: UUID,
        message_id: UUID,
    ) -> None:
        """Delete user feedback for an assistant message."""
        self._conversation_service.delete_feedback(
            user_id=user_id,
            message_id=message_id,
        )

    def delete_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""

        self._conversation_service.delete_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
