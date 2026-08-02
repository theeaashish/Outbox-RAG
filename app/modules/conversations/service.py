from __future__ import annotations

import logging
from collections.abc import Sequence
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundException
from app.db.models import Conversation, KnowledgeBase, Message
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """Application service responsible for conversation lifecycle management."""

    def __init__(
        self,
        *,
        db: Session,
        conversation_repository: ConversationRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        message_repository: MessageRepository,
    ) -> None:
        self._db = db
        self._conversation_repository = conversation_repository
        self._knowledge_base_repository = knowledge_base_repository
        self._message_repository = message_repository

    def _get_knowledge_base(self, *, knowledge_base_id: UUID) -> KnowledgeBase:
        """Retrieve a knowledge base or raise ResourceNotFoundException if not found."""

        knowledge_base = self._knowledge_base_repository.get_by_id(knowledge_base_id)

        if knowledge_base is None:
            raise ResourceNotFoundException(
                f"Knowledge base with ID {knowledge_base_id} not found"
            )
        return knowledge_base

    def _get_conversation(
        self,
        *,
        conversation_id: UUID,
    ) -> Conversation:
        """Retrieve a conversation or raise if it does not exist."""

        conversation = self._conversation_repository.get_by_id(
            conversation_id,
        )

        if conversation is None:
            raise ResourceNotFoundException("Conversation not found")

        return conversation

    def _create_conversation(
        self,
        *,
        knowledge_base: KnowledgeBase,
    ) -> Conversation:
        """Create a new conversation."""

        conversation = Conversation(
            knowledge_base_id=knowledge_base.id,
        )

        return self._conversation_repository.create(conversation)

    def create_conversation(
        self,
        *,
        knowledge_base_id: UUID,
    ) -> Conversation:
        """Create a conversation for a knowledge base."""

        knowledge_base = self._get_knowledge_base(
            knowledge_base_id=knowledge_base_id,
        )

        try:
            conversation = self._create_conversation(
                knowledge_base=knowledge_base,
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        self._conversation_repository.refresh(conversation)

        logger.info(
            "Conversation created",
            extra={
                "conversation_id": str(conversation.id),
                "knowledge_base_id": str(knowledge_base.id),
            },
        )

        return conversation

    def get_conversation(
        self,
        *,
        conversation_id: UUID,
    ) -> Conversation:
        """Retrieve a conversation."""

        return self._get_conversation(
            conversation_id=conversation_id,
        )

    def list_conversations(
        self,
        *,
        knowledge_base_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """List conversations for a knowledge base."""

        self._get_knowledge_base(
            knowledge_base_id=knowledge_base_id,
        )

        return self._conversation_repository.list_by_knowledge_base(
            knowledge_base_id=knowledge_base_id,
            limit=limit,
            offset=offset,
        )

    def list_messages(
        self,
        *,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """List messages for a conversation."""

        self._get_conversation(
            conversation_id=conversation_id,
        )

        return self._message_repository.list_by_conversation(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

    def delete_conversation(
        self,
        *,
        conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""

        conversation = self._get_conversation(
            conversation_id=conversation_id,
        )

        try:
            self._conversation_repository.delete(conversation)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        logger.info(
            "Conversation deleted",
            extra={
                "conversation_id": str(conversation.id),
                "knowledge_base_id": str(conversation.knowledge_base_id),
            },
        )
