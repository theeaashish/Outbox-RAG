from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.exceptions import ResourceNotFoundException, ValidationException
from app.core.pagination import (
    CursorCodec,
    CursorPage,
    CursorPosition,
    CursorResource,
    InvalidCursorError,
)
from app.db.models import Conversation, KnowledgeBase, Message
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository
from app.repositories.project import ProjectRepository

logger = logging.getLogger(__name__)


class ConversationService:
    """Application service responsible for conversation lifecycle management."""

    def __init__(
        self,
        *,
        db: Session,
        conversation_repository: ConversationRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
        project_repository: ProjectRepository,
        message_repository: MessageRepository,
        cursor_codec: CursorCodec,
    ) -> None:
        self._db = db
        self._conversation_repository = conversation_repository
        self._knowledge_base_repository = knowledge_base_repository
        self._project_repository = project_repository
        self._message_repository = message_repository
        self._cursor_codec = cursor_codec

    @property
    def cursor_codec(self) -> CursorCodec:
        """Expose the codec for response cursor construction."""

        return self._cursor_codec

    def _get_knowledge_base(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> KnowledgeBase:
        """Retrieve a knowledge base or raise ResourceNotFoundException if not found."""

        knowledge_base = self._knowledge_base_repository.get_by_user_and_id(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        if knowledge_base is None:
            raise ResourceNotFoundException(
                f"Knowledge base with ID {knowledge_base_id} not found"
            )
        return knowledge_base

    def _get_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Conversation:
        """Retrieve a conversation or raise if it does not exist."""

        conversation = self._conversation_repository.get_by_user_and_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        if conversation is None:
            raise ResourceNotFoundException("Conversation not found")

        return conversation

    def _create_conversation(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
        knowledge_base: KnowledgeBase,
    ) -> Conversation:
        """Create a new conversation after verifying relationship invariants."""

        if knowledge_base.project_id != project_id or knowledge_base.user_id != user_id:
            raise ResourceNotFoundException("Inconsistent resource ownership")

        conversation = Conversation(
            user_id=user_id,
            project_id=project_id,
            knowledge_base_id=knowledge_base.id,
        )

        return self._conversation_repository.create(conversation)

    def create_conversation(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> Conversation:
        """Create a conversation for a knowledge base."""

        knowledge_base = self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        project = self._project_repository.get_by_user_and_id(
            user_id=user_id,
            project_id=knowledge_base.project_id,
        )
        if project is None:
            raise ResourceNotFoundException("Project not found")

        try:
            conversation = self._create_conversation(
                user_id=user_id,
                project_id=project.id,
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

    def create_project_conversation(
        self,
        *,
        user_id: UUID,
        project_id: UUID,
    ) -> Conversation:
        """Create a conversation using the project's knowledge base."""
        project = self._project_repository.get_by_user_and_id(
            user_id=user_id,
            project_id=project_id,
        )
        if project is None:
            raise ResourceNotFoundException("Project not found")

        knowledge_base = self._knowledge_base_repository.get_by_user_and_project_id(
            user_id=user_id,
            project_id=project.id,
        )
        if knowledge_base is None:
            raise ResourceNotFoundException("Project knowledge base not found")

        try:
            conversation = self._create_conversation(
                user_id=user_id,
                project_id=project.id,
                knowledge_base=knowledge_base,
            )
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        self._conversation_repository.refresh(conversation)
        return conversation

    def get_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Conversation:
        """Retrieve a conversation."""

        return self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

    def list_conversations(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[Conversation]:
        """List conversations for a knowledge base."""

        self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        return self._conversation_repository.list_by_user_and_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            limit=limit,
            offset=offset,
        )

    def list_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        limit: int = 100,
        offset: int = 0,
    ) -> Sequence[Message]:
        """List messages for a conversation."""

        self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        return self._message_repository.list_by_conversation(
            conversation_id=conversation_id,
            limit=limit,
            offset=offset,
        )

    def _decode_page_cursors(
        self,
        *,
        resource: CursorResource,
        scope_id: UUID,
        after: str | None,
        before: str | None,
    ) -> tuple[CursorPosition | None, CursorPosition | None, datetime]:
        if after is not None and before is not None:
            raise ValidationException("Only one of after or before may be provided")

        try:
            after_position = (
                self._cursor_codec.decode(
                    cursor=after,
                    resource=resource,
                    scope_id=scope_id,
                )
                if after is not None
                else None
            )
            before_position = (
                self._cursor_codec.decode(
                    cursor=before,
                    resource=resource,
                    scope_id=scope_id,
                )
                if before is not None
                else None
            )
        except InvalidCursorError as exc:
            raise ValidationException("Invalid cursor") from exc

        snapshot_timestamp = (
            after_position.snapshot_timestamp
            if after_position is not None
            else (
                before_position.snapshot_timestamp
                if before_position is not None
                else datetime.now(UTC)
            )
        )
        return after_position, before_position, snapshot_timestamp

    def list_conversations_cursor(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ) -> CursorPage[Conversation]:
        """List knowledge-base conversations through a signed cursor page."""

        self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        after_position, before_position, snapshot_timestamp = self._decode_page_cursors(
            resource=CursorResource.CONVERSATIONS,
            scope_id=knowledge_base_id,
            after=after,
            before=before,
        )
        return self._conversation_repository.list_page_by_user_and_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            page_size=page_size,
            snapshot_timestamp=snapshot_timestamp,
            after=after_position,
            before=before_position,
        )

    def list_messages_cursor(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ) -> CursorPage[Message]:
        """List newest-first conversation messages through a signed cursor page."""

        self._get_conversation(
            user_id=user_id,
            conversation_id=conversation_id,
        )
        after_position, before_position, snapshot_timestamp = self._decode_page_cursors(
            resource=CursorResource.MESSAGES,
            scope_id=conversation_id,
            after=after,
            before=before,
        )
        return self._message_repository.list_page_by_conversation(
            conversation_id=conversation_id,
            page_size=page_size,
            snapshot_timestamp=snapshot_timestamp,
            after=after_position,
            before=before_position,
        )

    def delete_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> None:
        """Delete a conversation."""

        conversation = self._get_conversation(
            user_id=user_id,
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
