from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.context.models import AssembledContext
from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.message_adapter import to_chat_message
from app.core.ai.llm.models import ChatMessage
from app.core.ai.prompting.base import PromptBuilder
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.db.models import Conversation, Message
from app.db.models.enums import MessageRole
from app.modules.retrieval.service import RetrievalService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    """Application result for a completed, persisted chat turn."""

    assistant_message: Message
    context: AssembledContext


class ChatService:
    """Application service responsible for synchronous RAG chat turns."""

    def __init__(
        self,
        *,
        db: Session,
        conversation_repository: ConversationRepository,
        message_repository: MessageRepository,
        retrieval_service: RetrievalService,
        context_assembler: ContextAssembler,
        prompt_builder: PromptBuilder,
        llm_provider: LLMProvider,
        history_message_limit: int,
        retrieval_limit: int,
        similarity_threshold: float | None,
    ) -> None:
        self._db = db
        self._conversation_repository = conversation_repository
        self._message_repository = message_repository
        self._retrieval_service = retrieval_service
        self._context_assembler = context_assembler
        self._prompt_builder = prompt_builder
        self._llm_provider = llm_provider
        self._history_message_limit = history_message_limit
        self._retrieval_limit = retrieval_limit
        self._similarity_threshold = similarity_threshold

    def _get_conversation(self, *, conversation_id: UUID) -> Conversation:
        """Return a conversation or raise when it no longer exists."""

        conversation = self._conversation_repository.get_by_id(conversation_id)
        if conversation is None:
            raise ResourceNotFoundException("Conversation not found")

        return conversation

    def _load_history(self, *, conversation_id: UUID) -> list[ChatMessage]:
        """Load the bounded prompt history in chronological order."""

        messages: Sequence[Message] = (
            self._message_repository.list_recent_by_conversation(
                conversation_id=conversation_id,
                limit=self._history_message_limit,
            )
        )
        return [to_chat_message(message) for message in messages]

    @staticmethod
    def _normalize_assistant_content(*, content: str) -> str:
        """Reject provider responses that contain no usable assistant text."""

        normalized_content = content.strip()
        if not normalized_content:
            raise AIServiceException("LLM returned an empty response")

        return normalized_content

    def _persist_messages(
        self,
        *,
        conversation_id: UUID,
        user_content: str,
        assistant_content: str,
    ) -> Message:
        """Atomically persist the completed user and assistant turn."""

        try:
            if not self._conversation_repository.exists(conversation_id):
                raise ResourceNotFoundException("Conversation not found")

            user_message = Message(
                role=MessageRole.USER,
                content=user_content,
                conversation_id=conversation_id,
            )
            assistant_message = Message(
                role=MessageRole.ASSISTANT,
                content=assistant_content,
                conversation_id=conversation_id,
            )

            self._message_repository.create(user_message)
            self._message_repository.create(assistant_message)
            self._db.commit()
        except SQLAlchemyError as exc:
            self._db.rollback()
            logger.exception(
                "Chat message persistence failed",
                extra={"conversation_id": str(conversation_id)},
            )
            raise DatabaseException("Failed to persist chat messages") from exc
        except Exception:
            self._db.rollback()
            raise

        self._message_repository.refresh(assistant_message)

        logger.info(
            "Chat messages persisted",
            extra={
                "conversation_id": str(conversation_id),
                "user_message_id": str(user_message.id),
                "assistant_message_id": str(assistant_message.id),
            },
        )

        return assistant_message

    def send_message(
        self,
        *,
        conversation_id: UUID,
        content: str,
    ) -> ChatTurnResult:
        """Generate and persist one synchronous retrieval-augmented chat turn."""

        request_started_at = perf_counter()
        logger.info(
            "Chat request received",
            extra={"conversation_id": str(conversation_id)},
        )

        conversation = self._get_conversation(conversation_id=conversation_id)
        knowledge_base_id = conversation.knowledge_base_id

        history = self._load_history(conversation_id=conversation_id)
        logger.info(
            "Chat history loaded",
            extra={
                "conversation_id": str(conversation_id),
                "history_count": len(history),
            },
        )

        retrieval_started_at = perf_counter()
        retrieved_chunks = self._retrieval_service.retrieve(
            knowledge_base_id=knowledge_base_id,
            query=content,
            limit=self._retrieval_limit,
            threshold=self._similarity_threshold,
        )
        retrieval_latency_ms = round((perf_counter() - retrieval_started_at) * 1000)
        logger.info(
            "Chat retrieval completed",
            extra={
                "conversation_id": str(conversation_id),
                "knowledge_base_id": str(knowledge_base_id),
                "result_count": len(retrieved_chunks),
                "latency_ms": retrieval_latency_ms,
            },
        )

        context_started_at = perf_counter()
        context = self._context_assembler.assemble(
            query=content,
            retrieved_chunks=retrieved_chunks,
        )
        prompt = self._prompt_builder.build(
            context=context,
            conversation=history,
            user_query=content,
        )
        context_prompt_latency_ms = round((perf_counter() - context_started_at) * 1000)
        logger.info(
            "Chat prompt built",
            extra={
                "conversation_id": str(conversation_id),
                "context_count": len(context.chunks),
                "prompt_message_count": len(prompt),
                "latency_ms": context_prompt_latency_ms,
            },
        )

        # Release the read-only transaction before waiting on the remote provider.
        self._db.rollback()

        logger.info(
            "LLM generation started",
            extra={"conversation_id": str(conversation_id)},
        )
        llm_started_at = perf_counter()
        llm_response = self._llm_provider.generate(prompt)
        llm_latency_ms = round((perf_counter() - llm_started_at) * 1000)
        assistant_content = self._normalize_assistant_content(
            content=llm_response.content,
        )
        logger.info(
            "LLM generation completed",
            extra={
                "conversation_id": str(conversation_id),
                "model": llm_response.model,
                "finish_reason": llm_response.finish_reason,
                "latency_ms": llm_latency_ms,
            },
        )

        assistant_message = self._persist_messages(
            conversation_id=conversation_id,
            user_content=content,
            assistant_content=assistant_content,
        )

        request_latency_ms = round((perf_counter() - request_started_at) * 1000)
        logger.info(
            "Chat request completed",
            extra={
                "conversation_id": str(conversation_id),
                "latency_ms": request_latency_ms,
            },
        )

        return ChatTurnResult(
            assistant_message=assistant_message,
            context=context,
        )
