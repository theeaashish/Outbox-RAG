from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.context.models import AssembledContext
from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.message_adapter import to_chat_message
from app.core.ai.llm.models import (
    ChatMessage,
    LLMStream,
    LLMStreamCompletion,
    LLMStreamDelta,
    LLMUsage,
)
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


@dataclass(frozen=True, slots=True)
class PreparedChatTurn:
    """Validated, prompt-ready chat work with no active DB transaction."""

    conversation_id: UUID
    knowledge_base_id: UUID
    user_id: UUID
    user_content: str
    context: AssembledContext
    prompt: list[ChatMessage]


from enum import StrEnum


@dataclass(frozen=True, slots=True)
class ChatStreamMetadata:
    conversation_id: UUID
    source_count: int


@dataclass(frozen=True, slots=True)
class ChatStreamCitations:
    context: AssembledContext


@dataclass(frozen=True, slots=True)
class ChatStreamToken:
    delta: str


@dataclass(frozen=True, slots=True)
class ChatStreamComplete:
    assistant_message: Message
    context: AssembledContext
    model: str
    finish_reason: str
    usage: LLMUsage | None


class ChatStreamEventType(StrEnum):
    METADATA = "metadata"
    CITATIONS = "citations"
    TOKEN = "token"
    COMPLETE = "complete"


type ChatStreamPayload = (
    ChatStreamMetadata | ChatStreamCitations | ChatStreamToken | ChatStreamComplete
)


@dataclass(frozen=True, slots=True)
class ChatStreamEvent:
    type: ChatStreamEventType
    payload: ChatStreamPayload


class ChatEventStream:
    """Closable iterator over domain chat stream events."""

    def __init__(
        self, *, events: Iterator[ChatStreamEvent], provider_stream: LLMStream | None
    ):
        self._events = events
        self._provider_stream = provider_stream
        self._closed = False

    def __iter__(self) -> Iterator[ChatStreamEvent]:
        return self

    def __next__(self) -> ChatStreamEvent:
        return next(self._events)

    def close(self) -> None:
        """Release the provider stream exactly once."""

        if self._closed:
            return
        self._closed = True
        if self._provider_stream is not None:
            self._provider_stream.close()
        close = getattr(self._events, "close", None)
        if callable(close):
            close()


class ChatService:
    """Application service responsible for synchronous and streaming RAG turns."""

    _SUCCESSFUL_STREAM_FINISH_REASONS = frozenset({"stop", "length"})

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
        stream_max_buffered_characters: int,
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
        self._stream_max_buffered_characters = stream_max_buffered_characters

    def _get_conversation(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
    ) -> Conversation:
        conversation = self._conversation_repository.get_by_user_and_id(
            user_id=user_id,
            conversation_id=conversation_id,
        )

        if conversation is None:
            raise ResourceNotFoundException("Conversation not found")
        return conversation

    def _load_history(self, *, conversation_id: UUID) -> list[ChatMessage]:
        messages: Sequence[Message] = (
            self._message_repository.list_recent_by_conversation(
                conversation_id=conversation_id,
                limit=self._history_message_limit,
            )
        )
        return [to_chat_message(message) for message in messages]

    @staticmethod
    def _normalize_assistant_content(*, content: str) -> str:
        normalized_content = content.strip()
        if not normalized_content:
            raise AIServiceException("LLM returned an empty response")
        return normalized_content

    def prepare_turn(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
    ) -> PreparedChatTurn:
        """Build a prompt and release read resources before provider work begins."""

        preparation_started_at = perf_counter()
        try:
            conversation = self._get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            knowledge_base_id = conversation.knowledge_base_id
            history = self._load_history(conversation_id=conversation_id)
            retrieved_chunks = self._retrieval_service.retrieve(
                user_id=user_id,
                knowledge_base_id=knowledge_base_id,
                query=content,
                limit=self._retrieval_limit,
                threshold=self._similarity_threshold,
            )
            context = self._context_assembler.assemble(
                query=content,
                retrieved_chunks=retrieved_chunks,
            )
            prompt = self._prompt_builder.build(
                context=context,
                conversation=history,
                user_query=content,
            )
        except Exception:
            self._db.rollback()
            raise

        self._db.rollback()
        logger.info(
            "Chat turn prepared",
            extra={
                "conversation_id": str(conversation_id),
                "knowledge_base_id": str(knowledge_base_id),
                "history_count": len(history),
                "context_count": len(context.chunks),
                "prompt_message_count": len(prompt),
                "latency_ms": round((perf_counter() - preparation_started_at) * 1000),
            },
        )
        return PreparedChatTurn(
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            user_content=content,
            context=context,
            prompt=prompt,
        )

    def _persist_messages(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_content: str,
        assistant_content: str,
    ) -> Message:
        try:
            if (
                self._conversation_repository.get_by_user_and_id(
                    user_id=user_id,
                    conversation_id=conversation_id,
                )
                is None
            ):
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
        return assistant_message

    def send_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        content: str,
    ) -> ChatTurnResult:
        """Generate and atomically persist a non-streaming chat turn."""

        prepared = self.prepare_turn(
            user_id=user_id,
            conversation_id=conversation_id,
            content=content,
        )
        llm_response = self._llm_provider.generate(prepared.prompt)
        assistant_content = self._normalize_assistant_content(
            content=llm_response.content
        )
        assistant_message = self._persist_messages(
            user_id=user_id,
            conversation_id=prepared.conversation_id,
            user_content=prepared.user_content,
            assistant_content=assistant_content,
        )
        return ChatTurnResult(
            assistant_message=assistant_message,
            context=prepared.context,
        )

    def stream_prepared_turn(
        self,
        *,
        prepared: PreparedChatTurn,
    ) -> ChatEventStream:
        """Stream one prepared turn and persist it only after valid completion."""

        provider_stream = self._llm_provider.stream(prepared.prompt)

        def events() -> Iterator[ChatStreamEvent]:
            content_parts: list[str] = []
            buffered_characters = 0
            completion: LLMStreamCompletion | None = None
            delta_count = 0
            generation_started_at = perf_counter()
            first_token_at: float | None = None
            try:
                yield ChatStreamEvent(
                    type=ChatStreamEventType.METADATA,
                    payload=ChatStreamMetadata(
                        conversation_id=prepared.conversation_id,
                        source_count=len(prepared.context.chunks),
                    ),
                )
                yield ChatStreamEvent(
                    type=ChatStreamEventType.CITATIONS,
                    payload=ChatStreamCitations(context=prepared.context),
                )

                for event in provider_stream:
                    if isinstance(event, LLMStreamDelta):
                        if completion is not None:
                            raise AIServiceException(
                                "LLM emitted content after completion"
                            )
                        if not event.content:
                            continue
                        content_parts.append(event.content)
                        buffered_characters += len(event.content)
                        if buffered_characters > self._stream_max_buffered_characters:
                            raise AIServiceException(
                                "LLM response exceeded output limit"
                            )
                        delta_count += 1
                        if first_token_at is None:
                            first_token_at = perf_counter()
                            ttft_ms = round(
                                (first_token_at - generation_started_at) * 1000
                            )
                            logger.info(
                                "First chat token received",
                                extra={
                                    "conversation_id": str(prepared.conversation_id),
                                    "knowledge_base_id": str(
                                        prepared.knowledge_base_id
                                    ),
                                    "time_to_first_token_ms": ttft_ms,
                                },
                            )
                        yield ChatStreamEvent(
                            type=ChatStreamEventType.TOKEN,
                            payload=ChatStreamToken(delta=event.content),
                        )
                        continue

                    if completion is not None:
                        raise AIServiceException(
                            "LLM emitted multiple completion events"
                        )
                    completion = event

                if completion is None:
                    raise AIServiceException("LLM stream ended without completion")
                if (
                    completion.finish_reason
                    not in self._SUCCESSFUL_STREAM_FINISH_REASONS
                ):
                    raise AIServiceException("LLM stream ended unsuccessfully")

                assistant_content = self._normalize_assistant_content(
                    content="".join(content_parts),
                )
                persistence_started_at = perf_counter()
                assistant_message = self._persist_messages(
                    conversation_id=prepared.conversation_id,
                    user_content=prepared.user_content,
                    assistant_content=assistant_content,
                    user_id=prepared.user_id,
                )
                logger.info(
                    "Chat stream completed",
                    extra={
                        "conversation_id": str(prepared.conversation_id),
                        "knowledge_base_id": str(prepared.knowledge_base_id),
                        "model": completion.model,
                        "finish_reason": completion.finish_reason,
                        "delta_count": delta_count,
                        "streamed_characters": buffered_characters,
                        "citation_count": len(prepared.context.chunks),
                        "time_to_first_token_ms": (
                            None
                            if first_token_at is None
                            else round((first_token_at - generation_started_at) * 1000)
                        ),
                        "generation_latency_ms": round(
                            (persistence_started_at - generation_started_at) * 1000
                        ),
                        "persistence_latency_ms": round(
                            (perf_counter() - persistence_started_at) * 1000
                        ),
                        "prompt_tokens": (
                            None
                            if completion.usage is None
                            else completion.usage.prompt_tokens
                        ),
                        "completion_tokens": (
                            None
                            if completion.usage is None
                            else completion.usage.completion_tokens
                        ),
                        "total_tokens": (
                            None
                            if completion.usage is None
                            else completion.usage.total_tokens
                        ),
                    },
                )
                yield ChatStreamEvent(
                    type=ChatStreamEventType.COMPLETE,
                    payload=ChatStreamComplete(
                        assistant_message=assistant_message,
                        context=prepared.context,
                        model=completion.model,
                        finish_reason=completion.finish_reason,
                        usage=completion.usage,
                    ),
                )
            except Exception:
                logger.warning(
                    "Chat stream abandoned",
                    extra={
                        "conversation_id": str(prepared.conversation_id),
                        "knowledge_base_id": str(prepared.knowledge_base_id),
                        "delta_count": delta_count,
                        "streamed_characters": buffered_characters,
                    },
                )
                raise

        return ChatEventStream(
            events=events(),
            provider_stream=provider_stream,
        )
