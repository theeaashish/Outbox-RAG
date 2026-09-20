from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from threading import Lock
from time import perf_counter
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.context.citations import CitationValidator
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
from app.core.ai.prompting.conversational import ConversationalPromptBuilder
from app.core.ai.routing.models import ConversationMode, QueryRoutingDecision
from app.core.ai.routing.router import QueryRouter
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.db.models import Conversation, Message
from app.db.models.enums import MessageRole
from app.db.session import managed_session
from app.modules.chat.streaming import (
    ChatStreamCancelReason,
    ChatStreamLifecycle,
    ChatStreamSnapshot,
    ChatStreamState,
)
from app.modules.retrieval.service import RetrievalService
from app.repositories.conversation import ConversationRepository
from app.repositories.message import MessageRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatTurnResult:
    """Application result for a completed, persisted chat turn."""

    assistant_message: Message
    context: AssembledContext | None
    routing: QueryRoutingDecision


@dataclass(frozen=True, slots=True)
class PreparedChatTurn:
    """Validated, prompt-ready chat work with no active DB transaction."""

    conversation_id: UUID
    knowledge_base_id: UUID
    user_id: UUID
    user_message_id: UUID
    user_content: str
    context: AssembledContext | None
    prompt: list[ChatMessage]
    routing: QueryRoutingDecision
    assistant_message_id: UUID


from enum import StrEnum


@dataclass(frozen=True, slots=True)
class ChatStreamMetadata:
    conversation_id: UUID
    source_count: int


@dataclass(frozen=True, slots=True)
class ChatStreamCitations:
    context: AssembledContext | None


@dataclass(frozen=True, slots=True)
class ChatStreamToken:
    delta: str


@dataclass(frozen=True, slots=True)
class ChatStreamComplete:
    assistant_message: Message
    context: AssembledContext | None
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
        self,
        *,
        events: Iterator[ChatStreamEvent],
        lifecycle: ChatStreamLifecycle,
    ) -> None:
        self._events = events
        self._lifecycle = lifecycle
        self._lock = Lock()
        self._closed = False

    def __iter__(self) -> Iterator[ChatStreamEvent]:
        return self

    def __next__(self) -> ChatStreamEvent:
        with self._lock:
            if self._closed:
                raise StopIteration
            try:
                return next(self._events)
            except StopIteration:
                self._closed = True
                raise

    @property
    def lifecycle(self) -> ChatStreamLifecycle:
        return self._lifecycle

    @property
    def snapshot(self) -> ChatStreamSnapshot:
        return self._lifecycle.snapshot

    def request_cancel(self, reason: ChatStreamCancelReason) -> bool:
        """Attempt to cooperatively cancel an active stream."""
        return self._lifecycle.request_cancel(reason)

    def close(self) -> None:
        """Signal cooperative cancellation and serialize generator closure."""
        self._lifecycle.request_cancel(ChatStreamCancelReason.CLIENT_DISCONNECT)
        with self._lock:
            if self._closed:
                return
            self._closed = True
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
        conversational_prompt_builder: ConversationalPromptBuilder,
        query_router: QueryRouter,
        llm_provider: LLMProvider,
        history_message_limit: int,
        retrieval_limit: int,
        similarity_threshold: float | None,
        stream_max_buffered_characters: int,
        citation_validator: CitationValidator | None = None,
    ) -> None:
        self._db = db
        self._conversation_repository = conversation_repository
        self._message_repository = message_repository
        self._retrieval_service = retrieval_service
        self._context_assembler = context_assembler
        self._prompt_builder = prompt_builder
        self._conversational_prompt_builder = conversational_prompt_builder
        self._query_router = query_router
        self._llm_provider = llm_provider
        self._history_message_limit = history_message_limit
        self._retrieval_limit = retrieval_limit
        self._similarity_threshold = similarity_threshold
        self._stream_max_buffered_characters = stream_max_buffered_characters
        self._citation_validator = citation_validator or CitationValidator()

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

    def _load_history(
        self,
        *,
        conversation_id: UUID,
        exclude_message_id: UUID,
    ) -> list[ChatMessage]:
        messages: Sequence[Message] = (
            self._message_repository.list_recent_by_conversation(
                conversation_id=conversation_id,
                limit=self._history_message_limit,
                exclude_message_id=exclude_message_id,
            )
        )
        return [
            to_chat_message(message)
            for message in messages
            if message.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]

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
            user_message_id = self._persist_user_message(
                user_id=user_id,
                conversation_id=conversation_id,
                user_content=content,
            )
            conversation = self._get_conversation(
                user_id=user_id,
                conversation_id=conversation_id,
            )
            knowledge_base_id = conversation.knowledge_base_id
            history = self._load_history(
                conversation_id=conversation_id,
                exclude_message_id=user_message_id,
            )

            routing = self._query_router.route(
                query=content,
                conversation=history,
            )

            if routing.mode == ConversationMode.CASUAL:
                context = None
                prompt = self._conversational_prompt_builder.build(
                    conversation=history,
                    user_query=content,
                )
            else:
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
                "routing_mode": routing.mode.value,
                "routing_reason": routing.reason.value,
                "history_count": len(history),
                "context_count": 0 if context is None else len(context.chunks),
                "prompt_message_count": len(prompt),
                "latency_ms": round((perf_counter() - preparation_started_at) * 1000),
            },
        )
        return PreparedChatTurn(
            conversation_id=conversation_id,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            user_message_id=user_message_id,
            user_content=content,
            context=context,
            prompt=prompt,
            routing=routing,
            assistant_message_id=uuid4(),
        )

    def _persist_user_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        user_content: str,
    ) -> UUID:
        try:
            with managed_session() as db:
                conversation_repository = ConversationRepository(db=db)
                message_repository = MessageRepository(db=db)

                if (
                    conversation_repository.get_by_user_and_id(
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
                message_repository.create(user_message)
                message_repository.flush()
                user_message_id = user_message.id
                db.commit()
                return user_message_id
        except ResourceNotFoundException:
            raise
        except SQLAlchemyError as exc:
            logger.exception(
                "User message persistence failed",
                extra={"conversation_id": str(conversation_id)},
            )
            raise DatabaseException("Failed to persist user message") from exc

    def _persist_assistant_message(
        self,
        *,
        user_id: UUID,
        conversation_id: UUID,
        assistant_content: str,
        message_id: UUID | None = None,
    ) -> Message:
        try:
            with managed_session() as db:
                conversation_repository = ConversationRepository(db=db)
                message_repository = MessageRepository(db=db)

                if (
                    conversation_repository.get_by_user_and_id(
                        user_id=user_id,
                        conversation_id=conversation_id,
                    )
                    is None
                ):
                    raise ResourceNotFoundException("Conversation not found")

                assistant_message = Message(
                    id=message_id or uuid4(),
                    role=MessageRole.ASSISTANT,
                    content=assistant_content,
                    conversation_id=conversation_id,
                )
                message_repository.create(assistant_message)
                db.commit()
                message_repository.refresh(assistant_message)
                return assistant_message
        except ResourceNotFoundException:
            raise
        except SQLAlchemyError as exc:
            logger.exception(
                "Assistant message persistence failed",
                extra={"conversation_id": str(conversation_id)},
            )
            raise DatabaseException("Failed to persist assistant message") from exc

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

        if prepared.context is not None:
            valid_citations = {chunk.citation for chunk in prepared.context.chunks}
            validation = self._citation_validator.validate(
                content=assistant_content,
                valid_citations=valid_citations,
            )
            if not validation.is_valid:
                logger.warning(
                    "LLM generated invalid citation IDs",
                    extra={
                        "conversation_id": str(prepared.conversation_id),
                        "invalid_citations": validation.invalid_citations,
                        "valid_citations": list(valid_citations),
                    },
                )

        assistant_message = self._persist_assistant_message(
            user_id=prepared.user_id,
            conversation_id=prepared.conversation_id,
            assistant_content=assistant_content,
            message_id=prepared.assistant_message_id,
        )
        return ChatTurnResult(
            assistant_message=assistant_message,
            context=prepared.context,
            routing=prepared.routing,
        )

    def stream_prepared_turn(
        self,
        *,
        prepared: PreparedChatTurn,
    ) -> ChatEventStream:
        """Stream one prepared turn and persist it only after valid completion."""

        lifecycle = ChatStreamLifecycle()

        def events() -> Iterator[ChatStreamEvent]:
            content_parts: list[str] = []
            buffered_characters = 0
            completion: LLMStreamCompletion | None = None
            delta_count = 0
            generation_started_at = perf_counter()
            first_token_at: float | None = None
            provider_stream: LLMStream | None = None
            try:
                if not lifecycle.accepts_provider_output():
                    return

                yield ChatStreamEvent(
                    type=ChatStreamEventType.METADATA,
                    payload=ChatStreamMetadata(
                        conversation_id=prepared.conversation_id,
                        source_count=0
                        if prepared.context is None
                        else len(prepared.context.chunks),
                    ),
                )

                if not lifecycle.accepts_provider_output():
                    return

                yield ChatStreamEvent(
                    type=ChatStreamEventType.CITATIONS,
                    payload=ChatStreamCitations(context=prepared.context),
                )

                if not lifecycle.accepts_provider_output():
                    return

                provider_stream = self._llm_provider.stream(prepared.prompt)

                for event in provider_stream:
                    if isinstance(event, LLMStreamDelta):
                        if not event.content:
                            continue

                        if not lifecycle.accepts_provider_output():
                            return

                        if completion is not None:
                            raise AIServiceException(
                                "LLM emitted content after completion"
                            )

                        new_size = buffered_characters + len(event.content)
                        if new_size > self._stream_max_buffered_characters:
                            lifecycle.request_cancel(
                                ChatStreamCancelReason.OUTPUT_LIMIT
                            )
                            raise AIServiceException(
                                "LLM response exceeded output limit"
                            )

                        content_parts.append(event.content)
                        buffered_characters = new_size
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

                    if isinstance(event, LLMStreamCompletion):
                        if completion is not None:
                            raise AIServiceException(
                                "LLM emitted multiple completion events"
                            )
                        if not lifecycle.accept_provider_completion():
                            return
                        completion = event
                        continue

                if completion is None:
                    if not lifecycle.accepts_provider_output():
                        return
                    raise AIServiceException("LLM stream ended without completion")

                if (
                    completion.finish_reason
                    not in self._SUCCESSFUL_STREAM_FINISH_REASONS
                ):
                    raise AIServiceException("LLM stream ended unsuccessfully")

                assistant_content = self._normalize_assistant_content(
                    content="".join(content_parts),
                )

                lifecycle.begin_validation()
                if prepared.context is not None:
                    valid_citations = {
                        chunk.citation for chunk in prepared.context.chunks
                    }
                    validation = self._citation_validator.validate(
                        content=assistant_content,
                        valid_citations=valid_citations,
                    )
                    if not validation.is_valid:
                        logger.warning(
                            "LLM generated invalid citation IDs",
                            extra={
                                "conversation_id": str(prepared.conversation_id),
                                "invalid_citations": validation.invalid_citations,
                                "valid_citations": list(valid_citations),
                            },
                        )

                lifecycle.begin_finalization()
                persistence_started_at = perf_counter()
                try:
                    assistant_message = self._persist_assistant_message(
                        user_id=prepared.user_id,
                        conversation_id=prepared.conversation_id,
                        assistant_content=assistant_content,
                        message_id=prepared.assistant_message_id,
                    )
                except Exception:
                    lifecycle.mark_failed()
                    raise

                lifecycle.mark_assistant_durable()
                lifecycle.mark_done()

                logger.info(
                    "Chat stream completed",
                    extra={
                        "conversation_id": str(prepared.conversation_id),
                        "knowledge_base_id": str(prepared.knowledge_base_id),
                        "model": completion.model,
                        "finish_reason": completion.finish_reason,
                        "delta_count": delta_count,
                        "streamed_characters": buffered_characters,
                        "citation_count": 0
                        if prepared.context is None
                        else len(prepared.context.chunks),
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
                snapshot = lifecycle.snapshot
                if (
                    snapshot.state == ChatStreamState.CANCELLED
                    and snapshot.cancel_reason != ChatStreamCancelReason.OUTPUT_LIMIT
                ):
                    logger.info(
                        "Chat stream cancelled",
                        extra={
                            "conversation_id": str(prepared.conversation_id),
                            "knowledge_base_id": str(prepared.knowledge_base_id),
                            "cancel_reason": snapshot.cancel_reason.value
                            if snapshot.cancel_reason
                            else None,
                            "delta_count": delta_count,
                            "streamed_characters": buffered_characters,
                        },
                    )
                    return

                lifecycle.mark_failed()
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
            finally:
                if provider_stream is not None:
                    provider_stream.close()

        return ChatEventStream(
            events=events(),
            lifecycle=lifecycle,
        )
