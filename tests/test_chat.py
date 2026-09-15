from datetime import UTC, datetime, timedelta
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import Table, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.llm.models import (
    LLMResponse,
    LLMStreamCompletion,
    LLMStreamDelta,
    LLMUsage,
)
from app.core.ai.routing import ConversationMode, QueryRouter, RoutingReason
from app.core.config import settings
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.db.base import Base
from app.db.models import Conversation, KnowledgeBase, Message, User
from app.db.models.enums import MessageRole
from app.dependencies.auth import get_current_user
from app.dependencies.chat import get_chat_controller
from app.main import app
from app.modules.chat import mapper
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.chat.service import (
    ChatService,
    ChatStreamCitations,
    ChatStreamComplete,
    ChatStreamEvent,
    ChatStreamEventType,
    ChatStreamMetadata,
    ChatStreamToken,
)
from app.modules.conversations.schemas import MessageResponse
from app.repositories.message import MessageRepository


class _FakeLLMStream:
    def __init__(self, events: list[object]) -> None:
        self._events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            return next(self._events)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        self.closed = True


def test_chat_request_normalizes_and_validates_content():
    assert ChatRequest(content="  What changed?  ").content == "What changed?"

    with pytest.raises(ValidationError):
        ChatRequest(content="   ")

    with pytest.raises(ValidationError):
        ChatRequest(content="x" * (settings.chat_max_message_characters + 1))


def test_list_recent_by_conversation_returns_chronological_window():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(
        engine,
        tables=[
            cast(Table, KnowledgeBase.__table__),
            cast(Table, Conversation.__table__),
            cast(Table, Message.__table__),
        ],
    )
    session = sessionmaker(bind=engine)()
    user_id = uuid4()
    project_id = uuid4()
    knowledge_base = KnowledgeBase(
        name="Handbook", user_id=user_id, project_id=project_id
    )
    conversation = Conversation(
        knowledge_base=knowledge_base, user_id=user_id, project_id=project_id
    )
    session.add(conversation)
    session.commit()

    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    contents = ["first", "second", "third", "fourth"]
    for index, content in enumerate(contents):
        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=content,
                created_at=started_at + timedelta(seconds=index),
            )
        )
    session.commit()

    messages = MessageRepository(db=session).list_recent_by_conversation(
        conversation_id=conversation.id,
        limit=2,
    )

    assert [message.content for message in messages] == ["third", "fourth"]
    session.close()
    engine.dispose()


def _build_chat_service():
    db = MagicMock()
    user_id = uuid4()
    project_id = uuid4()
    conversation = Conversation(
        id=uuid4(),
        user_id=user_id,
        project_id=project_id,
        knowledge_base_id=uuid4(),
    )
    conversation_repository = MagicMock()
    conversation_repository.get_by_user_and_id.return_value = conversation
    conversation_repository.exists.return_value = True
    message_repository = MagicMock()
    message_repository.list_recent_by_conversation.return_value = [
        Message(
            id=uuid4(),
            conversation_id=conversation.id,
            role=MessageRole.USER,
            content="Earlier question",
        ),
        Message(
            id=uuid4(),
            conversation_id=conversation.id,
            role=MessageRole.ASSISTANT,
            content="Earlier answer",
        ),
    ]
    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = []
    context = AssembledContext(query="Current question", block="", chunks=[])
    context_assembler = MagicMock()
    context_assembler.assemble.return_value = context
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = []
    conversational_prompt_builder = MagicMock()
    conversational_prompt_builder.build.return_value = []
    query_router = QueryRouter()
    llm_provider = MagicMock()
    llm_provider.generate.return_value = LLMResponse(
        content="Grounded answer",
        model="test-model",
        finish_reason="stop",
        usage=None,
    )
    service = ChatService(
        db=db,
        conversation_repository=conversation_repository,
        message_repository=message_repository,
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        conversational_prompt_builder=conversational_prompt_builder,
        query_router=query_router,
        llm_provider=llm_provider,
        history_message_limit=20,
        retrieval_limit=5,
        similarity_threshold=0.7,
        stream_max_buffered_characters=65_536,
    )
    return {
        "service": service,
        "user_id": user_id,
        "db": db,
        "conversation": conversation,
        "conversation_repository": conversation_repository,
        "message_repository": message_repository,
        "retrieval_service": retrieval_service,
        "context": context,
        "context_assembler": context_assembler,
        "prompt_builder": prompt_builder,
        "conversational_prompt_builder": conversational_prompt_builder,
        "query_router": query_router,
        "llm_provider": llm_provider,
    }


def test_chat_service_generates_context_and_persists_a_message_pair():
    dependencies = _build_chat_service()

    result = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    dependencies[
        "message_repository"
    ].list_recent_by_conversation.assert_called_once_with(
        conversation_id=dependencies["conversation"].id,
        limit=20,
    )
    dependencies["retrieval_service"].retrieve.assert_called_once_with(
        user_id=dependencies["user_id"],
        knowledge_base_id=dependencies["conversation"].knowledge_base_id,
        query="Current question",
        limit=5,
        threshold=0.7,
    )
    dependencies["prompt_builder"].build.assert_called_once()
    dependencies["llm_provider"].generate.assert_called_once()
    assert dependencies["message_repository"].create.call_count == 2
    persisted_messages = [
        call.args[0]
        for call in dependencies["message_repository"].create.call_args_list
    ]
    assert [message.role for message in persisted_messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]
    assert [message.content for message in persisted_messages] == [
        "Current question",
        "Grounded answer",
    ]
    dependencies["db"].commit.assert_called_once()
    dependencies["message_repository"].refresh.assert_called_once_with(
        result.assistant_message
    )
    assert result.context == dependencies["context"]


def test_chat_service_does_not_persist_when_generation_fails():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].generate.side_effect = AIServiceException(
        "unavailable"
    )

    with pytest.raises(AIServiceException):
        dependencies["service"].send_message(
            user_id=dependencies["user_id"],
            conversation_id=dependencies["conversation"].id,
            content="Current question",
        )

    dependencies["message_repository"].create.assert_not_called()
    dependencies["db"].commit.assert_not_called()


def test_chat_service_rolls_back_message_pair_when_commit_fails():
    dependencies = _build_chat_service()
    dependencies["db"].commit.side_effect = SQLAlchemyError("write failed")

    with pytest.raises(DatabaseException):
        dependencies["service"].send_message(
            user_id=dependencies["user_id"],
            conversation_id=dependencies["conversation"].id,
            content="Current question",
        )

    assert dependencies["message_repository"].create.call_count == 2
    assert dependencies["db"].rollback.call_count == 2


def test_chat_service_rejects_missing_conversation_before_external_calls():
    dependencies = _build_chat_service()
    dependencies["conversation_repository"].get_by_user_and_id.return_value = None

    with pytest.raises(ResourceNotFoundException):
        dependencies["service"].send_message(
            user_id=dependencies["user_id"],
            conversation_id=uuid4(),
            content="Current question",
        )

    dependencies["retrieval_service"].retrieve.assert_not_called()
    dependencies["llm_provider"].generate.assert_not_called()
    dependencies["message_repository"].create.assert_not_called()


def test_chat_service_stream_orders_events_and_persists_atomically():
    dependencies = _build_chat_service()
    fake_stream = _FakeLLMStream(
        [
            LLMStreamDelta(content="Hel"),
            LLMStreamDelta(content="lo"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=LLMUsage(
                    prompt_tokens=3,
                    completion_tokens=2,
                    total_tokens=5,
                ),
            ),
        ]
    )
    dependencies["llm_provider"].stream.return_value = fake_stream
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    events = list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    assert events[0].type == ChatStreamEventType.METADATA
    assert (
        isinstance(events[0].payload, ChatStreamMetadata)
        and events[0].payload.conversation_id == dependencies["conversation"].id
    )
    assert events[1].type == ChatStreamEventType.CITATIONS
    assert events[2].type == ChatStreamEventType.TOKEN
    assert (
        isinstance(events[2].payload, ChatStreamToken)
        and events[2].payload.delta == "Hel"
    )
    assert events[3].type == ChatStreamEventType.TOKEN
    assert (
        isinstance(events[3].payload, ChatStreamToken)
        and events[3].payload.delta == "lo"
    )
    assert events[4].type == ChatStreamEventType.COMPLETE
    assert (
        isinstance(events[4].payload, ChatStreamComplete)
        and events[4].payload.finish_reason == "stop"
    )
    assert (
        isinstance(events[4].payload, ChatStreamComplete)
        and events[4].payload.assistant_message.content == "Hello"
    )
    assert dependencies["message_repository"].create.call_count == 2
    dependencies["db"].commit.assert_called_once()
    assert fake_stream.closed is True


def test_chat_service_stream_persists_length_finish_reason():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamDelta(content="Truncated"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="length",
                usage=None,
            ),
        ]
    )
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    events = list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    complete = events[-1]
    assert complete.type == ChatStreamEventType.COMPLETE
    assert (
        isinstance(complete.payload, ChatStreamComplete)
        and complete.payload.finish_reason == "length"
    )
    dependencies["db"].commit.assert_called_once()


def test_chat_service_stream_does_not_persist_on_empty_completion():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        ]
    )
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    with pytest.raises(AIServiceException):
        list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    dependencies["message_repository"].create.assert_not_called()
    dependencies["db"].commit.assert_not_called()


def test_chat_service_stream_rejects_invalid_finish_reason():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamDelta(content="Blocked"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="safety",
                usage=None,
            ),
        ]
    )
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    with pytest.raises(AIServiceException):
        list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    dependencies["db"].commit.assert_not_called()


def test_chat_service_stream_enforces_output_character_limit():
    dependencies = _build_chat_service()
    dependencies["service"]._stream_max_buffered_characters = 5
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamDelta(content="123456"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        ]
    )
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    with pytest.raises(AIServiceException, match="output limit"):
        list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    dependencies["db"].commit.assert_not_called()


def test_chat_service_stream_close_releases_provider_without_commit():
    dependencies = _build_chat_service()
    fake_stream = _FakeLLMStream(
        [
            LLMStreamDelta(content="partial"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        ]
    )
    dependencies["llm_provider"].stream.return_value = fake_stream
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

    stream = dependencies["service"].stream_prepared_turn(prepared=prepared)

    assert next(stream).type == ChatStreamEventType.METADATA
    assert next(stream).type == ChatStreamEventType.CITATIONS
    assert next(stream).type == ChatStreamEventType.TOKEN
    stream.close()

    assert fake_stream.closed is True
    dependencies["db"].commit.assert_not_called()
    dependencies["message_repository"].create.assert_not_called()


def test_chat_service_stream_does_not_persist_when_conversation_deleted():
    dependencies = _build_chat_service()
    dependencies["conversation_repository"].exists.return_value = False
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamDelta(content="Hello"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        ]
    )
    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )
    dependencies["conversation_repository"].get_by_user_and_id.return_value = None

    with pytest.raises(ResourceNotFoundException):
        list(dependencies["service"].stream_prepared_turn(prepared=prepared))

    dependencies["db"].commit.assert_not_called()


def test_chat_response_maps_citation_metadata():
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    assistant_message = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        role=MessageRole.ASSISTANT,
        content="Answer [1]",
        created_at=created_at,
        updated_at=created_at,
    )
    context = AssembledContext(
        query="Question",
        block="Context",
        chunks=[
            ContextChunk(
                citation=1,
                document_id=uuid4(),
                document_name="Handbook",
                chunk_index=2,
                similarity=0.91,
                content="Source text",
            )
        ],
    )

    response = mapper.to_chat_response(
        assistant_message=assistant_message,
        context=context,
    )

    assert response.assistant_message.content == "Answer [1]"
    assert response.sources[0].citation == 1
    assert response.sources[0].document_name == "Handbook"
    assert response.sources[0].score == 0.91


def test_to_sse_event_sequence_contract():
    conversation_id = uuid4()
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    context = AssembledContext(
        query="Question",
        block="Context",
        chunks=[
            ContextChunk(
                citation=1,
                document_id=uuid4(),
                document_name="Handbook",
                chunk_index=0,
                similarity=0.9,
                content="Source",
            )
        ],
    )
    assistant_message = Message(
        id=uuid4(),
        conversation_id=conversation_id,
        role=MessageRole.ASSISTANT,
        content="Hello",
        created_at=created_at,
        updated_at=created_at,
    )

    metadata = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.METADATA,
            payload=ChatStreamMetadata(conversation_id=conversation_id, source_count=1),
        )
    )
    citations = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.CITATIONS,
            payload=ChatStreamCitations(context=context),
        )
    )
    token = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.TOKEN,
            payload=ChatStreamToken(delta="Hi"),
        )
    )
    complete = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.COMPLETE,
            payload=ChatStreamComplete(
                assistant_message=assistant_message,
                context=context,
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        )
    )
    error = mapper.to_sse_error(code="generation_failed", message="failed")

    assert metadata["event"] == "metadata"
    assert citations["event"] == "citations"
    assert token["event"] == "token"
    assert complete["event"] == "complete"
    assert error["event"] == "error"


class _StubChatController:
    async def send_message(
        self,
        *,
        user_id: object = None,
        conversation_id: object,
        content: str,
    ) -> ChatResponse:
        if content == "missing":
            raise ResourceNotFoundException("Conversation not found")
        if content == "provider failure":
            raise AIServiceException("Failed to generate LLM response")

        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        return ChatResponse(
            assistant_message=MessageResponse(
                id=uuid4(),
                role=MessageRole.ASSISTANT,
                content=f"Answer to {content}",
                created_at=timestamp,
                updated_at=timestamp,
            ),
            sources=[],
        )


@pytest.fixture
def chat_client():
    mock_user = User(
        id=uuid4(),
        email="test@example.com",
        name="Test User",
    )
    app.dependency_overrides[get_chat_controller] = _StubChatController
    app.dependency_overrides[get_current_user] = lambda: mock_user
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def test_chat_service_logs_invalid_citations_without_mutating_content():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].generate.return_value = LLMResponse(
        content="Ground fact [1] and hallucinated [99]",
        model="test-model",
        finish_reason="stop",
        usage=None,
    )
    context = AssembledContext(
        query="Question",
        block='[Source 1] Document: "Doc"\n---\nGround fact',
        chunks=[
            ContextChunk(
                citation=1,
                document_id=uuid4(),
                document_name="Doc",
                chunk_index=0,
                similarity=0.9,
                content="Ground fact",
            )
        ],
    )
    dependencies["service"].prepare_turn = MagicMock(
        return_value=dependencies["service"].prepare_turn(
            user_id=dependencies["user_id"],
            conversation_id=dependencies["conversation"].id,
            content="Question",
        )
    )
    # Patch context in prepared turn
    prepared = dependencies["service"].prepare_turn.return_value
    object.__setattr__(prepared, "context", context)

    result = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Question",
    )

    # Non-destructive: content is preserved with [99] intact
    assert result.assistant_message.content == "Ground fact [1] and hallucinated [99]"


def test_chat_route_returns_response_and_validation_errors(chat_client: TestClient):
    conversation_id = uuid4()
    success = chat_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "  hello  "},
    )
    blank = chat_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "   "},
    )

    assert success.status_code == 200
    assert success.json()["assistant_message"]["content"] == "Answer to hello"
    assert blank.status_code == 422


def test_chat_route_uses_existing_exception_responses(chat_client: TestClient):
    conversation_id = uuid4()
    missing = chat_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "missing"},
    )
    provider_failure = chat_client.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        json={"content": "provider failure"},
    )

    assert missing.status_code == 404
    assert missing.json() == {
        "success": False,
        "error": {"message": "Conversation not found"},
    }
    assert provider_failure.status_code == 500
    assert provider_failure.json() == {
        "success": False,
        "error": {"message": "Failed to generate LLM response"},
    }


def test_chat_service_casual_turn_skips_retrieval_and_context_assembly():
    dependencies = _build_chat_service()

    result = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Hello there!",
    )

    dependencies["retrieval_service"].retrieve.assert_not_called()
    dependencies["context_assembler"].assemble.assert_not_called()
    dependencies["prompt_builder"].build.assert_not_called()
    dependencies["conversational_prompt_builder"].build.assert_called_once()
    assert result.context is None
    assert result.routing.mode == ConversationMode.CASUAL
    assert result.routing.reason == RoutingReason.EXPLICIT_CASUAL

    result.assistant_message.id = uuid4()
    result.assistant_message.created_at = datetime.now(UTC)
    result.assistant_message.updated_at = datetime.now(UTC)
    response = mapper.to_chat_response(
        assistant_message=result.assistant_message,
        context=result.context,
    )
    assert response.sources == []
    assert response.assistant_message.content == "Grounded answer"


def test_chat_service_knowledge_turn_executes_retrieval_and_context_assembly():
    dependencies = _build_chat_service()

    result = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="What is the refund policy?",
    )

    dependencies["retrieval_service"].retrieve.assert_called_once_with(
        user_id=dependencies["user_id"],
        knowledge_base_id=dependencies["conversation"].knowledge_base_id,
        query="What is the refund policy?",
        limit=5,
        threshold=0.7,
    )
    dependencies["context_assembler"].assemble.assert_called_once()
    dependencies["prompt_builder"].build.assert_called_once()
    dependencies["conversational_prompt_builder"].build.assert_not_called()
    assert result.context == dependencies["context"]
    assert result.routing.mode == ConversationMode.KNOWLEDGE
    assert result.routing.reason == RoutingReason.EXPLICIT_KNOWLEDGE


def test_chat_service_stream_casual_turn_emits_empty_sources_and_zero_count():
    dependencies = _build_chat_service()
    dependencies["llm_provider"].stream.return_value = _FakeLLMStream(
        [
            LLMStreamDelta(content="Hi! "),
            LLMStreamDelta(content="How can I help?"),
            LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        ]
    )

    prepared = dependencies["service"].prepare_turn(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Thank you!",
    )
    assert prepared.context is None
    assert prepared.routing.mode == ConversationMode.CASUAL
    assert prepared.routing.reason == RoutingReason.EXPLICIT_CASUAL
    dependencies["retrieval_service"].retrieve.assert_not_called()
    dependencies["context_assembler"].assemble.assert_not_called()

    events = list(dependencies["service"].stream_prepared_turn(prepared=prepared))
    assert len(events) == 5

    assert events[0].type == ChatStreamEventType.METADATA
    assert isinstance(events[0].payload, ChatStreamMetadata)
    assert events[0].payload.source_count == 0

    assert events[1].type == ChatStreamEventType.CITATIONS
    assert isinstance(events[1].payload, ChatStreamCitations)
    assert events[1].payload.context is None

    assert events[2].type == ChatStreamEventType.TOKEN
    assert isinstance(events[2].payload, ChatStreamToken)
    assert events[2].payload.delta == "Hi! "

    assert events[3].type == ChatStreamEventType.TOKEN
    assert isinstance(events[3].payload, ChatStreamToken)
    assert events[3].payload.delta == "How can I help?"

    assert events[4].type == ChatStreamEventType.COMPLETE
    assert isinstance(events[4].payload, ChatStreamComplete)
    assert events[4].payload.context is None
    assert events[4].payload.finish_reason == "stop"


def test_chat_service_follow_up_routing_continuity():
    dependencies = _build_chat_service()

    # Case A: Previous user message in history was a Knowledge query
    dependencies["message_repository"].list_recent_by_conversation.return_value = [
        Message(
            id=uuid4(),
            conversation_id=dependencies["conversation"].id,
            role=MessageRole.USER,
            content="What are the system requirements?",
        ),
        Message(
            id=uuid4(),
            conversation_id=dependencies["conversation"].id,
            role=MessageRole.ASSISTANT,
            content="You need 8GB RAM.",
        ),
    ]

    result_knowledge_follow_up = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Can you explain that in more detail?",
    )
    assert result_knowledge_follow_up.routing.mode == ConversationMode.KNOWLEDGE
    assert (
        result_knowledge_follow_up.routing.reason == RoutingReason.KNOWLEDGE_FOLLOW_UP
    )
    dependencies["retrieval_service"].retrieve.assert_called_once()
    dependencies["context_assembler"].assemble.assert_called_once()

    # Reset mocks
    dependencies["retrieval_service"].retrieve.reset_mock()
    dependencies["context_assembler"].assemble.reset_mock()
    dependencies["conversational_prompt_builder"].build.reset_mock()

    # Case B: Previous user message in history was Casual
    dependencies["message_repository"].list_recent_by_conversation.return_value = [
        Message(
            id=uuid4(),
            conversation_id=dependencies["conversation"].id,
            role=MessageRole.USER,
            content="Hello there!",
        ),
        Message(
            id=uuid4(),
            conversation_id=dependencies["conversation"].id,
            role=MessageRole.ASSISTANT,
            content="Hello! How can I assist you today?",
        ),
    ]

    result_casual_follow_up = dependencies["service"].send_message(
        user_id=dependencies["user_id"],
        conversation_id=dependencies["conversation"].id,
        content="Can you explain that in more detail?",
    )
    assert result_casual_follow_up.routing.mode == ConversationMode.CASUAL
    assert result_casual_follow_up.routing.reason == RoutingReason.CASUAL_FOLLOW_UP
    dependencies["retrieval_service"].retrieve.assert_not_called()
    dependencies["context_assembler"].assemble.assert_not_called()
    dependencies["conversational_prompt_builder"].build.assert_called_once()
    assert result_casual_follow_up.context is None


def test_to_sse_event_casual_turn_payload_contract():
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assistant_message = Message(
        id=uuid4(),
        conversation_id=uuid4(),
        role=MessageRole.ASSISTANT,
        content="You're welcome!",
        created_at=now,
        updated_at=now,
    )

    citations_event = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.CITATIONS,
            payload=ChatStreamCitations(context=None),
        )
    )
    assert citations_event["event"] == "citations"
    assert citations_event["data"] == "[]"

    complete_event = mapper.to_sse_event(
        ChatStreamEvent(
            type=ChatStreamEventType.COMPLETE,
            payload=ChatStreamComplete(
                assistant_message=assistant_message,
                context=None,
                model="test-model",
                finish_reason="stop",
                usage=None,
            ),
        )
    )
    assert complete_event["event"] == "complete"
    assert '"assistant_message"' in complete_event["data"]
    assert '"finish_reason": "stop"' in complete_event["data"]
