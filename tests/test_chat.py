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
from app.core.config import settings
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
)
from app.db.base import Base
from app.db.models import Conversation, KnowledgeBase, Message
from app.db.models.enums import MessageRole
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
    knowledge_base = KnowledgeBase(name="Handbook")
    conversation = Conversation(knowledge_base=knowledge_base)
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
    conversation = Conversation(id=uuid4(), knowledge_base_id=uuid4())
    conversation_repository = MagicMock()
    conversation_repository.get_by_id.return_value = conversation
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
        llm_provider=llm_provider,
        history_message_limit=20,
        retrieval_limit=5,
        similarity_threshold=0.7,
        stream_max_buffered_characters=65_536,
    )
    return {
        "service": service,
        "db": db,
        "conversation": conversation,
        "conversation_repository": conversation_repository,
        "message_repository": message_repository,
        "retrieval_service": retrieval_service,
        "context": context,
        "context_assembler": context_assembler,
        "prompt_builder": prompt_builder,
        "llm_provider": llm_provider,
    }


def test_chat_service_generates_context_and_persists_a_message_pair():
    dependencies = _build_chat_service()

    result = dependencies["service"].send_message(
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
            conversation_id=dependencies["conversation"].id,
            content="Current question",
        )

    assert dependencies["message_repository"].create.call_count == 2
    assert dependencies["db"].rollback.call_count == 2


def test_chat_service_rejects_missing_conversation_before_external_calls():
    dependencies = _build_chat_service()
    dependencies["conversation_repository"].get_by_id.return_value = None

    with pytest.raises(ResourceNotFoundException):
        dependencies["service"].send_message(
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
        conversation_id=dependencies["conversation"].id,
        content="Current question",
    )

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
    app.dependency_overrides[get_chat_controller] = _StubChatController
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


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
