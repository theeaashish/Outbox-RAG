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
from app.core.ai.llm.models import LLMResponse
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
from app.modules.chat.service import ChatService
from app.modules.conversations.schemas import MessageResponse
from app.repositories.message import MessageRepository


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
    conversation_repository.get_by_id.side_effect = [conversation, conversation]
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
    dependencies["conversation_repository"].get_by_id.side_effect = None

    with pytest.raises(ResourceNotFoundException):
        dependencies["service"].send_message(
            conversation_id=uuid4(),
            content="Current question",
        )

    dependencies["retrieval_service"].retrieve.assert_not_called()
    dependencies["llm_provider"].generate.assert_not_called()
    dependencies["message_repository"].create.assert_not_called()


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
