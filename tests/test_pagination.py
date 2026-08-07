from datetime import UTC, datetime, timedelta
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import Table, create_engine
from sqlalchemy.orm import sessionmaker

from app.core.exceptions import ValidationException
from app.core.pagination import CursorCodec, CursorResource, InvalidCursorError
from app.db.base import Base
from app.db.models import Conversation, KnowledgeBase, Message
from app.db.models.enums import MessageRole
from app.modules.conversations.service import ConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository


def _sqlite_session():
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
    return engine, session


def test_cursor_codec_round_trip_and_rejects_tampering():
    codec = CursorCodec(signing_key="secret-one")
    scope_id = uuid4()
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    entity_id = uuid4()
    snapshot_timestamp = datetime(2026, 1, 2, tzinfo=UTC)

    cursor = codec.encode(
        resource=CursorResource.MESSAGES,
        scope_id=scope_id,
        created_at=created_at,
        entity_id=entity_id,
        snapshot_timestamp=snapshot_timestamp,
    )
    position = codec.decode(
        cursor=cursor, resource=CursorResource.MESSAGES, scope_id=scope_id
    )

    assert position.entity_id == entity_id
    assert position.created_at == created_at
    assert position.snapshot_timestamp == snapshot_timestamp

    with pytest.raises(InvalidCursorError):
        codec.decode(
            cursor=cursor + "x", resource=CursorResource.MESSAGES, scope_id=scope_id
        )

    with pytest.raises(InvalidCursorError):
        codec.decode(
            cursor=cursor, resource=CursorResource.CONVERSATIONS, scope_id=scope_id
        )

    with pytest.raises(InvalidCursorError):
        codec.decode(cursor=cursor, resource=CursorResource.MESSAGES, scope_id=uuid4())


def test_cursor_codec_accepts_previous_signing_key():
    previous = CursorCodec(signing_key="old-key")
    current = CursorCodec(
        signing_key="new-key",
        previous_signing_key="old-key",
    )
    scope_id = uuid4()
    cursor = previous.encode(
        resource=CursorResource.CONVERSATIONS,
        scope_id=scope_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        entity_id=uuid4(),
        snapshot_timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )

    position = current.decode(
        cursor=cursor,
        resource=CursorResource.CONVERSATIONS,
        scope_id=scope_id,
    )
    assert position.snapshot_timestamp.year == 2026


def test_message_keyset_pagination_is_newest_first_and_stable():
    engine, session = _sqlite_session()
    knowledge_base = KnowledgeBase(name="Docs")
    conversation = Conversation(knowledge_base=knowledge_base)
    session.add(conversation)
    session.commit()

    started_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    for index in range(5):
        session.add(
            Message(
                conversation_id=conversation.id,
                role=MessageRole.USER,
                content=f"m{index}",
                created_at=started_at + timedelta(seconds=index),
            )
        )
    session.commit()

    repository = MessageRepository(db=session)
    snapshot_timestamp = started_at + timedelta(seconds=10)
    first_page = repository.list_page_by_conversation(
        conversation_id=conversation.id,
        page_size=2,
        snapshot_timestamp=snapshot_timestamp,
    )
    assert [message.content for message in first_page.items] == ["m4", "m3"]
    assert first_page.has_next_page is True
    assert first_page.has_previous_page is False

    after = first_page.items[-1]
    second_page = repository.list_page_by_conversation(
        conversation_id=conversation.id,
        page_size=2,
        snapshot_timestamp=snapshot_timestamp,
        after=type(first_page.items[0])  # placate type checkers via position below
        and None,
    )
    # rebuild using CursorPosition from codec path
    from app.core.pagination import CursorPosition

    second_page = repository.list_page_by_conversation(
        conversation_id=conversation.id,
        page_size=2,
        snapshot_timestamp=snapshot_timestamp,
        after=CursorPosition(
            created_at=after.created_at,
            entity_id=after.id,
            snapshot_timestamp=snapshot_timestamp,
        ),
    )
    assert [message.content for message in second_page.items] == ["m2", "m1"]
    assert second_page.has_next_page is True
    assert second_page.has_previous_page is True

    third_page = repository.list_page_by_conversation(
        conversation_id=conversation.id,
        page_size=2,
        snapshot_timestamp=snapshot_timestamp,
        after=CursorPosition(
            created_at=second_page.items[-1].created_at,
            entity_id=second_page.items[-1].id,
            snapshot_timestamp=snapshot_timestamp,
        ),
    )
    assert [message.content for message in third_page.items] == ["m0"]
    assert third_page.has_next_page is False
    assert third_page.has_previous_page is True

    newer_page = repository.list_page_by_conversation(
        conversation_id=conversation.id,
        page_size=2,
        snapshot_timestamp=snapshot_timestamp,
        before=CursorPosition(
            created_at=second_page.items[0].created_at,
            entity_id=second_page.items[0].id,
            snapshot_timestamp=snapshot_timestamp,
        ),
    )
    assert [message.content for message in newer_page.items] == ["m4", "m3"]
    assert newer_page.has_previous_page is False
    assert newer_page.has_next_page is True

    session.close()
    engine.dispose()


def test_conversation_keyset_pagination_scopes_to_knowledge_base():
    engine, session = _sqlite_session()
    kb_one = KnowledgeBase(name="One")
    kb_two = KnowledgeBase(name="Two")
    session.add_all([kb_one, kb_two])
    session.commit()

    started_at = datetime(2026, 2, 1, tzinfo=UTC)
    for index in range(3):
        session.add(
            Conversation(
                knowledge_base_id=kb_one.id,
                created_at=started_at + timedelta(seconds=index),
            )
        )
    session.add(
        Conversation(
            knowledge_base_id=kb_two.id,
            created_at=started_at + timedelta(seconds=10),
        )
    )
    session.commit()

    page = ConversationRepository(db=session).list_page_by_knowledge_base(
        knowledge_base_id=kb_one.id,
        page_size=2,
        snapshot_timestamp=started_at + timedelta(seconds=20),
    )
    assert len(page.items) == 2
    assert all(item.knowledge_base_id == kb_one.id for item in page.items)
    assert page.has_next_page is True

    session.close()
    engine.dispose()


def test_conversation_service_rejects_both_after_and_before():
    engine, session = _sqlite_session()
    knowledge_base = KnowledgeBase(name="Docs")
    session.add(knowledge_base)
    session.commit()

    service = ConversationService(
        db=session,
        conversation_repository=ConversationRepository(db=session),
        knowledge_base_repository=KnowledgeBaseRepository(db=session),
        message_repository=MessageRepository(db=session),
        cursor_codec=CursorCodec(signing_key="test-key"),
    )

    with pytest.raises(ValidationException, match="after or before"):
        service.list_conversations_cursor(
            knowledge_base_id=knowledge_base.id,
            page_size=10,
            after="cursor-a",
            before="cursor-b",
        )

    session.close()
    engine.dispose()


def test_conversation_service_rejects_invalid_cursor():
    engine, session = _sqlite_session()
    knowledge_base = KnowledgeBase(name="Docs")
    session.add(knowledge_base)
    session.commit()

    service = ConversationService(
        db=session,
        conversation_repository=ConversationRepository(db=session),
        knowledge_base_repository=KnowledgeBaseRepository(db=session),
        message_repository=MessageRepository(db=session),
        cursor_codec=CursorCodec(signing_key="test-key"),
    )

    with pytest.raises(ValidationException, match="Invalid cursor"):
        service.list_conversations_cursor(
            knowledge_base_id=knowledge_base.id,
            page_size=10,
            after="not-a-cursor",
        )

    session.close()
    engine.dispose()


class _StubConversationController:
    def list_conversations_cursor(
        self,
        *,
        knowledge_base_id: object,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ):
        from app.modules.conversations.schemas import (
            ConversationCursorPageResponse,
            CursorPageInfo,
        )

        return ConversationCursorPageResponse(
            items=[],
            page_info=CursorPageInfo(
                next_cursor=None,
                previous_cursor=None,
                has_next_page=False,
                has_previous_page=False,
                page_size=page_size,
            ),
        )

    def list_messages_cursor(
        self,
        *,
        conversation_id: object,
        page_size: int,
        after: str | None = None,
        before: str | None = None,
    ):
        from app.modules.conversations.schemas import (
            CursorPageInfo,
            MessageCursorPageResponse,
        )

        return MessageCursorPageResponse(
            items=[],
            page_info=CursorPageInfo(
                next_cursor=None,
                previous_cursor=None,
                has_next_page=False,
                has_previous_page=False,
                page_size=page_size,
            ),
        )

    def list_conversations(
        self,
        *,
        knowledge_base_id: object,
        limit: int = 50,
        offset: int = 0,
    ):
        from app.modules.conversations.schemas import ConversationListResponse

        return ConversationListResponse(results=[])

    def list_messages(
        self,
        *,
        conversation_id: object,
        limit: int = 100,
        offset: int = 0,
    ):
        from app.modules.conversations.schemas import MessageListResponse

        return MessageListResponse(results=[])


def test_v2_cursor_routes_and_v1_deprecation_headers():
    from fastapi.testclient import TestClient

    from app.dependencies.conversations import get_conversation_controller
    from app.main import app

    app.dependency_overrides[get_conversation_controller] = _StubConversationController
    with TestClient(app) as client:
        kb_id = uuid4()
        conv_id = uuid4()

        res_v2_conv = client.get(f"/api/v2/knowledge-bases/{kb_id}/conversations")
        assert res_v2_conv.status_code == 200
        assert "items" in res_v2_conv.json()

        res_v2_msg = client.get(f"/api/v2/conversations/{conv_id}/messages")
        assert res_v2_msg.status_code == 200
        assert "items" in res_v2_msg.json()

        res_v1_conv = client.get(f"/api/v1/knowledge-bases/{kb_id}/conversations")
        assert res_v1_conv.status_code == 200
        assert res_v1_conv.headers.get("Deprecation") == "true"

        res_v1_msg = client.get(f"/api/v1/conversations/{conv_id}/messages")
        assert res_v1_msg.status_code == 200
        assert res_v1_msg.headers.get("Deprecation") == "true"

    app.dependency_overrides.clear()
