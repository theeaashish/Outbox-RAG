from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.core.ai.context.models import AssembledContext, ContextChunk
from app.core.ai.llm.base import LLMProvider
from app.core.ai.llm.models import (
    LLMResponse,
    LLMStreamCompletion,
    LLMStreamDelta,
)
from app.core.ai.routing.models import (
    ConversationMode,
    QueryRoutingDecision,
    RoutingReason,
)
from app.core.ai.routing.router import QueryRouter
from app.core.exceptions import AIServiceException, DatabaseException
from app.core.pagination import CursorCodec
from app.db.database import engine
from app.db.models import (
    Conversation,
    Document,
    DocumentChunk,
    KnowledgeBase,
    Message,
    MessageSource,
    Project,
    User,
)
from app.db.models.enums import DocumentStatus, MessageRole
from app.modules.chat.service import ChatService
from app.modules.conversations.service import ConversationService
from app.repositories.conversation import ConversationRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository
from app.repositories.message_feedback import MessageFeedbackRepository
from app.repositories.project import ProjectRepository


@pytest.fixture
def db_session() -> Generator[Session]:
    SessionMaker = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = SessionMaker()
    created_user_ids: list[UUID] = []

    try:
        yield session
    finally:
        if created_user_ids:
            try:
                for uid in created_user_ids:
                    session.execute(
                        text("DELETE FROM users WHERE id = :uid"), {"uid": uid}
                    )
                session.commit()
            except SQLAlchemyError:
                session.rollback()
        session.close()


def _create_test_hierarchy(
    session: Session,
) -> tuple[User, Project, KnowledgeBase, Conversation, Document, list[DocumentChunk]]:
    user_suffix = uuid4().hex[:8]
    user = User(
        email=f"source_test_{user_suffix}@example.com",
        email_normalized=f"source_test_{user_suffix}@example.com",
        name="Sources Test User",
    )
    session.add(user)
    session.flush()

    project = Project(user_id=user.id, name=f"Proj_{user_suffix}")
    session.add(project)
    session.flush()

    kb = KnowledgeBase(
        user_id=user.id,
        project_id=project.id,
        name=f"KB_{user_suffix}",
    )
    session.add(kb)
    session.flush()

    conversation = Conversation(
        user_id=user.id,
        project_id=project.id,
        knowledge_base_id=kb.id,
    )
    session.add(conversation)
    session.flush()

    doc = Document(
        knowledge_base_id=kb.id,
        title="Architecture Handbook",
        filename="handbook.pdf",
        mime_type="application/pdf",
        storage_path=f"storage/{user_suffix}.pdf",
        sha256_hash=uuid4().hex,
        file_size=1024,
        status=DocumentStatus.READY,
    )
    session.add(doc)
    session.flush()

    from app.core.constants import EMBEDDING_DIMENSION

    chunks: list[DocumentChunk] = []
    dummy_vec = [0.0] * EMBEDDING_DIMENSION
    dummy_vec[0] = 1.0
    for i in range(2):
        chunk = DocumentChunk(
            document_id=doc.id,
            chunk_index=i,
            content=f"Content for chunk {i}",
            embedding=dummy_vec,
            chunk_metadata={"page_start": i + 1, "page_end": i + 1},
        )
        session.add(chunk)
        chunks.append(chunk)

    session.commit()
    return user, project, kb, conversation, doc, chunks


def _build_chat_service(
    session: Session,
    *,
    llm_content: str = "Default answer [1]",
    chunks: list[DocumentChunk] | None = None,
    doc: Document | None = None,
    citation_validation_strict: bool | None = None,
) -> ChatService:
    conv_repo = ConversationRepository(db=session)
    msg_repo = MessageRepository(db=session)

    context_chunks = []
    if chunks and doc:
        for idx, ch in enumerate(chunks):
            meta = ch.chunk_metadata or {}
            context_chunks.append(
                ContextChunk(
                    citation=idx + 1,
                    chunk_id=ch.id,
                    document_id=doc.id,
                    document_name=doc.title,
                    chunk_index=ch.chunk_index,
                    similarity=0.85 - (idx * 0.1),
                    content=ch.content,
                    page_start=meta.get("page_start"),
                    page_end=meta.get("page_end"),
                )
            )

    context = (
        AssembledContext(query="test", block="context", chunks=context_chunks)
        if context_chunks
        else None
    )

    retrieval_service = MagicMock()
    retrieval_service.retrieve.return_value = []
    context_assembler = MagicMock()
    context_assembler.assemble.return_value = context
    prompt_builder = MagicMock()
    prompt_builder.build.return_value = []
    conversational_prompt_builder = MagicMock()
    conversational_prompt_builder.build.return_value = []

    query_router = MagicMock(spec=QueryRouter)
    if context:
        query_router.route.return_value = QueryRoutingDecision(
            mode=ConversationMode.KNOWLEDGE,
            reason=RoutingReason.EXPLICIT_KNOWLEDGE,
        )
    else:
        query_router.route.return_value = QueryRoutingDecision(
            mode=ConversationMode.CASUAL,
            reason=RoutingReason.EXPLICIT_CASUAL,
        )

    llm_provider = MagicMock(spec=LLMProvider)
    llm_provider.generate.return_value = LLMResponse(
        content=llm_content,
        model="test-model",
        finish_reason="stop",
        usage=None,
    )

    class FakeStream:
        def __iter__(self):
            yield LLMStreamDelta(content=llm_content)
            yield LLMStreamCompletion(
                model="test-model",
                finish_reason="stop",
                usage=None,
            )

        def close(self):
            pass

    llm_provider.stream.return_value = FakeStream()

    return ChatService(
        db=session,
        conversation_repository=conv_repo,
        message_repository=msg_repo,
        retrieval_service=retrieval_service,
        context_assembler=context_assembler,
        prompt_builder=prompt_builder,
        conversational_prompt_builder=conversational_prompt_builder,
        query_router=query_router,
        llm_provider=llm_provider,
        history_message_limit=20,
        retrieval_limit=5,
        similarity_threshold=0.7,
        stream_max_buffered_characters=65536,
        citation_validation_strict=citation_validation_strict,
    )


def test_message_sources_atomic_persistence(db_session: Session):
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Postgres stores data. [1]",
            chunks=chunks,
            doc=doc,
        )

        result = service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Where is data stored?",
        )

        assert result.assistant_message is not None
        assistant_msg_id = result.assistant_message.id

        # Verify in fresh query from PostgreSQL
        with Session(engine) as verify_session:
            msg_repo = MessageRepository(db=verify_session)
            msg = msg_repo.get_by_id_with_sources(assistant_msg_id)
            assert msg is not None
            assert msg.role == MessageRole.ASSISTANT
            assert len(msg.sources) == 2

            # Deterministic ordering by source_order asc
            s0 = msg.sources[0]
            assert s0.citation_number == 1
            assert s0.is_cited is True
            assert s0.source_order == 0
            assert s0.document_id == doc.id
            assert s0.document_chunk_id == chunks[0].id
            assert s0.document_name_snapshot == "Architecture Handbook"
            assert s0.chunk_index_snapshot == 0
            assert s0.retrieval_score == 0.85
            assert s0.page_start_snapshot == 1

            s1 = msg.sources[1]
            assert s1.citation_number == 2
            assert s1.is_cited is False
            assert s1.source_order == 1
            assert s1.document_id == doc.id
            assert s1.document_chunk_id == chunks[1].id
            assert s1.document_name_snapshot == "Architecture Handbook"
            assert s1.chunk_index_snapshot == 1
            assert s1.retrieval_score == 0.75
            assert s1.page_start_snapshot == 2
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_invalid_citations_reject_and_persist_nothing(db_session: Session):
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        # LLM emits invalid citation [99] in strict mode
        service = _build_chat_service(
            db_session,
            llm_content="Data is stored in Redis. [99]",
            chunks=chunks,
            doc=doc,
            citation_validation_strict=True,
        )

        with pytest.raises(AIServiceException) as exc_info:
            service.send_message(
                user_id=user.id,
                conversation_id=conversation.id,
                content="Where is data stored?",
            )
        assert "citation validation" in str(exc_info.value).lower()

        # Invariant: Neither assistant Message nor MessageSource rows committed
        with Session(engine) as verify_session:
            messages = list(
                verify_session.scalars(
                    select(Message).where(Message.conversation_id == conversation.id)
                )
            )
            # Only user message persisted (from prepare_turn); NO assistant message
            assert len(messages) == 1
            assert messages[0].role == MessageRole.USER

            sources = list(
                verify_session.scalars(
                    select(MessageSource).where(
                        MessageSource.message_id == messages[0].id
                    )
                )
            )
            assert len(sources) == 0
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_casual_turn_persists_no_sources(db_session: Session):
    user, _project, _kb, conversation, _doc, _chunks = _create_test_hierarchy(
        db_session
    )
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Hello! How can I help you?",
            chunks=None,
            doc=None,
        )

        result = service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Hi there",
        )

        assert result.assistant_message is not None
        assistant_msg_id = result.assistant_message.id

        with Session(engine) as verify_session:
            msg_repo = MessageRepository(db=verify_session)
            msg = msg_repo.get_by_id_with_sources(assistant_msg_id)
            assert msg is not None
            assert len(msg.sources) == 0
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_message_sources_eager_loading_and_ordering_in_cursor_page(db_session: Session):
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Postgres stores data. [1]",
            chunks=chunks,
            doc=doc,
        )

        service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Where is data stored?",
        )

        # Query messages via ConversationService.list_messages_cursor
        conv_service = ConversationService(
            db=db_session,
            conversation_repository=ConversationRepository(db=db_session),
            knowledge_base_repository=KnowledgeBaseRepository(db=db_session),
            project_repository=ProjectRepository(db=db_session),
            message_repository=MessageRepository(db=db_session),
            cursor_codec=CursorCodec(signing_key="test-secret-salt-key-1234567890"),
            message_feedback_repository=MessageFeedbackRepository(db=db_session),
        )

        page, _feedback_map = conv_service.list_messages_cursor(
            user_id=user.id,
            conversation_id=conversation.id,
            page_size=10,
        )

        assert len(page.items) == 2
        assistant_msg = next(m for m in page.items if m.role == MessageRole.ASSISTANT)
        assert len(assistant_msg.sources) == 2
        assert assistant_msg.sources[0].source_order == 0
        assert assistant_msg.sources[0].citation_number == 1
        assert assistant_msg.sources[0].is_cited is True
        assert assistant_msg.sources[1].source_order == 1
        assert assistant_msg.sources[1].citation_number == 2
        assert assistant_msg.sources[1].is_cited is False
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_document_deletion_sets_fk_to_null_and_preserves_snapshot(db_session: Session):
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Data storage explained. [1]",
            chunks=chunks,
            doc=doc,
        )

        result = service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Tell me about storage",
        )
        assistant_msg_id = result.assistant_message.id

        # Delete the document (cascade deletes document_chunks)
        with Session(engine) as del_session:
            del_session.execute(
                text("DELETE FROM documents WHERE id = :doc_id"), {"doc_id": doc.id}
            )
            del_session.commit()

        # Check MessageSource rows
        with Session(engine) as verify_session:
            msg_repo = MessageRepository(db=verify_session)
            msg = msg_repo.get_by_id_with_sources(assistant_msg_id)
            assert msg is not None
            assert len(msg.sources) == 2
            for s in msg.sources:
                assert s.document_id is None  # ON DELETE SET NULL
                assert s.document_chunk_id is None  # ON DELETE SET NULL
                assert (
                    s.document_name_snapshot == "Architecture Handbook"
                )  # Snapshot preserved!
                assert s.chunk_index_snapshot in (0, 1)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_race_safe_message_finalization(db_session: Session):
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Content [1]",
            chunks=chunks,
            doc=doc,
        )

        msg_id = uuid4()
        first_call = service._finalize_assistant_turn(
            user_id=user.id,
            conversation_id=conversation.id,
            assistant_content="First response [1]",
            message_id=msg_id,
            context=None,
            cited_citations=set(),
        )
        assert first_call.id == msg_id

        # Concurrent / duplicate call with same content returns canonical existing message
        second_call = service._finalize_assistant_turn(
            user_id=user.id,
            conversation_id=conversation.id,
            assistant_content="First response [1]",
            message_id=msg_id,
            context=None,
            cited_citations=set(),
        )
        assert second_call.id == msg_id
        assert second_call.content == "First response [1]"

        # Different content for existing message_id raises DatabaseException
        with pytest.raises(Exception) as exc_info:
            service._finalize_assistant_turn(
                user_id=user.id,
                conversation_id=conversation.id,
                assistant_content="Different payload",
                message_id=msg_id,
                context=None,
                cited_citations=set(),
            )
        assert "Duplicate message ID conflict" in str(exc_info.value)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_conversation_deletion_cascade_in_postgres(db_session: Session):
    """Verify that deleting a conversation cascades messages, sources, and feedbacks with passive_deletes."""
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Response with sources [1]",
            chunks=chunks,
            doc=doc,
        )

        turn = service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Tell me about architecture",
        )
        asst_id = turn.assistant_message.id

        # Verify sources exist in PostgreSQL
        with Session(engine) as s:
            sources = s.scalars(
                select(MessageSource).where(MessageSource.message_id == asst_id)
            ).all()
            assert len(sources) > 0

        # Delete conversation
        conv_repo = ConversationRepository(db=db_session)
        conv = conv_repo.get_by_user_and_id(
            user_id=user.id,
            conversation_id=conversation.id,
        )
        assert conv is not None
        db_session.delete(conv)
        db_session.commit()

        # Verify messages and sources were cascaded in PostgreSQL
        with Session(engine) as s:
            msg = s.get(Message, asst_id)
            assert msg is None
            sources_after = s.scalars(
                select(MessageSource).where(MessageSource.message_id == asst_id)
            ).all()
            assert len(sources_after) == 0
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_to_message_response_raises_invalid_request_error_when_sources_unloaded(
    db_session: Session,
):
    from sqlalchemy.exc import InvalidRequestError

    from app.modules.conversations.mapper import to_message_response

    user, _project, _kb, conversation, _doc, _chunks = _create_test_hierarchy(
        db_session
    )
    try:
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="Hello world",
            conversation_id=conversation.id,
        )
        db_session.add(msg)
        db_session.commit()

        with Session(engine) as s:
            loaded_msg = s.scalar(select(Message).where(Message.id == msg.id))
            assert loaded_msg is not None
            with pytest.raises(InvalidRequestError):
                to_message_response(loaded_msg)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_fk_failure_is_not_reported_as_duplicate(db_session: Session):
    """Verify that an FK constraint violation raises DatabaseException without claiming duplicate message ID."""
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Response",
            chunks=chunks,
            doc=doc,
        )
        fake_conv_id = uuid4()
        with patch.object(
            ConversationRepository,
            "get_by_user_and_id",
            return_value=conversation,
        ):
            with pytest.raises(DatabaseException) as exc_info:
                service._finalize_assistant_turn(
                    user_id=user.id,
                    conversation_id=fake_conv_id,
                    assistant_content="Test message",
                    message_id=uuid4(),
                    context=None,
                    cited_citations=set(),
                )
            assert "Duplicate message ID conflict" not in str(exc_info.value)
            assert "Failed to finalize assistant turn" in str(exc_info.value)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_reingest_document_sets_chunks_to_null_and_preserves_sources(
    db_session: Session,
):
    """Verify that re-ingesting a document (deleting old chunks) sets document_chunk_id to NULL while keeping snapshots."""
    user, _project, _kb, conversation, doc, chunks = _create_test_hierarchy(db_session)
    try:
        service = _build_chat_service(
            db_session,
            llm_content="Data storage explained. [1]",
            chunks=chunks,
            doc=doc,
        )

        result = service.send_message(
            user_id=user.id,
            conversation_id=conversation.id,
            content="Tell me about storage",
        )
        assistant_msg_id = result.assistant_message.id

        # Simulate re-ingest: delete existing chunks for this document
        with Session(engine) as del_session:
            del_session.execute(
                text("DELETE FROM document_chunks WHERE document_id = :doc_id"),
                {"doc_id": doc.id},
            )
            del_session.commit()

        # Check MessageSource rows: document_chunk_id should be NULL, document_id remains intact
        with Session(engine) as verify_session:
            msg_repo = MessageRepository(db=verify_session)
            msg = msg_repo.get_by_id_with_sources(assistant_msg_id)
            assert msg is not None
            assert len(msg.sources) == 2
            for s in msg.sources:
                assert s.document_id == doc.id
                assert s.document_chunk_id is None
                assert s.document_name_snapshot == "Architecture Handbook"
                assert s.chunk_index_snapshot in (0, 1)
                assert isinstance(s.retrieval_score, float)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()
