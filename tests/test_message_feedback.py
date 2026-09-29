from __future__ import annotations

from collections.abc import Generator
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db.database import engine
from app.db.models import (
    Conversation,
    KnowledgeBase,
    Message,
    MessageFeedback,
    Project,
    User,
)
from app.db.models.enums import FeedbackRating, MessageRole, UserStatus
from app.dependencies.auth import get_current_user
from app.main import app
from app.repositories.message_feedback import MessageFeedbackRepository


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


def _create_user_and_conversation(
    session: Session,
    *,
    prefix: str = "user",
) -> tuple[User, Conversation, Message, Message]:
    user_suffix = uuid4().hex[:8]
    user = User(
        email=f"{prefix}_{user_suffix}@example.com",
        email_normalized=f"{prefix}_{user_suffix}@example.com",
        name=f"User {user_suffix}",
        status=UserStatus.ACTIVE,
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

    user_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.USER,
        content="What is RAG?",
    )
    session.add(user_msg)
    session.flush()

    assistant_msg = Message(
        conversation_id=conversation.id,
        role=MessageRole.ASSISTANT,
        content="RAG stands for Retrieval-Augmented Generation.",
    )
    session.add(assistant_msg)
    session.commit()

    return user, conversation, user_msg, assistant_msg


def test_feedback_upsert_and_toggle(db_session: Session):
    user, _conversation, _user_msg, assistant_msg = _create_user_and_conversation(
        db_session
    )
    feedback_repo = MessageFeedbackRepository(db=db_session)

    try:
        # 1. Initial up-vote
        fb_up = feedback_repo.upsert(
            message_id=assistant_msg.id,
            user_id=user.id,
            rating=FeedbackRating.UP,
        )
        db_session.commit()
        assert fb_up.rating == FeedbackRating.UP
        assert fb_up.message_id == assistant_msg.id
        assert fb_up.user_id == user.id

        # 2. Toggle to down-vote (same row updated)
        fb_down = feedback_repo.upsert(
            message_id=assistant_msg.id,
            user_id=user.id,
            rating=FeedbackRating.DOWN,
        )
        db_session.commit()
        assert fb_down.id == fb_up.id
        assert fb_down.rating == FeedbackRating.DOWN

        # 3. Query back
        loaded = feedback_repo.get_by_id(fb_down.id)
        assert loaded is not None
        assert loaded.rating == FeedbackRating.DOWN
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_feedback_delete(db_session: Session):
    user, _conversation, _user_msg, assistant_msg = _create_user_and_conversation(
        db_session
    )
    feedback_repo = MessageFeedbackRepository(db=db_session)

    try:
        fb = feedback_repo.upsert(
            message_id=assistant_msg.id,
            user_id=user.id,
            rating=FeedbackRating.UP,
        )
        fb_id = fb.id
        db_session.commit()

        deleted = feedback_repo.delete_by_message_and_user(
            message_id=assistant_msg.id,
            user_id=user.id,
        )
        db_session.commit()
        assert deleted is True

        # Second delete is a safe no-op
        deleted_again = feedback_repo.delete_by_message_and_user(
            message_id=assistant_msg.id,
            user_id=user.id,
        )
        db_session.commit()
        assert deleted_again is False

        # Verify not found
        loaded = feedback_repo.get_by_id(fb_id)
        assert loaded is None
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()


def test_feedback_api_endpoints_and_user_isolation(db_session: Session):
    user_a, conv_a, user_msg_a, asst_msg_a = _create_user_and_conversation(
        db_session, prefix="user_a"
    )
    user_b, conv_b, _user_msg_b, asst_msg_b = _create_user_and_conversation(
        db_session, prefix="user_b"
    )

    client = TestClient(app)

    try:
        # User A logs in
        app.dependency_overrides[get_current_user] = lambda: user_a

        # 1. User A rates their assistant message UP
        resp_up = client.put(
            f"/api/v2/messages/{asst_msg_a.id}/feedback",
            json={"rating": "up"},
        )
        assert resp_up.status_code == 200
        data_up = resp_up.json()
        assert data_up["rating"] == "up"
        assert data_up["message_id"] == str(asst_msg_a.id)
        assert data_up["user_id"] == str(user_a.id)

        # 2. User A cannot rate a User message
        resp_user_msg = client.put(
            f"/api/v2/messages/{user_msg_a.id}/feedback",
            json={"rating": "up"},
        )
        assert resp_user_msg.status_code in (400, 422)

        # 3. User A cannot rate User B's message
        resp_forbidden = client.put(
            f"/api/v2/messages/{asst_msg_b.id}/feedback",
            json={"rating": "up"},
        )
        assert resp_forbidden.status_code == 404

        # 4. User A checks message history: assistant message has feedback="up"
        resp_history_a = client.get(
            f"/api/v2/conversations/{conv_a.id}/messages",
        )
        assert resp_history_a.status_code == 200
        items_a = resp_history_a.json()["items"]
        asst_in_history_a = next(m for m in items_a if m["id"] == str(asst_msg_a.id))
        assert asst_in_history_a["feedback"] == "up"

        # 5. User B logs in
        app.dependency_overrides[get_current_user] = lambda: user_b

        # User B checks their own history: feedback is None
        resp_history_b = client.get(
            f"/api/v2/conversations/{conv_b.id}/messages",
        )
        assert resp_history_b.status_code == 200
        items_b = resp_history_b.json()["items"]
        asst_in_history_b = next(m for m in items_b if m["id"] == str(asst_msg_b.id))
        assert asst_in_history_b["feedback"] is None

        # User B rates their own assistant message DOWN
        resp_down = client.put(
            f"/api/v2/messages/{asst_msg_b.id}/feedback",
            json={"rating": "down"},
        )
        assert resp_down.status_code == 200
        assert resp_down.json()["rating"] == "down"

        # 6. User A logs back in: User A still sees "up" on message A
        app.dependency_overrides[get_current_user] = lambda: user_a
        resp_history_a2 = client.get(
            f"/api/v2/conversations/{conv_a.id}/messages",
        )
        items_a2 = resp_history_a2.json()["items"]
        asst_in_history_a2 = next(m for m in items_a2 if m["id"] == str(asst_msg_a.id))
        assert asst_in_history_a2["feedback"] == "up"

        # 7. User A deletes their feedback
        del_resp = client.delete(f"/api/v2/messages/{asst_msg_a.id}/feedback")
        assert del_resp.status_code == 204

        # Verify feedback is now None in message history
        resp_history_a3 = client.get(
            f"/api/v2/conversations/{conv_a.id}/messages",
        )
        items_a3 = resp_history_a3.json()["items"]
        asst_in_history_a3 = next(m for m in items_a3 if m["id"] == str(asst_msg_a.id))
        assert asst_in_history_a3["feedback"] is None

    finally:
        app.dependency_overrides.clear()
        with Session(engine) as cleanup:
            cleanup.execute(
                text("DELETE FROM users WHERE id IN (:u1, :u2)"),
                {"u1": user_a.id, "u2": user_b.id},
            )
            cleanup.commit()


def test_feedback_concurrent_upsert_in_postgres(db_session: Session):
    """Verify that multiple concurrent upserts against PostgreSQL ON CONFLICT succeed without error."""
    from concurrent.futures import ThreadPoolExecutor

    user, _conv, _user_msg, asst_msg = _create_user_and_conversation(db_session)
    asst_msg_id = asst_msg.id
    user_id = user.id

    def do_upsert(rating: FeedbackRating) -> None:
        with Session(engine) as session:
            repo = MessageFeedbackRepository(db=session)
            repo.upsert(
                message_id=asst_msg_id,
                user_id=user_id,
                rating=rating,
            )
            session.commit()

    try:
        ratings = [
            FeedbackRating.UP if i % 2 == 0 else FeedbackRating.DOWN for i in range(10)
        ]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(do_upsert, r) for r in ratings]
            for f in futures:
                f.result()  # Should not raise any IntegrityError or deadlock

        # Exactly 1 feedback row exists in PostgreSQL
        with Session(engine) as session:
            fb = session.scalar(
                select(MessageFeedback).where(
                    MessageFeedback.message_id == asst_msg.id,
                    MessageFeedback.user_id == user.id,
                )
            )
            assert fb is not None
            assert fb.rating in (FeedbackRating.UP, FeedbackRating.DOWN)
    finally:
        with Session(engine) as cleanup:
            cleanup.execute(text("DELETE FROM users WHERE id = :uid"), {"uid": user.id})
            cleanup.commit()
