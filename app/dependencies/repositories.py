from __future__ import annotations

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.repositories.conversation import ConversationRepository
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository
from app.repositories.message import MessageRepository
from app.repositories.outbox_event import OutboxEventRepository
from app.repositories.password_credential import PasswordCredentialRepository
from app.repositories.session import SessionRepository
from app.repositories.user import UserRepository

DBSession = Annotated[Session, Depends(get_db)]


def get_knowledge_base_repository(
    db: DBSession,
) -> KnowledgeBaseRepository:
    """Return a knowledge base repository."""
    return KnowledgeBaseRepository(db=db)


def get_document_repository(
    db: DBSession,
) -> DocumentRepository:
    """Return a document repository."""
    return DocumentRepository(db=db)


def get_document_chunk_repository(
    db: DBSession,
) -> DocumentChunkRepository:
    """Return a document chunk repository."""
    return DocumentChunkRepository(db=db)


def get_conversation_repository(
    db: DBSession,
) -> ConversationRepository:
    """Return a conversation repository."""
    return ConversationRepository(db=db)


def get_message_repository(
    db: DBSession,
) -> MessageRepository:
    """Return a message repository."""
    return MessageRepository(db=db)


def get_outbox_event_repository(
    db: DBSession,
) -> OutboxEventRepository:
    """Return an outbox event repository."""

    return OutboxEventRepository(db=db)


def get_user_repository(
    db: DBSession,
) -> UserRepository:
    return UserRepository(db=db)


def get_password_credential_repository(db: DBSession) -> PasswordCredentialRepository:

    return PasswordCredentialRepository(db=db)


def get_session_repository(
    db: DBSession,
) -> SessionRepository:
    """Return a session repository."""

    return SessionRepository(db=db)
