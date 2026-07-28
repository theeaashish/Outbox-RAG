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
