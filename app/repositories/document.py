from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.db.models import Document, KnowledgeBase
from app.db.models.enums import DocumentStatus
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=Document,
        )

    def get_by_user_and_id(
        self,
        *,
        user_id: UUID,
        document_id: UUID,
    ) -> Document | None:
        """Retrieve a document by user and document ID."""
        statement = (
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(
                Document.id == document_id,
                KnowledgeBase.user_id == user_id,
            )
        )
        return self.db.scalar(statement)

    def list_by_user_and_knowledge_base(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> list[Document]:
        """Retrieve all documents for a knowledge base owned by user."""
        statement = (
            select(Document)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                KnowledgeBase.user_id == user_id,
            )
            .order_by(Document.created_at.desc())
        )
        return list(self.db.scalars(statement))

    def get_for_update(
        self,
        *,
        document_id: UUID,
    ) -> Document | None:
        """Lock and retrieve a document row for exclusive processing."""
        statement = (
            select(Document)
            .where(Document.id == document_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
        return self.db.scalar(statement)

    def get_by_hash(
        self,
        *,
        knowledge_base_id: UUID,
        sha256_hash: str,
    ) -> Document | None:
        """Retrieve a document by its hash."""
        statement = (
            select(Document)
            .where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.sha256_hash == sha256_hash,
            )
            .execution_options(populate_existing=True)
        )
        return self.db.scalar(statement)

    def exists_by_storage_path(
        self,
        *,
        storage_path: str,
    ) -> bool:
        """Check whether a document referencing the storage path exists."""
        statement = select(exists().where(Document.storage_path == storage_path))
        return bool(self.db.scalar(statement))

    def list_stale_uploading_documents(
        self,
        *,
        updated_before: datetime,
        limit: int = 100,
    ) -> list[tuple[UUID, str]]:
        """Retrieve (id, storage_path) of documents in UPLOADING state prior to a cutoff."""
        statement = (
            select(Document.id, Document.storage_path)
            .where(
                Document.status == DocumentStatus.UPLOADING,
                Document.updated_at < updated_before,
            )
            .order_by(Document.updated_at.asc())
            .limit(limit)
        )
        return list(self.db.execute(statement).tuples().all())
