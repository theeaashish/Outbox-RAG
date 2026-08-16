from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Document
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[Document]):
    """Repository for Document model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=Document,
        )

    def get_for_update(
        self,
        *,
        document_id: UUID,
    ) -> Document | None:
        """Lock and retrieve a document row for exclusive processing."""
        statement = select(Document).where(Document.id == document_id).with_for_update()
        return self.db.scalar(statement)

    def get_by_hash(
        self,
        *,
        knowledge_base_id: UUID,
        sha256_hash: str,
    ) -> Document | None:
        """Retrieve a document by its hash."""
        statement = select(Document).where(
            Document.knowledge_base_id == knowledge_base_id,
            Document.sha256_hash == sha256_hash,
        )
        return self.db.scalar(statement)

    def get_by_knowledge_base(
        self,
        *,
        knowledge_base_id: UUID,
    ) -> list[Document]:
        """Retrieve all documents for a knowledge base."""
        statement = (
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(Document.created_at.desc())
        )
        return list(self.db.scalars(statement))
