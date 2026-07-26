from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DocumentChunk
from app.repositories.base import BaseRepository


class DocumentChunkRepository(BaseRepository[DocumentChunk]):
    """Repository for DocumentChunk model operations."""

    def __init__(self, *, db: Session) -> None:
        super().__init__(
            db=db,
            model=DocumentChunk,
        )

    def get_by_document(
        self,
        *,
        document_id: UUID,
    ) -> list[DocumentChunk]:
        """Retrieve all chunks for a document in chunk order."""
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        return list(self.db.scalars(statement))
