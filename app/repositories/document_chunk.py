from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, contains_eager

from app.core.ai.retrieval.models import RetrievedChunk
from app.db.models import Document, DocumentChunk, KnowledgeBase
from app.db.models.enums import DocumentStatus
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

    def count_by_document_id(self, *, document_id: UUID) -> int:
        """Return the number of chunks stored for a document."""
        statement = (
            select(func.count())
            .select_from(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
        )
        return int(self.db.scalar(statement) or 0)

    def delete_by_document_id(self, *, document_id: UUID) -> None:
        """Delete all chunks for a document in a single SQL statement."""
        statement = delete(DocumentChunk).where(
            DocumentChunk.document_id == document_id
        )
        self.db.execute(statement)

    def search_similar(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        embedding: list[float],
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most similar chunks within a knowledge base using
        cosine similarity. Only READY documents are searched.
        """

        distance = DocumentChunk.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("similarity")

        filters = [
            Document.knowledge_base_id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
            Document.status == DocumentStatus.READY,
        ]
        if threshold is not None:
            filters.append((1 - distance) >= threshold)

        statement = (
            select(DocumentChunk, similarity)
            .join(Document, DocumentChunk.document_id == Document.id)
            .join(KnowledgeBase, Document.knowledge_base_id == KnowledgeBase.id)
            .options(contains_eager(DocumentChunk.document))
            .where(*filters)
            .order_by(distance.asc())
            .limit(limit)
        )

        rows = self.db.execute(statement)

        return [
            RetrievedChunk(
                chunk=chunk,
                similarity=float(similarity_val),
            )
            for chunk, similarity_val in rows
        ]
