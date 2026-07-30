from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.ai.retrieval.models import RetrievedChunk
from app.db.models import Document, DocumentChunk
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

    def search_similar(
        self,
        *,
        knowledge_base_id: UUID,
        embedding: list[float],
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most similar chunks within a knowledge base using
        cosine similarity.
        """

        distance = DocumentChunk.embedding.cosine_distance(embedding)
        similarity = (1 - distance).label("similarity")

        statement = (
            select(DocumentChunk, similarity)
            .join(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
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
