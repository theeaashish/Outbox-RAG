from __future__ import annotations

from uuid import UUID

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.retrieval.models import RetrievedChunk
from app.repositories.document_chunk import DocumentChunkRepository


class RetrievalService:
    """Application service for semantic document retrieval."""

    def __init__(
        self,
        *,
        embedding_generator: EmbeddingGenerator,
        chunk_repository: DocumentChunkRepository,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.chunk_repository = chunk_repository

    def retrieve(
        self,
        *,
        knowledge_base_id: UUID,
        query: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a user query.
        """

        embedding = self.embedding_generator.embed_query(query)

        return self.chunk_repository.search_similar(
            knowledge_base_id=knowledge_base_id,
            embedding=embedding,
            limit=limit,
        )
