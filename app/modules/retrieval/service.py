from __future__ import annotations

import logging
from uuid import UUID

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.retrieval.models import RetrievedChunk
from app.core.exceptions import ResourceNotFoundException
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository

logger = logging.getLogger(__name__)


class RetrievalService:
    """Application service for semantic document retrieval."""

    def __init__(
        self,
        *,
        embedding_generator: EmbeddingGenerator,
        chunk_repository: DocumentChunkRepository,
        knowledge_base_repository: KnowledgeBaseRepository,
    ) -> None:
        self.embedding_generator = embedding_generator
        self.chunk_repository = chunk_repository
        self._knowledge_base_repository = knowledge_base_repository

    def retrieve(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a user query.

        When ``threshold`` is omitted, returns top-k by similarity only.
        """

        if (
            self._knowledge_base_repository.get_by_user_and_id(
                user_id=user_id, knowledge_base_id=knowledge_base_id
            )
            is None
        ):
            raise ResourceNotFoundException("Knowledge base not found")

        logger.info(
            "Retrieval started",
            extra={
                "user_id": str(user_id),
                "kb_id": str(knowledge_base_id),
                "limit": limit,
                "threshold": threshold,
            },
        )

        embedding = self.embedding_generator.embed_query(query)

        results = self.chunk_repository.search_similar(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            embedding=embedding,
            limit=limit,
            threshold=threshold,
        )

        logger.info(
            "Retrieval completed",
            extra={
                "kb_id": str(knowledge_base_id),
                "result_count": len(results),
            },
        )

        return results
