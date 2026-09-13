from __future__ import annotations

import logging
import math
from time import perf_counter
from uuid import UUID

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.retrieval.models import RetrievedChunk
from app.core.constants import MAX_RETRIEVAL_LIMIT
from app.core.exceptions import ResourceNotFoundException, ValidationException
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
        if not isinstance(query, str):
            raise ValidationException("Query must be a string")
        normalized_query = query.strip()
        if not normalized_query:
            raise ValidationException("Query must not be empty or whitespace-only")

        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or limit < 1
            or limit > MAX_RETRIEVAL_LIMIT
        ):
            raise ValidationException(
                f"Limit must be an integer between 1 and {MAX_RETRIEVAL_LIMIT}"
            )

        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not (0.0 <= threshold <= 1.0)
            or math.isnan(threshold)
        ):
            raise ValidationException("Threshold must be a float between 0.0 and 1.0")

        if (
            self._knowledge_base_repository.get_by_user_and_id(
                user_id=user_id, knowledge_base_id=knowledge_base_id
            )
            is None
        ):
            raise ResourceNotFoundException("Knowledge base not found")

        start_time = perf_counter()
        logger.info(
            "Retrieval started",
            extra={
                "user_id": str(user_id),
                "kb_id": str(knowledge_base_id),
                "query_length": len(normalized_query),
                "limit": limit,
                "threshold": threshold,
            },
        )

        embedding = self.embedding_generator.embed_query(normalized_query)

        results = self.chunk_repository.search_similar(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            embedding=embedding,
            limit=limit,
            threshold=threshold,
        )

        duration_ms = round((perf_counter() - start_time) * 1000, 2)
        top_score = round(results[0].similarity, 4) if results else None

        logger.info(
            "Retrieval completed",
            extra={
                "user_id": str(user_id),
                "kb_id": str(knowledge_base_id),
                "query_length": len(normalized_query),
                "result_count": len(results),
                "top_score": top_score,
                "duration_ms": duration_ms,
            },
        )

        return results
