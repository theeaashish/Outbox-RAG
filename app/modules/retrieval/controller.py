from __future__ import annotations

from uuid import UUID

from app.core.ai.retrieval.models import RetrievedChunk
from app.modules.retrieval.service import RetrievalService


class RetrievalController:
    """Controller for semantic document retrieval."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
    ) -> None:
        self.retrieval_service: RetrievalService

    def retrieve(
        self,
        *,
        knowledge_base_id: UUID,
        query: str,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        """

        return self.retrieval_service.retrieve(
            knowledge_base_id=knowledge_base_id, query=query, limit=limit
        )
