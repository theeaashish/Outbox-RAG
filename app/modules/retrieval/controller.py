from __future__ import annotations

from uuid import UUID

from app.modules.retrieval.schemas import RetrievalResponse, RetrievedChunkResponse
from app.modules.retrieval.service import RetrievalService


class RetrievalController:
    """Controller for semantic document retrieval."""

    def __init__(
        self,
        *,
        retrieval_service: RetrievalService,
    ) -> None:
        self.retrieval_service = retrieval_service

    def retrieve(
        self,
        *,
        knowledge_base_id: UUID,
        query: str,
        limit: int = 5,
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query and return RetrievalResponse.
        """

        retrieved_chunks = self.retrieval_service.retrieve(
            knowledge_base_id=knowledge_base_id, query=query, limit=limit
        )

        return RetrievalResponse(
            results=[
                RetrievedChunkResponse(
                    document_id=rc.chunk.document_id,
                    document_name=rc.chunk.document.title if rc.chunk.document else "",
                    chunk_index=rc.chunk.chunk_index,
                    content=rc.chunk.content,
                    score=rc.similarity,
                    char_start=rc.chunk.char_start if rc.chunk.char_start is not None else 0,
                    char_end=rc.chunk.char_end if rc.chunk.char_end is not None else 0,
                )
                for rc in retrieved_chunks
            ]
        )
