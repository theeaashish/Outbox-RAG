from __future__ import annotations

from uuid import UUID

from starlette.concurrency import run_in_threadpool

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

    async def retrieve(
        self,
        *,
        knowledge_base_id: UUID,
        query: str,
        limit: int = 5,
        threshold: float | None = None,
    ) -> RetrievalResponse:
        """
        Retrieve relevant chunks for a query and return RetrievalResponse.
        """

        retrieved_chunks = await run_in_threadpool(
            self.retrieval_service.retrieve,
            knowledge_base_id=knowledge_base_id,
            query=query,
            limit=limit,
            threshold=threshold,
        )

        return RetrievalResponse(
            results=[
                RetrievedChunkResponse(
                    document_id=rc.chunk.document_id,
                    document_name=rc.chunk.document.title if rc.chunk.document else "",
                    chunk_index=rc.chunk.chunk_index,
                    content=rc.chunk.content,
                    score=rc.similarity,
                    char_start=rc.chunk.char_start,
                    char_end=rc.chunk.char_end,
                )
                for rc in retrieved_chunks
            ]
        )
