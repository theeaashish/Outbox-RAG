from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(
        min_length=1,
        description="Natural language search query.",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve.",
    )


class RetrievedChunkResponse(BaseModel):
    """Response model for a retrieved chunk."""

    document_id: UUID
    document_name: str

    chunk_index: int

    content: str

    score: float

    char_start: int
    char_end: int


class RetrievalResponse(BaseModel):
    """Response wrapper."""

    results: list[RetrievedChunkResponse]
