from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.core.config import settings


class RetrievalRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(
        min_length=1,
        description="Natural language search query.",
    )

    limit: int = Field(
        default_factory=lambda: settings.default_top_k,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve.",
    )

    threshold: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Optional minimum cosine similarity. When omitted, top-k "
            "retrieval is used without a score floor."
        ),
    )


class RetrievedChunkResponse(BaseModel):
    """Response model for a retrieved chunk."""

    document_id: UUID
    document_name: str

    chunk_index: int

    content: str

    score: float

    char_start: int | None
    char_end: int | None


class RetrievalResponse(BaseModel):
    """Response wrapper."""

    results: list[RetrievedChunkResponse]
