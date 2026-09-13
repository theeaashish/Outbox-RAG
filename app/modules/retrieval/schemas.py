from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.constants import MAX_RETRIEVAL_LIMIT


class RetrievalRequest(BaseModel):
    """Request model for semantic search."""

    query: str = Field(
        min_length=1,
        description="Natural language search query.",
    )

    limit: int = Field(
        default_factory=lambda: settings.default_top_k,
        ge=1,
        le=MAX_RETRIEVAL_LIMIT,
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

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value: object) -> object:
        """Trim whitespace and reject blank or whitespace-only queries."""
        if not isinstance(value, str):
            return value
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace-only")
        return trimmed


class RetrievedChunkResponse(BaseModel):
    """Response model for a retrieved chunk."""

    document_id: UUID
    document_name: str

    chunk_index: int

    content: str

    score: float = Field(
        ...,
        description="Cosine similarity score in range [-1.0, 1.0], where higher is more similar.",
    )

    page_start: int | None = None
    page_end: int | None = None

    char_start: int | None = None
    char_end: int | None = None


class RetrievalResponse(BaseModel):
    """Response wrapper."""

    results: list[RetrievedChunkResponse]
