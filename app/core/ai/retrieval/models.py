from __future__ import annotations

from dataclasses import dataclass

from app.db.models import DocumentChunk


@dataclass(slots=True, frozen=True)
class RetrievedChunk:
    """Represents a retrieved document chunk and its similarity score."""

    chunk: DocumentChunk
    similarity: float
