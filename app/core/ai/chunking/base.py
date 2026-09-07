from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.document.models import Chunk, ParsedDocument


class DocumentChunker(ABC):
    """Abstract base class for document-aware chunking strategies."""

    @abstractmethod
    def split(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        """Split a parsed document into canonical domain chunks."""
        raise NotImplementedError


# Compatibility-only alias for existing references
TextChunker = DocumentChunker
