from __future__ import annotations

from abc import ABC, abstractmethod

from app.core.document.models import Block, BlockType, DocumentMetadata, ParsedDocument


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    """

    def parse(self, content: bytes) -> ParsedDocument:
        """Parse content into the canonical document representation."""

        text = self.extract_text(content=content)
        return ParsedDocument(
            metadata=DocumentMetadata(title=None, page_count=None),
            blocks=(
                Block(
                    type=BlockType.PARAGRAPH,
                    text=text,
                    page_number=None,
                    block_index=0,
                ),
            ),
        )

    @abstractmethod
    def extract_text(self, content: bytes) -> str:
        """Extract plain text from document content for legacy callers."""
        raise NotImplementedError
