from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.ai.chunking.base import TextChunker
from app.core.config import settings


class RecursiveTextChunker(TextChunker):
    """Chunk text using Langchain recursive character splitter"""

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks using Langchain recursive character splitter"""
        return self._splitter.split_text(text)
