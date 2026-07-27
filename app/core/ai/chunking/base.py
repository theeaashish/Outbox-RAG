from __future__ import annotations

from abc import ABC, abstractmethod


class TextChunker(ABC):
    """Abstract base class for text chunking strategies"""

    @abstractmethod
    def split(self, text: str) -> list[str]:
        """Split text into chunks."""
        raise NotImplementedError
