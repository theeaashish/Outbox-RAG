from __future__ import annotations

from abc import ABC, abstractmethod


class DocumentParser(ABC):
    """
    Abstract base class for document parsers.
    """

    @abstractmethod
    def extract_text(self, content: bytes) -> str:
        """Extract plain text from document content"""
        raise NotImplementedError
