from __future__ import annotations

from abc import ABC, abstractmethod


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators"""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query"""

        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple documents"""
        raise NotImplementedError
