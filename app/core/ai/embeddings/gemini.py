from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.config import settings
from app.core.constants import EMBEDDING_DIMENSION
from app.core.exceptions import AIServiceException


class GeminiEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator backed by Google gemini embedding model"""

    def __init__(self) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            api_key=settings.google_api_key,
            output_dimensionality=EMBEDDING_DIMENSION,
        )

    @staticmethod
    def _validate_dimension(embedding: list[float]) -> list[float]:
        """Ensure embedding dimension matches expected EMBEDDING_DIMENSION."""
        if len(embedding) != EMBEDDING_DIMENSION:
            raise AIServiceException(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}"
            )
        return embedding

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query"""
        embedding = self._embeddings.embed_query(text)
        return self._validate_dimension(embedding)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple document"""
        embeddings = self._embeddings.embed_documents(texts)
        for embedding in embeddings:
            self._validate_dimension(embedding)
        return embeddings

