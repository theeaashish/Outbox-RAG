from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.config import settings


class GeminiEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator backed by Google gemini embedding model"""

    def __init__(self) -> None:
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            api_key=settings.google_api_key,
        )

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query"""
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple document"""
        return self._embeddings.embed_documents(texts)
