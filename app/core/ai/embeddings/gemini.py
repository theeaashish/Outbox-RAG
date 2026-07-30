from __future__ import annotations

import logging

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.config import settings
from app.core.constants import EMBEDDING_DIMENSION
from app.core.exceptions import AIServiceException

logger = logging.getLogger(__name__)


class GeminiEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator backed by Google gemini embedding model"""

    def __init__(self) -> None:
        self._model_name = settings.gemini_embedding_model
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model_name,
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

        try:
            embedding = self._embeddings.embed_query(text)
            result = self._validate_dimension(embedding)
        except AIServiceException:
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed",
                extra={"model": self._model_name, "count": 1},
            )
            raise AIServiceException("Embedding generation failed") from exc

        logger.info(
            "Embedding generated",
            extra={"model": self._model_name, "count": 1},
        )
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple document"""

        try:
            embeddings = self._embeddings.embed_documents(texts)
            for embedding in embeddings:
                self._validate_dimension(embedding)
        except AIServiceException:
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed",
                extra={"model": self._model_name, "count": len(texts)},
            )
            raise AIServiceException("Embedding generation failed") from exc

        logger.info(
            "Embeddings generated",
            extra={"model": self._model_name, "count": len(texts)},
        )
        return embeddings
