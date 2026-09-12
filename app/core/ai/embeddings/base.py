from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class EmbeddingGenerator(ABC):
    """Abstract base class for embedding generators.

    Invariants:
    - Queries and documents share the same embedding dimension (EMBEDDING_DIMENSION).
    - Query and document embeddings are semantically distinct entry points.
    - Batches must be non-empty and preserve input order: output[i] corresponds to texts[i].
    - Exactly one output embedding vector is returned per input text.
    """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text.

        Args:
            text: Non-empty query string.

        Returns:
            A list of floats with length equal to EMBEDDING_DIMENSION.

        Raises:
            ValidationException: If text is not a non-empty string.
            TransientAIServiceException: If a transient/retryable provider failure occurs.
            AIServiceException: If a permanent failure or output contract violation occurs.
        """
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for a non-empty batch of document texts.

        Preserves input order: output[i] corresponds to texts[i].

        Args:
            texts: Non-empty sequence of non-empty strings.

        Returns:
            A list of float lists, each with length equal to EMBEDDING_DIMENSION,
            where len(output) == len(texts).

        Raises:
            ValidationException: If texts is empty or contains non-strings or empty strings.
            TransientAIServiceException: If a transient/retryable provider failure occurs.
            AIServiceException: If a permanent failure, output cardinality mismatch, or
                output contract violation occurs.
        """
        raise NotImplementedError
