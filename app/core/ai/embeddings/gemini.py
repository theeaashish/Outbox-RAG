from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from app.core.ai.classification import classify_ai_exception
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.config import settings
from app.core.constants import EMBEDDING_DIMENSION
from app.core.exceptions import AIServiceException, ValidationException

logger = logging.getLogger(__name__)


class GeminiEmbeddingGenerator(EmbeddingGenerator):
    """Embedding generator backed by Google Gemini embedding model."""

    def __init__(
        self,
        *,
        embeddings: Embeddings | None = None,
        model_name: str | None = None,
    ) -> None:
        self._model_name = model_name or settings.gemini_embedding_model
        if embeddings is not None:
            self._embeddings = embeddings
        else:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=self._model_name,
                api_key=SecretStr(settings.google_api_key),
                output_dimensionality=EMBEDDING_DIMENSION,
            )

    @staticmethod
    def _validate_vector(embedding: Any) -> list[float]:
        """Validate embedding vector structure, dimension, and numeric finiteness.

        Treats missing, non-numeric, wrong-dimension, or non-finite values
        as permanent contract violations.
        """
        if (
            embedding is None
            or not isinstance(embedding, Sequence)
            or isinstance(embedding, (str, bytes))
        ):
            raise AIServiceException(
                "Provider returned malformed non-sequence embedding vector"
            )

        if len(embedding) != EMBEDDING_DIMENSION:
            raise AIServiceException(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, "
                f"got {len(embedding)}"
            )

        validated: list[float] = []
        for val in embedding:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise AIServiceException(
                    f"Embedding vector contains non-numeric value: {val!r}"
                )
            fval = float(val)
            if not math.isfinite(fval):
                raise AIServiceException(
                    f"Embedding vector contains non-finite value: {fval}"
                )
            validated.append(fval)

        return validated

    def embed_query(self, text: str) -> list[float]:
        """Generate an embedding for a single query text."""
        if not isinstance(text, str):
            raise ValidationException("Query text must be a string")
        if not text.strip():
            raise ValidationException("Query text cannot be empty")

        try:
            raw_embedding = self._embeddings.embed_query(text)
            result = self._validate_vector(raw_embedding)
        except (ValidationException, AIServiceException):
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed",
                extra={"model": self._model_name, "count": 1},
            )
            raise classify_ai_exception(
                exc,
                transient_message="Embedding service temporarily unavailable",
                permanent_message="Embedding generation failed",
            ) from exc

        logger.info(
            "Embedding generated",
            extra={"model": self._model_name, "count": 1},
        )
        return result

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate embeddings for multiple document chunks in batch.

        Guarantees input order preservation: output[i] corresponds to texts[i].
        """
        if not isinstance(texts, Sequence) or isinstance(texts, (str, bytes)):
            raise ValidationException("Document batch must be a sequence of strings")
        if not texts:
            raise ValidationException("Document batch cannot be empty")

        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValidationException(f"Document at index {idx} must be a string")
            if not text.strip():
                raise ValidationException(f"Document at index {idx} cannot be empty")

        text_list = list(texts)

        try:
            raw_embeddings = self._embeddings.embed_documents(text_list)
            if (
                raw_embeddings is None
                or not isinstance(raw_embeddings, Sequence)
                or isinstance(raw_embeddings, (str, bytes))
            ):
                raise AIServiceException(
                    "Provider returned invalid embedding batch format"
                )

            if len(raw_embeddings) != len(text_list):
                raise AIServiceException(
                    f"Embedding batch cardinality mismatch: expected {len(text_list)} embeddings, "
                    f"got {len(raw_embeddings)}"
                )

            result = [self._validate_vector(emb) for emb in raw_embeddings]
        except (ValidationException, AIServiceException):
            raise
        except Exception as exc:
            logger.exception(
                "Embedding generation failed",
                extra={"model": self._model_name, "count": len(text_list)},
            )
            raise classify_ai_exception(
                exc,
                transient_message="Embedding service temporarily unavailable",
                permanent_message="Embedding generation failed",
            ) from exc

        logger.info(
            "Embeddings generated",
            extra={"model": self._model_name, "count": len(text_list)},
        )
        return result
