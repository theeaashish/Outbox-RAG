from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from google.genai.errors import APIError, ClientError, ServerError
from langchain_core.embeddings import Embeddings

from app.core.ai.embeddings.gemini import GeminiEmbeddingGenerator
from app.core.constants import EMBEDDING_DIMENSION
from app.core.exceptions import (
    AIServiceException,
    TransientAIServiceException,
    ValidationException,
)
from app.db.models import DocumentChunk
from app.repositories.document_chunk import DocumentChunkRepository


def _make_vector(dim: int = EMBEDDING_DIMENSION, val: float = 0.05) -> list[float]:
    """Helper to generate a valid float vector of specified dimension."""
    return [val] * dim


_UNSET = object()


class _MockEmbeddings(Embeddings):
    """Mock LangChain Embeddings implementation for contract testing."""

    def __init__(
        self,
        query_vector: Any = _UNSET,
        doc_vectors: Any = _UNSET,
    ) -> None:
        self.query_vector = query_vector
        self.doc_vectors = doc_vectors

    def embed_query(self, text: str) -> list[float]:
        if isinstance(self.query_vector, Exception):
            raise self.query_vector
        if self.query_vector is _UNSET:
            return _make_vector()
        return self.query_vector  # type: ignore[reportReturnType]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if isinstance(self.doc_vectors, Exception):
            raise self.doc_vectors
        if self.doc_vectors is _UNSET:
            return [_make_vector(val=0.01 * (i + 1)) for i in range(len(texts))]
        return self.doc_vectors  # type: ignore[reportReturnType]


def test_embedding_dimension_single_shared_constant():
    """Verify EMBEDDING_DIMENSION is 768 and shared by model and generator."""
    assert EMBEDDING_DIMENSION == 768
    # DocumentChunk.embedding column type dimension should match constant
    assert DocumentChunk.embedding.type.dim == EMBEDDING_DIMENSION  # type: ignore[reportUnknownMemberType]


# ==============================================================================
# Query Input Semantics
# ==============================================================================


@pytest.mark.parametrize(
    "invalid_query",
    [
        "",
        "   ",
        "\t\n  \r",
    ],
)
def test_embed_query_rejects_empty_or_whitespace_strings(invalid_query: str):
    generator = GeminiEmbeddingGenerator(embeddings=_MockEmbeddings())
    with pytest.raises(ValidationException, match="Query text cannot be empty"):
        generator.embed_query(invalid_query)


@pytest.mark.parametrize(
    "invalid_query",
    [
        None,
        123,
        ["query in a list"],
        {"query": "text"},
    ],
)
def test_embed_query_rejects_non_string_types(invalid_query: Any):
    generator = GeminiEmbeddingGenerator(embeddings=_MockEmbeddings())
    with pytest.raises(ValidationException, match="Query text must be a string"):
        generator.embed_query(invalid_query)  # type: ignore[reportArgumentType]


def test_embed_query_happy_path():
    expected_vector = _make_vector(val=0.42)
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=expected_vector)
    )
    result = generator.embed_query("valid search query")
    assert result == expected_vector
    assert len(result) == EMBEDDING_DIMENSION


# ==============================================================================
# Batch Document Input Semantics
# ==============================================================================


def test_embed_documents_rejects_empty_batch():
    """Choice 1: Reject empty document batches."""
    generator = GeminiEmbeddingGenerator(embeddings=_MockEmbeddings())
    with pytest.raises(ValidationException, match="Document batch cannot be empty"):
        generator.embed_documents([])


@pytest.mark.parametrize(
    "invalid_batch",
    [
        "not a list",
        b"raw bytes",
        None,
        12345,
    ],
)
def test_embed_documents_rejects_non_sequences(invalid_batch: Any):
    generator = GeminiEmbeddingGenerator(embeddings=_MockEmbeddings())
    with pytest.raises(
        ValidationException, match="Document batch must be a sequence of strings"
    ):
        generator.embed_documents(invalid_batch)  # type: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "invalid_batch",
    [
        ["valid chunk", ""],
        ["  ", "valid chunk"],
        ["valid chunk", None],
        [123, "valid chunk"],
    ],
)
def test_embed_documents_rejects_empty_or_non_string_elements(
    invalid_batch: list[Any],
):
    generator = GeminiEmbeddingGenerator(embeddings=_MockEmbeddings())
    with pytest.raises(ValidationException):
        generator.embed_documents(invalid_batch)  # type: ignore[reportArgumentType]


# ==============================================================================
# Output Integrity and Contract Validation
# ==============================================================================


def test_embed_query_rejects_wrong_dimension():
    """Verify dimension mismatch on query is treated as permanent AIServiceException."""
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=_make_vector(dim=512))
    )
    with pytest.raises(AIServiceException, match="Embedding dimension mismatch"):
        generator.embed_query("query")


def test_embed_documents_rejects_wrong_dimension():
    """Verify dimension mismatch on batch is treated as permanent AIServiceException."""
    bad_vectors = [
        _make_vector(dim=EMBEDDING_DIMENSION),
        _make_vector(dim=1024),
    ]
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=bad_vectors)
    )
    with pytest.raises(AIServiceException, match="Embedding dimension mismatch"):
        generator.embed_documents(["doc1", "doc2"])


def test_embed_documents_rejects_cardinality_mismatch():
    """Batch output cardinality must exactly match input count."""
    # Input has 2 texts, provider returns 3 vectors
    extra_vectors = [_make_vector() for _ in range(3)]
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=extra_vectors)
    )
    with pytest.raises(
        AIServiceException, match="Embedding batch cardinality mismatch"
    ):
        generator.embed_documents(["doc1", "doc2"])

    # Input has 2 texts, provider returns 1 vector
    missing_vectors = [_make_vector()]
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=missing_vectors)
    )
    with pytest.raises(
        AIServiceException, match="Embedding batch cardinality mismatch"
    ):
        generator.embed_documents(["doc1", "doc2"])


@pytest.mark.parametrize(
    "malformed_output",
    [
        None,
        "not a list of vectors",
        123,
    ],
)
def test_embed_documents_rejects_malformed_batch_output(malformed_output: Any):
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=malformed_output)
    )
    with pytest.raises(AIServiceException):
        generator.embed_documents(["doc1"])


@pytest.mark.parametrize(
    "bad_element",
    [
        "string_instead_of_number",
        True,
        False,
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_validate_vector_rejects_non_numeric_and_non_finite_values(bad_element: Any):
    vector = _make_vector()
    vector[10] = bad_element
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=vector)
    )
    with pytest.raises(AIServiceException, match="non-numeric|non-finite"):
        generator.embed_query("query")


def test_order_preservation_invariant():
    """Inputs[i] -> Embeddings[i] invariant must strictly hold."""
    vec_0 = _make_vector(val=0.10)
    vec_1 = _make_vector(val=0.20)
    vec_2 = _make_vector(val=0.30)

    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=[vec_0, vec_1, vec_2])
    )
    results = generator.embed_documents(["chunk_0", "chunk_1", "chunk_2"])

    assert len(results) == 3
    assert results[0] == vec_0
    assert results[1] == vec_1
    assert results[2] == vec_2


# ==============================================================================
# Provider Error Classification & Cause Preservation
# ==============================================================================


def test_transient_error_classification_preserves_cause():
    """Transient errors (429, 503, timeouts) raise TransientAIServiceException with __cause__."""
    # Test HTTP 429 Rate limit
    provider_error = APIError(429, {"message": "Resource exhausted"})
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=provider_error)
    )
    with pytest.raises(TransientAIServiceException) as exc_info:
        generator.embed_query("query")
    assert exc_info.value.__cause__ is provider_error

    # Test HTTP 503 Server Error
    server_error = ServerError(503, {"message": "Service unavailable"})
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=server_error)
    )
    with pytest.raises(TransientAIServiceException) as exc_info:
        generator.embed_documents(["doc"])
    assert exc_info.value.__cause__ is server_error

    # Test httpx Network Timeout
    timeout_error = httpx.ConnectTimeout("Connection timed out")
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=timeout_error)
    )
    with pytest.raises(TransientAIServiceException) as exc_info:
        generator.embed_query("query")
    assert exc_info.value.__cause__ is timeout_error


def test_permanent_error_classification_preserves_cause():
    """Permanent errors (400, 401, 404) raise AIServiceException with __cause__."""
    client_error = ClientError(400, {"message": "Invalid argument"})
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(query_vector=client_error)
    )
    with pytest.raises(AIServiceException) as exc_info:
        generator.embed_query("query")
    # Must NOT be transient
    assert not isinstance(exc_info.value, TransientAIServiceException)
    assert exc_info.value.__cause__ is client_error


def test_no_provider_specific_exceptions_leak():
    """Google or LangChain internal exceptions must not leak to caller."""
    gemini_error = ClientError(401, {"message": "Unauthenticated"})
    generator = GeminiEmbeddingGenerator(
        embeddings=_MockEmbeddings(doc_vectors=gemini_error)
    )
    with pytest.raises(AIServiceException):
        generator.embed_documents(["doc"])


# ==============================================================================
# Repository search_similar Contract
# ==============================================================================


def test_search_similar_rejects_wrong_dimension():
    db = MagicMock()
    repo = DocumentChunkRepository(db=db)
    wrong_dim_vec = _make_vector(dim=512)

    with pytest.raises(ValidationException, match="Embedding dimension mismatch"):
        repo.search_similar(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            embedding=wrong_dim_vec,
        )


def test_search_similar_rejects_non_positive_limit():
    db = MagicMock()
    repo = DocumentChunkRepository(db=db)
    valid_vec = _make_vector()

    with pytest.raises(ValidationException, match="Limit must be positive"):
        repo.search_similar(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            embedding=valid_vec,
            limit=0,
        )
