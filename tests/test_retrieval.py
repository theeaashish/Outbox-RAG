from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.core.ai.context.assembler import ContextAssembler
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.ai.retrieval.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
)
from app.core.ai.retrieval.models import ChunkProvenance, RetrievedChunk
from app.core.constants import MAX_RETRIEVAL_LIMIT
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
    TransientAIServiceException,
    ValidationException,
)
from app.db.models import Document, DocumentChunk, KnowledgeBase
from app.modules.retrieval.controller import RetrievalController
from app.modules.retrieval.schemas import RetrievalRequest, RetrievalResponse
from app.modules.retrieval.service import RetrievalService
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository

# ==============================================================================
# 1. RetrievedChunk Model & Canonical Score Semantics
# ==============================================================================


def test_retrieved_chunk_canonical_semantics():
    doc = Document(id=uuid4(), title="Test Doc", filename="test.pdf")
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=doc.id,
        chunk_index=0,
        content="Sample content",
        document=doc,
        chunk_metadata={"page_start": 2, "page_end": 4, "source_block_indexes": [0, 1]},
    )

    retrieved = RetrievedChunk(chunk=chunk, similarity=0.95)
    assert retrieved.chunk == chunk
    assert retrieved.similarity == 0.95
    # Provenance is a typed property view
    assert isinstance(retrieved.provenance, ChunkProvenance)
    assert retrieved.provenance.page_start == 2
    assert retrieved.provenance.page_end == 4
    assert retrieved.provenance.source_block_indexes == (0, 1)


# ==============================================================================
# 2. ChunkProvenance Trust Boundary & Strict Validation
# ==============================================================================


def test_chunk_provenance_from_none_or_empty_metadata():
    p_none = ChunkProvenance.from_metadata(None)
    assert p_none.page_start is None
    assert p_none.page_end is None
    assert p_none.source_block_indexes == ()

    p_empty = ChunkProvenance.from_metadata({})
    assert p_empty.page_start is None
    assert p_empty.page_end is None
    assert p_empty.source_block_indexes == ()


def test_chunk_provenance_valid_metadata():
    meta = {
        "page_start": 1,
        "page_end": 3,
        "source_block_indexes": [0, 2, 5],
    }
    provenance = ChunkProvenance.from_metadata(meta)
    assert provenance.page_start == 1
    assert provenance.page_end == 3
    assert provenance.source_block_indexes == (0, 2, 5)


def test_chunk_provenance_missing_optional_fields():
    meta = {"source_block_indexes": [1]}
    provenance = ChunkProvenance.from_metadata(meta)
    assert provenance.page_start is None
    assert provenance.page_end is None
    assert provenance.source_block_indexes == (1,)


@pytest.mark.parametrize(
    "invalid_meta, expected_message",
    [
        ("not-a-dict", "expected dictionary or None"),
        (123, "expected dictionary or None"),
        (True, "expected dictionary or None"),
        (
            {"page_start": 1},
            "page_start and page_end must either both be present or both be absent",
        ),
        (
            {"page_end": 2},
            "page_start and page_end must either both be present or both be absent",
        ),
        (
            {"page_start": "1", "page_end": 2},
            "page_start must be a non-negative integer",
        ),
        (
            {"page_start": True, "page_end": 2},
            "page_start must be a non-negative integer",
        ),
        (
            {"page_start": -1, "page_end": 2},
            "page_start must be a non-negative integer",
        ),
        (
            {"page_start": 1, "page_end": "5"},
            "page_end must be a non-negative integer",
        ),
        (
            {"page_start": 1, "page_end": False},
            "page_end must be a non-negative integer",
        ),
        (
            {"page_start": 1, "page_end": -2},
            "page_end must be a non-negative integer",
        ),
        ({"page_start": 5, "page_end": 2}, "page_start cannot exceed page_end"),
        (
            {"source_block_indexes": []},
            "source_block_indexes must be non-empty when present",
        ),
        ({"source_block_indexes": None}, "source_block_indexes must be a sequence"),
        ({"source_block_indexes": "0,1"}, "source_block_indexes must be a sequence"),
        (
            {"source_block_indexes": ["0", "1"]},
            "source_block_indexes must contain non-negative integers",
        ),
        (
            {"source_block_indexes": [1, -2]},
            "source_block_indexes must contain non-negative integers",
        ),
        (
            {"source_block_indexes": [True, 2]},
            "source_block_indexes must contain non-negative integers",
        ),
        (
            {"source_block_indexes": [5, 2]},
            "source_block_indexes must be non-decreasing",
        ),
    ],
)
def test_chunk_provenance_rejects_malformed_metadata(invalid_meta, expected_message):
    with pytest.raises(DatabaseException, match=expected_message):
        ChunkProvenance.from_metadata(invalid_meta)  # type: ignore[arg-type]


# ==============================================================================
# 3. RetrievalService Input Contract & Defensive Validation
# ==============================================================================


def _build_service_with_mocks():
    mock_embedder = MagicMock(spec=EmbeddingGenerator)
    mock_embedder.embed_query.return_value = [0.1] * 768

    mock_chunk_repo = MagicMock(spec=DocumentChunkRepository)
    mock_chunk_repo.search_similar.return_value = []

    mock_kb_repo = MagicMock(spec=KnowledgeBaseRepository)
    mock_kb = KnowledgeBase(
        id=uuid4(), user_id=uuid4(), project_id=uuid4(), name="Test KB"
    )
    mock_kb_repo.get_by_user_and_id.return_value = mock_kb

    service = RetrievalService(
        embedding_generator=mock_embedder,
        chunk_repository=mock_chunk_repo,
        knowledge_base_repository=mock_kb_repo,
    )
    return service, mock_embedder, mock_chunk_repo, mock_kb_repo


@pytest.mark.parametrize(
    "invalid_query",
    [
        "",
        "   ",
        "\t\n  ",
        123,
        None,
    ],
)
def test_retrieval_service_rejects_invalid_queries(invalid_query):
    service, _, _, _ = _build_service_with_mocks()
    with pytest.raises(ValidationException):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query=invalid_query,  # type: ignore[arg-type]
        )


def test_retrieval_service_normalizes_whitespace():
    service, mock_embedder, _, _ = _build_service_with_mocks()
    service.retrieve(
        user_id=uuid4(),
        knowledge_base_id=uuid4(),
        query="  database pooling   \n",
    )
    mock_embedder.embed_query.assert_called_once_with("database pooling")


@pytest.mark.parametrize(
    "invalid_limit",
    [
        0,
        -1,
        MAX_RETRIEVAL_LIMIT + 1,
        True,
        "5",
        2.5,
    ],
)
def test_retrieval_service_rejects_invalid_limits(invalid_limit):
    service, _, _, _ = _build_service_with_mocks()
    with pytest.raises(ValidationException, match="Limit must be an integer"):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query="valid query",
            limit=invalid_limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.01,
        1.01,
        True,
        False,
        "0.5",
        float("nan"),
    ],
)
def test_retrieval_service_rejects_invalid_thresholds(invalid_threshold):
    service, _, _, _ = _build_service_with_mocks()
    with pytest.raises(ValidationException, match="Threshold must be a float"):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query="valid query",
            threshold=invalid_threshold,  # type: ignore[arg-type]
        )


# ==============================================================================
# 4. RetrievalService Authorization & Error Propagation
# ==============================================================================


def test_retrieval_service_authorizes_knowledge_base():
    service, _, _, mock_kb_repo = _build_service_with_mocks()
    mock_kb_repo.get_by_user_and_id.return_value = None

    with pytest.raises(ResourceNotFoundException, match="Knowledge base not found"):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query="valid query",
        )


def test_retrieval_service_propagates_embedding_exceptions():
    service, mock_embedder, _, _ = _build_service_with_mocks()
    mock_embedder.embed_query.side_effect = TransientAIServiceException("Rate limited")

    with pytest.raises(TransientAIServiceException, match="Rate limited"):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query="valid query",
        )

    mock_embedder.embed_query.side_effect = AIServiceException("Permanent model error")
    with pytest.raises(AIServiceException, match="Permanent model error"):
        service.retrieve(
            user_id=uuid4(),
            knowledge_base_id=uuid4(),
            query="valid query",
        )


def test_retrieval_service_returns_empty_list_on_no_matches():
    service, _, mock_chunk_repo, _ = _build_service_with_mocks()
    mock_chunk_repo.search_similar.return_value = []

    results = service.retrieve(
        user_id=uuid4(),
        knowledge_base_id=uuid4(),
        query="no matches query",
    )
    assert results == []


# ==============================================================================
# 5. Controller Mapping & API Schema Validation
# ==============================================================================


@pytest.mark.anyio
async def test_retrieval_controller_response_mapping():
    doc = Document(id=uuid4(), title="Test Doc", filename="test.pdf")
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=doc.id,
        chunk_index=0,
        content="Sample content",
        char_start=None,
        char_end=None,
        document=doc,
        chunk_metadata={"page_start": 3, "page_end": 5, "source_block_indexes": [1, 2]},
    )
    retrieved_item = RetrievedChunk(chunk=chunk, similarity=0.88)

    mock_service = MagicMock()
    mock_service.retrieve.return_value = [retrieved_item]

    controller = RetrievalController(retrieval_service=mock_service)
    response = await controller.retrieve(
        user_id=uuid4(), knowledge_base_id=uuid4(), query="test query", limit=5
    )

    assert isinstance(response, RetrievalResponse)
    assert len(response.results) == 1
    item = response.results[0]
    assert item.document_id == doc.id
    assert item.document_name == "Test Doc"
    assert item.score == 0.88
    assert item.content == "Sample content"
    assert item.page_start == 3
    assert item.page_end == 5
    assert item.char_start is None
    assert item.char_end is None


def test_retrieval_request_schema_normalization_and_validation():
    req = RetrievalRequest(query="  rag architecture  ", limit=10)
    assert req.query == "rag architecture"
    assert req.limit == 10

    with pytest.raises(ValidationError):
        RetrievalRequest(query="    ")

    with pytest.raises(ValidationError):
        RetrievalRequest(query="valid", limit=MAX_RETRIEVAL_LIMIT + 1)

    with pytest.raises(ValidationError):
        RetrievalRequest(query="valid", threshold=1.5)


# ==============================================================================
# 6. ContextAssembler Integration
# ==============================================================================


def test_context_assembler_consumes_typed_provenance():
    doc = Document(id=uuid4(), title="Doc A", filename="doc_a.pdf")
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=doc.id,
        chunk_index=1,
        content="Context text content",
        document=doc,
        chunk_metadata={"page_start": 4, "page_end": 6, "source_block_indexes": [0, 1]},
    )
    retrieved = RetrievedChunk(chunk=chunk, similarity=0.91)

    assembler = ContextAssembler()
    assembled = assembler.assemble(
        query="what is context?", retrieved_chunks=[retrieved]
    )

    assert len(assembled.chunks) == 1
    cc = assembled.chunks[0]
    assert cc.citation == 1
    assert cc.document_name == "Doc A"
    assert cc.chunk_index == 1
    assert cc.similarity == 0.91
    assert cc.page_start == 4
    assert cc.page_end == 6
    assert not hasattr(
        cc, "source_block_indexes"
    )  # Internal block index not in ContextChunk


# ==============================================================================
# 7. Pure Evaluation Metrics (calculate_recall_at_k, calculate_precision_at_k)
# ==============================================================================


def test_retrieval_metrics():
    retrieved = ["chunk_a", "chunk_b", "chunk_c", "chunk_d", "chunk_e"]
    relevant = {"chunk_a", "chunk_c", "chunk_z"}

    # Top-3: retrieved = ['chunk_a', 'chunk_b', 'chunk_c']
    # Intersection with relevant: {'chunk_a', 'chunk_c'} (2 items)
    assert calculate_recall_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)
    assert calculate_precision_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)

    # Top-1: retrieved = ['chunk_a']
    assert calculate_recall_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 3)
    assert calculate_precision_at_k(retrieved, relevant, k=1) == pytest.approx(1 / 1)

    # Edge cases
    assert calculate_recall_at_k(retrieved, set(), k=5) == 0.0
    assert calculate_recall_at_k(retrieved, relevant, k=0) == 0.0
    assert calculate_precision_at_k(retrieved, relevant, k=0) == 0.0
    assert calculate_precision_at_k([], relevant, k=5) == 0.0
