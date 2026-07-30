from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.ai.retrieval.models import RetrievedChunk
from app.db.models import Document, DocumentChunk
from app.modules.retrieval.controller import RetrievalController
from app.modules.retrieval.schemas import RetrievalResponse


def test_retrieved_chunk_instantiation():
    doc = Document(id=uuid4(), title="Test Doc", filename="test.pdf")
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=doc.id,
        chunk_index=0,
        content="Sample content",
        char_start=0,
        char_end=14,
        document=doc,
    )
    
    retrieved = RetrievedChunk(chunk=chunk, similarity=0.95)
    assert retrieved.chunk == chunk
    assert retrieved.similarity == 0.95


@pytest.mark.anyio
async def test_retrieval_controller_response_mapping():
    doc = Document(id=uuid4(), title="Test Doc", filename="test.pdf")
    chunk = DocumentChunk(
        id=uuid4(),
        document_id=doc.id,
        chunk_index=0,
        content="Sample content",
        char_start=0,
        char_end=14,
        document=doc,
    )
    retrieved_item = RetrievedChunk(chunk=chunk, similarity=0.88)
    
    mock_service = MagicMock()
    mock_service.retrieve.return_value = [retrieved_item]
    
    controller = RetrievalController(retrieval_service=mock_service)
    response = await controller.retrieve(knowledge_base_id=uuid4(), query="test query", limit=5)
    
    assert isinstance(response, RetrievalResponse)
    assert len(response.results) == 1
    assert response.results[0].document_id == doc.id
    assert response.results[0].document_name == "Test Doc"
    assert response.results[0].score == 0.88
    assert response.results[0].content == "Sample content"
