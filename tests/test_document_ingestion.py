from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.exceptions import AIServiceException, ValidationException
from app.db.models.enums import DocumentStatus
from app.modules.document.ingestion import DocumentIngestionService


def _build_service(
    *,
    document: SimpleNamespace,
    chunk_count: int = 0,
    storage_content: bytes = b"%PDF-1.4",
    chunks: list[str] | None = None,
    embeddings: list[list[float]] | None = None,
    embed_side_effect: Exception | None = None,
) -> tuple[DocumentIngestionService, MagicMock, MagicMock, MagicMock]:
    db = MagicMock()
    document_repository = MagicMock()
    chunk_repository = MagicMock()
    parser_registry = MagicMock()
    chunker = MagicMock()
    embedding_generator = MagicMock()
    storage = MagicMock()

    document_repository.get_for_update.return_value = document
    chunk_repository.count_by_document_id.return_value = chunk_count

    storage.read.return_value = storage_content

    parser = MagicMock()
    parser.extract_text.return_value = "extracted document text"
    parser_registry.get_parser.return_value = parser

    produced_chunks = chunks if chunks is not None else ["chunk-one", "chunk-two"]
    chunker.split.return_value = produced_chunks

    if embed_side_effect is not None:
        embedding_generator.embed_documents.side_effect = embed_side_effect
    else:
        produced_embeddings = (
            embeddings
            if embeddings is not None
            else [[0.1, 0.2] for _ in produced_chunks]
        )
        embedding_generator.embed_documents.return_value = produced_embeddings

    # After insert, count should match the produced chunks for the happy path.
    def count_side_effect(*, document_id):
        _ = document_id
        # First calls may be the READY guard; after create, return len(chunks).
        if chunk_repository.create.called:
            return len(produced_chunks)
        return chunk_count

    chunk_repository.count_by_document_id.side_effect = count_side_effect

    service = DocumentIngestionService(
        db=db,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        parser_registry=parser_registry,
        chunker=chunker,
        embedding_generator=embedding_generator,
        storage=storage,
    )
    return service, db, document_repository, chunk_repository


def _pending_document(**overrides) -> SimpleNamespace:
    data = {
        "id": uuid4(),
        "filename": "resume.pdf",
        "storage_path": "kb/hash.pdf",
        "status": DocumentStatus.PENDING,
        "retry_count": 0,
        "last_error": None,
        "processing_started_at": None,
        "processed_at": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_skips_document_already_processing() -> None:
    document = _pending_document(status=DocumentStatus.PROCESSING)
    service, db, document_repository, chunk_repository = _build_service(
        document=document
    )

    service.process_document(document_id=document.id)

    document_repository.get_for_update.assert_called_once_with(document_id=document.id)
    db.commit.assert_not_called()
    chunk_repository.create.assert_not_called()
    assert document.status == DocumentStatus.PROCESSING


def test_skips_document_already_ready_with_chunks() -> None:
    document = _pending_document(status=DocumentStatus.READY)
    service, db, _, chunk_repository = _build_service(
        document=document,
        chunk_count=3,
    )

    service.process_document(document_id=document.id)

    db.commit.assert_not_called()
    chunk_repository.create.assert_not_called()
    assert document.status == DocumentStatus.READY


def test_claims_processing_and_persists_immediately() -> None:
    document = _pending_document()
    service, db, _, chunk_repository = _build_service(document=document)

    service.process_document(document_id=document.id)

    # First commit is the PROCESSING claim; second is READY + chunks.
    assert db.commit.call_count == 2
    assert document.status == DocumentStatus.READY
    assert document.last_error is None
    assert document.processed_at is not None
    assert document.processing_started_at is not None
    assert chunk_repository.create.call_count == 2
    chunk_repository.delete_by_document_id.assert_called_once_with(
        document_id=document.id
    )


def test_marks_failed_with_error_and_increments_retry_count() -> None:
    document = _pending_document(retry_count=1)
    service, db, document_repository, _ = _build_service(
        document=document,
        embed_side_effect=RuntimeError("rate limited by Gemini"),
    )

    # After rollback, failure path re-locks the same document.
    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(RuntimeError, match="rate limited by Gemini"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert document.last_error == "rate limited by Gemini"
    assert document.retry_count == 2
    # Claim commit + failed commit
    assert db.commit.call_count == 2
    db.rollback.assert_called()


def test_raises_when_persisted_chunk_count_mismatches() -> None:
    document = _pending_document()
    service, db, document_repository, chunk_repository = _build_service(
        document=document,
        chunks=["only-one"],
        embeddings=[[0.1]],
    )

    # Force post-insert validation failure.
    def count_side_effect(*, document_id):
        _ = document_id
        if chunk_repository.create.called:
            return 0
        return 0

    chunk_repository.count_by_document_id.side_effect = count_side_effect
    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(AIServiceException, match="Persisted chunk count mismatch"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert "Persisted chunk count mismatch" in (document.last_error or "")
    assert db.rollback.called


def test_raises_when_parser_returns_empty_text() -> None:
    document = _pending_document()
    service, db, document_repository, _ = _build_service(document=document)

    parser = MagicMock()
    parser.extract_text.return_value = "   "
    parser_registry = MagicMock()
    parser_registry.get_parser.return_value = parser
    service._parser_registry = parser_registry
    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(ValidationException, match="did not extract any text"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert db.rollback.called
