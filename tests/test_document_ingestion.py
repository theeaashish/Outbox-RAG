from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import psycopg.errors
import pytest
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
    StorageException,
    TransientAIServiceException,
    TransientDatabaseException,
    TransientStorageException,
    ValidationException,
)
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
    storage_read_side_effect: Exception | None = None,
    db_flush_side_effect: Exception | None = None,
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

    if storage_read_side_effect is not None:
        storage.read.side_effect = storage_read_side_effect
    else:
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

    if db_flush_side_effect is not None:
        db.flush.side_effect = db_flush_side_effect

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
        "processing_token": None,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_skips_document_already_processing_when_active() -> None:
    now = datetime.now(UTC)
    active_token = uuid4()
    document = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_started_at=now,
        processing_token=active_token,
        last_error=None,
    )
    service, db, document_repository, chunk_repository = _build_service(
        document=document
    )

    token = service.process_document(document_id=document.id)

    assert token is None
    document_repository.get_for_update.assert_called_once_with(document_id=document.id)
    db.commit.assert_not_called()
    chunk_repository.create.assert_not_called()
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == active_token


def test_claim_document_state_machine_transitions() -> None:
    now = datetime.now(UTC)

    # 1. PENDING document -> claimed, last_error cleared, processing_started_at set, token assigned
    pending_doc = _pending_document(status=DocumentStatus.PENDING, last_error="old")
    service, _db, _, _ = _build_service(document=pending_doc)
    claimed, token = service._claim_document(document=cast(Any, pending_doc))
    assert claimed is True
    assert token is not None
    assert pending_doc.status == DocumentStatus.PROCESSING
    assert pending_doc.processing_started_at is not None
    assert pending_doc.processing_token == token
    assert pending_doc.last_error is None

    # 2. Actively processing document -> skipped
    active_token = uuid4()
    active_doc = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_started_at=now,
        processing_token=active_token,
        last_error=None,
    )
    claimed, token = service._claim_document(document=cast(Any, active_doc))
    assert claimed is False
    assert token is None
    assert active_doc.processing_token == active_token

    # 3. Retrying document (waiting for retry: processing_started_at is set, last_error is set)
    old_token = uuid4()
    retrying_doc = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_started_at=now,
        processing_token=old_token,
        last_error="Temporary 429",
    )
    claimed, token = service._claim_document(document=cast(Any, retrying_doc))
    assert claimed is True
    assert token is not None
    assert token != old_token
    assert retrying_doc.status == DocumentStatus.PROCESSING
    assert retrying_doc.processing_started_at is not None
    assert retrying_doc.processing_token == token
    assert retrying_doc.last_error is None


def test_skips_document_already_ready_with_chunks() -> None:
    document = _pending_document(status=DocumentStatus.READY)
    service, db, _, chunk_repository = _build_service(
        document=document,
        chunk_count=3,
    )

    token = service.process_document(document_id=document.id)

    assert token is None
    db.commit.assert_not_called()
    chunk_repository.create.assert_not_called()
    assert document.status == DocumentStatus.READY


def test_claims_processing_and_persists_immediately() -> None:
    document = _pending_document()
    service, db, _, chunk_repository = _build_service(document=document)

    token = service.process_document(document_id=document.id)

    assert token is not None
    # First commit is the PROCESSING claim; second is READY + chunks.
    assert db.commit.call_count == 2
    assert document.status == DocumentStatus.READY
    assert document.last_error is None
    assert document.processed_at is not None
    assert document.processing_started_at is None
    assert document.processing_token is None
    assert chunk_repository.create.call_count == 2
    chunk_repository.delete_by_document_id.assert_called_once_with(
        document_id=document.id
    )


def test_permanent_ai_failure_marks_failed_and_increments_retry_count() -> None:
    document = _pending_document(retry_count=1)
    service, db, document_repository, _ = _build_service(
        document=document,
        embed_side_effect=AIServiceException("Invalid API key / 401 Unauthorized"),
    )

    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(AIServiceException, match="Invalid API key"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert "Invalid API key" in (document.last_error or "")
    assert document.retry_count == 2
    assert document.processing_started_at is None
    assert db.commit.call_count == 2
    db.rollback.assert_called()


def test_transient_ai_failure_keeps_processing_and_increments_retry_count() -> None:
    document = _pending_document(retry_count=0)
    service, db, document_repository, _ = _build_service(
        document=document,
        embed_side_effect=TransientAIServiceException(
            "Rate limited / 429 Too Many Requests"
        ),
    )

    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(TransientAIServiceException, match="Rate limited"):
        service.process_document(document_id=document.id)

    # Should remain PROCESSING with processing_started_at preserved for Celery retry & zombie detection!
    assert document.status == DocumentStatus.PROCESSING
    assert "Rate limited" in (document.last_error or "")
    assert document.retry_count == 1
    assert document.processing_started_at is not None
    assert db.commit.call_count == 2
    db.rollback.assert_called()


def test_retry_after_transient_failure_succeeds_and_marks_ready() -> None:
    # State after a transient failure: status=PROCESSING, retry_count=1, last_error set, processing_started_at preserved
    document = _pending_document(
        status=DocumentStatus.PROCESSING,
        retry_count=1,
        last_error="Rate limited / 429",
        processing_started_at=datetime.now(UTC),
    )
    service, db, _, chunk_repository = _build_service(document=document)

    # Retry should claim the document and process it to READY
    service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.READY
    assert document.last_error is None
    assert document.processed_at is not None
    assert chunk_repository.create.call_count == 2
    assert db.commit.call_count == 2


def test_file_not_found_permanent_failure() -> None:
    document = _pending_document()
    service, _db, document_repository, _ = _build_service(
        document=document,
        storage_read_side_effect=StorageException("Stored file not found"),
    )
    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(StorageException, match="Stored file not found"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert "Stored file not found" in (document.last_error or "")
    assert document.retry_count == 1


def test_transient_storage_failure_keeps_processing() -> None:
    document = _pending_document()
    service, _db, document_repository, _ = _build_service(
        document=document,
        storage_read_side_effect=TransientStorageException("Disk I/O timeout"),
    )
    document_repository.get_for_update.side_effect = [document, document]

    with pytest.raises(TransientStorageException, match="Disk I/O timeout"):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.PROCESSING
    assert "Disk I/O timeout" in (document.last_error or "")
    assert document.retry_count == 1


def test_transient_database_operational_failure_keeps_processing() -> None:
    document = _pending_document()

    # Simulate psycopg deadlock / operational error wrapped in SQLAlchemy
    psycopg_deadlock = psycopg.errors.DeadlockDetected("deadlock detected")
    sa_op_error = OperationalError("statement", {}, psycopg_deadlock)

    service, _db, _document_repository, _ = _build_service(
        document=document,
        db_flush_side_effect=sa_op_error,
    )

    with pytest.raises(TransientDatabaseException):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.PROCESSING
    assert document.retry_count == 1


def test_permanent_database_integrity_failure_marks_failed() -> None:
    document = _pending_document()

    # Simulate unique constraint violation
    psycopg_unique = psycopg.errors.UniqueViolation(
        "duplicate key value violates unique constraint"
    )
    sa_integrity_error = IntegrityError("statement", {}, psycopg_unique)

    service, _db, _document_repository, _ = _build_service(
        document=document,
        db_flush_side_effect=sa_integrity_error,
    )

    with pytest.raises(DatabaseException):
        service.process_document(document_id=document.id)

    assert document.status == DocumentStatus.FAILED
    assert document.retry_count == 1


def test_raises_when_persisted_chunk_count_mismatches() -> None:
    document = _pending_document()
    service, db, _document_repository, chunk_repository = _build_service(
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

    with pytest.raises(DatabaseException, match="Persisted chunk count mismatch"):
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


def test_transient_database_failure_during_initial_fetch_raises_transient_database_exception() -> (
    None
):
    doc_id = uuid4()
    document = _pending_document(id=doc_id)
    psycopg_conn = psycopg.errors.ConnectionException("connection closed unexpectedly")
    sa_op_error = OperationalError("statement", {}, psycopg_conn)

    service, db, document_repository, _ = _build_service(document=document)
    document_repository.get_for_update.side_effect = sa_op_error

    with pytest.raises(TransientDatabaseException):
        service.process_document(document_id=doc_id)

    assert db.rollback.called


def test_transient_database_failure_during_claim_commit_raises_transient_database_exception() -> (
    None
):
    document = _pending_document()
    psycopg_deadlock = psycopg.errors.DeadlockDetected("deadlock detected on claim")
    sa_op_error = OperationalError("statement", {}, psycopg_deadlock)

    service, db, _document_repository, _ = _build_service(document=document)
    # First get_for_update succeeds, claim commit fails with deadlock
    db.commit.side_effect = sa_op_error

    with pytest.raises(TransientDatabaseException):
        service.process_document(document_id=document.id)

    assert db.rollback.called


def test_resource_not_found_on_fetch_raises_resource_not_found_exception() -> None:
    doc_id = uuid4()
    document = _pending_document(id=doc_id)
    service, db, document_repository, _ = _build_service(document=document)
    document_repository.get_for_update.return_value = None

    with pytest.raises(
        ResourceNotFoundException, match=f"Document with ID {doc_id} not found"
    ):
        service.process_document(document_id=doc_id)

    assert db.rollback.called


def test_zombie_reclamation_after_timeout() -> None:
    from datetime import timedelta

    now = datetime.now(UTC)
    old_started = now - timedelta(seconds=901)
    old_token = uuid4()

    # Zombie document: PROCESSING for > 900s without error
    zombie_doc = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_started_at=old_started,
        processing_token=old_token,
        last_error=None,
    )
    service, _db, _, _ = _build_service(document=zombie_doc)
    claimed, token = service._claim_document(document=cast(Any, zombie_doc))

    assert claimed is True
    assert token is not None
    assert token != old_token
    assert zombie_doc.status == DocumentStatus.PROCESSING
    assert zombie_doc.processing_token == token
    assert zombie_doc.processing_started_at is not None
    assert zombie_doc.processing_started_at > old_started


def test_fenced_persistence_aborts_when_token_superseded() -> None:
    document = _pending_document()
    service, _db, document_repository, chunk_repository = _build_service(
        document=document
    )

    initial_get = True
    superseded_token = uuid4()

    def get_document_side_effect(*, document_id):
        nonlocal initial_get
        if initial_get:
            initial_get = False
            return document
        # Second get_for_update (persistence transaction): simulate another worker reclaimed it!
        document.processing_token = superseded_token
        return document

    document_repository.get_for_update.side_effect = get_document_side_effect

    result_token = service.process_document(document_id=document.id)

    # Persistence should have aborted
    assert result_token is None
    chunk_repository.create.assert_not_called()
    chunk_repository.delete_by_document_id.assert_not_called()
    assert document.processing_token == superseded_token
    assert document.status == DocumentStatus.PROCESSING


def test_stale_worker_fail_document_aborts_when_token_superseded() -> None:
    current_token = uuid4()
    stale_token = uuid4()
    document = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_token=current_token,
        last_error=None,
    )
    service, db, _, _ = _build_service(document=document)

    # Stale worker tries to mark failed with stale_token
    service._fail_document(
        document_id=document.id,
        token=stale_token,
        error="Stale worker failed",
    )

    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == current_token
    assert document.last_error is None
    db.commit.assert_not_called()


def test_stale_worker_record_transient_failure_aborts_when_token_superseded() -> None:
    current_token = uuid4()
    stale_token = uuid4()
    document = _pending_document(
        status=DocumentStatus.PROCESSING,
        processing_token=current_token,
        retry_count=0,
        last_error=None,
    )
    service, db, _, _ = _build_service(document=document)

    # Stale worker tries to record transient failure with stale_token
    service._record_transient_failure(
        document_id=document.id,
        token=stale_token,
        error="Stale transient error",
    )

    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == current_token
    assert document.retry_count == 0
    assert document.last_error is None
    db.commit.assert_not_called()


def test_token_invariants_throughout_lifecycle() -> None:
    # 1. PENDING: processing_token must be NULL
    document = _pending_document(status=DocumentStatus.PENDING)
    assert document.processing_token is None

    service, _db, _, _ = _build_service(document=document)

    # 2. PROCESSING: processing_token must be NOT NULL
    claimed, token = service._claim_document(document=cast(Any, document))
    assert claimed is True
    assert token is not None
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_token == token

    # 3. READY: processing_token must be NULL
    service._mark_ready(document=cast(Any, document))
    assert document.status == DocumentStatus.READY
    assert document.processing_token is None
    assert document.processing_started_at is None

    # Reset to PROCESSING with token
    document.status = DocumentStatus.PROCESSING
    document.processing_token = token
    # 4. FAILED: processing_token must be NULL
    service._mark_failed(document=cast(Any, document), error="Terminal failure")
    assert document.status == DocumentStatus.FAILED
    assert document.processing_token is None
    assert document.processing_started_at is None


def test_on_token_claimed_callback_invoked_on_claim() -> None:
    document = _pending_document()
    service, _db, _, _ = _build_service(document=document)

    captured_tokens: list[UUID] = []

    def callback(token: UUID) -> None:
        captured_tokens.append(token)

    token = service.process_document(
        document_id=document.id,
        on_token_claimed=callback,
    )

    assert token is not None
    assert len(captured_tokens) == 1
    assert captured_tokens[0] == token
