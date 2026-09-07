from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.ai.chunking.base import DocumentChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.config import settings
from app.core.document.models import Chunk, ParsedDocument
from app.core.document.normalizer import DocumentNormalizer
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.exceptions import (
    AIServiceException,
    DatabaseException,
    ResourceNotFoundException,
    TransientAIServiceException,
    TransientDatabaseException,
    TransientStorageException,
    ValidationException,
)
from app.core.storage.base import StorageService
from app.db.classification import classify_database_exception
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.enums import DocumentStatus
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository

logger = logging.getLogger(__name__)

# Keep stored error text bounded so a pathological traceback cannot bloat rows.
_MAX_ERROR_LENGTH: Final[int] = 4000


class DocumentIngestionService:
    """Application service responsible for asynchronous document ingestion."""

    def __init__(
        self,
        *,
        db: Session,
        document_repository: DocumentRepository,
        chunk_repository: DocumentChunkRepository,
        parser_registry: DocumentParserRegistry,
        chunker: DocumentChunker,
        embedding_generator: EmbeddingGenerator,
        storage: StorageService,
        normalizer: DocumentNormalizer | None = None,
        processing_timeout_seconds: int | None = None,
    ) -> None:
        self._db = db
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_generator = embedding_generator
        self._storage = storage
        self._normalizer = (
            normalizer if normalizer is not None else DocumentNormalizer()
        )
        timeout_sec = (
            processing_timeout_seconds
            if processing_timeout_seconds is not None
            else settings.document_processing_timeout_seconds
        )
        self._processing_timeout = timedelta(seconds=timeout_sec)

    def _get_document_for_update(self, *, document_id: UUID) -> Document:
        """Lock and retrieve the document that should be processed."""

        document = self._document_repository.get_for_update(document_id=document_id)

        if document is None:
            raise ResourceNotFoundException(
                f"Document with ID {document_id} not found."
            )

        return document

    def _mark_processing(self, *, document: Document, token: UUID) -> None:
        """Mark the document as currently being processed with an active fencing token."""

        now = datetime.now(UTC)
        document.status = DocumentStatus.PROCESSING
        document.processing_started_at = now
        document.processing_token = token
        document.processed_at = None
        document.last_error = None

    def _mark_ready(self, *, document: Document) -> None:
        """Mark the document as successfully processed, clearing the claim token."""

        document.status = DocumentStatus.READY
        document.processed_at = datetime.now(UTC)
        document.processing_started_at = None
        document.processing_token = None
        document.last_error = None

    def _mark_failed(self, *, document: Document, error: str) -> None:
        """Mark the document as failed and clear the claim token."""

        document.status = DocumentStatus.FAILED
        document.last_error = error[:_MAX_ERROR_LENGTH]
        document.retry_count = int(document.retry_count or 0) + 1
        document.processing_started_at = None
        document.processing_token = None

    @staticmethod
    def _ensure_document_has_content(*, document: ParsedDocument) -> None:
        """Ensure the parser produced meaningful content."""

        if not document.blocks or not any(
            block.text.strip() for block in document.blocks
        ):
            raise ValidationException(
                "The document parser did not extract any text from the document."
            )

    @staticmethod
    def _ensure_embeddings_match_chunks(
        *,
        chunks: Sequence[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Ensure every chunk has exactly one embedding."""

        if len(chunks) != len(embeddings):
            raise AIServiceException("Chunk and embedding count mismatch")

    def _ensure_chunks_persisted(
        self,
        *,
        document_id: UUID,
        expected_count: int,
    ) -> None:
        """Ensure the expected number of chunks were written to the database."""

        persisted_count = self._chunk_repository.count_by_document_id(
            document_id=document_id
        )
        if persisted_count != expected_count:
            raise DatabaseException(
                "Persisted chunk count mismatch: "
                f"expected {expected_count}, got {persisted_count}"
            )

    def _create_chunks(
        self,
        *,
        document: Document,
        chunks: Sequence[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        """Persist the chunks and their embeddings in the database."""

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            document_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
                char_start=None,
                char_end=None,
                chunk_metadata={
                    "source_block_indexes": list(chunk.source_block_indexes),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                },
            )

            self._chunk_repository.create(document_chunk)

    def _claim_document(self, *, document: Document) -> tuple[bool, UUID | None]:
        """
        Decide whether this worker should process the document.

        Returns (True, token) when the document was claimed for processing.
        """

        if document.status == DocumentStatus.READY:
            chunk_count = self._chunk_repository.count_by_document_id(
                document_id=document.id
            )
            if chunk_count > 0:
                logger.info(
                    "Document already processed",
                    extra={
                        "document_id": str(document.id),
                        "chunk_count": chunk_count,
                    },
                )
                return False, None

        now = datetime.now(UTC)

        # Explicit state meaning:
        # PROCESSING + last_error != NULL means the document is awaiting a Celery retry attempt,
        # not necessarily that a worker is currently actively executing it.
        # If status is PROCESSING and last_error IS None, a worker is actively executing.
        if (
            document.status == DocumentStatus.PROCESSING
            and document.processing_started_at is not None
            and document.last_error is None
        ):
            if (now - document.processing_started_at) < self._processing_timeout:
                logger.info(
                    "Document already processing",
                    extra={"document_id": str(document.id)},
                )
                return False, None

            logger.warning(
                "Reclaiming stale processing document (zombie recovery)",
                extra={
                    "document_id": str(document.id),
                    "started_at": str(document.processing_started_at),
                    "timeout_seconds": self._processing_timeout.total_seconds(),
                },
            )

        token = uuid4()
        self._mark_processing(document=document, token=token)
        self._db.commit()
        return True, token

    def _record_transient_failure(
        self, *, document_id: UUID, token: UUID | None, error: str
    ) -> None:
        """Best-effort persistence of transient failure metadata while keeping document PROCESSING."""

        if token is None:
            return

        try:
            document = self._get_document_for_update(document_id=document_id)
            if (
                document is None
                or document.status != DocumentStatus.PROCESSING
                or document.processing_token != token
            ):
                logger.warning(
                    "Transient error not recorded: claim superseded or missing",
                    extra={
                        "document_id": str(document_id),
                        "expected_token": str(token),
                        "current_token": str(document.processing_token)
                        if document
                        else None,
                    },
                )
                self._db.rollback()
                return

            document.status = DocumentStatus.PROCESSING
            document.last_error = error[:_MAX_ERROR_LENGTH]
            document.retry_count = int(document.retry_count or 0) + 1
            # processing_token remains token (invariant: NOT NULL on PROCESSING)
            self._db.commit()
        except Exception:
            self._db.rollback()
            logger.exception(
                "Failed to record transient document error",
                extra={"document_id": str(document_id)},
            )

    def _fail_document(
        self, *, document_id: UUID, token: UUID | None, error: str
    ) -> None:
        """Best-effort transition of a document to FAILED after a permanent failure or rollback."""

        if token is None:
            return

        try:
            document = self._get_document_for_update(document_id=document_id)
            if (
                document is None
                or document.status == DocumentStatus.READY
                or document.processing_token != token
            ):
                logger.warning(
                    "Permanent failure not recorded: claim superseded or already READY",
                    extra={
                        "document_id": str(document_id),
                        "expected_token": str(token),
                        "current_token": str(document.processing_token)
                        if document
                        else None,
                    },
                )
                self._db.rollback()
                return

            self._mark_failed(document=document, error=error)
            self._db.commit()
        except Exception:
            self._db.rollback()
            logger.exception(
                "Failed to mark document as failed",
                extra={"document_id": str(document_id)},
            )

    def process_document(
        self,
        *,
        document_id: UUID,
        on_token_claimed: Callable[[UUID], None] | None = None,
    ) -> UUID | None:
        """
        Process a persisted document into searchable vector chunks.

        The document must already exist in the database and its file
        must already exist in storage.

        If on_token_claimed is provided, it is invoked immediately after the
        document is atomically claimed in the database and before parsing/chunking,
        allowing callers (e.g. worker tasks) to record the processing token for failure fencing.
        """

        started_at = time.perf_counter()
        token: UUID | None = None

        try:
            document = self._get_document_for_update(document_id=document_id)

            claimed, token = self._claim_document(document=document)
            if not claimed or token is None:
                return None

            if on_token_claimed is not None:
                on_token_claimed(token)

            logger.info(
                "Document ingestion started",
                extra={
                    "document_id": str(document.id),
                    "status": document.status.value,
                    "retry_count": document.retry_count,
                    "token": str(token),
                },
            )

            content = self._storage.read(document.storage_path)

            extension = Path(document.filename).suffix.lower()

            parser = self._parser_registry.get_parser(extension=extension)

            logger.info(
                "Document parser selected",
                extra={
                    "document_id": str(document.id),
                    "extension": extension,
                    "parser": type(parser).__name__,
                },
            )

            parsed_document = parser.parse(content=content)
            normalized_document = self._normalizer.normalize(parsed_document)

            self._ensure_document_has_content(document=normalized_document)

            chunks = self._chunker.split(document=normalized_document)

            if not chunks:
                raise ValidationException(
                    "The document chunker did not produce any chunks from the document."
                )

            logger.info(
                "Document chunks generated",
                extra={
                    "document_id": str(document.id),
                    "chunk_count": len(chunks),
                },
            )

            chunk_texts = [chunk.content for chunk in chunks]
            embedding_started_at = time.perf_counter()
            embeddings = self._embedding_generator.embed_documents(chunk_texts)
            embedding_duration_ms = int(
                (time.perf_counter() - embedding_started_at) * 1000
            )

            self._ensure_embeddings_match_chunks(chunks=chunks, embeddings=embeddings)

            # Final persistence transaction: re-acquire row lock and verify processing_token
            document = self._get_document_for_update(document_id=document_id)
            if (
                document is None
                or document.status != DocumentStatus.PROCESSING
                or document.processing_token != token
            ):
                logger.warning(
                    "Document processing claim expired or superseded by another worker; aborting persistence",
                    extra={
                        "document_id": str(document_id),
                        "expected_token": str(token),
                        "current_token": str(document.processing_token)
                        if document
                        else None,
                    },
                )
                self._db.rollback()
                return None

            self._chunk_repository.delete_by_document_id(document_id=document.id)

            self._create_chunks(
                document=document,
                chunks=chunks,
                embeddings=embeddings,
            )

            # Flush so the subsequent count query sees the new rows.
            self._db.flush()
            self._ensure_chunks_persisted(
                document_id=document.id,
                expected_count=len(chunks),
            )

            self._mark_ready(document=document)

            self._db.commit()

        except ResourceNotFoundException:
            self._db.rollback()
            raise

        except Exception as exc:
            self._db.rollback()

            error_to_raise: Exception = exc
            if isinstance(exc, SQLAlchemyError):
                error_to_raise = classify_database_exception(exc)

            is_transient = isinstance(
                error_to_raise,
                (
                    TransientAIServiceException,
                    TransientDatabaseException,
                    TransientStorageException,
                ),
            )

            if is_transient:
                self._record_transient_failure(
                    document_id=document_id, token=token, error=str(error_to_raise)
                )
            else:
                self._fail_document(
                    document_id=document_id, token=token, error=str(error_to_raise)
                )

            logger.exception(
                "Document ingestion failed",
                extra={
                    "document_id": str(document_id),
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                    "is_transient": is_transient,
                    "token": str(token) if token else None,
                },
            )

            if error_to_raise is not exc:
                raise error_to_raise from exc
            raise

        duration_ms = int((time.perf_counter() - started_at) * 1000)

        logger.info(
            "Document ingestion completed",
            extra={
                "document_id": str(document.id),
                "chunk_count": len(chunks),
                "duration_ms": duration_ms,
                "embedding_time_ms": embedding_duration_ms,
            },
        )
        return token
