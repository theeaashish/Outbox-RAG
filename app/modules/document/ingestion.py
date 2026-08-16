from __future__ import annotations

import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.exceptions import (
    AIServiceException,
    ResourceNotFoundException,
    ValidationException,
)
from app.core.storage.base import StorageService
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.enums import DocumentStatus
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository

logger = logging.getLogger(__name__)

# Keep stored error text bounded so a pathological traceback cannot bloat rows.
_MAX_ERROR_LENGTH = 4000


class DocumentIngestionService:
    """Application service responsible for asynchronous document ingestion."""

    def __init__(
        self,
        *,
        db: Session,
        document_repository: DocumentRepository,
        chunk_repository: DocumentChunkRepository,
        parser_registry: DocumentParserRegistry,
        chunker: TextChunker,
        embedding_generator: EmbeddingGenerator,
        storage: StorageService,
    ) -> None:
        self._db = db
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_generator = embedding_generator
        self._storage = storage

    def _get_document_for_update(self, *, document_id: UUID) -> Document:
        """Lock and retrieve the document that should be processed."""

        document = self._document_repository.get_for_update(document_id=document_id)

        if document is None:
            raise ResourceNotFoundException(
                f"Document with ID {document_id} not found."
            )

        return document

    def _mark_processing(self, *, document: Document) -> None:
        """Mark the document as currently being processed."""

        now = datetime.now(UTC)
        document.status = DocumentStatus.PROCESSING
        document.processing_started_at = now
        document.processed_at = None

    def _mark_ready(self, *, document: Document) -> None:
        """Mark the document as successfully processed."""

        document.status = DocumentStatus.READY
        document.processed_at = datetime.now(UTC)
        document.last_error = None

    def _mark_failed(self, *, document: Document, error: str) -> None:
        """Mark the document as failed and record operational metadata."""

        document.status = DocumentStatus.FAILED
        document.last_error = error[:_MAX_ERROR_LENGTH]
        document.retry_count = int(document.retry_count or 0) + 1

    @staticmethod
    def _ensure_text_was_extracted(*, text: str) -> None:
        """Ensure the parser produced meaningful text."""

        if not text.strip():
            raise ValidationException(
                "The document parser did not extract any text from the document."
            )

    @staticmethod
    def _ensure_embeddings_match_chunks(
        *,
        chunks: list[str],
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
            raise AIServiceException(
                "Persisted chunk count mismatch: "
                f"expected {expected_count}, got {persisted_count}"
            )

    def _create_chunks(
        self,
        *,
        document: Document,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Persist the chunks and their embeddings in the database."""

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings, strict=True)
        ):
            document_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                char_start=None,
                char_end=None,
                chunk_metadata=None,
            )

            self._chunk_repository.create(document_chunk)

    def _claim_document(self, *, document: Document) -> bool:
        """
        Decide whether this worker should process the document.

        Returns True when the document was claimed for processing.
        """

        if document.status == DocumentStatus.PROCESSING:
            logger.info(
                "Document already processing",
                extra={"document_id": str(document.id)},
            )
            return False

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
                return False

        self._mark_processing(document=document)
        self._db.commit()
        return True

    def _fail_document(self, *, document_id: UUID, error: str) -> None:
        """Best-effort transition of a document to FAILED after a rollback."""

        try:
            document = self._get_document_for_update(document_id=document_id)
            self._mark_failed(document=document, error=error)
            self._db.commit()
        except Exception:
            self._db.rollback()
            logger.exception(
                "Failed to mark document as failed",
                extra={"document_id": str(document_id)},
            )

    def process_document(self, *, document_id: UUID) -> None:
        """
        Process a persisted document into searchable vector chunks.

        The document must already exist in the database and its file
        must already exist in storage.
        """

        started_at = time.perf_counter()
        document = self._get_document_for_update(document_id=document_id)

        if not self._claim_document(document=document):
            return

        logger.info(
            "Document ingestion started",
            extra={
                "document_id": str(document.id),
                "status": document.status.value,
                "retry_count": document.retry_count,
            },
        )

        try:
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

            text = parser.extract_text(content=content)

            self._ensure_text_was_extracted(text=text)

            chunks = self._chunker.split(text=text)

            if not chunks:
                raise ValidationException(
                    "The document chunker did not produce any chunks from the text."
                )

            logger.info(
                "Document chunks generated",
                extra={
                    "document_id": str(document.id),
                    "chunk_count": len(chunks),
                },
            )

            embedding_started_at = time.perf_counter()
            embeddings = self._embedding_generator.embed_documents(chunks)
            embedding_duration_ms = int(
                (time.perf_counter() - embedding_started_at) * 1000
            )

            self._ensure_embeddings_match_chunks(chunks=chunks, embeddings=embeddings)

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

        except Exception as exc:
            self._db.rollback()
            self._fail_document(document_id=document_id, error=str(exc))

            logger.exception(
                "Document ingestion failed",
                extra={
                    "document_id": str(document_id),
                    "duration_ms": int((time.perf_counter() - started_at) * 1000),
                },
            )

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
