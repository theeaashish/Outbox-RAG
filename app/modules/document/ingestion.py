from __future__ import annotations

import logging
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

    def _get_document(self, *, document_id: UUID) -> Document:
        """Retrieve the document that should be processed."""

        document = self._document_repository.get_by_id(document_id)

        if document is None:
            raise ResourceNotFoundException(
                f"Document with ID {document_id} not found."
            )

        return document

    def _mark_processing(self, *, document: Document) -> None:
        """Mark the document as currently being processed."""

        document.status = DocumentStatus.PROCESSING
        self._db.commit()

    def _mark_ready(self, *, document: Document) -> None:
        """Mark the document as successfully processed."""

        document.status = DocumentStatus.READY

    def _mark_failed(self, *, document: Document) -> None:
        """Mark the document as failed."""

        document.status = DocumentStatus.FAILED

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

    def process_document(self, *, document_id: UUID) -> None:
        """
        Process a persisted document into searchable vector chunks.

        The document must already exist in the database and its file
        must already exist in storage.
        """

        document = self._get_document(document_id=document_id)

        logger.info(
            "Document ingestion started",
            extra={
                "document_id": str(document.id),
                "status": document.status.value,
            },
        )

        try:
            self._mark_processing(document=document)

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

            embeddings = self._embedding_generator.embed_documents(chunks)

            self._ensure_embeddings_match_chunks(chunks=chunks, embeddings=embeddings)

            self._create_chunks(
                document=document,
                chunks=chunks,
                embeddings=embeddings,
            )

            self._mark_ready(document=document)

            self._db.commit()

        except Exception:
            self._db.rollback()

            try:
                document = self._get_document(document_id=document_id)

                self._mark_failed(document=document)

                self._db.commit()

            except Exception:
                self._db.rollback()

                logger.exception(
                    "Failed to mark document as failed",
                    extra={
                        "document_id": str(document_id),
                    },
                )

            logger.exception(
                "Document ingestion failed",
                extra={
                    "document_id": str(document_id),
                },
            )

            raise

        logger.info(
            "Document ingestion completed",
            extra={
                "document_id": str(document.id),
            },
        )
