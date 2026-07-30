from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.hasher import FileHasher
from app.core.document.incoming_file import IncomingFile
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator
from app.core.exceptions import (
    AIServiceException,
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)
from app.core.storage.base import StorageService
from app.db.models.document import Document
from app.db.models.document_chunk import DocumentChunk
from app.db.models.enums import DocumentStatus
from app.db.models.knowledge_base import KnowledgeBase
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository

logger = logging.getLogger(__name__)


class DocumentService:
    """Application service responsible for document ingestion."""

    def __init__(
        self,
        *,
        db: Session,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        chunk_repository: DocumentChunkRepository,
        parser_registry: DocumentParserRegistry,
        validator: UploadValidator,
        hasher: FileHasher,
        chunker: TextChunker,
        embedding_generator: EmbeddingGenerator,
        storage: StorageService,
    ) -> None:
        self._db = db

        self._knowledge_base_repository = knowledge_base_repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository

        self._parser_registry = parser_registry

        self._validator = validator
        self._hasher = hasher
        self._chunker = chunker
        self._embedding_generator = embedding_generator
        self._storage = storage

    def _get_knowledge_base(self, *, knowledge_base_id: UUID) -> KnowledgeBase:
        knowledge_base = self._knowledge_base_repository.get_by_id(knowledge_base_id)

        if knowledge_base is None:
            raise ResourceNotFoundException("Knowledge base not found")

        return knowledge_base

    def _ensure_document_is_unique(
        self, *, knowledge_base_id: UUID, sha256_hash: str
    ) -> None:
        """Ensure document is unique"""

        existing_document = self._document_repository.get_by_hash(
            knowledge_base_id=knowledge_base_id,
            sha256_hash=sha256_hash,
        )

        if existing_document is not None:
            raise ValidationException("Document with same content already exists")

    def _ensure_text_was_extracted(self, *, text: str) -> None:
        """Ensure the parser extracted meaningful text"""

        if not text.strip():
            raise ValidationException("Parser did not extract any text")

    def _ensure_embeddings_match_chunks(
        self, *, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        """Ensure every chunk has a corresponding embedding"""

        if len(chunks) != len(embeddings):
            raise AIServiceException("Chunk and embedding count mismatch")

    def _build_storage_path(
        self,
        *,
        knowledge_base_id: UUID,
        sha256_hash: str,
        filename: str,
    ) -> str:
        """Return the relative storage key for a document."""

        extension = Path(filename).suffix
        return f"{knowledge_base_id}/{sha256_hash}{extension}"

    def _create_document(
        self,
        *,
        knowledge_base: KnowledgeBase,
        file: IncomingFile,
        sha256_hash: str,
        storage_path: str,
    ) -> Document:
        """Create and persist a document."""

        document = Document(
            title=Path(file.filename).stem,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            storage_path=storage_path,
            sha256_hash=sha256_hash,
            file_size=file.size,
            status=DocumentStatus.PENDING,
            knowledge_base_id=knowledge_base.id,
        )

        return self._document_repository.create(document)

    def _create_chunks(
        self,
        *,
        document: Document,
        chunks: list[str],
        embeddings: list[list[float]],
    ) -> None:
        """Create and persist document chunks with embeddings"""

        for index, (chunk, embedding) in enumerate(
            zip(chunks, embeddings, strict=True)
        ):
            doc_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                char_start=None,
                char_end=None,
                chunk_metadata=None,
            )
            self._chunk_repository.create(doc_chunk)

    def _mark_document_ready(
        self,
        *,
        document: Document,
    ) -> None:
        """Mark the document as successfully processed."""

        document.status = DocumentStatus.READY

    def _mark_document_failed(
        self,
        *,
        document: Document,
    ) -> None:
        """
        Mark the document as failed.

        Reserved for a future async ingestion worker that creates a row
        before processing completes. Not used on the sync request path.
        """

        document.status = DocumentStatus.FAILED

    def _cleanup_storage(self, *, storage_path: str) -> None:
        """Best-effort delete of a stored file after a failed DB commit."""

        try:
            self._storage.delete(storage_path)
        except Exception:
            logger.exception(
                "Failed to clean up stored file after DB failure",
                extra={"path": storage_path},
            )

    def upload_document(
        self,
        *,
        knowledge_base_id: UUID,
        file: IncomingFile,
    ) -> Document:
        """Upload and ingest a document into a knowledge base"""

        logger.info(
            "Upload started",
            extra={
                "kb_id": str(knowledge_base_id),
                "filename": file.filename,
                "size": file.size,
            },
        )

        extension = self._validator.validate(file)

        knowledge_base = self._get_knowledge_base(knowledge_base_id=knowledge_base_id)

        content = file.content
        file_hash = self._hasher.hash(content)

        self._ensure_document_is_unique(
            knowledge_base_id=knowledge_base.id,
            sha256_hash=file_hash,
        )

        parser = self._parser_registry.get_parser(extension)
        logger.info(
            "Parser selected",
            extra={"extension": extension, "parser": type(parser).__name__},
        )

        text = parser.extract_text(content)
        self._ensure_text_was_extracted(text=text)

        chunks = self._chunker.split(text)
        if not chunks:
            raise ValidationException("Document produced no chunks")

        logger.info("Chunks generated", extra={"chunk_count": len(chunks)})

        embeddings = self._embedding_generator.embed_documents(chunks)
        self._ensure_embeddings_match_chunks(chunks=chunks, embeddings=embeddings)

        storage_path = self._build_storage_path(
            knowledge_base_id=knowledge_base.id,
            sha256_hash=file_hash,
            filename=file.filename,
        )
        storage_saved = False

        try:
            self._storage.save(storage_path, content)
            storage_saved = True

            document = self._create_document(
                knowledge_base=knowledge_base,
                file=file,
                sha256_hash=file_hash,
                storage_path=storage_path,
            )

            self._db.flush()
            self._create_chunks(document=document, chunks=chunks, embeddings=embeddings)
            self._mark_document_ready(document=document)
            self._db.commit()
        except IntegrityError as exc:
            self._db.rollback()
            if storage_saved:
                self._cleanup_storage(storage_path=storage_path)
            raise ConflictException(
                "Document with same content already exists"
            ) from exc
        except Exception:
            self._db.rollback()
            if storage_saved:
                self._cleanup_storage(storage_path=storage_path)
            raise

        self._db.refresh(document)

        logger.info(
            "Upload completed",
            extra={
                "kb_id": str(knowledge_base_id),
                "document_id": str(document.id),
                "chunk_count": len(chunks),
            },
        )

        return document
