from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.hasher import FileHasher
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator
from app.core.exceptions import (
    AIServiceException,
    ResourceNotFoundException,
    ValidationException,
)
from app.db.models.document import Document
from app.db.models.enums import DocumentStatus
from app.db.models.knowledge_base import KnowledgeBase
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository


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

    def _create_document(
        self,
        *,
        knowledge_base: KnowledgeBase,
        file: UploadFile,
        sha256_hash: str,
        file_size: int,
    ) -> Document:
        """Create and persist a document."""

        extension = Path(file.filename).suffix

        storage_path = f"uploads/{knowledge_base.id}/{sha256_hash}{extension}"

        return self._document_repository.create(
            title=Path(file.filename).stem,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            storage_path=storage_path,
            sha256_hash=sha256_hash,
            file_size=file_size,
            status=DocumentStatus.PENDING,
            knowledge_base_id=knowledge_base.id,
        )

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
            self._chunk_repository.create(
                document_id=document.id,
                chunk_index=index,
                content=chunk,
                embedding=embedding,
                char_start=None,
                char_end=None,
                chunk_metadata=None,
            )

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
        """Mark the document as failed."""

        document.status = DocumentStatus.FAILED

    async def upload_document(
        self,
        *,
        knowledge_base_id: UUID,
        file: UploadFile,
    ) -> Document:
        """Upload and ingest a document into a knowledge base"""

        # Validate upload
        extension = self._validator.validate(file)

        # Retrieve knowledge base
        knowledge_base = self._get_knowledge_base(knowledge_base_id=knowledge_base_id)

        # Read uploaded file
        content = await file.read()

        # Generate file hash
        file_hash = self._hasher.hash(content)

        # Check duplicate document
        self._ensure_document_is_unique(
            knowledge_base_id=knowledge_base.id,
            sha256_hash=file_hash,
        )

        # Resolve parser
        parser = self._parser_registry.get_parser(extension)

        # Extract text
        text = parser.extract_text(content)

        # Ensure text was extracted
        self._ensure_text_was_extracted(text=text)

        # Chunk text
        chunks = self._chunker.split(text)

        # Generate embeddings
        embeddings = self._embedding_generator.embed_documents(chunks)

        # Ensure embeddings match chunks
        self._ensure_embeddings_match_chunks(chunks=chunks, embeddings=embeddings)

        # Database transaction phase
        try:
            # Create document
            document = self._create_document(
                knowledge_base=knowledge_base,
                file=file,
                sha256_hash=file_hash,
                file_size=len(content),
            )

            # Flush document
            self._db.flush()

            # Create chunks
            self._create_chunks(document=document, chunks=chunks, embeddings=embeddings)

            # Mark document as ready
            self._mark_document_ready(document=document)

            # Commit transaction
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        # Refresh document
        self._db.refresh(document)

        # Return document
        return document
