from __future__ import annotations

from uuid import UUID

from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.parsers.registry import DocumentParserRegistry
from app.db.models.document import Document
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository


class DocumentService:
    """Service responsible for document ingestion"""

    def __init__(
        self,
        *,
        db: Session,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        chunk_repository: DocumentChunkRepository,
        parser_registry: DocumentParserRegistry,
        chunker: TextChunker,
        embedding_generator: EmbeddingGenerator,
    ) -> None:
        self._db = db
        self._knowledge_base_repository = knowledge_base_repository
        self._document_repository = document_repository
        self._chunk_repository = chunk_repository
        self._parser_registry = parser_registry
        self._chunker = chunker
        self._embedding_generator = embedding_generator

    def upload_document(
        self,
        *,
        knowledge_base_id: UUID,
        file: UploadFile,
    ) -> Document:
        """Upload and ingest a document."""
        raise NotImplementedError
