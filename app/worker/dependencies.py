from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.ai.chunking.base import DocumentChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.normalizer import DocumentNormalizer
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.storage import StorageService
from app.dependencies.core import (
    get_document_chunker,
    get_document_normalizer,
    get_document_parser_registry,
    get_embedding_generator,
    get_storage_service,
)
from app.modules.document.ingestion import DocumentIngestionService
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository


def build_document_ingestion_service(
    *,
    db: Session,
) -> DocumentIngestionService:
    """Build the document ingestion service for a Celery worker."""

    document_repository = DocumentRepository(db=db)

    chunk_repository = DocumentChunkRepository(db=db)

    parser_registry: DocumentParserRegistry = get_document_parser_registry()

    normalizer: DocumentNormalizer = get_document_normalizer()

    chunker: DocumentChunker = get_document_chunker()

    embedding_generator: EmbeddingGenerator = get_embedding_generator()

    storage: StorageService = get_storage_service()

    return DocumentIngestionService(
        db=db,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        parser_registry=parser_registry,
        chunker=chunker,
        embedding_generator=embedding_generator,
        storage=storage,
        normalizer=normalizer,
    )
