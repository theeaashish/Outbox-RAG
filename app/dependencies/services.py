from __future__ import annotations

from typing import Annotated

from fastapi import Depends

from app.core.ai.chunking.base import TextChunker
from app.core.ai.embeddings.base import EmbeddingGenerator
from app.core.document.hasher import FileHasher
from app.core.document.parsers.registry import DocumentParserRegistry
from app.core.document.validator import UploadValidator
from app.dependencies.core import (
    get_document_parser_registry,
    get_embedding_generator,
    get_file_hasher,
    get_text_chunker,
    get_upload_validator,
)
from app.dependencies.repositories import (
    DBSession,
    get_document_chunk_repository,
    get_document_repository,
    get_knowledge_base_repository,
)
from app.modules.document.service import DocumentService
from app.repositories.document import DocumentRepository
from app.repositories.document_chunk import DocumentChunkRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository

KnowledgeBaseRepositoryDep = Annotated[
    KnowledgeBaseRepository,
    Depends(get_knowledge_base_repository),
]

DocumentRepositoryDep = Annotated[
    DocumentRepository,
    Depends(get_document_repository),
]

DocumentChunkRepositoryDep = Annotated[
    DocumentChunkRepository,
    Depends(get_document_chunk_repository),
]

TextChunkerDep = Annotated[
    TextChunker,
    Depends(get_text_chunker),
]

EmbeddingGeneratorDep = Annotated[
    EmbeddingGenerator,
    Depends(get_embedding_generator),
]

ParserRegistryDep = Annotated[
    DocumentParserRegistry,
    Depends(get_document_parser_registry),
]

UploadValidatorDep = Annotated[
    UploadValidator,
    Depends(get_upload_validator),
]

FileHasherDep = Annotated[
    FileHasher,
    Depends(get_file_hasher),
]


def get_document_service(
    db: DBSession,
    knowledge_base_repository: KnowledgeBaseRepositoryDep,
    document_repository: DocumentRepositoryDep,
    chunk_repository: DocumentChunkRepositoryDep,
    parser_registry: ParserRegistryDep,
    validator: UploadValidatorDep,
    hasher: FileHasherDep,
    chunker: TextChunkerDep,
    embedding_generator: EmbeddingGeneratorDep,
) -> DocumentService:
    """Return a configured DocumentService."""

    return DocumentService(
        db=db,
        knowledge_base_repository=knowledge_base_repository,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
        parser_registry=parser_registry,
        validator=validator,
        hasher=hasher,
        chunker=chunker,
        embedding_generator=embedding_generator,
    )


DocumentServiceDep = Annotated[
    DocumentService,
    Depends(get_document_service),
]
