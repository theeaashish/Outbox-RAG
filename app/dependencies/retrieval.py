from typing import Annotated

from fastapi import Depends

from app.dependencies.database import DbSessionDep
from app.dependencies.embeddings import EmbeddingGeneratorDep
from app.modules.retrieval.controller import RetrievalController
from app.modules.retrieval.service import RetrievalService
from app.repositories.document_chunk import DocumentChunkRepository


def get_document_chunk_repository(
    db: DbSessionDep,
) -> DocumentChunkRepository:
    return DocumentChunkRepository(db=db)


DocumentChunkRepositoryDep = Annotated[
    DocumentChunkRepository,
    Depends(get_document_chunk_repository),
]


def get_retrieval_service(
    embedding_generator: EmbeddingGeneratorDep,
    chunk_repository: DocumentChunkRepositoryDep,
) -> RetrievalService:
    return RetrievalService(
        embedding_generator=embedding_generator,
        chunk_repository=chunk_repository,
    )


RetrievalServiceDep = Annotated[
    RetrievalService,
    Depends(get_retrieval_service),
]


def get_retrieval_controller(
    retrieval_service: RetrievalServiceDep,
) -> RetrievalController:
    return RetrievalController(
        retrieval_service=retrieval_service,
    )


RetrievalControllerDep = Annotated[
    RetrievalController,
    Depends(get_retrieval_controller),
]
