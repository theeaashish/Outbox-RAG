from typing import Annotated

from fastapi import Depends

from app.dependencies.services import (
    DocumentChunkRepositoryDep,
    EmbeddingGeneratorDep,
)
from app.modules.retrieval.controller import RetrievalController
from app.modules.retrieval.service import RetrievalService


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
