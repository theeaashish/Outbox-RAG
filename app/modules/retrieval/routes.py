from uuid import UUID

from fastapi import APIRouter

from app.dependencies.retrieval import RetrievalControllerDep
from app.modules.retrieval.schemas import RetrievalRequest, RetrievalResponse

router = APIRouter(
    prefix="/knowledge-bases",
    tags=["Retrieval"],
)


@router.post("/{knowledge_base_id}/search", response_model=RetrievalResponse)
def retrieve(
    *,
    knowledge_base_id: UUID,
    request: RetrievalRequest,
    controller: RetrievalControllerDep,
) -> RetrievalResponse:
    """Perform semantic search within a knowledge base"""

    return controller.retrieve(
        knowledge_base_id=knowledge_base_id, query=request.query, limit=request.limit
    )
