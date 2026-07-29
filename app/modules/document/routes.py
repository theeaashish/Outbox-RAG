from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from app.dependencies.controllers import DocumentControllerDep
from app.modules.document.schemas import DocumentResponse

router = APIRouter(prefix="/knowledge-bases", tags=["Documents"])


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    *,
    knowledge_base_id: UUID,
    file: UploadFile = File(...),  # noqa: B008
    controller: DocumentControllerDep,
) -> DocumentResponse:
    document = await controller.upload_document(
        knowledge_base_id=knowledge_base_id,
        file=file,
    )

    return DocumentResponse.model_validate(document)
