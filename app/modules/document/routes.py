from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from app.dependencies.auth import CurrentUser
from app.dependencies.controllers import DocumentControllerDep
from app.modules.document.schemas import DocumentListResponse, DocumentResponse

router = APIRouter(prefix="/knowledge-bases", tags=["Documents"])


@router.get(
    "/{knowledge_base_id}/documents",
    response_model=DocumentListResponse,
)
def list_documents(
    *,
    knowledge_base_id: UUID,
    controller: DocumentControllerDep,
    current_user: CurrentUser,
) -> DocumentListResponse:
    return controller.list_documents(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
    )


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
    current_user: CurrentUser,
) -> DocumentResponse:
    document = await controller.upload_document(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        file=file,
    )

    return DocumentResponse.model_validate(document)
