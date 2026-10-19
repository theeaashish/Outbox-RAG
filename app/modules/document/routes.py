from uuid import UUID

from fastapi import APIRouter, File, UploadFile, status

from app.dependencies.auth import CurrentUser
from app.dependencies.controllers import DocumentControllerDep
from app.modules.document.schemas import (
    CompleteDocumentUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    InitiateDocumentUploadRequest,
    InitiateDocumentUploadResponse,
    RefreshUploadAuthorizationResponse,
)

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


@router.post(
    "/{knowledge_base_id}/documents/upload/initiate",
    response_model=InitiateDocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def initiate_upload(
    *,
    knowledge_base_id: UUID,
    request: InitiateDocumentUploadRequest,
    controller: DocumentControllerDep,
    current_user: CurrentUser,
) -> InitiateDocumentUploadResponse:
    return controller.initiate_upload(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        request=request,
    )


@router.post(
    "/{knowledge_base_id}/documents/{document_id}/complete",
    response_model=CompleteDocumentUploadResponse,
    status_code=status.HTTP_200_OK,
)
def complete_upload(
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    controller: DocumentControllerDep,
    current_user: CurrentUser,
) -> CompleteDocumentUploadResponse:
    return controller.complete_upload(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )


@router.post(
    "/{knowledge_base_id}/documents/{document_id}/upload/refresh",
    response_model=RefreshUploadAuthorizationResponse,
    status_code=status.HTTP_200_OK,
)
def refresh_upload_authorization(
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    controller: DocumentControllerDep,
    current_user: CurrentUser,
) -> RefreshUploadAuthorizationResponse:
    return controller.refresh_upload_authorization(
        user_id=current_user.id,
        knowledge_base_id=knowledge_base_id,
        document_id=document_id,
    )
