from uuid import UUID

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.document.incoming_file import IncomingFile
from app.db.models import Document
from app.modules.document.schemas import (
    CompleteDocumentUploadResponse,
    DocumentListResponse,
    DocumentResponse,
    InitiateDocumentUploadRequest,
    InitiateDocumentUploadResponse,
    RefreshUploadAuthorizationResponse,
    UploadAuthorizationResponse,
)
from app.modules.document.service import DocumentService


class DocumentController:
    def __init__(self, service: DocumentService) -> None:
        self._service = service

    async def upload_document(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        file: UploadFile,
    ) -> Document:
        content = await file.read()
        incoming = IncomingFile(
            filename=file.filename or "",
            content_type=file.content_type,
            size=len(content),
            content=content,
        )
        return await run_in_threadpool(
            self._service.upload_document,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            file=incoming,
        )

    def list_documents(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> DocumentListResponse:
        """List documents owned by the current user in a knowledge base."""

        documents = self._service.list_documents(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        return DocumentListResponse(
            results=[
                DocumentResponse.model_validate(document) for document in documents
            ]
        )

    def initiate_upload(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        request: InitiateDocumentUploadRequest,
    ) -> InitiateDocumentUploadResponse:
        """Initiate direct-to-storage document upload."""

        document, presigned_url = self._service.initiate_upload(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            filename=request.filename,
            content_type=request.content_type,
            file_size=request.file_size,
        )
        return InitiateDocumentUploadResponse(
            document_id=document.id,
            upload=UploadAuthorizationResponse(
                url=presigned_url.url,
                method=presigned_url.method,
                expires_in=presigned_url.expires_in,
                headers=presigned_url.headers,
                form_fields=presigned_url.form_fields,
            ),
        )

    def complete_upload(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> CompleteDocumentUploadResponse:
        """Complete and finalize direct-to-storage document upload."""

        document = self._service.complete_upload(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        return CompleteDocumentUploadResponse.model_validate(document)

    def refresh_upload_authorization(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> RefreshUploadAuthorizationResponse:
        """Refresh upload authorization for an existing UPLOADING document."""

        doc_id, presigned_url = self._service.refresh_upload_authorization(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
        )
        return RefreshUploadAuthorizationResponse(
            document_id=doc_id,
            upload=UploadAuthorizationResponse(
                url=presigned_url.url,
                method=presigned_url.method,
                expires_in=presigned_url.expires_in,
                headers=presigned_url.headers,
                form_fields=presigned_url.form_fields,
            ),
        )
