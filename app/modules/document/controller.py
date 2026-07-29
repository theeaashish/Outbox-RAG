from uuid import UUID

from fastapi import UploadFile

from app.db.models import Document
from app.modules.document.service import DocumentService


class DocumentController:
    def __init__(self, service: DocumentService) -> None:
        self._service = service

    async def upload_document(
        self,
        *,
        knowledge_base_id: UUID,
        file: UploadFile,
    ) -> Document:
        return await self._service.upload_document(
            knowledge_base_id=knowledge_base_id, file=file
        )
