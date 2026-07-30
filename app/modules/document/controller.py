from uuid import UUID

from fastapi import UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.document.incoming_file import IncomingFile
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
        content = await file.read()
        incoming = IncomingFile(
            filename=file.filename or "",
            content_type=file.content_type,
            size=len(content),
            content=content,
        )
        return await run_in_threadpool(
            self._service.upload_document,
            knowledge_base_id=knowledge_base_id,
            file=incoming,
        )
