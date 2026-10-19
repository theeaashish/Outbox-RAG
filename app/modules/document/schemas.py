from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.models.enums import DocumentStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    filename: str
    mime_type: str
    file_size: int
    status: DocumentStatus
    last_error: str | None = None
    retry_count: int = 0
    processing_started_at: datetime | None = None
    processed_at: datetime | None = None
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    """Documents belonging to a knowledge base."""

    results: list[DocumentResponse]


class InitiateDocumentUploadRequest(BaseModel):
    """Metadata required to initiate a direct-to-storage upload."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(
        default="application/octet-stream",
        min_length=1,
        max_length=100,
    )
    file_size: int = Field(gt=0)


class UploadAuthorizationResponse(BaseModel):
    """Temporary authorization for a direct storage upload."""

    url: str
    method: str
    expires_in: int
    headers: dict[str, str] = Field(default_factory=dict)
    form_fields: dict[str, str] = Field(default_factory=dict)


class InitiateDocumentUploadResponse(BaseModel):
    """Response returned after an upload is initiated."""

    document_id: UUID
    upload: UploadAuthorizationResponse


class CompleteDocumentUploadResponse(DocumentResponse):
    """Response returned after a direct-to-storage document upload is completed."""


class RefreshUploadAuthorizationResponse(BaseModel):
    """Response returned when upload authorization is refreshed."""

    document_id: UUID
    upload: UploadAuthorizationResponse
