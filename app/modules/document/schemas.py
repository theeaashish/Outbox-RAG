from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

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
