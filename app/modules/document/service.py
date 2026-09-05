from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.document.hasher import FileHasher
from app.core.document.incoming_file import IncomingFile
from app.core.document.validator import UploadValidator
from app.core.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    ValidationException,
)
from app.core.storage.base import StorageService
from app.db.models.document import Document
from app.db.models.enums import DocumentStatus, OutboxEventType
from app.db.models.knowledge_base import KnowledgeBase
from app.db.models.outbox_event import OutboxEvent
from app.repositories.document import DocumentRepository
from app.repositories.knowledge_base import KnowledgeBaseRepository

logger = logging.getLogger(__name__)


class DocumentService:
    """Application service responsible for document upload and persistence."""

    def __init__(
        self,
        *,
        db: Session,
        knowledge_base_repository: KnowledgeBaseRepository,
        document_repository: DocumentRepository,
        validator: UploadValidator,
        hasher: FileHasher,
        storage: StorageService,
    ) -> None:
        self._db = db
        self._knowledge_base_repository = knowledge_base_repository
        self._document_repository = document_repository
        self._validator = validator
        self._hasher = hasher
        self._storage = storage

    def _get_knowledge_base(
        self, *, user_id: UUID, knowledge_base_id: UUID
    ) -> KnowledgeBase:
        knowledge_base = self._knowledge_base_repository.get_by_user_and_id(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )

        if knowledge_base is None:
            raise ResourceNotFoundException("Knowledge base not found")

        return knowledge_base

    def _ensure_document_is_unique(
        self, *, knowledge_base_id: UUID, sha256_hash: str
    ) -> None:
        """Ensure document is unique within the knowledge base."""

        existing_document = self._document_repository.get_by_hash(
            knowledge_base_id=knowledge_base_id,
            sha256_hash=sha256_hash,
        )

        if existing_document is not None:
            raise ValidationException("Document with same content already exists")

    def _build_storage_path(
        self,
        *,
        knowledge_base_id: UUID,
        sha256_hash: str,
        filename: str,
    ) -> str:
        """Return the relative storage key for a document."""

        extension = Path(filename).suffix
        return f"{knowledge_base_id}/{sha256_hash}{extension}"

    def _create_document(
        self,
        *,
        knowledge_base: KnowledgeBase,
        file: IncomingFile,
        sha256_hash: str,
        storage_path: str,
    ) -> Document:
        """Create and persist a document."""

        document = Document(
            title=Path(file.filename).stem,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            storage_path=storage_path,
            sha256_hash=sha256_hash,
            file_size=file.size,
            status=DocumentStatus.PENDING,
            knowledge_base_id=knowledge_base.id,
        )

        return self._document_repository.create(document)

    def _create_outbox_event(
        self,
        *,
        document: Document,
    ) -> OutboxEvent:
        """Create a durable event for asynchronous document processing."""

        event = OutboxEvent(
            event_type=OutboxEventType.DOCUMENT_PROCESS,
            aggregate_type="document",
            aggregate_id=document.id,
            payload={
                "document_id": str(document.id),
            },
        )

        self._db.add(event)

        return event

    def _cleanup_storage(self, *, storage_path: str) -> None:
        """Best-effort delete of a stored file after a failed DB commit."""

        try:
            self._storage.delete(storage_path)
        except Exception:
            logger.exception(
                "Failed to clean up stored file after DB failure",
                extra={"path": storage_path},
            )

    def upload_document(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        file: IncomingFile,
    ) -> Document:
        """Accept and persist a document for asynchronous ingestion."""

        logger.info(
            "Upload Started",
            extra={
                "user_id": str(user_id),
                "kb_id": str(knowledge_base_id),
                "file_name": file.filename,
                "size": file.size,
            },
        )

        # validate the incoming file
        self._validator.validate(file=file)

        # ensure the knowledge base exists and belongs to the user.
        knowledge_base = self._get_knowledge_base(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
        )

        # hash the file for content-based deduplication
        file_hash = self._hasher.hash(file.content)

        self._ensure_document_is_unique(
            knowledge_base_id=knowledge_base.id,
            sha256_hash=file_hash,
        )

        # build the storage key
        storage_path = self._build_storage_path(
            knowledge_base_id=knowledge_base.id,
            sha256_hash=file_hash,
            filename=file.filename,
        )

        storage_saved = False

        try:
            # save the original file.
            self._storage.save(storage_path, file.content)

            storage_saved = True

            # Create the document row
            document = self._create_document(
                knowledge_base=knowledge_base,
                file=file,
                sha256_hash=file_hash,
                storage_path=storage_path,
            )

            self._create_outbox_event(document=document)

            # commit the document transaction
            self._db.commit()

        except IntegrityError as exc:
            self._db.rollback()

            if storage_saved:
                self._cleanup_storage(
                    storage_path=storage_path,
                )

            raise ConflictException(
                "Document with same content already exists"
            ) from exc

        except Exception:
            self._db.rollback()

            if storage_saved:
                self._cleanup_storage(
                    storage_path=storage_path,
                )

            raise

        self._db.refresh(document)

        logger.info(
            "Document uploaded successfully",
            extra={
                "kb_id": str(knowledge_base_id),
                "document_id": str(document.id),
                "size": file.size,
            },
        )

        return document
