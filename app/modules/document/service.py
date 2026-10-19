from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.document.hasher import FileHasher
from app.core.document.incoming_file import IncomingFile
from app.core.document.validator import UploadValidator
from app.core.exceptions import (
    ConflictException,
    ResourceNotFoundException,
    StorageException,
    ValidationException,
)
from app.core.storage.base import StorageService
from app.core.storage.presigned import PresignedUrl, PresignedUrlService
from app.db.classification import is_unique_constraint_violation
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
        presigned_url_service: PresignedUrlService | None = None,
    ) -> None:
        self._db = db
        self._knowledge_base_repository = knowledge_base_repository
        self._document_repository = document_repository
        self._validator = validator
        self._hasher = hasher
        self._storage = storage
        self._presigned_url_service = presigned_url_service

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
            raise ConflictException("Document with same content already exists")

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
            id=uuid4(),
            title=Path(file.filename).stem,
            filename=file.filename,
            mime_type=file.content_type or "application/octet-stream",
            storage_path=storage_path,
            sha256_hash=sha256_hash,
            file_size=file.size,
            status=DocumentStatus.PENDING,
            knowledge_base_id=knowledge_base.id,
        )

        created_document = self._document_repository.create(document)
        self._document_repository.flush()
        return created_document

    def _create_outbox_event(
        self,
        *,
        document: Document,
    ) -> OutboxEvent:
        """Create a durable event for asynchronous document processing."""

        if document.id is None:
            raise ValueError("Cannot create outbox event for document without an ID")

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
        """Record a potentially orphaned content-addressed final object without deleting it.

        In a content-addressed storage scheme, an object may be referenced by
        another document or concurrent insert race. Immediate deletion cannot
        guarantee that a concurrent request has not committed a reference to the
        same object. To prevent accidental data loss, the object is preserved and
        logged for background reconciliation/garbage collection.
        """

        logger.warning(
            "Preserving potentially orphaned storage object after database failure",
            extra={"storage_path": storage_path},
        )

    def _cleanup_staging(self, *, staging_key: str) -> None:
        """Best-effort deletion of a per-document staging object."""

        try:
            self._storage.delete(staging_key)
        except Exception:
            logger.exception(
                "Failed to delete staging object",
                extra={"staging_key": staging_key},
            )

    def _detach_and_rollback(self, document: Document) -> Document:
        """Detach a document so it survives rollback with its attributes loaded."""

        self._db.expunge(document)

        if self._db.in_transaction():
            self._db.rollback()

        return document

    def _end_read_transaction(self) -> None:
        """Close the read transaction opened by the last query, if any."""

        if self._db.in_transaction():
            self._db.rollback()

    def _discard_uploading_document(self, *, document_id: UUID) -> None:
        """Best-effort removal of an UPLOADING row rejected by duplicate checks.

        A row that concurrently transitioned away from UPLOADING is left alone.
        """

        try:
            self._end_read_transaction()
            document = self._document_repository.get_for_update(document_id=document_id)

            if document is None or document.status != DocumentStatus.UPLOADING:
                self._db.rollback()
                return

            self._db.delete(document)
            self._db.commit()
        except Exception:
            self._db.rollback()
            logger.exception(
                "Failed to remove rejected UPLOADING document record",
                extra={"document_id": str(document_id)},
            )

    def _warn_unreferenced_final_object(
        self,
        *,
        document_id: UUID,
        knowledge_base_id: UUID,
        storage_path: str,
        reason: str,
    ) -> None:
        """Record a possible orphan without deleting a potentially shared object."""

        logger.warning(
            "Final storage object may be unreferenced: %s",
            reason,
            extra={
                "document_id": str(document_id),
                "knowledge_base_id": str(knowledge_base_id),
                "storage_path": storage_path,
            },
        )

    def _get_presigned_url_service(self) -> PresignedUrlService:
        """Return the upload-authorization capability for the active backend."""

        if self._presigned_url_service is None:
            raise StorageException("Direct uploads require the S3 storage backend")

        return self._presigned_url_service

    def _persist_final_object(
        self,
        *,
        storage_path: str,
        content: bytes,
        expected_hash: str,
    ) -> None:
        """Write verified bytes to the content-addressed final key.

        Reuses an existing object only if both its size and content hash match
        the verified incoming payload. If an existing object is missing or its
        integrity cannot be confirmed, writes the verified in-memory bytes.
        """

        try:
            existing = self._storage.head(storage_path)
        except ResourceNotFoundException:
            self._storage.save(storage_path, content)
            return

        if existing.size == len(content):
            try:
                existing_bytes = self._storage.read(storage_path)
                if self._hasher.hash(existing_bytes) == expected_hash:
                    return
            except (ResourceNotFoundException, StorageException):
                logger.warning(
                    "Could not verify existing storage object hash; overwriting with verified bytes",
                    extra={"storage_path": storage_path},
                )

        self._storage.save(storage_path, content)

    def initiate_upload(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        filename: str,
        content_type: str,
        file_size: int,
    ) -> tuple[Document, PresignedUrl]:
        """Create a staged document and its temporary upload authorization.

        Authorization is issued before the row is persisted: a persistence
        failure then leaves an unused authorization that was never returned to
        the client, which is preferable to an orphaned UPLOADING row. No
        database transaction is open while signing.
        """

        filename = filename.strip()
        content_type = content_type.strip()

        if not content_type:
            raise ValidationException("Content type is required")

        self._validator.validate_metadata(filename=filename, size=file_size)

        knowledge_base = self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        kb_id = knowledge_base.id

        presigned_url_service = self._get_presigned_url_service()
        self._end_read_transaction()

        document_id = uuid4()
        staging_key = (
            f"staging/knowledge-bases/{kb_id}/documents/{document_id}/original"
        )

        upload_authorization = presigned_url_service.create_upload_url(
            key=staging_key,
            content_type=content_type,
            expires_in=settings.presigned_url_expiry_seconds,
            expected_size_bytes=file_size,
        )

        document = Document(
            id=document_id,
            title=Path(filename).stem,
            filename=filename,
            mime_type=content_type,
            storage_path=staging_key,
            sha256_hash=None,
            file_size=file_size,
            status=DocumentStatus.UPLOADING,
            knowledge_base_id=kb_id,
        )

        try:
            self._document_repository.create(document)
            self._db.commit()
        except Exception:
            self._db.rollback()
            raise

        logger.info(
            "Document upload initiated",
            extra={
                "user_id": str(user_id),
                "kb_id": str(kb_id),
                "document_id": str(document.id),
                "size": file_size,
            },
        )

        return document, upload_authorization

    def refresh_upload_authorization(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> tuple[UUID, PresignedUrl]:
        """Issue a new short-lived upload authorization for an existing UPLOADING document.

        Verifies ownership, preserves the recorded document ID, staging key,
        filename, content type, and expected file size, and issues a fresh
        authorization without performing storage I/O during open transactions.
        """

        knowledge_base = self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        kb_id = knowledge_base.id
        self._end_read_transaction()

        try:
            document = self._document_repository.get_for_update(document_id=document_id)

            if document is None or document.knowledge_base_id != kb_id:
                self._db.rollback()
                raise ResourceNotFoundException("Document not found")

            if self._is_finalized(document.status):
                self._db.rollback()
                raise ConflictException(
                    f"Cannot refresh upload authorization for already finalized document in {document.status.value} status"
                )

            if document.status != DocumentStatus.UPLOADING:
                status = document.status
                self._db.rollback()
                raise ConflictException(
                    f"Upload authorization cannot be refreshed for document in {status.value} status"
                )

            # Capture scalars before committing to avoid expired attribute reloads
            doc_id = document.id
            staging_key = document.storage_path
            content_type = document.mime_type
            file_size = document.file_size

            # Update timestamp under lock to reset cleanup eligibility
            document.updated_at = datetime.now(UTC)
            self._db.commit()

        except Exception:
            self._db.rollback()
            raise

        presigned_url_service = self._get_presigned_url_service()
        upload_authorization = presigned_url_service.create_upload_url(
            key=staging_key,
            content_type=content_type,
            expires_in=settings.presigned_url_expiry_seconds,
            expected_size_bytes=file_size,
        )

        logger.info(
            "Document upload authorization refreshed",
            extra={
                "user_id": str(user_id),
                "kb_id": str(kb_id),
                "document_id": str(doc_id),
                "size": file_size,
            },
        )

        return doc_id, upload_authorization

    def cleanup_stale_uploading_documents(
        self,
        *,
        older_than: timedelta | None = None,
        batch_size: int = 100,
    ) -> int:
        """Safely clean up abandoned documents remaining in UPLOADING status.

        A document is eligible for cleanup only if it has remained un-updated
        past the retention cutoff (defaulting to 24 hours to ensure any issued
        presigned URL has long expired).

        To prevent race conditions with concurrent refresh or finalization requests,
        each candidate is re-checked under a row lock (FOR UPDATE). If the document
        was refreshed or transitioned away from UPLOADING, it is left alone.
        Otherwise, the row is deleted and its temporary staging object is
        deleted best-effort after the transaction commits.
        """

        cutoff = datetime.now(UTC) - (older_than or timedelta(hours=24))
        candidates = self._document_repository.list_stale_uploading_documents(
            updated_before=cutoff,
            limit=batch_size,
        )
        self._end_read_transaction()

        cleaned_count = 0
        for doc_id, staging_key in candidates:
            try:
                locked_doc = self._document_repository.get_for_update(
                    document_id=doc_id
                )
                if (
                    locked_doc is None
                    or locked_doc.status != DocumentStatus.UPLOADING
                    or locked_doc.updated_at >= cutoff
                ):
                    self._db.rollback()
                    continue

                self._db.delete(locked_doc)
                self._db.commit()
            except Exception:
                self._db.rollback()
                logger.exception(
                    "Failed to delete stale UPLOADING document row",
                    extra={"document_id": str(doc_id)},
                )
                continue

            self._cleanup_staging(staging_key=staging_key)
            cleaned_count += 1

        return cleaned_count

    @staticmethod
    def _is_finalized(status: DocumentStatus) -> bool:
        """Return whether a document has already left the UPLOADING state."""

        return status in (
            DocumentStatus.PENDING,
            DocumentStatus.PROCESSING,
            DocumentStatus.READY,
        )

    def _read_staged_content(
        self,
        *,
        staging_key: str,
        filename: str,
        declared_size: int,
    ) -> bytes:
        """Verify and read the staged object, then validate its content.

        Runs with no database transaction open. Validation failures remove the
        staging object; the row stays UPLOADING because the schema forbids a
        NULL hash outside that state and the upload authorization remains valid
        until it expires, so the client may retry the upload.
        """

        try:
            metadata = self._storage.head(staging_key)
        except ResourceNotFoundException as exc:
            raise ValidationException("Uploaded file not found in storage") from exc

        max_bytes = settings.max_upload_size_mb * 1024 * 1024

        if metadata.size < 1:
            self._cleanup_staging(staging_key=staging_key)
            raise ValidationException("Uploaded file is empty")

        if metadata.size != declared_size:
            self._cleanup_staging(staging_key=staging_key)
            raise ValidationException(
                "Uploaded file size does not match the declared size"
            )

        if metadata.size > max_bytes:
            self._cleanup_staging(staging_key=staging_key)
            raise ValidationException(
                f"File exceeds maximum size of {settings.max_upload_size_mb}MB"
            )

        try:
            content = self._storage.read(staging_key)
        except ResourceNotFoundException as exc:
            raise ValidationException("Uploaded file not found in storage") from exc

        if len(content) != metadata.size:
            self._cleanup_staging(staging_key=staging_key)
            raise ValidationException(
                "Stored object size changed while reading the upload"
            )

        try:
            self._validator.validate_metadata(
                filename=filename,
                size=len(content),
            )
        except ValidationException:
            self._cleanup_staging(staging_key=staging_key)
            raise

        return content

    def complete_upload(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
        document_id: UUID,
    ) -> Document:
        """Verify a staged object, promote it, and enqueue ingestion.

        The lookup and idempotency decision run under a short row lock that
        refreshes the document's timestamp to protect it against concurrent
        cleanup during storage I/O. Verification and promotion happen with no
        transaction open; finalization then runs in its own short transaction
        with a row lock and an atomic outbox insert.
        """

        knowledge_base = self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        kb_id = knowledge_base.id
        self._end_read_transaction()

        try:
            document = self._document_repository.get_for_update(document_id=document_id)

            if document is None or document.knowledge_base_id != kb_id:
                self._db.rollback()
                raise ResourceNotFoundException("Document not found")

            if self._is_finalized(document.status):
                # Idempotent: return the existing result without promoting again
                # or creating a second outbox event.
                logger.info(
                    "Document already finalized; returning existing record",
                    extra={
                        "document_id": str(document_id),
                        "status": document.status.value,
                    },
                )
                staging_key = (
                    f"staging/knowledge-bases/{kb_id}/documents/{document_id}/original"
                )
                document = self._detach_and_rollback(document)
                self._cleanup_staging(staging_key=staging_key)
                return document

            if document.status != DocumentStatus.UPLOADING:
                status = document.status
                self._db.rollback()
                raise ConflictException(
                    f"Document upload cannot be completed in {status.value} status"
                )

            # Capture scalars before committing to avoid expired attribute reloads.
            staging_key = document.storage_path
            filename = document.filename
            declared_size = document.file_size

            # Touch updated_at under lock to protect this document from concurrent cleanup
            # while storage I/O and verification are running.
            document.updated_at = datetime.now(UTC)
            self._db.commit()

        except Exception:
            self._db.rollback()
            raise

        content = self._read_staged_content(
            staging_key=staging_key,
            filename=filename,
            declared_size=declared_size,
        )

        file_hash = self._hasher.hash(content)
        final_storage_path = self._build_storage_path(
            knowledge_base_id=kb_id,
            sha256_hash=file_hash,
            filename=filename,
        )

        # Advisory duplicate check. The unique constraint is the final authority.
        try:
            self._ensure_document_is_unique(
                knowledge_base_id=kb_id,
                sha256_hash=file_hash,
            )
        except ConflictException:
            self._end_read_transaction()
            self._cleanup_staging(staging_key=staging_key)
            self._discard_uploading_document(document_id=document_id)
            raise

        self._end_read_transaction()

        self._persist_final_object(
            storage_path=final_storage_path,
            content=content,
            expected_hash=file_hash,
        )

        finalized = self._finalize_upload(
            document_id=document_id,
            knowledge_base_id=kb_id,
            staging_key=staging_key,
            storage_path=final_storage_path,
            sha256_hash=file_hash,
            file_size=len(content),
        )

        # The staged object is never the processing source, so removing it is
        # safe regardless of which request won finalization.
        self._cleanup_staging(staging_key=staging_key)

        logger.info(
            "Document upload completed successfully",
            extra={
                "user_id": str(user_id),
                "kb_id": str(kb_id),
                "document_id": str(finalized.id),
                "size": len(content),
            },
        )

        return finalized

    def _finalize_upload(
        self,
        *,
        document_id: UUID,
        knowledge_base_id: UUID,
        staging_key: str,
        storage_path: str,
        sha256_hash: str,
        file_size: int,
    ) -> Document:
        """Atomically finalize an UPLOADING document and enqueue processing.

        Runs in a short row-locked transaction with no storage I/O. Only the
        request that observes UPLOADING transitions the row and creates the
        outbox event; later requests observe the finalized state instead.
        """

        try:
            document = self._document_repository.get_for_update(document_id=document_id)

            if document is None or document.knowledge_base_id != knowledge_base_id:
                self._db.rollback()
                self._cleanup_staging(staging_key=staging_key)
                self._warn_unreferenced_final_object(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    storage_path=storage_path,
                    reason="document disappeared during finalization",
                )
                raise ResourceNotFoundException("Document not found")

            if self._is_finalized(document.status):
                # A concurrent request won finalization; do not duplicate work.
                document = self._detach_and_rollback(document)
                self._cleanup_staging(staging_key=staging_key)
                return document

            if document.status != DocumentStatus.UPLOADING:
                status = document.status
                self._db.rollback()
                self._cleanup_staging(staging_key=staging_key)
                self._warn_unreferenced_final_object(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    storage_path=storage_path,
                    reason=f"document left UPLOADING with status {status.value}",
                )
                raise ConflictException(
                    f"Document upload cannot be completed in {status.value} status"
                )

            existing_document = self._document_repository.get_by_hash(
                knowledge_base_id=knowledge_base_id,
                sha256_hash=sha256_hash,
            )

            if existing_document is not None and existing_document.id != document.id:
                self._db.delete(document)
                self._db.commit()
                self._cleanup_staging(staging_key=staging_key)
                self._warn_unreferenced_final_object(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    storage_path=storage_path,
                    reason="duplicate content detected under lock",
                )
                raise ConflictException("Document with same content already exists")

            document.storage_path = storage_path
            document.sha256_hash = sha256_hash
            document.file_size = file_size
            document.status = DocumentStatus.PENDING
            document.last_error = None

            # Same transaction as the transition: never dispatch anywhere else.
            self._create_outbox_event(document=document)
            self._db.commit()

            return document

        except IntegrityError as exc:
            self._db.rollback()

            # The content-addressed final object may be shared with the winning
            # document, so it is never deleted here.
            if is_unique_constraint_violation(exc, "uq_document_kb_sha256"):
                self._cleanup_staging(staging_key=staging_key)
                self._discard_uploading_document(document_id=document_id)
                self._warn_unreferenced_final_object(
                    document_id=document_id,
                    knowledge_base_id=knowledge_base_id,
                    storage_path=storage_path,
                    reason="lost duplicate content race",
                )
                raise ConflictException(
                    "Document with same content already exists"
                ) from exc

            self._warn_unreferenced_final_object(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                storage_path=storage_path,
                reason="database integrity error during finalization",
            )
            raise

        except (ResourceNotFoundException, ConflictException):
            raise

        except Exception:
            self._db.rollback()
            self._warn_unreferenced_final_object(
                document_id=document_id,
                knowledge_base_id=knowledge_base_id,
                storage_path=storage_path,
                reason="unexpected finalization failure",
            )
            raise

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
        self._end_read_transaction()

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

            # For content-addressed storage, an integrity conflict on uq_document_kb_sha256
            # guarantees that another document with the same content hash already exists
            # or won the insert race. Never delete the shared storage file on duplicate conflict.
            if is_unique_constraint_violation(exc, "uq_document_kb_sha256"):
                raise ConflictException(
                    "Document with same content already exists"
                ) from exc

            if storage_saved:
                self._cleanup_storage(
                    storage_path=storage_path,
                )

            raise

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

    def list_documents(
        self,
        *,
        user_id: UUID,
        knowledge_base_id: UUID,
    ) -> list[Document]:
        """List documents in a knowledge base after enforcing ownership."""

        knowledge_base = self._get_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
        )
        return self._document_repository.list_by_user_and_knowledge_base(
            user_id=user_id,
            knowledge_base_id=knowledge_base.id,
        )
