from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from celery import Task

from app.core.exceptions import (
    TransientAIServiceException,
    TransientDatabaseException,
    TransientStorageException,
)
from app.db.models.enums import DocumentStatus
from app.db.session import SessionLocal
from app.repositories.document import DocumentRepository
from app.worker.celery_app import celery_app
from app.worker.dependencies import build_document_ingestion_service

logger = logging.getLogger(__name__)


class DocumentProcessTask(Task):
    """Custom Celery task for document ingestion handling final failure transitions."""

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        einfo: Any,
    ) -> None:
        """Mark document as FAILED when retry budget is exhausted or on unhandled failure."""
        super().on_failure(exc, task_id, args, kwargs, einfo)

        document_id_str = args[0] if args else kwargs.get("document_id")
        if not document_id_str:
            return

        try:
            document_id = UUID(str(document_id_str))
        except (ValueError, TypeError):
            return

        db = SessionLocal()
        try:
            document_repo = DocumentRepository(db=db)
            document = document_repo.get_for_update(document_id=document_id)
            if document is not None and document.status != DocumentStatus.READY:
                document.status = DocumentStatus.FAILED
                document.last_error = str(exc)[:4000]
                document.processing_started_at = None
                db.commit()
                logger.info(
                    "Document marked as FAILED in Celery on_failure",
                    extra={
                        "document_id": str(document_id),
                        "task_id": task_id,
                    },
                )
        except Exception:
            db.rollback()
            logger.exception(
                "Failed to transition document to FAILED in Celery on_failure",
                extra={"document_id": str(document_id)},
            )
        finally:
            db.close()


@celery_app.task(
    base=DocumentProcessTask,
    bind=True,
    name="document.process",
    acks_late=True,
    reject_on_worker_lost=True,
    autoretry_for=(
        TransientAIServiceException,
        TransientDatabaseException,
        TransientStorageException,
    ),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def process_document(self, document_id: str) -> None:
    """Process a document asynchronously."""

    celery_retries = int(getattr(self.request, "retries", 0))

    logger.info(
        "Starting document processing task",
        extra={
            "document_id": document_id,
            "celery_retries": celery_retries,
        },
    )

    db = SessionLocal()

    try:
        ingestion_service = build_document_ingestion_service(db=db)

        ingestion_service.process_document(document_id=UUID(document_id))

        logger.info(
            "Document processing task completed",
            extra={
                "document_id": document_id,
                "celery_retries": celery_retries,
            },
        )

    finally:
        db.close()
