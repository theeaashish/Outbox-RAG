from __future__ import annotations

import logging
from uuid import UUID

from app.db.session import SessionLocal
from app.worker.celery_app import celery_app
from app.worker.dependencies import build_document_ingestion_service

logger = logging.getLogger(__name__)


@celery_app.task(
    name="document.process",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def process_document(document_id: str) -> None:
    """Process a document asynchronously."""

    db = SessionLocal()

    try:
        ingestion_service = build_document_ingestion_service(db=db)

        ingestion_service.process_document(document_id=UUID(document_id))

        logger.info(
            "Document processing task completed",
            extra={"document_id": document_id},
        )

    finally:
        db.close()
