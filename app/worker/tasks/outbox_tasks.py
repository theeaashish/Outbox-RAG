from __future__ import annotations

import logging

from sqlalchemy.exc import SQLAlchemyError

from app.db.models.enums import OutboxEventType
from app.db.session import SessionLocal
from app.repositories.outbox_event import OutboxEventRepository
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="outbox.publish")
def publish_outbox_events() -> None:
    """Publish pending outbox events to Celery."""

    db = SessionLocal()

    try:
        repository = OutboxEventRepository(db=db)
        events = repository.list_unpublished(limit=100)

        for event in events:
            try:
                payload = event.payload

                if event.event_type == OutboxEventType.DOCUMENT_PROCESS:
                    document_id = (
                        payload.get("document_id")
                        if isinstance(payload, dict)
                        else None
                    )
                    if not document_id:
                        raise ValueError(
                            f"Missing 'document_id' in payload for outbox event {event.id}"
                        )

                    celery_app.send_task("document.process", args=[str(document_id)])

                    repository.mark_published(event=event)
                    db.commit()

                    logger.info(
                        "Outbox event published",
                        extra={
                            "event_id": str(event.id),
                            "event_type": event.event_type.value,
                            "aggregate_id": str(event.aggregate_id),
                        },
                    )
                else:
                    raise ValueError(
                        f"Unsupported outbox event type: {event.event_type}"
                    )

            except Exception as exc:
                db.rollback()
                repository.record_failure(event=event, error=str(exc))
                try:
                    db.commit()
                except SQLAlchemyError:
                    db.rollback()

                logger.exception(
                    "Outbox event publication failed",
                    extra={
                        "event_id": str(event.id),
                        "event_type": event.event_type.value,
                    },
                )

    finally:
        db.close()
