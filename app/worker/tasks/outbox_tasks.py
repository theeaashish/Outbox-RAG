from __future__ import annotations

import logging
from typing import Final

from sqlalchemy.exc import SQLAlchemyError

from app.db.models.enums import OutboxEventType
from app.db.session import SessionLocal
from app.repositories.outbox_event import OutboxEventRepository
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_BATCH_SIZE: Final[int] = 100
_CLAIM_LEASE_SECONDS: Final[int] = 300


@celery_app.task(name="outbox.publish")
def publish_outbox_events() -> None:
    """Claim and publish pending outbox events."""

    db = SessionLocal()

    try:
        repository = OutboxEventRepository(db=db)

        claim_token, events = repository.claim_batch(
            limit=_BATCH_SIZE,
            lease_seconds=_CLAIM_LEASE_SECONDS,
        )

        if not events:
            db.rollback()
            return

        # Persist the claim before performing any broker/network I/O
        db.commit()

        for event in events:
            try:
                payload = event.payload

                if event.event_type != OutboxEventType.DOCUMENT_PROCESS:
                    raise ValueError(
                        f"Unsupported outbox event type: {event.event_type}"
                    )

                document_id = (
                    payload.get("document_id") if isinstance(payload, dict) else None
                )

                if not document_id:
                    raise ValueError(
                        f"Missing document_id in outbox event payload: {payload}"
                    )

                celery_app.send_task(
                    "document.process",
                    args=[str(document_id)],
                )

                db.begin()

                published = repository.mark_published(
                    event_id=event.id, claim_token=claim_token
                )

                if not published:
                    db.rollback()

                    logger.warning(
                        "Outbox event publication acknowledgement lost claim",
                        extra={
                            "event_id": str(event.id),
                            "event_type": event.event_type.value,
                            "aggregate_id": str(event.aggregate_id),
                        },
                    )
                    continue

                db.commit()

                logger.info(
                    "Outbox event published",
                    extra={
                        "event_id": str(event.id),
                        "event_type": event.event_type.value,
                        "aggregate_id": str(event.aggregate_id),
                    },
                )

            except Exception as exc:
                db.rollback()

                try:
                    db.begin()

                    recorded = repository.record_failure(
                        event_id=event.id,
                        claim_token=claim_token,
                        error=str(exc),
                    )

                    if recorded:
                        db.commit()
                    else:
                        db.rollback()

                        logger.warning(
                            "Outbox failure could not update event claim",
                            extra={
                                "event_id": str(event.id),
                                "event_type": event.event_type.value,
                            },
                        )
                except SQLAlchemyError:
                    db.rollback()

                    logger.exception(
                        "Failed to record outbox publication failure",
                        extra={
                            "event_id": str(event.id),
                            "event_type": event.event_type.value,
                        },
                    )

                logger.exception(
                    "Outbox event publication failed",
                    extra={
                        "event_id": str(event.id),
                        "event_type": event.event_type.value,
                    },
                )

    finally:
        db.close()
