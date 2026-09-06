from __future__ import annotations

import logging
import time
from typing import Final
from uuid import UUID

from app.core.config import settings
from app.db.models.enums import OutboxEventType
from app.db.session import SessionLocal
from app.repositories.outbox_event import OutboxEventRepository
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_BATCH_SIZE: Final[int] = 100


@celery_app.task(name="outbox.publish")
def publish_outbox_events() -> None:
    """Claim and publish pending outbox events using short per-event transactions."""

    started_at = time.monotonic()
    lease_seconds = settings.outbox_claim_lease_seconds
    max_attempts = settings.max_outbox_attempts

    db = SessionLocal()
    published_count = 0

    try:
        repository = OutboxEventRepository(db=db)

        # Sweep stale final-attempt claims before selecting new work.
        try:
            db.begin()
            stale_exhausted_count = repository.dead_letter_stale_exhausted(
                lease_seconds=lease_seconds,
                max_attempts=max_attempts,
                limit=_BATCH_SIZE,
            )
            db.commit()
            if stale_exhausted_count > 0:
                logger.warning(
                    "Transitioned stale exhausted outbox events to DEAD_LETTER",
                    extra={"count": stale_exhausted_count},
                )
        except Exception:
            db.rollback()
            logger.exception("Failed to sweep stale exhausted outbox events")

        claim_token, events = repository.claim_batch(
            limit=_BATCH_SIZE,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
        )

        if not events:
            db.rollback()
            return

        # Persist the claim before broker I/O.
        db.commit()

        for event in events:
            # Reject permanent payload errors before broker I/O.
            is_malformed = False
            validation_error = ""
            document_id: UUID | None = None

            try:
                payload = event.payload

                if event.event_type != OutboxEventType.DOCUMENT_PROCESS:
                    raise ValueError(
                        f"Unsupported outbox event type: {event.event_type}"
                    )

                if not isinstance(payload, dict):
                    raise TypeError(
                        f"Payload is not a valid dictionary: {type(payload)}"
                    )

                raw_doc_id = payload.get("document_id")
                if not raw_doc_id:
                    raise ValueError(
                        f"Missing document_id in outbox event payload: {payload}"
                    )

                document_id = UUID(str(raw_doc_id))

            except (TypeError, ValueError, KeyError, AttributeError) as val_exc:
                is_malformed = True
                validation_error = str(val_exc)

            if is_malformed:
                try:
                    db.rollback()
                    db.begin()
                    dead_lettered = repository.mark_dead_letter(
                        event_id=event.id,
                        claim_token=claim_token,
                        error=f"MALFORMED_PAYLOAD: {validation_error}",
                    )
                    if not dead_lettered:
                        db.rollback()
                        logger.warning(
                            "Outbox event dead-letter lost claim lease",
                            extra={"event_id": str(event.id)},
                        )
                        continue

                    db.commit()
                    logger.error(
                        "Outbox event payload malformed; moved to DEAD_LETTER",
                        extra={
                            "event_id": str(event.id),
                            "event_type": event.event_type.value,
                            "error": validation_error,
                        },
                    )
                except Exception:
                    db.rollback()
                    logger.exception(
                        "Failed to record dead-letter status for malformed outbox event",
                        extra={"event_id": str(event.id)},
                    )
                continue

            # Publish outside the database transaction.
            try:
                celery_app.send_task(
                    "document.process",
                    args=[str(document_id)],
                )
            except Exception as broker_exc:
                logger.exception(
                    "Outbox broker publish failed",
                    extra={
                        "event_id": str(event.id),
                        "event_type": event.event_type.value,
                        "attempt_count": event.attempt_count,
                        "max_attempts": max_attempts,
                        "error": str(broker_exc),
                    },
                )
                try:
                    db.rollback()
                    db.begin()
                    if event.attempt_count >= max_attempts:
                        dead_lettered = repository.mark_dead_letter(
                            event_id=event.id,
                            claim_token=claim_token,
                            error=f"EXHAUSTED_PUBLICATION_ATTEMPTS ({event.attempt_count}/{max_attempts}): {broker_exc!s}",
                        )
                        if not dead_lettered:
                            db.rollback()
                            logger.warning(
                                "Outbox event dead-letter lost claim lease",
                                extra={"event_id": str(event.id)},
                            )
                            continue

                        db.commit()
                        logger.error(
                            "Outbox event exhausted retry budget and was moved to DEAD_LETTER",
                            extra={
                                "event_id": str(event.id),
                                "attempt_count": event.attempt_count,
                                "max_attempts": max_attempts,
                            },
                        )
                    else:
                        recorded = repository.record_failure(
                            event_id=event.id,
                            claim_token=claim_token,
                            error=str(broker_exc),
                        )
                        if not recorded:
                            db.rollback()
                            logger.warning(
                                "Outbox event record_failure lost claim lease",
                                extra={"event_id": str(event.id)},
                            )
                            continue

                        db.commit()
                except Exception:
                    db.rollback()
                    logger.exception(
                        "Failed to update outbox event after broker failure",
                        extra={"event_id": str(event.id)},
                    )
                continue

            # Acknowledge publication in its own short transaction.
            try:
                db.rollback()
                db.begin()
                published = repository.mark_published(
                    event_id=event.id,
                    claim_token=claim_token,
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
                published_count += 1
                logger.info(
                    "Outbox event published",
                    extra={
                        "event_id": str(event.id),
                        "event_type": event.event_type.value,
                        "aggregate_id": str(event.aggregate_id),
                    },
                )
            except Exception:
                db.rollback()
                logger.exception(
                    "Failed to commit published status for outbox event",
                    extra={"event_id": str(event.id)},
                )

        duration_ms = int((time.monotonic() - started_at) * 1000)
        logger.info(
            "Outbox batch processing completed",
            extra={
                "batch_size": len(events),
                "published_count": published_count,
                "duration_ms": duration_ms,
                "lease_seconds": lease_seconds,
            },
        )
        if duration_ms > (lease_seconds * 1000) / 2:
            logger.warning(
                "Outbox batch duration exceeded 50% of claim lease",
                extra={
                    "duration_ms": duration_ms,
                    "lease_seconds": lease_seconds,
                },
            )

    finally:
        db.close()
