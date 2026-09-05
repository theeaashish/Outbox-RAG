from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from time import monotonic

from app.core.config import settings
from app.db.session import SessionLocal
from app.repositories.session import SessionRepository
from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

_SESSION_CLEANUP_BATCH_SIZE = 500
_MAX_BATCHES_PER_RUN = 20
_MAX_RUNTIME_SECONDS = 30.0


@celery_app.task(name="session.cleanup", acks_late=True)
def cleanup_sessions() -> None:
    """Delete bounded batches of sessions that are dead beyond retention."""

    started_at = monotonic()
    now = datetime.now(UTC)

    retention = timedelta(
        days=settings.session_cleanup_retention_days,
    )
    idle_timeout = timedelta(
        days=settings.session_idle_timeout_days,
    )

    db = SessionLocal()
    total_deleted = 0
    batches_executed = 0

    try:
        repository = SessionRepository(
            db=db,
            idle_timeout=idle_timeout,
        )

        while batches_executed < _MAX_BATCHES_PER_RUN:
            if (
                batches_executed > 0
                and (monotonic() - started_at) >= _MAX_RUNTIME_SECONDS
            ):
                logger.info(
                    "Session cleanup reached time budget ceiling",
                    extra={
                        "batches_executed": batches_executed,
                        "total_deleted": total_deleted,
                        "elapsed_seconds": round(monotonic() - started_at, 2),
                    },
                )
                break

            deleted_count = repository.delete_cleanup_batch(
                now=now,
                retention=retention,
                limit=_SESSION_CLEANUP_BATCH_SIZE,
            )

            db.commit()

            total_deleted += deleted_count
            batches_executed += 1

            if deleted_count < _SESSION_CLEANUP_BATCH_SIZE:
                break

        logger.info(
            "Session cleanup completed",
            extra={
                "total_deleted": total_deleted,
                "batches_executed": batches_executed,
                "batch_size": _SESSION_CLEANUP_BATCH_SIZE,
                "duration_ms": round(
                    (monotonic() - started_at) * 1000,
                    2,
                ),
            },
        )
    except Exception:
        db.rollback()

        logger.exception(
            "Session cleanup failed",
            extra={
                "total_deleted_so_far": total_deleted,
                "batches_executed": batches_executed,
                "batch_size": _SESSION_CLEANUP_BATCH_SIZE,
            },
        )

        raise
    finally:
        db.close()
