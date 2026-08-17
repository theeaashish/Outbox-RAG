from celery import Celery
from kombu import Queue

from app.core.config import settings
from app.worker.queues import QueueNames

celery_app = Celery(
    "basic_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    imports=(
        "app.worker.tasks.document_tasks",
        "app.worker.tasks.outbox_tasks",
    ),
    task_queues=(
        Queue(QueueNames.DOCUMENTS),
        Queue(QueueNames.MAINTENANCE),
    ),
    task_routes={
        "document.process": {"queue": QueueNames.DOCUMENTS},
        "outbox.publish": {"queue": QueueNames.MAINTENANCE},
    },
    beat_schedule={
        "publish-outbox-events": {
            "task": "outbox.publish",
            "schedule": 5.0,
            "options": {"queue": QueueNames.MAINTENANCE},
        },
    },
)
