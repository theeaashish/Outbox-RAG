from celery import Celery
from kombu import Queue

from app.core.config import settings

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
)

celery_app.conf.imports = (
    "app.worker.tasks.document_tasks",
    "app.worker.tasks.outbox_tasks",
)

celery_app.conf.task_queues = (
    Queue("documents"),
    Queue("maintenance"),
)

celery_app.conf.task_routes = {
    "document.process": {"queue": "documents"},
    "outbox.publish": {"queue": "maintenance"},
}

celery_app.conf.beat_schedule = {
    "publish-outbox-events": {
        "task": "outbox.publish",
        "schedule": 5.0,
        "options": {"queue": "maintenance"},
    },
}
