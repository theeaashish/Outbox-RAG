from typing import Final


class QueueNames:
    """Named Celery task queues used across workers."""

    DOCUMENTS: Final[str] = "documents"
    MAINTENANCE: Final[str] = "maintenance"
