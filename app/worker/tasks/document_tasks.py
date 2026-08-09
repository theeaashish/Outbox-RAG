from app.worker.celery_app import celery_app


@celery_app.task
def process_document(document_id: str) -> None:
    print(document_id)
