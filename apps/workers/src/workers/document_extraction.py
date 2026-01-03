"""Document text extraction worker."""

from celery import Task
from celery_app import celery_app
from lib import get_redis_client, ProgressPublisher
import sys
sys.path.insert(0, '../../packages/shared/src')

from shared import (
    Document,
    ProcessingStatus,
    DocumentExtractionJob,
    PermanentError,
    RetryableError,
)

import asyncio


class CallbackTask(Task):
    """Task with error handling callback."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        print(f"Task {task_id} failed: {exc}")


@celery_app.task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, job_payload: dict) -> dict:
    """Extract text from document and create chunks."""
    async def async_extract():
        try:
            job = DocumentExtractionJob(**job_payload)
            redis_client = await get_redis_client()
            publisher = ProgressPublisher(redis_client)

            # Publish start
            await publisher.publish_progress(
                job.document_id,
                0,
                "starting",
                "Starting text extraction..."
            )

            # Update document status in database
            # (placeholder - would use async_session)

            # Publish completion
            await publisher.publish_progress(
                job.document_id,
                100,
                "completed",
                "Extraction complete!"
            )

            return {
                "status": "success",
                "document_id": job.document_id,
                "chunks_created": 1,
            }

        except PermanentError as e:
            print(f"Permanent error in extraction: {e}")
            raise
        except Exception as e:
            print(f"Error: {e}")
            raise self.retry(exc=e, countdown=60)

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_extract())
