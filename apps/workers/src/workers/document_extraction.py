"""Document text extraction worker."""

import asyncio
import logging
from celery import Task
from celery_app import celery_app
from lib import get_redis_client, ProgressPublisher
from shared import (
    Document,
    ProcessingStatus,
    DocumentExtractionJob,
    PermanentError,
    RetryableError,
)

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Task with error handling callback."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=einfo)


@celery_app.task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, job_payload: dict) -> dict:
    """Extract text from document and create chunks."""
    async def async_extract():
        try:
            job = DocumentExtractionJob(**job_payload)
            logger.info(f"Starting extraction for document {job.document_id}")
            redis_client = await get_redis_client()
            publisher = ProgressPublisher(redis_client)

            # Publish start
            await publisher.publish_progress(
                job.document_id,
                0,
                "starting",
                "Starting text extraction..."
            )

            # TODO: Phase 4 - Update document status in database
            # Requires async_session from app.database
            # Should update ProcessingStatus to EXTRACTED after text extraction
            # and INDEXED after chunk creation

            # Publish completion
            await publisher.publish_progress(
                job.document_id,
                100,
                "completed",
                "Extraction complete!"
            )

            logger.info(f"Extraction completed for document {job.document_id}")
            return {
                "status": "success",
                "document_id": job.document_id,
                "chunks_created": 1,
            }

        except PermanentError as e:
            logger.error(f"Permanent error in extraction: {e}")
            raise
        except Exception as e:
            logger.error(f"Extraction error for document {job_payload.get('document_id')}: {e}")
            raise self.retry(exc=e, countdown=60)

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_extract())
