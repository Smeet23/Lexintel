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
    async_session,
    chunk_text,
    clean_text,
    extract_file,
    create_document_chunks,
)
from sqlalchemy.future import select

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
            logger.info(f"[extract_text] Starting for document {job.document_id}")
            redis_client = await get_redis_client()
            publisher = ProgressPublisher(redis_client)

            # Publish start
            await publisher.publish_progress(
                job.document_id,
                0,
                "starting",
                "Starting text extraction..."
            )

            # Get document from database
            async with async_session() as db:
                stmt = select(Document).where(Document.id == job.document_id)
                result = await db.execute(stmt)
                doc = result.scalar_one_or_none()

                if not doc:
                    raise FileNotFoundError(f"Document {job.document_id} not found")

                if not doc.file_path:
                    raise ValueError(f"Document {job.document_id} has no file_path")

                # Publish extraction start
                await publisher.publish_progress(
                    job.document_id,
                    10,
                    "extracting",
                    "Extracting text from file..."
                )

                # Extract text
                text = await extract_file(doc.file_path)
                text = clean_text(text)
                doc.extracted_text = text

                logger.info(f"[extract_text] Extracted {len(text)} chars for document {job.document_id}")

                # Create chunks
                await publisher.publish_progress(
                    job.document_id,
                    50,
                    "chunking",
                    "Creating text chunks..."
                )

                chunk_texts = chunk_text(text, chunk_size=4000, overlap=400)
                chunk_ids = await create_document_chunks(db, job.document_id, chunk_texts)

                # Update document status
                doc.processing_status = ProcessingStatus.EXTRACTED
                await db.commit()

                logger.info(
                    f"[extract_text] Created {len(chunk_ids)} chunks for document {job.document_id}"
                )

            # Publish completion
            await publisher.publish_progress(
                job.document_id,
                100,
                "completed",
                f"Extraction complete! Created {len(chunk_ids)} chunks"
            )

            logger.info(f"[extract_text] Completed for document {job.document_id}")
            return {
                "status": "success",
                "document_id": job.document_id,
                "text_length": len(text),
                "chunks_created": len(chunk_ids),
            }

        except FileNotFoundError as e:
            logger.error(f"[extract_text] File not found: {e}")
            raise PermanentError(f"File not found: {e}") from e
        except ValueError as e:
            logger.error(f"[extract_text] Invalid input: {e}")
            raise PermanentError(f"Invalid input: {e}") from e
        except PermanentError as e:
            logger.error(f"[extract_text] Permanent error: {e}")
            raise
        except Exception as e:
            logger.error(f"[extract_text] Error for document {job_payload.get('document_id')}: {e}")
            raise self.retry(exc=e, countdown=60)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_extract())
