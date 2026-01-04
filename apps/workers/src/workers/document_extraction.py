"""Document text extraction worker."""

import asyncio
import logging
from celery import Task
from ..celery_app import celery_app
from ..lib import get_redis_client, ProgressPublisher
from shared import (
    Document,
    ProcessingStatus,
    DocumentExtractionJob,
    PermanentError,
    async_session,
    chunk_text,
    clean_text,
    extract_file,
    create_document_chunks,
)
from sqlalchemy.future import select

logger = logging.getLogger(__name__)

# Import embeddings service - deferred to avoid circular imports
async def _get_create_chunk_embeddings():
    """Lazy import of embeddings service from shared package."""
    from shared import create_chunk_embeddings
    return create_chunk_embeddings


class CallbackTask(Task):
    """Task with error handling callback."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}", exc_info=einfo)


@celery_app.task(
    base=CallbackTask,
    bind=True,
    max_retries=3,
    name="workers.document_extraction.extract_text_from_document"
)
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

                if not doc.blob_url:
                    raise ValueError(f"Document {job.document_id} has no blob_url")

                # Publish extraction start
                await publisher.publish_progress(
                    job.document_id,
                    10,
                    "extracting",
                    "Extracting text from file..."
                )

                # Download file from presigned URL
                import aiohttp
                import tempfile
                from pathlib import Path as PathlibPath

                async with aiohttp.ClientSession() as session:
                    async with session.get(doc.blob_url) as response:
                        if response.status != 200:
                            raise ValueError(f"Failed to download file: HTTP {response.status}")
                        file_content = await response.read()

                # Save to temporary file for extraction
                file_extension = PathlibPath(doc.filename).suffix
                with tempfile.NamedTemporaryFile(suffix=file_extension, delete=False) as tmp_file:
                    tmp_file.write(file_content)
                    tmp_file_path = tmp_file.name

                try:
                    # Extract text from temporary file
                    text = await extract_file(tmp_file_path)
                    text = clean_text(text)
                    doc.extracted_text = text
                finally:
                    # Clean up temporary file
                    PathlibPath(tmp_file_path).unlink(missing_ok=True)

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

                # Update document status to EXTRACTED
                doc.processing_status = ProcessingStatus.EXTRACTED
                await db.commit()

                logger.info(
                    f"[extract_text] Created {len(chunk_ids)} chunks for document {job.document_id}"
                )

                # Step 3: Generate embeddings (NEW - Phase 4)
                try:
                    await publisher.publish_progress(
                        job.document_id,
                        70,
                        "generating_embeddings",
                        f"Generating embeddings for {len(chunk_texts)} chunks..."
                    )

                    create_chunk_embeddings = await _get_create_chunk_embeddings()
                    await create_chunk_embeddings(
                        db,
                        job.document_id,
                        chunk_texts,
                        batch_size=20
                    )

                    # Mark as complete (INDEXED status)
                    doc.processing_status = ProcessingStatus.INDEXED
                    await db.commit()

                    logger.info(
                        f"[extract_text] Generated embeddings for {len(chunk_ids)} chunks"
                    )

                except Exception as e:
                    # If embedding fails, keep document in EXTRACTED status
                    # This allows the text to be preserved for a retry via the reingest API
                    error_message = (
                        f"Embedding generation failed: {type(e).__name__}: {str(e)}"
                    )
                    logger.error(
                        f"[extract_text] {error_message} for {job.document_id}",
                        exc_info=True
                    )
                    doc.processing_status = ProcessingStatus.EXTRACTED
                    doc.error_message = error_message
                    await db.commit()

                    # Publish error to progress stream
                    await publisher.publish_progress(
                        job.document_id,
                        70,
                        "embedding_failed",
                        f"Embedding generation failed: {str(e)}"
                    )

                    # Re-raise the exception so it's handled by outer error handling
                    raise

            # Publish completion
            await publisher.publish_progress(
                job.document_id,
                100,
                "completed",
                f"Extraction and embedding complete! {len(chunk_ids)} chunks indexed"
            )

            logger.info(f"[extract_text] Completed for document {job.document_id}")
            return {
                "status": "success",
                "document_id": job.document_id,
                "text_length": len(text),
                "chunks_created": len(chunk_ids),
            }

        except FileNotFoundError as e:
            error_msg = f"Document not found in database: {e}"
            logger.error(f"[extract_text] {error_msg}")
            raise PermanentError(error_msg) from e
        except ValueError as e:
            error_msg = f"Invalid input - missing blob_url or invalid file: {e}"
            logger.error(f"[extract_text] {error_msg}")
            raise PermanentError(error_msg) from e
        except PermanentError as e:
            logger.error(f"[extract_text] Permanent error (no retry): {e}")
            raise
        except Exception as e:
            error_type = type(e).__name__
            error_msg = f"Unexpected error ({error_type}): {str(e)}"
            logger.error(
                f"[extract_text] {error_msg} for document {job_payload.get('document_id')}",
                exc_info=True
            )
            # Retry with exponential backoff (60s first retry)
            logger.info(f"[extract_text] Retrying task (attempt {self.request.retries + 1}/3)")
            raise self.retry(exc=e, countdown=60)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(async_extract())
