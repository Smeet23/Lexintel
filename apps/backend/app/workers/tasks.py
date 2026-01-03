import logging
from celery import shared_task, Task
from app.database import async_session
from app.models import Document, ProcessingStatus
from app.config import settings
from app.services.extraction import extract_file, create_text_chunks
import asyncio
import os

logger = logging.getLogger(__name__)

class CallbackTask(Task):
    """Task with on_success and on_failure callbacks"""
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"[workers] Task {task_id} succeeded")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"[workers] Task {task_id} failed: {exc}")

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    """
    Extract text from document (PDF, DOCX, TXT, etc.) and create chunks
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        # Run async operations
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            _extract_and_chunk_document(document_id)
        )

        logger.info(f"[extract_text] Completed for document {document_id}")
        return result
    except FileNotFoundError as e:
        # File not found: permanent error, don't retry
        logger.error(f"[extract_text] File not found for {document_id}: {e}")
        raise
    except Exception as exc:
        # Other errors: retry with backoff
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)


async def _extract_and_chunk_document(document_id: str) -> dict:
    """
    Async helper to extract text and create chunks

    Flow:
    1. Get Document from DB
    2. Validate file exists
    3. Extract text from file
    4. Clean and split into chunks
    5. Create DocumentChunk records
    6. Update Document status
    7. Queue embeddings task
    """
    from sqlalchemy import select

    async with async_session() as session:
        # Step 1: Get document from database
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        document = result.scalars().first()

        if not document:
            raise ValueError(f"Document {document_id} not found")

        logger.info(
            f"[extract_text] Processing document {document_id}: {document.filename}"
        )

        # Step 2: Validate file exists
        if not document.file_path or not os.path.exists(document.file_path):
            raise FileNotFoundError(
                f"File not found for document {document_id}: {document.file_path}"
            )

        try:
            # Step 3: Extract text from file
            logger.info(f"[extract_text] Extracting text from {document.file_path}")
            text = await extract_file(document.file_path)
            logger.info(
                f"[extract_text] Extracted {len(text)} characters from {document_id}"
            )

            # Step 4 & 5: Split text into chunks and create records
            logger.info(f"[extract_text] Creating chunks for document {document_id}")
            chunk_ids = await create_text_chunks(
                document_id=document_id,
                text=text,
                session=session,
            )

            # Step 6: Update document with extracted text and status
            document.extracted_text = text
            document.processing_status = ProcessingStatus.EXTRACTED

            await session.commit()

            logger.info(
                f"[extract_text] Successfully extracted {len(chunk_ids)} chunks "
                f"for document {document_id}"
            )

            # Step 7: Queue embeddings task (Phase 4)
            from app.workers.tasks import generate_embeddings
            generate_embeddings.delay(document_id)
            logger.info(f"[extract_text] Queued embeddings task for {document_id}")

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": len(chunk_ids),
                "text_length": len(text),
            }

        except FileNotFoundError as e:
            # File errors are permanent, don't retry
            logger.error(f"[extract_text] File error for {document_id}: {e}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
            raise

        except Exception as e:
            # Other errors: update status and re-raise for retry
            logger.error(f"[extract_text] Error processing {document_id}: {e}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
            raise

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def generate_embeddings(self, document_id: str):
    """
    Generate embeddings for document chunks using OpenAI
    """
    try:
        logger.info(f"[generate_embeddings] Starting for document {document_id}")

        # TODO: Implement embedding generation
        # 1. Get document chunks from DB
        # 2. Call OpenAI embedding API
        # 3. Store embeddings in DocumentChunk.embedding
        # 4. Update Document status to INDEXED

        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[generate_embeddings] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)

@shared_task(base=CallbackTask, bind=True)
def process_document_pipeline(self, document_id: str):
    """
    Complete document processing pipeline:
    1. Extract text
    2. Create chunks
    3. Generate embeddings
    """
    try:
        logger.info(f"[process_pipeline] Starting for document {document_id}")

        # Queue text extraction first
        extract_text_from_document.delay(document_id)

        return {"status": "processing", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[process_pipeline] Error: {exc}")
        raise
