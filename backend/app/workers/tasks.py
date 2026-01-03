import logging
from celery import shared_task, Task
from app.database import async_session
from app.models import Document, ProcessingStatus
from app.config import settings
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
    Extract text from document (PDF, DOCX, TXT, etc.)
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        # TODO: Implement text extraction
        # 1. Get document from DB
        # 2. Download from local file path
        # 3. Extract text based on file type
        # 4. Update Document.extracted_text
        # 5. Queue embedding generation task

        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)

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
