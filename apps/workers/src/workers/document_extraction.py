"""Document text extraction worker."""

import asyncio
import logging
import re
from pathlib import Path
from uuid import uuid4
from celery import Task
from celery_app import celery_app
from lib import get_redis_client, ProgressPublisher
from shared import (
    Document,
    DocumentChunk,
    ProcessingStatus,
    DocumentExtractionJob,
    PermanentError,
    RetryableError,
    async_session,
)
from sqlalchemy.future import select

logger = logging.getLogger(__name__)


# ===== TEXT EXTRACTION FUNCTIONS =====
def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
    min_size: int = 200,
) -> list:
    """Split text into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        slice_text = text[start:end].rstrip()

        if len(slice_text) >= min_size or end >= text_len:
            chunks.append(slice_text)
            logger.debug(
                f"[chunking] Created chunk {len(chunks)}: "
                f"{len(slice_text)} chars at position {start}"
            )

        next_start = end - overlap
        if next_start <= start:
            break

        start = next_start

    logger.info(
        f"[chunking] Split {text_len} chars into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )
    return chunks


def clean_text(text: str) -> str:
    """Clean extracted text by removing artifacts and normalizing whitespace."""
    if not text:
        return text

    # Remove NULL bytes
    text = text.replace('\x00', '')

    # Normalize line endings (CRLF to LF)
    text = text.replace('\r\n', '\n')

    # Convert form feeds to newlines
    text = text.replace('\f', '\n')

    # Remove control characters (except \n, \t)
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    # Remove page numbers
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove page headers
    text = re.sub(r'\nPage\s+\d+.*\n', '\n', text)

    # Normalize multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Final trim
    text = text.strip()

    logger.debug(f"[extraction] Cleaned text: {len(text)} chars")
    return text


async def extract_file(file_path: str) -> str:
    """Extract text from file based on file type."""
    file_lower = file_path.lower()

    if file_lower.endswith('.pdf'):
        return await extract_pdf(file_path)
    elif file_lower.endswith('.docx'):
        return await extract_docx(file_path)
    elif file_lower.endswith('.txt'):
        return await extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


async def extract_pdf(file_path: str) -> str:
    """Extract text from PDF file."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise PermanentError("PyPDF2 not installed in worker environment")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.warning(f"[extraction] No text on PDF page {page_num + 1}: {file_path}")

            text = "\n".join(text_parts)
            logger.info(f"[extraction] Extracted {len(text)} chars from PDF: {file_path}")
            return text

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract PDF {file_path}: {e}")
        raise ValueError(f"Failed to extract PDF: {e}") from e


async def extract_docx(file_path: str) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
    except ImportError:
        raise PermanentError("python-docx not installed in worker environment")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        doc = Document(file_path)
        text_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        text = "\n".join(text_parts)
        logger.info(f"[extraction] Extracted {len(text)} chars from DOCX: {file_path}")
        return text

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract DOCX {file_path}: {e}")
        raise ValueError(f"Failed to extract DOCX: {e}") from e


async def extract_txt(file_path: str) -> str:
    """Extract text from TXT file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    try:
        text = path.read_text(encoding='utf-8')
        logger.info(f"[extraction] Extracted {len(text)} chars from TXT: {file_path}")
        return text
    except UnicodeDecodeError:
        text = path.read_text(encoding='latin-1')
        logger.warning(f"[extraction] Used latin-1 encoding for TXT: {file_path}")
        return text
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract TXT {file_path}: {e}")
        raise


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
                chunk_ids = []

                for chunk_index, chunk_text_item in enumerate(chunk_texts):
                    chunk = DocumentChunk(
                        id=str(uuid4()),
                        document_id=job.document_id,
                        chunk_text=chunk_text_item,
                        chunk_index=chunk_index,
                    )
                    db.add(chunk)
                    chunk_ids.append(chunk.id)

                    logger.debug(
                        f"[chunks] Created chunk {chunk_index}: {len(chunk_text_item)} chars"
                    )

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
