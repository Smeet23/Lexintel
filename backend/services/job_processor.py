"""Background job processor for case document analysis"""
import logging
from datetime import datetime, timezone
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from backend.models import ProcessingJob, Case, Chunk
from backend.services.storage import download_pdf_from_blob
from backend.services.chunking import chunk_pdf_from_blob
from backend.services.embeddings import embed_chunks
from backend.services.vector_store import upsert_vectors, create_collection

logger = logging.getLogger(__name__)


def get_pending_jobs(db: Session, limit: int = 5) -> List[ProcessingJob]:
    """Get pending jobs in FIFO order"""
    return db.query(ProcessingJob).filter(
        ProcessingJob.status == "pending"
    ).order_by(ProcessingJob.created_at).limit(limit).all()


def calculate_retry_delay(attempt: int) -> int:
    """Calculate retry delay in seconds based on attempt number"""
    delays = {1: 0, 2: 5, 3: 10}
    return delays.get(attempt, 10)


def mark_job_complete(case_id: str, db: Session) -> bool:
    """Mark a job as complete"""
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.case_id == case_id).first()
        if not job:
            return False
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error marking job complete: {str(e)}")
        db.rollback()
        return False


def mark_job_failed(
    case_id: str,
    db: Session,
    error_message: str,
    next_retry_at: datetime = None
) -> bool:
    """Mark a job as failed and schedule retry if attempts remaining"""
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.case_id == case_id).first()
        if not job:
            return False

        job.attempts += 1
        job.error_message = error_message
        job.next_retry_at = next_retry_at

        # Check if max attempts exceeded
        if job.attempts >= job.max_attempts:
            job.status = "failed"
        else:
            job.status = "pending"

        db.commit()
        return True
    except Exception as e:
        logger.error(f"Error marking job failed: {str(e)}")
        db.rollback()
        return False


async def process_case(case_id: str, db: Session) -> Dict:
    """
    Process a case: download PDF, chunk, embed, and store vectors.

    Args:
        case_id: UUID of the case to process
        db: Database session

    Returns:
        Dict with keys: success (bool), chunks_created (int), error (str if failed)
    """
    try:
        # Get case from database
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            raise ValueError(f"Case not found: {case_id}")

        logger.info(f"Processing case: {case_id}")

        # Download PDF from blob storage
        pdf_bytes = await download_pdf_from_blob(case.blob_storage_path)
        logger.info(f"Downloaded PDF ({len(pdf_bytes)} bytes)")

        # Chunk PDF
        chunks_data = chunk_pdf_from_blob(pdf_bytes)
        if not chunks_data:
            raise ValueError("No chunks created from PDF")
        logger.info(f"Created {len(chunks_data)} chunks")

        # Delete old chunks for this case (for reprocessing)
        db.query(Chunk).filter(Chunk.case_id == case_id).delete()
        db.commit()

        # Embed chunks
        chunk_contents = [c["content"] for c in chunks_data]
        embeddings = embed_chunks(chunk_contents)
        logger.info(f"Created {len(embeddings)} embeddings")

        # Create vector store collection
        create_collection(str(case_id))

        # Upsert vectors
        vectors_count = upsert_vectors(str(case_id), chunks_data, embeddings)
        logger.info(f"Upserted {vectors_count} vectors")

        # Create Chunk records in database
        for i, chunk_data in enumerate(chunks_data):
            chunk = Chunk(
                case_id=case_id,
                page_num=chunk_data.get("page_num"),
                section_name=chunk_data.get("section_name"),
                content=chunk_data.get("content"),
                chunk_sequence=i + 1
            )
            db.add(chunk)

        # Update case status to "ready"
        case.status = "ready"
        db.commit()

        # Update job status to "completed"
        job = db.query(ProcessingJob).filter(ProcessingJob.case_id == case_id).first()
        if job:
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

        logger.info(f"Successfully processed case {case_id}")
        return {"success": True, "chunks_created": len(chunks_data)}

    except Exception as e:
        logger.error(f"Error processing case {case_id}: {str(e)}")
        db.rollback()
        return {"success": False, "error": str(e)}
