"""Background job processor for matter document analysis"""
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from backend.models import ProcessingJob, Matter, Chunk
from backend.services.storage import download_document_from_blob
from backend.services.chunking import chunk_document_from_blob
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


def mark_job_complete(matter_id: str, db: Session) -> bool:
    """Mark a job as complete"""
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.matter_id == matter_id).first()
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
    matter_id: str,
    db: Session,
    error_message: str,
    next_retry_at: datetime = None
) -> bool:
    """Mark a job as failed and schedule retry if attempts remaining"""
    try:
        job = db.query(ProcessingJob).filter(ProcessingJob.matter_id == matter_id).first()
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


async def process_matter(matter_id: str, db: Session) -> Dict:
    """
    Process a matter: download document, chunk, embed, and store vectors.

    Args:
        matter_id: UUID of the matter to process
        db: Database session

    Returns:
        Dict with keys: success (bool), chunks_created (int), error (str if failed)
    """
    try:
        # Get matter from database
        matter = db.query(Matter).filter(Matter.id == matter_id).first()
        if not matter:
            raise ValueError(f"Matter not found: {matter_id}")

        logger.info(f"Processing matter: {matter_id}")

        # Get file type
        file_type = matter.file_type or "pdf"

        # Download document from blob storage
        document_bytes = await download_document_from_blob(matter.blob_storage_path)
        logger.info(f"Downloaded {file_type.upper()} ({len(document_bytes)} bytes)")

        # Chunk document
        chunks_data = chunk_document_from_blob(document_bytes, file_type=file_type)
        if not chunks_data:
            raise ValueError(f"No chunks created from {file_type.upper()}")
        logger.info(f"Created {len(chunks_data)} chunks")

        # Delete old chunks for this matter (for reprocessing)
        db.query(Chunk).filter(Chunk.matter_id == matter_id).delete()
        db.commit()

        # Embed chunks
        chunk_contents = [c["content"] for c in chunks_data]
        embeddings = embed_chunks(chunk_contents)
        logger.info(f"Created {len(embeddings)} embeddings")

        # Create vector store collection
        create_collection(str(matter_id))

        # Upsert vectors
        vectors_count = upsert_vectors(str(matter_id), chunks_data, embeddings)
        logger.info(f"Upserted {vectors_count} vectors")

        # Create Chunk records in database
        for i, chunk_data in enumerate(chunks_data):
            chunk = Chunk(
                matter_id=matter_id,
                page_num=chunk_data.get("page_num"),
                section_name=chunk_data.get("section_name"),
                content=chunk_data.get("content"),
                chunk_sequence=i + 1
            )
            db.add(chunk)

        # Update matter status to "ready"
        matter.status = "ready"
        db.commit()

        # Update job status to "completed"
        job = db.query(ProcessingJob).filter(ProcessingJob.matter_id == matter_id).first()
        if job:
            job.status = "completed"
            job.completed_at = datetime.now(timezone.utc)
            db.commit()

        logger.info(f"Successfully processed matter {matter_id}")
        return {"success": True, "chunks_created": len(chunks_data)}

    except Exception as e:
        logger.error(f"Error processing matter {matter_id}: {str(e)}")
        db.rollback()

        # Update matter status to failed
        matter = db.query(Matter).filter(Matter.id == matter_id).first()
        if matter:
            matter.status = "error"
            db.commit()

        return {"success": False, "error": str(e)}


async def run_job_worker(
    db: Session,
    max_jobs_per_batch: int = 5,
    sleep_interval: int = 10,
    max_iterations: Optional[int] = None
) -> None:
    """
    Run job worker that continuously processes pending jobs in batches.

    Args:
        db: Database session
        max_jobs_per_batch: Maximum jobs to process per batch (default: 5)
        sleep_interval: Seconds to sleep between batches (default: 10)
        max_iterations: For testing: stop after N iterations (default: None for infinite)
    """
    iteration = 0
    while True:
        try:
            # Check if we should stop (for testing)
            if max_iterations is not None and iteration >= max_iterations:
                logger.info(f"Job worker reached max iterations ({max_iterations}), stopping")
                break

            iteration += 1
            logger.debug(f"Job worker iteration {iteration}")

            # Get pending jobs
            pending_jobs = get_pending_jobs(db, limit=max_jobs_per_batch)
            if not pending_jobs:
                logger.debug("No pending jobs, sleeping")
                await asyncio.sleep(sleep_interval)
                continue

            logger.info(f"Processing {len(pending_jobs)} pending jobs")

            # Process each job sequentially
            for job in pending_jobs:
                try:
                    logger.info(f"Processing job {job.id} for matter {job.matter_id}")

                    # Update job to processing
                    job.status = "processing"
                    job.started_at = datetime.now(timezone.utc)
                    db.commit()

                    # Process the matter
                    result = await process_matter(job.matter_id, db)

                    if result["success"]:
                        logger.info(f"Job {job.id} completed successfully")
                    else:
                        logger.error(f"Job {job.id} failed: {result.get('error')}")
                        # Schedule retry
                        delay_seconds = calculate_retry_delay(job.attempts + 1)
                        retry_time = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                        mark_job_failed(job.matter_id, db, result.get("error"), next_retry_at=retry_time)

                except Exception as e:
                    logger.error(f"Error processing job {job.id}: {str(e)}")
                    delay_seconds = calculate_retry_delay(job.attempts + 1)
                    retry_time = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
                    mark_job_failed(job.matter_id, db, str(e), next_retry_at=retry_time)

            # Sleep before next batch
            logger.debug(f"Batch complete, sleeping {sleep_interval} seconds")
            await asyncio.sleep(sleep_interval)

        except Exception as e:
            logger.error(f"Job worker error: {str(e)}")
            await asyncio.sleep(sleep_interval)
