"""Background worker for processing cases"""
import asyncio
import logging
import time
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.database import SessionLocal
from backend.models import ProcessingJob, Case, Chunk
from backend.services.storage import download_pdf_from_blob
from backend.services.chunking import chunk_pdf_from_blob
from backend.services.embeddings import embed_chunks
from backend.services.vector_store import upsert_vectors, create_collection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_job(job: ProcessingJob, db: Session) -> bool:
    """Process a single job"""
    try:
        logger.info(f"Processing job {job.id} for case {job.case_id}")

        # Get case
        case = db.query(Case).filter(Case.id == job.case_id).first()
        if not case:
            logger.error(f"Case {job.case_id} not found")
            job.status = "failed"
            job.error_message = "Case not found"
            db.commit()
            return False

        # Update job to processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        db.commit()

        # Download PDF from blob storage
        logger.info(f"Downloading PDF from {case.blob_storage_path}")
        pdf_content = await download_pdf_from_blob(case.blob_storage_path)

        # Chunk PDF
        logger.info("Chunking PDF")
        chunks = await chunk_pdf_from_blob(pdf_content)
        logger.info(f"Created {len(chunks)} chunks")

        # Extract content for embeddings
        chunk_contents = [chunk["content"] for chunk in chunks]

        # Create embeddings
        logger.info("Generating embeddings")
        embeddings = embed_chunks(chunk_contents)

        # Ensure collection exists
        await create_collection("legal_rag")

        # Create collection first
        logger.info("Creating Qdrant collection")
        create_collection(str(case.id))

        # Add IDs to chunks for vector store
        chunks_with_ids = []
        for idx, chunk in enumerate(chunks):
            chunk["id"] = f"{case.id}:{idx}"
            chunks_with_ids.append(chunk)

        # Store vectors in Qdrant
        logger.info("Storing vectors in Qdrant")
        upsert_vectors(
            case_id=str(case.id),
            chunks=chunks_with_ids,
            embeddings=embeddings
        )

        # Store chunk metadata in PostgreSQL
        logger.info("Storing chunk metadata in database")
        for idx, chunk in enumerate(chunks):
            db_chunk = Chunk(
                id=None,  # Will be auto-generated
                case_id=case.id,
                page_num=chunk.get("page_num"),
                section_name=chunk.get("section_name"),
                content=chunk.get("content"),
                chunk_sequence=idx
            )
            db.add(db_chunk)

        # Update case status to ready
        case.status = "ready"

        # Mark job as completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)

        db.commit()
        logger.info(f"Successfully processed job {job.id}")
        return True

    except Exception as e:
        logger.error(f"Error processing job {job.id}: {str(e)}", exc_info=True)

        job.status = "failed"
        job.error_message = str(e)
        job.attempts = job.attempts + 1

        # Update case status to error if max attempts reached
        case = db.query(Case).filter(Case.id == job.case_id).first()
        if case and job.attempts >= job.max_attempts:
            case.status = "error"

        db.commit()
        return False


async def worker_loop(check_interval: int = 5):
    """Main worker loop - continuously processes pending jobs"""
    logger.info("Starting worker loop")

    while True:
        db = SessionLocal()
        try:
            # Get pending jobs
            pending_jobs = db.query(ProcessingJob).filter(
                ProcessingJob.status == "pending"
            ).order_by(ProcessingJob.created_at).limit(5).all()

            if pending_jobs:
                logger.info(f"Found {len(pending_jobs)} pending jobs")
                for job in pending_jobs:
                    await process_job(job, db)
            else:
                logger.debug("No pending jobs")

            await asyncio.sleep(check_interval)

        except Exception as e:
            logger.error(f"Error in worker loop: {str(e)}", exc_info=True)
            await asyncio.sleep(check_interval)
        finally:
            db.close()


if __name__ == "__main__":
    asyncio.run(worker_loop())
