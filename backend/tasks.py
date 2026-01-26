"""Celery tasks for document processing"""
import logging
from datetime import datetime, timezone
from uuid import UUID

# Handle both module import styles
try:
    from backend.celery_app import celery_app
    from backend.database import get_session_factory
    from backend.models import Case, Chunk
    from backend.services.storage import download_pdf_from_blob
    from backend.services.chunking import chunk_pdf_from_blob
    from backend.services.embeddings import embed_chunks
    from backend.services.vector_store import upsert_vectors, create_collection
except ImportError:
    from celery_app import celery_app
    from database import get_session_factory
    from models import Case, Chunk
    from services.storage import download_pdf_from_blob
    from services.chunking import chunk_pdf_from_blob
    from services.embeddings import embed_chunks
    from services.vector_store import upsert_vectors, create_collection

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
    track_started=True
)
def process_document_task(self, case_id: str):
    """
    Process a case document: chunk, embed, and store in vector DB.

    Args:
        case_id: UUID of the case to process

    Returns:
        dict with status and result
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        logger.info(f"[Task {self.request.id}] Processing case {case_id}")

        # Get case
        case = db.query(Case).filter(Case.id == UUID(case_id)).first()
        if not case:
            logger.error(f"Case {case_id} not found")
            return {"status": "failed", "error": "Case not found"}

        # Update case status to processing
        case.status = "processing"
        db.commit()

        # 1. Download PDF from blob storage
        logger.info(f"[Task {self.request.id}] Downloading PDF from {case.blob_storage_path}")
        pdf_content = download_pdf_from_blob(case.blob_storage_path)

        # 2. Chunk PDF
        logger.info(f"[Task {self.request.id}] Chunking PDF")
        chunks = chunk_pdf_from_blob(pdf_content)
        logger.info(f"[Task {self.request.id}] Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No chunks extracted from PDF")

        # 3. Extract content for embeddings
        chunk_contents = [chunk["content"] for chunk in chunks]

        # 4. Generate embeddings
        logger.info(f"[Task {self.request.id}] Generating embeddings for {len(chunks)} chunks")
        embeddings = embed_chunks(chunk_contents)

        # 5. Create Qdrant collection
        logger.info(f"[Task {self.request.id}] Creating vector collection")
        create_collection(case_id)

        # 6. Add IDs to chunks
        chunks_with_ids = []
        for idx, chunk in enumerate(chunks):
            chunk["id"] = f"{case_id}:{idx}"
            chunks_with_ids.append(chunk)

        # 7. Store vectors in Qdrant
        logger.info(f"[Task {self.request.id}] Upserting vectors to Qdrant")
        upsert_vectors(
            case_id=case_id,
            chunks=chunks_with_ids,
            embeddings=embeddings
        )

        # 8. Store chunk metadata in PostgreSQL
        logger.info(f"[Task {self.request.id}] Storing chunks in database")
        for idx, chunk in enumerate(chunks):
            db_chunk = Chunk(
                case_id=UUID(case_id),
                page_num=chunk.get("page_num"),
                section_name=chunk.get("section_name"),
                content=chunk.get("content"),
                chunk_sequence=idx
            )
            db.add(db_chunk)

        # 9. Update case status to ready
        case.status = "ready"
        case.updated_at = datetime.now(timezone.utc)
        db.commit()

        logger.info(f"[Task {self.request.id}] Successfully processed case {case_id}")
        return {
            "status": "success",
            "case_id": case_id,
            "chunks_processed": len(chunks)
        }

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Error processing case {case_id}: {str(exc)}", exc_info=True)

        # Update case status to error
        try:
            case = db.query(Case).filter(Case.id == UUID(case_id)).first()
            if case:
                case.status = "error"
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update case status: {str(e)}")

        # Retry with exponential backoff
        try:
            raise self.retry(exc=exc)
        except self.MaxRetriesExceededError:
            logger.error(f"[Task {self.request.id}] Max retries exceeded for case {case_id}")
            return {
                "status": "failed",
                "case_id": case_id,
                "error": str(exc),
                "retries_exhausted": True
            }

    finally:
        db.close()


# Export task
__all__ = ["process_document_task"]
