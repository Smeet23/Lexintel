"""Celery tasks for document processing"""
import logging
from datetime import datetime, timezone
from uuid import UUID
from celery import shared_task

# Handle both module import styles
try:
    from backend.database import get_session_factory
    from backend.models import Matter, Chunk
    from backend.services.storage import download_document_from_blob
    from backend.services.chunking import chunk_document_from_blob
    from backend.services.embeddings import embed_chunks
    from backend.services.vector_store import upsert_vectors, create_collection
    from backend.services.progress import (
        publish_downloading, publish_chunking, publish_embedding,
        publish_indexing, publish_storing, publish_ready,
        publish_error, publish_retrying
    )
except ImportError:
    from database import get_session_factory
    from models import Matter, Chunk
    from services.storage import download_document_from_blob
    from services.chunking import chunk_document_from_blob
    from services.embeddings import embed_chunks
    from services.vector_store import upsert_vectors, create_collection
    from services.progress import (
        publish_downloading, publish_chunking, publish_embedding,
        publish_indexing, publish_storing, publish_ready,
        publish_error, publish_retrying
    )

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
    track_started=True
)
def process_document_task(self, matter_id: str):
    """
    Process a matter document: chunk, embed, and store in vector DB.

    Args:
        matter_id: UUID of the matter to process

    Returns:
        dict with status and result
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        logger.info(f"[Task {self.request.id}] Processing matter {matter_id}")

        # Get matter
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        if not matter:
            logger.error(f"Matter {matter_id} not found")
            return {"status": "failed", "error": "Matter not found"}

        # Update matter status to processing
        matter.status = "processing"
        db.commit()

        # Get file type from matter
        file_type = matter.file_type or "pdf"

        # 1. Download document from blob storage
        logger.info(f"[Task {self.request.id}] Downloading {file_type.upper()} from {matter.blob_storage_path}")
        publish_downloading(matter_id)
        document_content = download_document_from_blob(matter.blob_storage_path)

        # 2. Chunk document
        logger.info(f"[Task {self.request.id}] Chunking {file_type.upper()}")
        publish_chunking(matter_id, progress=0)
        chunks = chunk_document_from_blob(document_content, file_type=file_type)
        publish_chunking(matter_id, progress=100, current=len(chunks), total=len(chunks))
        logger.info(f"[Task {self.request.id}] Created {len(chunks)} chunks")

        if not chunks:
            raise ValueError("No chunks extracted from document")

        # 3. Extract content for embeddings
        chunk_contents = [chunk["content"] for chunk in chunks]

        # 4. Generate embeddings (with progress updates)
        logger.info(f"[Task {self.request.id}] Generating embeddings for {len(chunks)} chunks")
        publish_embedding(matter_id, progress=0, current=0, total=len(chunks))

        # Process embeddings in batches for progress reporting
        embeddings = []
        batch_size = 20
        for i in range(0, len(chunk_contents), batch_size):
            batch = chunk_contents[i:i + batch_size]
            batch_embeddings = embed_chunks(batch)
            embeddings.extend(batch_embeddings)

            # Update progress
            processed = min(i + batch_size, len(chunk_contents))
            progress = int((processed / len(chunk_contents)) * 100)
            publish_embedding(matter_id, progress=progress, current=processed, total=len(chunks))

        # 5. Create Qdrant collection
        logger.info(f"[Task {self.request.id}] Creating vector collection")
        publish_indexing(matter_id, progress=0, detail="Creating collection...")
        create_collection(matter_id)
        publish_indexing(matter_id, progress=30, detail="Collection created")

        # 6. Add IDs to chunks
        chunks_with_ids = []
        for idx, chunk in enumerate(chunks):
            chunk["id"] = f"{matter_id}:{idx}"
            chunks_with_ids.append(chunk)

        # 7. Store vectors in Qdrant
        logger.info(f"[Task {self.request.id}] Upserting vectors to Qdrant")
        publish_indexing(matter_id, progress=50, detail="Upserting vectors...")
        upsert_vectors(
            matter_id=matter_id,
            chunks=chunks_with_ids,
            embeddings=embeddings
        )
        publish_indexing(matter_id, progress=100, detail=f"{len(chunks)} vectors indexed")

        # 8. Store chunk metadata in PostgreSQL
        logger.info(f"[Task {self.request.id}] Storing chunks in database")
        publish_storing(matter_id)
        for idx, chunk in enumerate(chunks):
            db_chunk = Chunk(
                matter_id=UUID(matter_id),
                page_num=chunk.get("page_num"),
                section_name=chunk.get("section_name"),
                content=chunk.get("content"),
                chunk_sequence=idx
            )
            db.add(db_chunk)

        # 9. Update matter status to ready
        matter.status = "ready"
        matter.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Publish ready event
        publish_ready(matter_id, len(chunks))

        logger.info(f"[Task {self.request.id}] Successfully processed matter {matter_id}")
        return {
            "status": "success",
            "matter_id": matter_id,
            "chunks_processed": len(chunks)
        }

    except Exception as exc:
        logger.error(f"[Task {self.request.id}] Error processing matter {matter_id}: {str(exc)}", exc_info=True)

        retry_count = self.request.retries
        max_retries = self.max_retries

        # Check if we should retry
        if retry_count < max_retries:
            # Publish retrying event
            publish_retrying(matter_id, retry_count + 1, max_retries, str(exc))
            logger.info(f"[Task {self.request.id}] Retrying matter {matter_id} (attempt {retry_count + 1}/{max_retries})")
            raise self.retry(exc=exc)

        # Max retries exceeded - update matter status to error
        try:
            matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
            if matter:
                matter.status = "error"
                db.commit()
        except Exception as e:
            logger.error(f"Failed to update matter status: {str(e)}")

        # Publish error event
        publish_error(matter_id, str(exc), retry_count)

        logger.error(f"[Task {self.request.id}] Max retries exceeded for matter {matter_id}")
        return {
            "status": "failed",
            "matter_id": matter_id,
            "error": str(exc),
            "retries_exhausted": True
        }

    finally:
        db.close()


# Export task
__all__ = ["process_document_task"]
