"""Celery tasks for document processing"""
import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4
from celery import shared_task

# Handle both module import styles
try:
    from backend.database import get_session_factory
    from backend.models import Matter, Chunk, Document
    from backend.services.storage import download_document_from_blob
    from backend.services.chunking import chunk_document_from_blob
    from backend.services.embeddings import embed_chunks
    from backend.services.vector_store import upsert_vectors, create_collection
    from backend.services.keyword_extractor import extract_chunk_keywords
    from backend.services.document_summary import generate_doc_summary, classify_document
    from backend.services.progress import (
        publish_downloading, publish_chunking, publish_embedding,
        publish_indexing, publish_storing, publish_enriching, publish_ready,
        publish_error, publish_retrying
    )
    from backend.services.audit import log_activity
except ImportError:
    from database import get_session_factory
    from models import Matter, Chunk, Document
    from services.storage import download_document_from_blob
    from services.chunking import chunk_document_from_blob
    from services.embeddings import embed_chunks
    from services.vector_store import upsert_vectors, create_collection
    from services.keyword_extractor import extract_chunk_keywords
    from services.document_summary import generate_doc_summary, classify_document
    from services.progress import (
        publish_downloading, publish_chunking, publish_embedding,
        publish_indexing, publish_storing, publish_enriching, publish_ready,
        publish_error, publish_retrying
    )
    from services.audit import log_activity

logger = logging.getLogger(__name__)


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
    track_started=True
)
def process_document_task(self, matter_id: str, document_id: str):
    """
    Process a document: chunk, embed, and store in vector DB.

    Reads blob path from Document record and updates Document.status.
    Matter status is derived from all its documents' statuses.

    Args:
        matter_id: UUID of the matter
        document_id: UUID of the document to process

    Returns:
        dict with status and result
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        logger.info(f"[Task {self.request.id}] Processing matter {matter_id}, document {document_id}")

        # Get matter
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        if not matter:
            logger.error(f"Matter {matter_id} not found")
            return {"status": "failed", "error": "Matter not found"}

        # Get document
        document = db.query(Document).filter(Document.id == UUID(document_id)).first()
        if not document:
            logger.error(f"Document {document_id} not found")
            return {"status": "failed", "error": "Document not found"}

        document.status = "processing"
        matter.status = "processing"
        db.commit()

        file_type = document.file_type or "pdf"
        blob_path = document.blob_storage_path

        # 1. Download document from blob storage
        logger.info(f"[Task {self.request.id}] Downloading {file_type.upper()} from {blob_path}")
        publish_downloading(matter_id)
        document_content = download_document_from_blob(blob_path)

        # 2. Chunk document
        logger.info(f"[Task {self.request.id}] Chunking {file_type.upper()}")
        publish_chunking(matter_id, progress=0)
        chunks = chunk_document_from_blob(document_content, file_type=file_type)
        del document_content  # Free raw file bytes early
        publish_chunking(matter_id, progress=100, current=len(chunks), total=len(chunks))
        logger.info(f"[Task {self.request.id}] Created {len(chunks)} chunks")
        log_activity(db, matter_id, "document_chunked", details=f"Extracted {len(chunks)} chunks from {document.name}")

        if not chunks:
            raise ValueError("No chunks extracted from document")

        # 2b. Extract keywords from each chunk using YAKE (local, fast)
        logger.info(f"[Task {self.request.id}] Extracting keywords from {len(chunks)} chunks")
        for chunk in chunks:
            chunk["concepts"] = extract_chunk_keywords(chunk.get("content", ""))

        # 2c. Enrich document: summary + classification via Gemini (parallel)
        publish_enriching(matter_id, detail="Generating summary and classification...")
        full_text = "\n".join(chunk.get("content", "") for chunk in chunks)

        async def _enrich():
            return await asyncio.gather(
                generate_doc_summary(full_text),
                classify_document(full_text),
            )

        doc_summary, classification = asyncio.run(_enrich())

        # Store enrichment results on Document record
        document.summary = doc_summary
        document.document_type = classification["document_type"]
        document.jurisdiction = classification["jurisdiction"]
        db.commit()

        logger.info(
            f"[Task {self.request.id}] Enrichment complete: "
            f"summary={'yes' if doc_summary else 'no'}, "
            f"type={classification['document_type']}, "
            f"jurisdiction={classification['jurisdiction']}"
        )

        # Propagate document-level metadata to chunk dicts for Qdrant payload
        for chunk in chunks:
            chunk["document_type"] = classification["document_type"]
            chunk["jurisdiction"] = classification["jurisdiction"]

        # 3. Store chunk metadata in PostgreSQL FIRST with client-side UUIDs
        logger.info(f"[Task {self.request.id}] Storing {len(chunks)} chunks in database")
        publish_storing(matter_id)
        chunk_mappings = []
        doc_uuid = UUID(document_id)
        for idx, chunk in enumerate(chunks):
            chunk_id = uuid4()
            chunk["id"] = str(chunk_id)
            chunk["chunk_sequence"] = idx
            chunk["document_id"] = str(document_id)
            chunk["document_name"] = document.name
            chunk_mappings.append({
                "id": chunk_id,
                "matter_id": UUID(matter_id),
                "document_id": doc_uuid,
                "page_num": chunk.get("page_num"),
                "section_name": chunk.get("section_name"),
                "section_type": chunk.get("section_type"),
                "content": chunk.get("content"),
                "concepts": chunk.get("concepts"),
                "chunk_sequence": idx,
            })

        db.bulk_insert_mappings(Chunk, chunk_mappings)
        db.flush()
        del chunk_mappings  # Free mapping dicts after DB insert

        # 4. Build texts for embedding (Summary-Augmented Chunking)
        # Prepend doc summary to each chunk for embedding ONLY — original
        # content is stored unchanged in PostgreSQL and Qdrant payload.
        if doc_summary:
            logger.info(f"[Task {self.request.id}] Using SAC: prepending summary to {len(chunks)} chunks for embedding")
            chunk_contents = [f"{doc_summary}\n{chunk['content']}" for chunk in chunks]
        else:
            chunk_contents = [chunk["content"] for chunk in chunks]

        # 5. Generate embeddings (with progress updates and per-batch retry)
        logger.info(f"[Task {self.request.id}] Generating embeddings for {len(chunks)} chunks")
        publish_embedding(matter_id, progress=0, current=0, total=len(chunks))

        embeddings = []
        batch_size = 96  # Align with Cohere's per-call limit of 96 texts
        for i in range(0, len(chunk_contents), batch_size):
            batch = chunk_contents[i:i + batch_size]
            batch_embeddings = embed_chunks(batch)
            embeddings.extend(batch_embeddings)

            # Update progress
            processed = min(i + batch_size, len(chunk_contents))
            progress = int((processed / len(chunk_contents)) * 100)
            publish_embedding(matter_id, progress=progress, current=processed, total=len(chunks))

        del chunk_contents  # Free duplicated text strings
        log_activity(db, matter_id, "embeddings_generated", details=f"Generated {len(embeddings)} embeddings for {document.name}")

        # 6. Create Qdrant collection
        logger.info(f"[Task {self.request.id}] Creating vector collection")
        publish_indexing(matter_id, progress=0, detail="Creating collection...")
        create_collection(matter_id)
        publish_indexing(matter_id, progress=30, detail="Collection created")

        # 7. Store vectors in Qdrant (chunks now have UUID IDs from DB)
        logger.info(f"[Task {self.request.id}] Upserting vectors to Qdrant")
        publish_indexing(matter_id, progress=50, detail="Upserting vectors...")
        upsert_vectors(
            matter_id=matter_id,
            chunks=chunks,
            embeddings=embeddings
        )
        del embeddings  # Free embedding vectors
        num_chunks = len(chunks)
        del chunks  # Free chunk dicts
        publish_indexing(matter_id, progress=100, detail=f"{num_chunks} vectors indexed")
        log_activity(db, matter_id, "vectors_indexed", details=f"Indexed {num_chunks} vectors for {document.name}")

        # 8. Update document + matter status to ready (unless user cancelled)
        # Lock the matter row to prevent race conditions when multiple
        # documents for the same matter complete simultaneously.
        matter = db.query(Matter).filter(
            Matter.id == UUID(matter_id)
        ).with_for_update().first()

        if not matter:
            logger.error(f"[Task {self.request.id}] Matter {matter_id} not found during status update")
            return {"status": "failed", "error": "Matter not found"}

        if matter.status == "cancelled":
            logger.info(f"[Task {self.request.id}] Matter {matter_id} was cancelled, skipping status update")
            document.status = "cancelled"
            db.commit()
            return {"status": "cancelled", "matter_id": matter_id}

        # Update document status
        document.status = "ready"
        document.updated_at = datetime.now(timezone.utc)

        # Derive matter status: "ready" only when all documents are ready
        all_docs = db.query(Document).filter(Document.matter_id == UUID(matter_id)).all()
        all_ready = all(d.status == "ready" for d in all_docs)
        any_error = any(d.status == "error" for d in all_docs)
        if all_ready:
            matter.status = "ready"
        elif any_error:
            matter.status = "error"
        # else: still "processing" — other docs in flight

        matter.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Publish ready event
        publish_ready(matter_id, num_chunks)
        log_activity(db, matter_id, "document_processed", details=f"{document.name} ready with {num_chunks} chunks")

        logger.info(f"[Task {self.request.id}] Successfully processed matter {matter_id}")
        return {
            "status": "success",
            "matter_id": matter_id,
            "document_id": document_id,
            "chunks_processed": num_chunks
        }

    except Exception as exc:
        db.rollback()
        logger.error(f"[Task {self.request.id}] Error processing matter {matter_id}: {str(exc)}", exc_info=True)

        retry_count = self.request.retries
        max_retries = self.max_retries

        # Check if we should retry
        if retry_count < max_retries:
            publish_retrying(matter_id, retry_count + 1, max_retries, str(exc))
            logger.info(f"[Task {self.request.id}] Retrying matter {matter_id} (attempt {retry_count + 1}/{max_retries})")
            raise self.retry(exc=exc, countdown=5)

        # Max retries exceeded - update document status to error,
        # derive matter status from all documents (don't blindly set to error)
        try:
            doc = db.query(Document).filter(Document.id == UUID(document_id)).first()
            if doc:
                doc.status = "error"
            matter = db.query(Matter).filter(
                Matter.id == UUID(matter_id)
            ).with_for_update().first()
            if matter:
                all_docs = db.query(Document).filter(Document.matter_id == UUID(matter_id)).all()
                all_ready = all(d.status == "ready" for d in all_docs)
                any_processing = any(d.status == "processing" for d in all_docs)
                if all_ready:
                    matter.status = "ready"
                elif any_processing:
                    pass  # Keep "processing" — other docs still in flight
                else:
                    matter.status = "error"
                matter.updated_at = datetime.now(timezone.utc)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to update status on error: {str(e)}")

        # Publish error event
        publish_error(matter_id, str(exc), retry_count)
        log_activity(db, matter_id, "processing_failed", details=f"Failed after {retry_count} retries: {str(exc)[:200]}")

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
