"""Celery tasks for document processing"""
import asyncio
import logging
from datetime import datetime, timezone
from uuid import UUID, uuid4, uuid5, NAMESPACE_DNS
from celery import shared_task

# Handle both module import styles
try:
    from backend.database import get_session_factory
    from backend.models import Matter, Chunk, Document
    from backend.services.storage import download_document_from_blob
    from backend.services.chunking import chunk_document_from_blob
    from backend.services.embeddings import embed_chunks, embed_chunks_with_provider, active_embedding_model
    from backend.services.vector_store import upsert_vectors, create_collection
    from backend.services.keyword_extractor import extract_chunk_keywords
    from backend.services.document_summary import generate_doc_summary, classify_document
    from backend.services.authority_detector import detect_authority
    from backend.services.temporal_extractor import extract_temporal_metadata
    from backend.services.progress import (
        publish_downloading, publish_chunking, publish_embedding,
        publish_indexing, publish_storing, publish_enriching, publish_ready,
        publish_error, publish_retrying
    )
    from backend.services.audit import log_activity
except ImportError:
    try:
        from database import get_session_factory
        from models import Matter, Chunk, Document
        from services.storage import download_document_from_blob
        from services.chunking import chunk_document_from_blob
        from services.embeddings import embed_chunks, embed_chunks_with_provider, active_embedding_model
        from services.vector_store import upsert_vectors, create_collection
        from services.keyword_extractor import extract_chunk_keywords
        from services.document_summary import generate_doc_summary, classify_document
        from services.authority_detector import detect_authority
        from services.temporal_extractor import extract_temporal_metadata
        from services.progress import (
            publish_downloading, publish_chunking, publish_embedding,
            publish_indexing, publish_storing, publish_enriching, publish_ready,
            publish_error, publish_retrying
        )
        from services.audit import log_activity
    except ImportError:
        from ..database import get_session_factory
        from ..models import Matter, Chunk, Document
        from ..services.storage import download_document_from_blob
        from ..services.chunking import chunk_document_from_blob
        from ..services.embeddings import embed_chunks, embed_chunks_with_provider, active_embedding_model
        from ..services.vector_store import upsert_vectors, create_collection
        from ..services.keyword_extractor import extract_chunk_keywords
        from ..services.document_summary import generate_doc_summary, classify_document
        from ..services.authority_detector import detect_authority
        from ..services.temporal_extractor import extract_temporal_metadata
        from ..services.progress import (
            publish_downloading, publish_chunking, publish_embedding,
            publish_indexing, publish_storing, publish_enriching, publish_ready,
            publish_error, publish_retrying
        )
        from ..services.audit import log_activity

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

        # Stamp the provider now — before any enrichment/asyncio.run that could
        # raise — so matter.embedding_model is always bound even if a later step
        # fails. This keeps the re-index drift tracker accurate.
        ingest_embedding_model = active_embedding_model()

        # Idempotency guard: clean up any partial results from prior failed attempts
        existing_chunks = db.query(Chunk).filter(
            Chunk.document_id == UUID(document_id)
        ).count()
        if existing_chunks > 0:
            logger.info(
                f"[Task {self.request.id}] Found {existing_chunks} existing chunks for document "
                f"{document_id}, cleaning up for retry"
            )
            # Collect chunk IDs before deletion so we can purge Qdrant points too.
            existing_chunk_ids = [
                str(row.id)
                for row in db.query(Chunk.id).filter(
                    Chunk.document_id == UUID(document_id)
                ).all()
            ]
            db.query(Chunk).filter(Chunk.document_id == UUID(document_id)).delete()
            db.commit()
            # Best-effort purge of corresponding Qdrant points so a schema-mismatch
            # (dense-only collection vs sparse retry) cannot cause an infinite crash loop.
            if existing_chunk_ids:
                try:
                    from backend.services.vector_store import delete_vectors_by_document
                    deleted = delete_vectors_by_document(matter_id, existing_chunk_ids)
                    logger.info(
                        f"[Task {self.request.id}] Purged {deleted} stale Qdrant points for retry"
                    )
                except Exception as _qdrant_del_err:
                    logger.warning(
                        f"[Task {self.request.id}] Best-effort Qdrant purge failed (non-blocking): "
                        f"{_qdrant_del_err}"
                    )

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

        # 2c. Enrich document: summary + classification + authority + temporal (parallel)
        publish_enriching(matter_id, detail="Generating summary, classification, authority, and temporal metadata...")
        full_text = "\n".join(chunk.get("content", "") for chunk in chunks)

        # Default results in case authority/temporal detection fails
        authority_result = {
            "source_type": "other",
            "court_level": "unknown",
            "court_name": "unknown",
            "jurisdiction_code": "unknown",
            "binding_authority": False,
            "authority_score": 0.5,
            "confidence": 0.0,
        }
        temporal_result = None

        async def _enrich():
            return await asyncio.gather(
                generate_doc_summary(full_text),
                classify_document(full_text),
                detect_authority(full_text),
                extract_temporal_metadata(full_text, document.name),
            )

        try:
            doc_summary, classification, authority_result, temporal_result = asyncio.run(_enrich())
        except Exception as e:
            # If the full gather fails, fall back to just summary + classification
            logger.warning(f"[Task {self.request.id}] Full enrichment gather failed, retrying core only: {e}")

            async def _enrich_core():
                return await asyncio.gather(
                    generate_doc_summary(full_text),
                    classify_document(full_text),
                )

            doc_summary, classification = asyncio.run(_enrich_core())

        # Store enrichment results on Document record
        document.summary = doc_summary
        document.document_type = classification["document_type"]
        document.jurisdiction = classification["jurisdiction"]

        # Store temporal metadata on Document (columns must exist via migration)
        if temporal_result:
            try:
                if temporal_result.effective_date:
                    document.effective_date = temporal_result.effective_date
                if temporal_result.superseded_date:
                    document.superseded_date = temporal_result.superseded_date
                if temporal_result.version_number:
                    document.version_number = temporal_result.version_number
                if temporal_result.document_status:
                    document.document_status = temporal_result.document_status
            except AttributeError as e:
                logger.warning(
                    f"[Task {self.request.id}] Could not store temporal fields on Document "
                    f"(columns may not exist yet): {e}"
                )

        # Link document to amendment chain if it supersedes an existing document
        try:
            from backend.services.amendment_chain_manager import detect_supersession
            existing_docs = db.query(Document).filter(
                Document.matter_id == UUID(matter_id),
                Document.id != document.id,
            ).all()
            chain = detect_supersession(document, existing_docs, db)
            if chain:
                logger.info(
                    f"[Task {self.request.id}] Document linked to amendment chain: {chain.canonical_name}"
                )
        except ImportError:
            logger.debug("amendment_chain_manager not available, skipping")
        except Exception as e:
            logger.warning(f"[Task {self.request.id}] Amendment chain detection failed (non-blocking): {e}")

        # Citation knowledge graph indexing (non-blocking)
        try:
            from backend.config import get_settings as _get_settings
            _settings = _get_settings()
            if _settings.citation_graph_enabled:
                try:
                    from backend.services.citation_graph import extract_and_index_citations
                except ImportError:
                    from services.citation_graph import extract_and_index_citations
                jurisdiction = classification.get("jurisdiction", "unknown")
                import asyncio as _asyncio
                _asyncio.run(extract_and_index_citations(
                    db,
                    matter_id,
                    document_id,
                    document.name,
                    jurisdiction,
                    chunks,
                ))
                logger.info(f"[Task {self.request.id}] Citation graph indexing complete for {document.name}")
        except ImportError:
            logger.debug("citation_graph module not available, skipping graph indexing")
        except Exception as e:
            logger.warning(f"[Task {self.request.id}] Citation graph indexing failed (non-blocking): {e}")

        db.commit()

        logger.info(
            f"[Task {self.request.id}] Enrichment complete: "
            f"summary={'yes' if doc_summary else 'no'}, "
            f"type={classification['document_type']}, "
            f"jurisdiction={classification['jurisdiction']}, "
            f"authority={authority_result.get('court_level', 'unknown')}/{authority_result.get('authority_score', 0.5)}, "
            f"temporal={temporal_result.extraction_method if temporal_result else 'none'}"
        )

        # Propagate document-level metadata to chunk dicts for Qdrant payload
        for chunk in chunks:
            chunk["document_type"] = classification["document_type"]
            chunk["jurisdiction"] = classification["jurisdiction"]
            # Authority metadata — stored on every chunk for Qdrant filtering/reranking
            chunk["court_level"] = authority_result.get("court_level", "unknown")
            chunk["jurisdiction_code"] = authority_result.get("jurisdiction_code", "unknown")
            chunk["authority_score"] = authority_result.get("authority_score", 0.5)
            chunk["binding_authority"] = authority_result.get("binding_authority", False)
            chunk["source_type"] = authority_result.get("source_type", "other")
            # Temporal metadata — stored for freshness-aware retrieval
            if temporal_result:
                chunk["effective_date"] = temporal_result.effective_date.isoformat() if temporal_result.effective_date else None
                chunk["superseded_date"] = temporal_result.superseded_date.isoformat() if temporal_result.superseded_date else None
                chunk["document_status"] = temporal_result.document_status

        # 3. Store chunk metadata in PostgreSQL FIRST with client-side UUIDs
        logger.info(f"[Task {self.request.id}] Storing {len(chunks)} chunks in database")
        publish_storing(matter_id)
        chunk_mappings = []
        doc_uuid = UUID(document_id)
        for idx, chunk in enumerate(chunks):
            # Deterministic ID: same document + same sequence = same UUID
            # This makes retries idempotent — duplicate inserts become no-ops
            chunk_id = uuid5(NAMESPACE_DNS, f"{document_id}:{idx}")
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
                "embedding_model": ingest_embedding_model,
                "authority_metadata": {
                    "source_type": authority_result.get("source_type", "other"),
                    "court_level": authority_result.get("court_level", "unknown"),
                    "court_name": authority_result.get("court_name", "unknown"),
                    "jurisdiction_code": authority_result.get("jurisdiction_code", "unknown"),
                    "binding_authority": authority_result.get("binding_authority", False),
                    "authority_score": authority_result.get("authority_score", 0.5),
                    "confidence": authority_result.get("confidence", 0.0),
                },
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

        # 5b. Generate BM25 sparse vectors for hybrid search (non-blocking)
        sparse_vectors = None
        try:
            from backend.services.hybrid_search import generate_sparse_vectors_batch
            texts_for_sparse = [chunk["content"] for chunk in chunks]
            sparse_vectors = generate_sparse_vectors_batch(texts_for_sparse)
            del texts_for_sparse  # Free duplicated text strings
            # Verify we got valid sparse vectors (not all None)
            valid_count = sum(1 for sv in sparse_vectors if sv is not None)
            if valid_count > 0:
                logger.info(f"[Task {self.request.id}] Generated {valid_count}/{len(sparse_vectors)} BM25 sparse vectors")
            else:
                logger.warning(f"[Task {self.request.id}] All BM25 sparse vectors are None, falling back to dense-only")
                sparse_vectors = None
        except ImportError:
            logger.info(f"[Task {self.request.id}] hybrid_search module not available, using dense-only indexing")
        except Exception as e:
            logger.warning(f"[Task {self.request.id}] BM25 sparse vector generation failed (non-blocking): {e}")
            sparse_vectors = None

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
            embeddings=embeddings,
            sparse_vectors=sparse_vectors
        )
        del embeddings  # Free embedding vectors
        del sparse_vectors  # Free sparse vectors
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

        # Stamp the provider used so re-index drift tracking is accurate.
        matter.embedding_model = ingest_embedding_model
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


# ═══════════════════════════════════════════════════════════════
# EMBEDDING RE-INDEX / MIGRATION TASKS
# ═══════════════════════════════════════════════════════════════

# Re-index batch size — aligns with Cohere's 96-text per-call limit; Voyage
# (128) tolerates this too, so one constant covers both providers.
_REINDEX_BATCH_SIZE = 96


def _provider_for_model(target_model: str) -> str:
    """Map a target embedding model name to its provider.

    Both providers emit 1024-dim vectors so the Qdrant collection stays valid;
    only the embedding call differs. Unknown models default to 'cohere' (the
    safe general-purpose fallback) so a typo never silently routes to Voyage.
    """
    if target_model and target_model.lower().startswith("voyage"):
        return "voyage"
    return "cohere"


def _chunk_to_payload_dict(chunk, document) -> dict:
    """Build the upsert_vectors-shaped chunk dict from ORM Chunk + Document.

    Hydrates the same payload fields fresh ingestion writes so re-upserted
    vectors carry identical metadata. document_name/type/jurisdiction live on
    the Document; authority + temporal live inside Chunk.authority_metadata
    (a copy is read, never mutated). Returns a NEW dict.
    """
    authority = dict(chunk.authority_metadata or {})
    return {
        "id": str(chunk.id),
        "content": chunk.content or "",
        "page_num": chunk.page_num,
        "section_name": chunk.section_name,
        "section_type": chunk.section_type,
        "chunk_sequence": chunk.chunk_sequence if chunk.chunk_sequence is not None else 0,
        "concepts": chunk.concepts or [],
        "document_id": str(chunk.document_id),
        "document_name": document.name if document else "",
        "document_type": (document.document_type if document else None) or "",
        "jurisdiction": (document.jurisdiction if document else None) or "",
        # Authority hierarchy metadata (copied from Chunk.authority_metadata)
        "source_type": authority.get("source_type", "other"),
        "court_level": authority.get("court_level", "unknown"),
        "jurisdiction_code": authority.get("jurisdiction_code", "unknown"),
        "authority_score": authority.get("authority_score", 0.5),
        "binding_authority": authority.get("binding_authority", False),
        # Temporal metadata (from Document)
        "effective_date": (
            document.effective_date.isoformat()
            if document and document.effective_date else None
        ),
        "superseded_date": (
            document.superseded_date.isoformat()
            if document and document.superseded_date else None
        ),
        "document_status": (document.document_status if document else None) or "unknown",
    }


@shared_task(
    bind=True,
    max_retries=3,
    default_retry_delay=5,
    acks_late=True,
    track_started=True,
)
def reindex_matter_task(self, matter_id: str, target_model: str = "voyage-law-2"):
    """Re-embed a matter's chunks with ``target_model``, in place.

    Resume-safe (per-batch commit + filter on chunks not yet on target),
    idempotent (a clean matter re-embeds 0 chunks), and SAC-correct (embeds
    summary-augmented text, matching fresh ingestion). Point IDs are
    deterministic so each chunk's vector+payload is overwritten in place.
    Sparse (BM25) vectors are regenerated so hybrid search is preserved.

    Matter.embedding_model is only stamped on FULL success, so a half-done
    matter never falsely reports "done".
    """
    provider = _provider_for_model(target_model)
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        logger.info(
            f"[Reindex {self.request.id}] matter={matter_id} target={target_model} provider={provider}"
        )
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        if not matter:
            logger.error(f"[Reindex {self.request.id}] Matter {matter_id} not found")
            return {"status": "failed", "error": "Matter not found"}

        # Idempotent collection ensure (1024-dim collection stays valid).
        create_collection(matter_id)

        # Resume-safe selection: only chunks not already on the target model,
        # ordered by sequence for deterministic batching.
        total_pending = (
            db.query(Chunk)
            .filter(
                Chunk.matter_id == UUID(matter_id),
                (Chunk.embedding_model != target_model) | (Chunk.embedding_model.is_(None)),
            )
            .count()
        )
        logger.info(f"[Reindex {self.request.id}] {total_pending} chunks pending re-embed")

        if total_pending == 0:
            # Already fully on target — just stamp the matter (idempotent).
            matter.embedding_model = target_model
            matter.updated_at = datetime.now(timezone.utc)
            db.commit()
            publish_ready(matter_id, 0)
            return {"status": "success", "matter_id": matter_id, "reindexed": 0, "target_model": target_model}

        # Cache Document lookups so we don't re-query per chunk.
        doc_cache: dict = {}

        def _doc_for(chunk):
            key = chunk.document_id
            if key not in doc_cache:
                doc_cache[key] = db.query(Document).filter(Document.id == key).first()
            return doc_cache[key]

        processed = 0
        publish_embedding(matter_id, progress=0, current=0, total=total_pending)

        # Limit/offset-free pagination: re-query each page after commit.
        # The filter excludes chunks already stamped to target_model, so once
        # a batch is committed (stamp applied) those rows disappear from the
        # next query — the loop terminates when no unstamped chunks remain.
        # This is safe across commits because no server-side cursor is held.
        while True:
            batch = (
                db.query(Chunk)
                .filter(
                    Chunk.matter_id == UUID(matter_id),
                    (Chunk.embedding_model != target_model) | (Chunk.embedding_model.is_(None)),
                )
                .order_by(Chunk.chunk_sequence.asc())
                .limit(_REINDEX_BATCH_SIZE)
                .all()
            )
            if not batch:
                break

            # SAC-correct text: prepend the owning document's summary, mirroring
            # ingestion (tasks.py SAC). Raw content alone would diverge from
            # fresh-ingest vectors.
            sac_texts = []
            payload_chunks = []
            raw_contents = []
            for chunk in batch:
                document = _doc_for(chunk)
                summary = document.summary if document else None
                content = chunk.content or ""
                sac_texts.append(f"{summary}\n{content}" if summary else content)
                payload_chunks.append(_chunk_to_payload_dict(chunk, document))
                raw_contents.append(content)

            # Force the target provider (bypasses _detect_provider).
            embeddings = embed_chunks_with_provider(sac_texts, provider)

            # Regenerate sparse vectors to preserve hybrid search.
            sparse_vectors = None
            try:
                from backend.services.hybrid_search import generate_sparse_vectors_batch
                sparse_vectors = generate_sparse_vectors_batch(raw_contents)
                if not any(sv is not None for sv in sparse_vectors):
                    sparse_vectors = None
            except ImportError:
                logger.info(f"[Reindex {self.request.id}] hybrid_search unavailable, dense-only")
            except Exception as e:
                logger.warning(f"[Reindex {self.request.id}] sparse regen failed (non-blocking): {e}")
                sparse_vectors = None

            # Deterministic point-id overwrite in Qdrant.
            upsert_vectors(
                matter_id=matter_id,
                chunks=payload_chunks,
                embeddings=embeddings,
                sparse_vectors=sparse_vectors,
            )

            # Per-batch checkpoint: stamp chunk provenance + commit so a crash
            # resumes from here (these chunks won't be re-selected).
            for chunk in batch:
                chunk.embedding_model = target_model
            db.commit()

            processed += len(batch)
            progress = int((processed / total_pending) * 100)
            publish_embedding(matter_id, progress=progress, current=processed, total=total_pending)
            logger.info(f"[Reindex {self.request.id}] committed {processed}/{total_pending}")

        # Only on FULL success: stamp the matter as authoritatively on target.
        matter.embedding_model = target_model
        matter.updated_at = datetime.now(timezone.utc)
        db.commit()

        publish_ready(matter_id, processed)
        logger.info(f"[Reindex {self.request.id}] matter {matter_id} re-indexed ({processed} chunks)")
        return {
            "status": "success",
            "matter_id": matter_id,
            "reindexed": processed,
            "target_model": target_model,
        }

    except ValueError as exc:
        # Dimension mismatch / wrong-model — fatal, never retry.
        db.rollback()
        logger.error(f"[Reindex {self.request.id}] Fatal (non-retryable): {exc}", exc_info=True)
        publish_error(matter_id, str(exc))
        return {"status": "failed", "matter_id": matter_id, "error": str(exc), "fatal": True}
    except Exception as exc:
        db.rollback()
        logger.error(f"[Reindex {self.request.id}] Error: {exc}", exc_info=True)
        retry_count = self.request.retries
        if retry_count < self.max_retries:
            raise self.retry(exc=exc, countdown=5)
        publish_error(matter_id, str(exc), retry_count)
        return {"status": "failed", "matter_id": matter_id, "error": str(exc), "retries_exhausted": True}
    finally:
        db.close()


@shared_task(
    bind=True,
    acks_late=True,
    track_started=True,
)
def reindex_all_matters_task(self, target_model: str = "voyage-law-2"):
    """Dispatch a reindex_matter_task per non-deleted, ready, off-target matter.

    Recency-first (most-recently-updated matters re-index first); dispatched
    sequentially via apply_async so the worker pool rate-limits naturally.
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        matters = (
            db.query(Matter)
            .filter(
                Matter.is_deleted.is_(False),
                Matter.status == "ready",
                (Matter.embedding_model != target_model) | (Matter.embedding_model.is_(None)),
            )
            .order_by(Matter.updated_at.desc())
            .all()
        )
        matter_ids = [str(m.id) for m in matters]
        logger.info(
            f"[ReindexAll {self.request.id}] dispatching {len(matter_ids)} matters -> {target_model}"
        )
        for mid in matter_ids:
            # MUST match the worker queue (-Q default / task_default_queue="default").
            # The default Celery queue name previously stranded reindex child tasks.
            reindex_matter_task.apply_async(args=(mid, target_model), queue="default")
        return {"status": "dispatched", "matters": matter_ids, "count": len(matter_ids), "target_model": target_model}
    finally:
        db.close()


# Export tasks
__all__ = ["process_document_task", "reindex_matter_task", "reindex_all_matters_task"]
