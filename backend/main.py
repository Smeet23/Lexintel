import hmac
from typing import List
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Body, Query as QueryParam, Depends, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import ProgrammingError
from datetime import datetime, timezone
from uuid import UUID
import uuid
import json
import logging
import asyncio
import io
import time
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

try:
    from backend.config import get_settings
    from backend.database import get_db
    from backend.models import Matter, Chunk, Query, Document, ContractReview, Draft, AuditLog, SavedPrecedent, Conversation
    from backend.services.contract_review import analyze_contract
    from backend.services.draft_service import generate_draft
    from backend.services.audit import log_activity
    from backend.services.embeddings import embed_query
    from backend.services.vector_store import search_vectors
    from backend.services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
    from backend.services.rag_engine import query_matter
    from backend.validators import validate_filename, validate_matter_name, validate_question, validate_file_type
    from backend.services.progress import publish_uploaded
    from backend.exceptions import BlobDownloadException, BlobNotFoundException
    from backend.schemas import QueryCreate, AskResponse
except ImportError:
    try:
        from config import get_settings
        from database import get_db
        from models import Matter, Chunk, Query, Document, ContractReview, Draft, AuditLog, SavedPrecedent, Conversation
        from services.contract_review import analyze_contract
        from services.draft_service import generate_draft
        from services.audit import log_activity
        from services.embeddings import embed_query
        from services.vector_store import search_vectors
        from services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
        from services.rag_engine import query_matter
        from validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from services.progress import publish_uploaded
        from exceptions import BlobDownloadException, BlobNotFoundException
        from schemas import QueryCreate, AskResponse
    except ImportError:
        from .config import get_settings
        from .database import get_db
        from .models import Matter, Chunk, Query, Document, ContractReview, Draft, AuditLog, SavedPrecedent, Conversation
        from .services.contract_review import analyze_contract
        from .services.draft_service import generate_draft
        from .services.audit import log_activity
        from .services.embeddings import embed_query
        from .services.vector_store import search_vectors
        from .services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
        from .services.rag_engine import query_matter
        from .validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from .services.progress import publish_uploaded
        from .exceptions import BlobDownloadException, BlobNotFoundException
        from .schemas import QueryCreate, AskResponse

settings = get_settings()

MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50MB


def get_cors_origins() -> list:
    """Get allowed CORS origins from settings"""
    origins = settings.get_allowed_origins_list()
    return origins


app = FastAPI(
    title="Legal RAG API",
    description="RAG system for legal document analysis",
    version="0.1.0"
)

# CORS middleware
cors_origins = get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    # MVP / local dev: also accept the frontend on ANY localhost port (the dev
    # server often runs on 3000/3001/3100 etc.). This is scoped to loopback only,
    # so it never widens production exposure, and it works with
    # allow_credentials (unlike the wildcard "*"). Production origins still come
    # from the explicit allowed_origins list above.
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    # no-referrer: matter UUIDs live in the SPA URL path; with no auth they act as
    # bearer secrets, so never leak them via the Referer header to third parties.
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


# ============================================
# CITATION KNOWLEDGE GRAPH ENDPOINTS
# ============================================

@app.get("/graph/case/{citation}/good-law")
async def get_good_law_status(
    citation: str = Path(..., max_length=300),
    db: Session = Depends(get_db)
):
    """Check if a case citation is still good law (not overruled/reversed)."""
    try:
        from backend.services.citation_graph import is_good_law
    except ImportError:
        try:
            from services.citation_graph import is_good_law
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Citation graph not available"
            )
    try:
        result = is_good_law(db, citation)
        return result
    except Exception as e:
        logger.error(f"Error checking good law status for '{citation[:100]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to check good law status"
        )


@app.get("/graph/case/{citation}/chain")
async def get_citation_chain(
    citation: str = Path(..., max_length=300),
    depth: int = QueryParam(default=2, ge=1, le=5),
    db: Session = Depends(get_db)
):
    """Get the citation chain (ancestor/descendant treatments) for a case."""
    try:
        from backend.services.citation_graph import get_citation_chain as _get_citation_chain
    except ImportError:
        try:
            from services.citation_graph import get_citation_chain as _get_citation_chain
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Citation graph not available"
            )
    try:
        result = _get_citation_chain(db, citation, depth)
        return result
    except Exception as e:
        logger.error(f"Error getting citation chain for '{citation[:100]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve citation chain"
        )


@app.get("/graph/case/{citation}/network")
async def get_citation_network_endpoint(
    citation: str = Path(..., max_length=300),
    depth: int = QueryParam(default=2, ge=1, le=3),
    max_nodes: int = QueryParam(default=100, ge=10, le=500),
    db: Session = Depends(get_db)
):
    """Get the full citation network (nodes + edges) centered on a case."""
    try:
        from backend.services.citation_graph import get_citation_network as _get_citation_network
    except ImportError:
        try:
            from services.citation_graph import get_citation_network as _get_citation_network
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Citation graph not available"
            )
    try:
        result = _get_citation_network(db, citation, depth, max_nodes)
        return result
    except Exception as e:
        logger.error(f"Error getting citation network for '{citation[:100]}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve citation network"
        )


@app.get("/matters/{matter_id}/graph")
async def get_matter_graph(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Get the citation knowledge graph for all cases in a matter."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    try:
        from backend.services.citation_graph import get_matter_graph as _get_matter_graph
    except ImportError:
        try:
            from services.citation_graph import get_matter_graph as _get_matter_graph
        except ImportError:
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Citation graph not available"
            )
    try:
        return _get_matter_graph(db, matter_uuid)
    except Exception as e:
        logger.error(f"Error building matter graph for '{matter_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to build matter citation graph"
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ============================================
# MATTER MANAGEMENT ENDPOINTS
# ============================================

@app.get("/matters", response_model=list)
async def list_matters(
    db: Session = Depends(get_db)
):
    """List all matters"""
    try:
        matters = db.query(Matter).filter(Matter.is_deleted == False).all()
    except ProgrammingError as e:
        if "does not exist" in (str(e.orig) if e.orig else "") or "does not exist" in str(e):
            logger.warning("Matters table missing. Run: make db-init")
            return []
        raise
    return [
        {
            "id": str(matter.id),
            "name": matter.name,
            "status": matter.status,
            "file_type": matter.file_type,
            "created_at": matter.created_at.isoformat(),
            "updated_at": matter.updated_at.isoformat() if matter.updated_at else None
        }
        for matter in matters
    ]


@app.get("/matters/{matter_id}", response_model=dict)
async def get_matter(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Get a single matter by ID"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    from sqlalchemy import func
    docs_count = db.query(func.count(Document.id)).filter(
        Document.matter_id == matter_uuid
    ).scalar()
    queries_count = db.query(func.count(Query.id)).filter(
        Query.matter_id == matter_uuid
    ).scalar()

    return {
        "id": str(matter.id),
        "name": matter.name,
        "status": matter.status,
        "file_type": matter.file_type,
        "blob_storage_path": matter.blob_storage_path,
        "documents_count": docs_count,
        "queries_count": queries_count,
        "created_at": matter.created_at.isoformat(),
        "updated_at": matter.updated_at.isoformat() if matter.updated_at else None
    }


@app.post("/matters", response_model=dict)
async def upload_matter(
    request: Request,
    name: str = Form(...),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    """Create a matter with one or more documents (PDF, DOCX, TXT)"""
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]

    # Validate all files upfront
    for file in files:
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only PDF, DOCX, and TXT files are allowed. Got: {file.filename}"
            )
        if file.filename:
            validate_filename(file.filename)

    validate_matter_name(name)

    try:
        # Create matter record
        matter_id = uuid.uuid4()
        first_file_type = validate_file_type(files[0].content_type, files[0].filename)
        matter = Matter(
            id=matter_id,
            name=name,
            blob_storage_path="",
            file_type=first_file_type,
            status="processing"
        )
        db.add(matter)
        db.commit()

        try:
            from .celery_app import celery_app
        except ImportError:
            from celery_app import celery_app

        task_ids = []

        # Process each file
        for file in files:
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE} byte limit")
            file_content = await file.read()
            if len(file_content) > MAX_UPLOAD_SIZE:
                raise HTTPException(status_code=413, detail="File too large. Maximum 50MB.")

            try:
                file_type = validate_file_type(file.content_type, file.filename)
            except ValueError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )

            if not validate_file_format(file_content, file_type):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid {file_type.upper()} file: {file.filename}"
                )

            # Upload to blob storage
            blob_path = await upload_document_to_blob(file_content, str(matter_id), file.filename)

            # Create document record
            document_id = uuid.uuid4()
            document = Document(
                id=document_id,
                matter_id=matter_id,
                name=file.filename or name,
                blob_storage_path=blob_path,
                file_type=file_type,
                status="processing"
            )
            db.add(document)
            db.flush()

            # Queue processing task for this document
            task = celery_app.send_task(
                'backend.tasks.process_document_task',
                args=(str(matter_id), str(document_id)),
                queue='default'
            )
            document.celery_task_id = task.id
            task_ids.append(task.id)
            logger.info(f"Queued processing task {task.id} for document {document_id} ({file.filename})")

        # Update matter blob_storage_path to first document's path
        matter.blob_storage_path = blob_path
        db.commit()
        log_activity(db, str(matter_id), "matter_created", details=f"Created matter '{name}' with {len(files)} document(s)")
        db.refresh(matter)

        publish_uploaded(str(matter_id), f"{len(files)} document(s)")

        return {
            "id": str(matter.id),
            "name": matter.name,
            "status": matter.status,
            "file_type": matter.file_type,
            "documents_count": len(files),
            "task_ids": task_ids,
            "created_at": matter.created_at.isoformat()
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Failed to upload matter: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload matter. Please try again."
        )


@app.delete("/matters/{matter_id}", response_model=dict)
async def delete_matter(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Soft delete a matter"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    matter.is_deleted = True
    matter.updated_at = datetime.now(timezone.utc)
    db.commit()

    # Best-effort Qdrant collection cleanup after soft-delete
    try:
        from backend.services.vector_store import delete_collection
    except ImportError:
        try:
            from services.vector_store import delete_collection
        except ImportError:
            from .services.vector_store import delete_collection
    try:
        delete_collection(str(matter_uuid))
        logger.info(f"Cleaned up Qdrant collection for deleted matter {matter_id}")
    except Exception as cleanup_err:
        logger.warning(f"Qdrant cleanup on delete failed (non-fatal): {cleanup_err}")

    return {"id": str(matter.id), "deleted": True}


@app.post("/matters/{matter_id}/cancel", response_model=dict)
def cancel_matter_processing(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Cancel ongoing document processing for a matter.

    Uses a plain `def` (not async) so FastAPI runs it in a threadpool,
    avoiding event-loop blocking from the `with_for_update()` row lock.
    """
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    # Row-level lock prevents race between cancel and task completion
    matter = db.query(Matter).filter(
        Matter.id == matter_uuid, Matter.is_deleted == False
    ).with_for_update().first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    if matter.status != "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter is not processing"
        )

    # Revoke Celery tasks for ALL processing documents (not just matter-level task)
    try:
        from .celery_app import celery_app
    except ImportError:
        from celery_app import celery_app

    processing_docs = db.query(Document).filter(
        Document.matter_id == matter_uuid,
        Document.status == "processing"
    ).all()
    for doc in processing_docs:
        if doc.celery_task_id:
            celery_app.control.revoke(doc.celery_task_id, terminate=True)
            logger.info(f"Revoked Celery task {doc.celery_task_id} for document {doc.id}")
        doc.status = "cancelled"
        doc.celery_task_id = None

    # Update matter status and commit BEFORE network cleanup calls
    matter.status = "cancelled"
    matter.celery_task_id = None
    matter.updated_at = datetime.now(timezone.utc)
    db.commit()
    log_activity(db, str(matter_uuid), "processing_cancelled", details="Processing cancelled")

    # Clean up any partial vectors in Qdrant (best-effort, after commit)
    try:
        from backend.services.vector_store import delete_collection
    except ImportError:
        try:
            from services.vector_store import delete_collection
        except ImportError:
            from .services.vector_store import delete_collection
    try:
        delete_collection(str(matter_uuid))
        logger.info(f"Cleaned up Qdrant collection for cancelled matter {matter_id}")
    except Exception as cleanup_err:
        logger.warning(f"Qdrant cleanup on cancel failed (non-fatal): {cleanup_err}")

    return {"id": str(matter.id), "cancelled": True, "status": "cancelled"}


# ============================================
# DOCUMENT MANAGEMENT ENDPOINTS
# ============================================

@app.get("/matters/{matter_id}/documents", response_model=list)
async def list_matter_documents(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """List all documents for a matter"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    # Use a subquery to get chunk counts in a single query (avoids N+1)
    from sqlalchemy import func
    chunk_counts = (
        db.query(Chunk.document_id, func.count(Chunk.id).label("chunk_count"))
        .filter(Chunk.matter_id == matter_uuid)
        .group_by(Chunk.document_id)
        .subquery()
    )
    documents = (
        db.query(Document, chunk_counts.c.chunk_count)
        .outerjoin(chunk_counts, Document.id == chunk_counts.c.document_id)
        .filter(Document.matter_id == matter_uuid)
        .order_by(Document.created_at.asc())
        .all()
    )
    return [
        {
            "id": str(doc.id),
            "name": doc.name,
            "file_type": doc.file_type,
            "status": doc.status,
            "chunk_count": count or 0,
            "summary": doc.summary,
            "document_type": doc.document_type,
            "jurisdiction": doc.jurisdiction,
            "created_at": doc.created_at.isoformat(),
        }
        for doc, count in documents
    ]


@app.post("/matters/{matter_id}/documents", response_model=dict)
async def upload_matter_document(
    request: Request,
    matter_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload an additional document to an existing matter"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    # Validate MIME type
    allowed_types = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain"
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF, DOCX, and TXT files are allowed"
        )

    if file.filename:
        validate_filename(file.filename)

    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_UPLOAD_SIZE} byte limit")
    try:
        file_content = await file.read()
        if len(file_content) > MAX_UPLOAD_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum 50MB.")

        try:
            file_type = validate_file_type(file.content_type, file.filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        if not validate_file_format(file_content, file_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {file_type.upper()} file format"
            )

        # Upload to blob storage
        blob_path = await upload_document_to_blob(file_content, str(matter_uuid), file.filename)

        # Create document record
        document_id = uuid.uuid4()
        document = Document(
            id=document_id,
            matter_id=matter_uuid,
            name=file.filename or "document",
            blob_storage_path=blob_path,
            file_type=file_type,
            status="processing"
        )
        db.add(document)

        # Set matter back to processing
        matter.status = "processing"
        matter.updated_at = datetime.now(timezone.utc)
        db.commit()

        # Send processing task
        try:
            from .celery_app import celery_app
        except ImportError:
            from celery_app import celery_app

        task = celery_app.send_task(
            'backend.tasks.process_document_task',
            args=(str(matter_uuid), str(document_id)),
            queue='default'
        )
        document.celery_task_id = task.id
        db.commit()
        log_activity(db, str(matter_uuid), "document_uploaded", details=f"Uploaded '{file.filename}'")
        logger.info(f"Queued processing task {task.id} for document {document_id} in matter {matter_id}")

        return {
            "id": str(document.id),
            "matter_id": str(matter_uuid),
            "name": document.name,
            "file_type": document.file_type,
            "status": document.status,
            "task_id": task.id,
            "created_at": document.created_at.isoformat()
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Failed to upload document: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document. Please try again."
        )


@app.get("/matters/{matter_id}/documents/{document_id}/download")
async def download_document(
    matter_id: str,
    document_id: str,
    db: Session = Depends(get_db)
):
    """Download a specific document file"""
    try:
        matter_uuid = UUID(matter_id)
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )

    # Verify parent matter is not soft-deleted
    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    document = db.query(Document).filter(
        Document.id == doc_uuid,
        Document.matter_id == matter_uuid
    ).first()
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    content_type_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
    }
    media_type = content_type_map.get(document.file_type, "application/octet-stream")

    ext = document.file_type if document.file_type else "bin"
    name_stem = document.name
    if name_stem.lower().endswith(f".{ext}"):
        name_stem = name_stem[: -(len(ext) + 1)]
    filename = f"{name_stem}.{ext}"

    try:
        file_bytes = download_document_from_blob(document.blob_storage_path)
    except BlobNotFoundException as e:
        logger.warning(f"Document blob missing for document {document_id}: {e.detail}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found in storage."
        )
    except BlobDownloadException as e:
        logger.error(f"Failed to download document {document_id}: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document from storage."
        )

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(file_bytes)),
        }
    )


@app.delete("/matters/{matter_id}/documents/{document_id}", response_model=dict)
def delete_document(matter_id: str, document_id: str, db: Session = Depends(get_db)):
    """Delete a document and its chunks from a matter.

    Uses a plain `def` (not async) so FastAPI runs it in a threadpool,
    avoiding event-loop blocking from the potential Celery revoke call.
    """
    try:
        matter_uuid = UUID(matter_id)
        doc_uuid = UUID(document_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    document = db.query(Document).filter(
        Document.id == doc_uuid, Document.matter_id == matter_uuid
    ).first()
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # If document is processing, revoke its Celery task
    if document.status == "processing" and document.celery_task_id:
        try:
            from .celery_app import celery_app
        except ImportError:
            from celery_app import celery_app
        celery_app.control.revoke(document.celery_task_id, terminate=True)
        logger.info(f"Revoked Celery task {document.celery_task_id} for document {doc_uuid}")

    # Collect chunk IDs for vector cleanup (before deleting from DB)
    chunk_rows = db.query(Chunk.id).filter(Chunk.document_id == doc_uuid).all()
    chunk_ids = [str(row[0]) for row in chunk_rows]
    blob_path = document.blob_storage_path

    # Delete chunks and document from PostgreSQL FIRST (transactional)
    doc_name = document.name
    db.query(Chunk).filter(Chunk.document_id == doc_uuid).delete()
    db.delete(document)

    # Re-derive matter status after deletion
    remaining_statuses = [
        row[0] for row in db.query(Document.status).filter(
            Document.matter_id == matter_uuid, Document.id != doc_uuid
        ).all()
    ]
    if not remaining_statuses or all(s == "ready" for s in remaining_statuses):
        matter.status = "ready"
    elif any(s == "processing" for s in remaining_statuses):
        matter.status = "processing"
    elif any(s == "error" for s in remaining_statuses):
        matter.status = "error"
    matter.updated_at = datetime.now(timezone.utc)

    db.commit()
    log_activity(db, str(matter_uuid), "document_deleted", details=f"Deleted document '{doc_name}'")

    # Best-effort cleanup AFTER commit: Qdrant vectors + blob storage
    if chunk_ids:
        try:
            from backend.services.vector_store import delete_vectors_by_document
        except ImportError:
            try:
                from services.vector_store import delete_vectors_by_document
            except ImportError:
                from .services.vector_store import delete_vectors_by_document
        try:
            delete_vectors_by_document(str(matter_uuid), chunk_ids)
        except Exception as e:
            logger.warning(f"Qdrant vector cleanup failed (non-fatal): {e}")

    if blob_path:
        try:
            from backend.services.storage import delete_blob
        except ImportError:
            try:
                from services.storage import delete_blob
            except ImportError:
                from .services.storage import delete_blob
        try:
            delete_blob(blob_path)
        except Exception as e:
            logger.warning(f"Blob storage cleanup failed (non-fatal): {e}")

    return {"id": str(doc_uuid), "deleted": True}


# ============================================
# CONVERSATION ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/conversations", response_model=dict)
async def create_conversation(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Create a new conversation thread for a matter."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    conversation = Conversation(
        id=uuid.uuid4(),
        matter_id=matter_uuid,
        title=None,
    )
    db.add(conversation)
    db.commit()

    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
    }


@app.get("/matters/{matter_id}/conversations", response_model=list)
async def list_conversations(
    matter_id: str,
    limit: int = QueryParam(50, ge=1, le=200),
    offset: int = QueryParam(0, ge=0),
    db: Session = Depends(get_db)
):
    """List all conversations for a matter, newest first."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    from sqlalchemy import func

    # Subquery: message count + latest question per conversation (single query)
    stats = (
        db.query(
            Query.conversation_id,
            func.count(Query.id).label("msg_count"),
            func.max(Query.created_at).label("last_at"),
        )
        .filter(Query.conversation_id.isnot(None))
        .group_by(Query.conversation_id)
        .subquery()
    )

    conversations = (
        db.query(Conversation, stats.c.msg_count, stats.c.last_at)
        .outerjoin(stats, Conversation.id == stats.c.conversation_id)
        .filter(
            Conversation.matter_id == matter_uuid,
            Conversation.is_deleted == False,
        )
        .order_by(Conversation.updated_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    # Batch fetch last question preview for all conversations with messages (1 query)
    conv_ids_with_msgs = [conv.id for conv, count, _ in conversations if count]
    last_previews: dict[str, str] = {}
    if conv_ids_with_msgs:
        from sqlalchemy import and_
        # Subquery to get max created_at per conversation
        latest_sq = (
            db.query(
                Query.conversation_id,
                func.max(Query.created_at).label("max_at"),
            )
            .filter(Query.conversation_id.in_(conv_ids_with_msgs))
            .group_by(Query.conversation_id)
            .subquery()
        )
        latest_queries = (
            db.query(Query.conversation_id, Query.question)
            .join(
                latest_sq,
                and_(
                    Query.conversation_id == latest_sq.c.conversation_id,
                    Query.created_at == latest_sq.c.max_at,
                ),
            )
            .all()
        )
        for cid, question in latest_queries:
            last_previews[str(cid)] = question[:100]

    return [
        {
            "id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "message_count": msg_count or 0,
            "last_message_preview": last_previews.get(str(conv.id)),
        }
        for conv, msg_count, last_at in conversations
    ]


@app.get("/matters/{matter_id}/conversations/{conversation_id}", response_model=dict)
async def get_conversation(
    matter_id: str,
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Get a conversation with all its messages."""
    try:
        matter_uuid = UUID(matter_id)
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    conversation = db.query(Conversation).filter(
        Conversation.id == conv_uuid,
        Conversation.matter_id == matter_uuid,
        Conversation.is_deleted == False,
    ).first()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    queries = (
        db.query(Query)
        .filter(Query.conversation_id == conv_uuid)
        .order_by(Query.created_at.asc())
        .all()
    )

    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "queries": [
            {
                "id": str(q.id),
                "question": q.question,
                "answer": q.answer,
                "citations": q.citations,
                "citation_verification": q.citation_verification,
                "claim_verification": q.claim_verification,
                "conflict_analysis": q.conflict_analysis,
                "created_at": q.created_at.isoformat(),
            }
            for q in queries
        ],
    }


@app.patch("/matters/{matter_id}/conversations/{conversation_id}", response_model=dict)
async def update_conversation_title(
    matter_id: str,
    conversation_id: str,
    title: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Update a conversation's title."""
    try:
        matter_uuid = UUID(matter_id)
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    conversation = db.query(Conversation).filter(
        Conversation.id == conv_uuid,
        Conversation.matter_id == matter_uuid,
        Conversation.is_deleted == False,
    ).first()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    conversation.title = title[:255]
    db.commit()

    from sqlalchemy import func
    msg_count = db.query(func.count(Query.id)).filter(
        Query.conversation_id == conv_uuid
    ).scalar() or 0

    return {
        "id": str(conversation.id),
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
        "message_count": msg_count,
        "last_message_preview": None,
    }


@app.delete("/matters/{matter_id}/conversations/{conversation_id}", response_model=dict)
async def delete_conversation(
    matter_id: str,
    conversation_id: str,
    db: Session = Depends(get_db)
):
    """Soft-delete a conversation and dissociate its queries."""
    try:
        matter_uuid = UUID(matter_id)
        conv_uuid = UUID(conversation_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    conversation = db.query(Conversation).filter(
        Conversation.id == conv_uuid,
        Conversation.matter_id == matter_uuid,
        Conversation.is_deleted == False,
    ).first()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    # Soft delete: mark conversation as deleted and clear conversation_id from queries
    conversation.is_deleted = True
    conversation.updated_at = datetime.now(timezone.utc)
    db.query(Query).filter(Query.conversation_id == conv_uuid).update(
        {"conversation_id": None}
    )
    db.commit()
    log_activity(db, str(matter_uuid), "conversation_deleted", details="Deleted a conversation thread")

    return {"id": str(conv_uuid), "deleted": True}


# ============================================
# RAG QUERY ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/ask", response_model=AskResponse)
async def ask_question(
    request: Request,
    matter_id: str,
    body: QueryCreate = Body(...),
    db: Session = Depends(get_db)
):
    """Ask a question about a matter"""
    # Validate question input
    validate_question(body.question)

    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    # Check if matter exists (exclude soft-deleted)
    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    # Check if matter is ready for querying
    if matter.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter is still being processed. Please try again in a moment."
        )
    elif matter.status == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter processing failed. Please re-upload the document."
        )
    elif matter.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Matter processing was cancelled. Please re-upload the document."
        )

    # Resolve conversation if provided (Pydantic already validated UUID format)
    conversation_uuid = body.conversation_id
    if conversation_uuid:
        conversation = db.query(Conversation).filter(
            Conversation.id == conversation_uuid,
            Conversation.matter_id == matter_uuid,
            Conversation.is_deleted == False,
        ).first()
        if not conversation:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )
    else:
        conversation = None

    # Fetch recent conversation history for follow-up context.
    # If a conversation_id is given, use queries from that conversation;
    # otherwise fall back to the most recent matter-level queries.
    if conversation_uuid:
        recent_queries = (
            db.query(Query)
            .filter(Query.conversation_id == conversation_uuid)
            .order_by(Query.created_at.desc())
            .limit(5)
            .all()
        )
    else:
        recent_queries = (
            db.query(Query)
            .filter(Query.matter_id == matter_uuid)
            .order_by(Query.created_at.desc())
            .limit(5)
            .all()
        )
    conversation_history = [
        {"question": q.question, "answer": q.answer}
        for q in reversed(recent_queries)
    ]

    # Get RAG response — use agentic pipeline when enabled
    _agentic_t0 = time.monotonic()
    try:
        # Try agentic RAG if enabled (multi-agent with self-correction).
        # Use the module-level `settings` (already imported with the working
        # import-fallback). A bare `from backend.config import ...` here raised
        # ModuleNotFoundError when uvicorn runs from the backend/ cwd, silently
        # forcing use_agentic=False so the agentic path was never reachable via HTTP.
        use_agentic = bool(getattr(settings, 'agentic_rag_enabled', False))

        if use_agentic:
            try:
                try:
                    from backend.services.agentic_rag import agentic_query_matter
                except ImportError:
                    try:
                        from services.agentic_rag import agentic_query_matter
                    except ImportError:
                        from .services.agentic_rag import agentic_query_matter
                rag_result = await agentic_query_matter(
                    str(matter_uuid), body.question, db,
                    conversation_history=conversation_history,
                    include_legal_research=body.include_legal_research,
                )
            except Exception as e:
                logger.warning(f"Agentic RAG failed, falling back to standard: {e}")
                rag_result = await query_matter(
                    str(matter_uuid), body.question, db,
                    conversation_history=conversation_history,
                    include_legal_research=body.include_legal_research,
                    as_of_date=body.as_of_date,
                )
        else:
            rag_result = await query_matter(
                str(matter_uuid), body.question, db,
                conversation_history=conversation_history,
                include_legal_research=body.include_legal_research,
                as_of_date=body.as_of_date,
            )

        # Only store if answer was generated successfully
        if rag_result.get("answer"):
            # Agentic observability — present only when the agentic pipeline ran.
            # Non-agentic queries leave these NULL (additive, backwards-compatible).
            agentic_meta = rag_result.get("agentic_metadata")
            agent_latency_ms = None
            agent_iterations = None
            routing_decision = None
            if agentic_meta:
                agent_latency_ms = int((time.monotonic() - _agentic_t0) * 1000)
                agent_iterations = int(agentic_meta.get("rewrite_count", 0)) + 1
                routing_decision = agentic_meta.get("complexity")
                # Enrich the persisted blob with the derived top-level signals.
                agentic_meta = {
                    **agentic_meta,
                    "routing_decision": routing_decision,
                    "agent_iterations": agent_iterations,
                    "agent_latency_ms": agent_latency_ms,
                }

            db_query = Query(
                id=uuid.uuid4(),
                matter_id=matter_uuid,
                conversation_id=conversation_uuid,
                question=body.question,
                answer=rag_result.get("answer", ""),
                citations=rag_result.get("sources", []),
                citation_verification=rag_result.get("citation_verification"),
                claim_verification=rag_result.get("claim_verification"),
                conflict_analysis=rag_result.get("conflict_analysis"),
                routing_decision=routing_decision,
                agent_iterations=agent_iterations,
                agent_latency_ms=agent_latency_ms,
                agentic_metadata=agentic_meta,
                created_at=datetime.now(timezone.utc)
            )
            db.add(db_query)

            # Auto-set conversation title from first question if not yet titled
            if conversation and not conversation.title:
                conversation.title = body.question[:50]

            # Bump conversation updated_at so the list sorts by most recent activity
            if conversation:
                conversation.updated_at = datetime.now(timezone.utc)

            db.commit()
            log_activity(db, str(matter_uuid), "query", details=body.question)
            rag_result["query_id"] = str(db_query.id)

        return rag_result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process query. Please try again."
        )


@app.get("/matters/{matter_id}/queries", response_model=list)
async def get_query_history(
    matter_id: str,
    limit: int = QueryParam(50, ge=1, le=200),
    db: Session = Depends(get_db)
):
    """Get conversation history for a matter, ordered by created_at ascending."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    queries = (
        db.query(Query)
        .filter(Query.matter_id == matter_uuid)
        .order_by(Query.created_at.asc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": str(q.id),
            "question": q.question,
            "answer": q.answer,
            "citations": q.citations,
            "created_at": q.created_at.isoformat(),
        }
        for q in queries
    ]


@app.delete("/matters/{matter_id}/queries", response_model=dict)
async def delete_all_queries(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Delete all query history for a matter"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    deleted_count = db.query(Query).filter(Query.matter_id == matter_uuid).delete()
    # Also soft-delete all conversations to avoid orphaned empty shells
    conv_count = (
        db.query(Conversation)
        .filter(Conversation.matter_id == matter_uuid, Conversation.is_deleted == False)
        .update({"is_deleted": True})
    )
    db.commit()
    log_activity(db, str(matter_uuid), "queries_cleared", details=f"Cleared {deleted_count} queries and {conv_count} conversations")

    return {"matter_id": str(matter_uuid), "deleted_count": deleted_count}


@app.delete("/matters/{matter_id}/queries/{query_id}", response_model=dict)
async def delete_query(
    matter_id: str,
    query_id: str,
    db: Session = Depends(get_db)
):
    """Delete a single query from history"""
    try:
        matter_uuid = UUID(matter_id)
        query_uuid = UUID(query_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )

    query = db.query(Query).filter(Query.id == query_uuid, Query.matter_id == matter_uuid).first()
    if not query:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Query not found")

    db.delete(query)
    db.commit()
    log_activity(db, str(matter_uuid), "query_deleted", details="Deleted a query")

    return {"id": str(query_uuid), "deleted": True}


@app.get("/matters/{matter_id}/status", response_model=dict)
async def get_matter_status(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Get the status of a matter"""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    return {
        "id": str(matter.id),
        "name": matter.name,
        "status": matter.status,
        "created_at": matter.created_at.isoformat()
    }


@app.get("/matters/{matter_id}/document")
async def download_matter_document(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """Download or view the original document for a matter.

    Returns the raw file bytes with the correct Content-Type and a
    Content-Disposition header so browsers can display (PDF) or prompt
    a download (DOCX, TXT).
    """
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    content_type_map = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "txt": "text/plain",
    }
    media_type = content_type_map.get(matter.file_type, "application/octet-stream")

    # Derive a safe filename
    ext = matter.file_type if matter.file_type else "bin"
    name_stem = matter.name
    if name_stem.lower().endswith(f".{ext}"):
        name_stem = name_stem[: -(len(ext) + 1)]
    filename = f"{name_stem}.{ext}"

    try:
        file_bytes = download_document_from_blob(matter.blob_storage_path)
    except BlobNotFoundException as e:
        # The document no longer exists in storage (deleted/expired blob) — a
        # permanent 404, not a transient failure. Retrying will not help.
        logger.warning(f"Document blob missing for matter {matter_id}: {e.detail}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found in storage."
        )
    except BlobDownloadException as e:
        logger.error(f"Failed to download document for matter {matter_id}: {e.message} — {e.detail}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document from storage. Please try again."
        )

    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=media_type,
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Content-Length": str(len(file_bytes)),
        }
    )


@app.get("/matters/{matter_id}/chunks", response_model=list)
async def get_matter_chunks(
    matter_id: str,
    document_id: str = QueryParam(None, description="Filter chunks by document ID"),
    db: Session = Depends(get_db)
):
    """Return chunks for a matter ordered by chunk_sequence.

    Optionally filter by document_id. Without it, returns all chunks for the matter.

    Response items: {id, document_id, page_num, section_name, section_type, content, chunk_sequence}
    """
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    query = db.query(Chunk).filter(Chunk.matter_id == matter_uuid)

    if document_id:
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format"
            )
        query = query.filter(Chunk.document_id == doc_uuid)

    chunks = query.order_by(Chunk.chunk_sequence.asc().nullslast()).all()

    return [
        {
            "id": str(chunk.id),
            "document_id": str(chunk.document_id) if chunk.document_id else None,
            "page_num": chunk.page_num,
            "section_name": chunk.section_name,
            "section_type": chunk.section_type,
            "content": chunk.content,
            "concepts": chunk.concepts or [],
            "chunk_sequence": chunk.chunk_sequence,
        }
        for chunk in chunks
    ]


# ============================================
# SSE PROGRESS STREAMING ENDPOINT
# ============================================

@app.get("/matters/{matter_id}/progress")
async def matter_progress_stream(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """
    SSE endpoint for real-time document processing progress updates.

    Subscribes to Redis pub/sub channel and streams progress events to the client.
    """
    # Validate matter ID
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid matter ID format"
        )

    # Check matter exists (exclude soft-deleted)
    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Matter not found"
        )

    async def event_generator():
        """Generate SSE events from Redis pub/sub with cached state replay."""
        redis_client = await aioredis.from_url(
            settings.celery_broker_url,
            decode_responses=True
        )
        pubsub = redis_client.pubsub()
        channel = f"lexintel:matter:{matter_id}:progress"
        cache_key = f"lexintel:matter:{matter_id}:progress:latest"

        try:
            # 1. Subscribe FIRST to avoid missing events during cache read
            await pubsub.subscribe(channel)
            logger.info(f"SSE client subscribed to {channel}")

            # 2. Replay cached latest state (survives page refresh)
            cached = await redis_client.get(cache_key)
            last_overall = -1
            if cached:
                yield {"event": "progress", "data": cached}
                try:
                    parsed = json.loads(cached)
                    last_overall = parsed.get("overall_progress", 0)
                    if parsed.get("stage") in ("ready", "error"):
                        logger.info(f"SSE replay shows completed state for {channel}")
                        return
                except json.JSONDecodeError:
                    pass

            # 3. Send connection event
            yield {
                "event": "connected",
                "data": json.dumps({"matter_id": matter_id, "status": "connected"})
            }

            # 4. Listen for live messages with heartbeat and timeout
            max_duration = 600  # 10 minutes max connection
            heartbeat_interval = 15  # seconds
            import time
            start_time = time.monotonic()

            while True:
                # Check max duration
                if time.monotonic() - start_time > max_duration:
                    logger.info(f"SSE max duration reached for {channel}")
                    yield {"event": "timeout", "data": json.dumps({"message": "Connection timeout, please reconnect"})}
                    break

                try:
                    message = await asyncio.wait_for(
                        pubsub.get_message(ignore_subscribe_messages=True),
                        timeout=heartbeat_interval
                    )
                except asyncio.TimeoutError:
                    # Send heartbeat
                    yield {"event": "heartbeat", "data": json.dumps({"status": "connected"})}
                    continue

                if message and message["type"] == "message":
                    data = message["data"]

                    # Deduplicate: skip if already covered by cached replay
                    try:
                        parsed = json.loads(data)
                        current_overall = parsed.get("overall_progress", 0)
                        if current_overall <= last_overall:
                            continue
                        last_overall = current_overall
                    except json.JSONDecodeError:
                        pass

                    yield {"event": "progress", "data": data}

                    # Check if processing is complete
                    try:
                        parsed = json.loads(data)
                        if parsed.get("stage") in ("ready", "error"):
                            logger.info(f"SSE stream ending for {channel}: {parsed.get('stage')}")
                            break
                    except json.JSONDecodeError:
                        pass

        except asyncio.CancelledError:
            logger.info(f"SSE client disconnected from {channel}")
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.close()
            await redis_client.close()

    return EventSourceResponse(event_generator())


# ============================================
# CONTRACT REVIEW ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/contract-review", response_model=dict)
async def run_contract_review(
    matter_id: str,
    document_id: str = Body(None, embed=True),
    db: Session = Depends(get_db)
):
    """Run contract risk analysis on a document using Gemini."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    # Resolve document_id: use provided or pick first document
    if document_id:
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID")
    else:
        first_doc = db.query(Document).filter(
            Document.matter_id == matter_uuid
        ).order_by(Document.created_at.asc()).first()
        if not first_doc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No documents found for this matter")
        doc_uuid = first_doc.id

    # Delete any existing review for this document (re-run)
    db.query(ContractReview).filter(
        ContractReview.matter_id == matter_uuid,
        ContractReview.document_id == doc_uuid
    ).delete()
    db.commit()

    try:
        result = await analyze_contract(str(matter_uuid), str(doc_uuid), db)
    except Exception as e:
        logger.error(f"Contract review failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Contract review analysis failed")

    # Persist the review
    review = ContractReview(
        id=uuid.uuid4(),
        matter_id=matter_uuid,
        document_id=doc_uuid,
        risks=result.get("risks", []),
        summary=result.get("summary", {}),
        missing_clauses=result.get("missing_clauses", []),
        overall_score=result.get("overall_score"),
    )
    db.add(review)
    db.commit()

    log_activity(db, str(matter_uuid), "contract_review", details=f"Analyzed document for contract risks")

    return {
        "id": str(review.id),
        "matter_id": str(matter_uuid),
        "document_id": str(doc_uuid),
        "risks": review.risks,
        "summary": review.summary,
        "missing_clauses": review.missing_clauses,
        "overall_score": review.overall_score,
        "created_at": review.created_at.isoformat(),
    }


@app.get("/matters/{matter_id}/contract-review", response_model=dict)
async def get_contract_review(
    matter_id: str,
    document_id: str = QueryParam(None),
    db: Session = Depends(get_db)
):
    """Get the most recent contract review for a matter/document."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    query = db.query(ContractReview).filter(ContractReview.matter_id == matter_uuid)
    if document_id:
        try:
            doc_uuid = UUID(document_id)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID")
        query = query.filter(ContractReview.document_id == doc_uuid)

    review = query.order_by(ContractReview.created_at.desc()).first()
    if not review:
        return {"exists": False}

    return {
        "exists": True,
        "id": str(review.id),
        "matter_id": str(review.matter_id),
        "document_id": str(review.document_id),
        "risks": review.risks,
        "summary": review.summary,
        "missing_clauses": review.missing_clauses,
        "overall_score": review.overall_score,
        "created_at": review.created_at.isoformat(),
    }


# ============================================
# DRAFT ASSISTANT ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/drafts", response_model=dict)
async def create_draft(
    matter_id: str,
    document_type: str = Body(..., embed=True, max_length=100),
    instructions: str = Body(..., embed=True, max_length=5000),
    db: Session = Depends(get_db)
):
    """Generate a legal document draft using matter context and Gemini."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

    try:
        result = await generate_draft(str(matter_uuid), document_type, instructions, db)
    except Exception as e:
        logger.error(f"Draft generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Draft generation failed")

    # Persist the draft
    draft = Draft(
        id=uuid.uuid4(),
        matter_id=matter_uuid,
        document_type=document_type,
        instructions=instructions,
        content=result.get("content", ""),
        sources=result.get("sources", []),
    )
    db.add(draft)
    db.commit()

    log_activity(db, str(matter_uuid), "draft_generated", details=f"Generated {document_type}")

    return {
        "id": str(draft.id),
        "matter_id": str(matter_uuid),
        "document_type": draft.document_type,
        "instructions": draft.instructions,
        "content": draft.content,
        "sources": draft.sources,
        "created_at": draft.created_at.isoformat(),
    }


@app.get("/matters/{matter_id}/drafts", response_model=list)
async def list_drafts(
    matter_id: str,
    db: Session = Depends(get_db)
):
    """List all drafts for a matter, newest first."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    drafts = (
        db.query(Draft)
        .filter(Draft.matter_id == matter_uuid)
        .order_by(Draft.created_at.desc())
        .all()
    )

    return [
        {
            "id": str(d.id),
            "document_type": d.document_type,
            "instructions": d.instructions,
            "content": d.content,
            "sources": d.sources,
            "created_at": d.created_at.isoformat(),
        }
        for d in drafts
    ]


@app.get("/matters/{matter_id}/drafts/{draft_id}", response_model=dict)
async def get_draft(
    matter_id: str,
    draft_id: str,
    db: Session = Depends(get_db)
):
    """Get a single draft by ID."""
    try:
        matter_uuid = UUID(matter_id)
        draft_uuid = UUID(draft_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    draft = db.query(Draft).filter(Draft.id == draft_uuid, Draft.matter_id == matter_uuid).first()
    if not draft:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Draft not found")

    return {
        "id": str(draft.id),
        "matter_id": str(draft.matter_id),
        "document_type": draft.document_type,
        "instructions": draft.instructions,
        "content": draft.content,
        "sources": draft.sources,
        "created_at": draft.created_at.isoformat(),
    }


# ============================================
# AUDIT LOG ENDPOINT
# ============================================

@app.get("/matters/{matter_id}/audit-log", response_model=list)
async def get_audit_log(
    matter_id: str,
    limit: int = QueryParam(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get activity audit log for a matter, newest first."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")

    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=404, detail="Matter not found")

    logs = (
        db.query(AuditLog)
        .filter(AuditLog.matter_id == matter_uuid)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "id": str(log.id),
            "action": log.action,
            "user": log.user,
            "details": log.details,
            "sources": log.sources,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


# ============================================
# PRECEDENTS ENDPOINTS
# ============================================

@app.post("/precedents/search", response_model=dict)
async def search_precedents(
    query: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Search across all matters for relevant legal precedents using vector similarity."""
    if not query or len(query.strip()) < 3:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query must be at least 3 characters")

    # Get non-deleted, ready matters. Cap the fan-out: one request otherwise
    # amplifies into one Qdrant search PER matter, so bound it (most-recent first)
    # to keep cost/latency predictable as the corpus grows.
    _PRECEDENT_SEARCH_MAX_MATTERS = 50
    matters = (
        db.query(Matter)
        .filter(Matter.is_deleted == False, Matter.status == "ready")
        .order_by(Matter.updated_at.desc())
        .limit(_PRECEDENT_SEARCH_MAX_MATTERS)
        .all()
    )
    if not matters:
        return {"results": [], "total": 0}

    # Embed the query once
    try:
        query_embedding = embed_query(query)
    except Exception as e:
        logger.error(f"Embedding failed for precedent search: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Search embedding failed")

    # Fan-out search across all matter collections
    async def _search_matter(m):
        try:
            results = await asyncio.to_thread(
                search_vectors,
                matter_id=str(m.id),
                query_embedding=query_embedding,
                limit=5,
            )
            return [(m, r) for r in results]
        except Exception as e:
            logger.warning(f"Precedent search failed for matter {m.id}: {e}")
            return []

    tasks = [_search_matter(m) for m in matters]
    gathered = await asyncio.gather(*tasks)

    # Flatten, format and sort
    all_results = []
    for batch in gathered:
        for matter_obj, result in batch:
            all_results.append({
                "matter_id": str(matter_obj.id),
                "matter_name": matter_obj.name,
                "document_name": result.get("document_name", "Unknown"),
                "page_num": result.get("page_num", ""),
                "section_name": result.get("section_name", ""),
                "content": result.get("content", ""),
                "relevance_score": round(result.get("score", 0), 3),
            })

    all_results.sort(key=lambda x: x["relevance_score"], reverse=True)
    top_results = all_results[:20]

    return {"results": top_results, "total": len(top_results)}


@app.post("/precedents/save", response_model=dict)
async def save_precedent(
    title: str = Body(..., embed=True),
    query: str = Body(..., embed=True),
    document_name: str = Body(None, embed=True),
    matter_id: str = Body(None, embed=True),
    chunk_content: str = Body(None, embed=True),
    page_num: int = Body(None, embed=True),
    section_name: str = Body(None, embed=True),
    relevance_score: str = Body(None, embed=True),
    tags: list = Body([], embed=True),
    notes: str = Body(None, embed=True),
    db: Session = Depends(get_db)
):
    """Save a search result as a precedent bookmark."""
    matter_uuid = None
    if matter_id:
        try:
            matter_uuid = UUID(matter_id)
        except ValueError:
            pass

    precedent = SavedPrecedent(
        id=uuid.uuid4(),
        title=title,
        query=query,
        document_name=document_name,
        matter_id=matter_uuid,
        chunk_content=chunk_content,
        page_num=page_num,
        section_name=section_name,
        relevance_score=relevance_score,
        tags=tags,
        notes=notes,
    )
    db.add(precedent)
    db.commit()

    return {
        "id": str(precedent.id),
        "title": precedent.title,
        "created_at": precedent.created_at.isoformat(),
    }


@app.get("/precedents", response_model=list)
async def list_precedents(
    db: Session = Depends(get_db)
):
    """List all saved precedents, newest first."""
    precedents = db.query(SavedPrecedent).order_by(SavedPrecedent.created_at.desc()).all()

    return [
        {
            "id": str(p.id),
            "title": p.title,
            "query": p.query,
            "document_name": p.document_name,
            "matter_id": str(p.matter_id) if p.matter_id else None,
            "chunk_content": p.chunk_content,
            "page_num": p.page_num,
            "section_name": p.section_name,
            "relevance_score": p.relevance_score,
            "tags": p.tags or [],
            "notes": p.notes,
            "created_at": p.created_at.isoformat(),
        }
        for p in precedents
    ]


@app.delete("/precedents/{precedent_id}", response_model=dict)
async def delete_precedent(
    precedent_id: str,
    db: Session = Depends(get_db)
):
    """Delete a saved precedent."""
    try:
        p_uuid = UUID(precedent_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid ID format")

    precedent = db.query(SavedPrecedent).filter(SavedPrecedent.id == p_uuid).first()
    if not precedent:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Precedent not found")

    db.delete(precedent)
    db.commit()
    return {"id": str(p_uuid), "deleted": True}


# ============================================
# ADMIN: EMBEDDING RE-INDEX / MIGRATION
# ============================================
#
# SECURITY (hard prereq, out of scope for this pass): the app has no auth
# layer. These /admin/* endpoints MUST be gated (API key / IP allowlist /
# reverse-proxy) before production. _require_admin is a stub seam: when an
# ADMIN_API_KEY env var is configured it enforces the X-Admin-Key header;
# otherwise it is a permissive no-op (dev convenience). Replace before prod.

def _require_admin(request: Request) -> None:
    """Admin gate enforced on every request to /admin/* endpoints.

    Behaviour:
    - ADMIN_API_KEY set: caller must supply matching ``X-Admin-Api-Key`` header
      (constant-time comparison via ``hmac.compare_digest``); mismatch → 401.
    - ADMIN_API_KEY not set, debug=True: access allowed but a loud warning is
      emitted so the operator knows the endpoints are open.
    - ADMIN_API_KEY not set, debug=False (production): → 503.
    """
    import os
    expected = os.environ.get("ADMIN_API_KEY")
    if not expected:
        if settings.debug:
            logger.warning(
                "ADMIN_API_KEY not set — admin endpoints OPEN in debug mode"
            )
            return
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin API not configured",
        )
    provided = request.headers.get("X-Admin-Api-Key", "")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin auth required",
        )


@app.get("/admin/embedding-status", response_model=dict)
async def get_embedding_status(
    request: Request,
    db: Session = Depends(get_db),
):
    """Per-matter embedding model + drift, plus aggregate provider status.

    Drift = chunks not yet on the target model = partial-reindex detector.
    """
    _require_admin(request)
    from sqlalchemy import func

    target_model = settings.embedding_model

    matters = (
        db.query(Matter)
        .filter(Matter.is_deleted.is_(False))
        .order_by(Matter.updated_at.desc())
        .all()
    )

    per_matter = []
    reindexed = 0
    pending = 0
    for matter in matters:
        total_chunks = (
            db.query(func.count(Chunk.id))
            .filter(Chunk.matter_id == matter.id)
            .scalar()
        ) or 0
        on_target = (
            db.query(func.count(Chunk.id))
            .filter(Chunk.matter_id == matter.id, Chunk.embedding_model == target_model)
            .scalar()
        ) or 0
        drift = total_chunks - on_target
        is_on_target = (matter.embedding_model == target_model) and drift == 0
        if is_on_target:
            reindexed += 1
        else:
            pending += 1
        per_matter.append({
            "matter_id": str(matter.id),
            "name": matter.name,
            "status": matter.status,
            "embedding_model": matter.embedding_model,
            "total_chunks": total_chunks,
            "chunks_on_target": on_target,
            "drift": drift,
        })

    try:
        from backend.services.embeddings import _detect_provider
    except ImportError:
        from services.embeddings import _detect_provider

    return {
        "aggregate": {
            "active_provider": _detect_provider(),
            "target_model": target_model,
            "reindexed_matters": reindexed,
            "pending_matters": pending,
            "dimensions": 1024,
        },
        "matters": per_matter,
    }


@app.post("/admin/reindex", response_model=dict)
async def trigger_reindex(
    request: Request,
    matter_id: str = Body(default=None, embed=True),
    target_model: str = Body(default=None, embed=True),
):
    """Dispatch a re-index. With matter_id: that matter; without: all matters.

    Poll progress via /admin/embedding-status (drift -> 0) and the existing SSE
    progress channel. Returns the dispatched Celery task id.
    """
    _require_admin(request)
    resolved_model = target_model or settings.embedding_model

    _ALLOWED_EMBEDDING_MODELS: frozenset = frozenset({
        "voyage-law-2",
        "embed-english-v3.0",
    })
    if resolved_model not in _ALLOWED_EMBEDDING_MODELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid target_model. Allowed values: {sorted(_ALLOWED_EMBEDDING_MODELS)}",
        )

    try:
        from backend.celery_app import celery_app
    except ImportError:
        from celery_app import celery_app

    if matter_id:
        try:
            UUID(matter_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter_id")
        task = celery_app.send_task(
            "backend.tasks.reindex_matter_task",
            args=(matter_id, resolved_model),
            queue="default",
        )
        scope = "matter"
    else:
        task = celery_app.send_task(
            "backend.tasks.reindex_all_matters_task",
            args=(resolved_model,),
            queue="default",
        )
        scope = "all"

    logger.info(f"Dispatched reindex task {task.id} (scope={scope}, model={resolved_model})")
    return {
        "task_id": task.id,
        "scope": scope,
        "matter_id": matter_id,
        "target_model": resolved_model,
    }


# ════════════════════════════════════════════════════════════════════════════
# Standalone agent-native primitives — expose each RAG capability as its own
# endpoint so an agent (or integrator) can call verification, conflict detection,
# issue spotting, and citation lookup independently of /ask. (todos/002)
# ════════════════════════════════════════════════════════════════════════════

async def _retrieve_chunks_for(matter_uuid, query: str, top_k: int = 8):
    """Embed the query and retrieve top chunks for a matter — the retrieval stage
    shared by the standalone verify/conflict primitives below."""
    try:
        from backend.services.rag_engine import retrieve_chunks
    except ImportError:
        try:
            from services.rag_engine import retrieve_chunks
        except ImportError:
            from .services.rag_engine import retrieve_chunks
    embedding = await asyncio.to_thread(embed_query, query)
    return await asyncio.to_thread(retrieve_chunks, str(matter_uuid), embedding, top_k)


def _guard_matter(db, matter_id: str):
    """Parse matter_id and ensure the (non-deleted) matter exists. Returns UUID."""
    try:
        matter_uuid = UUID(matter_id)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid matter ID format")
    matter = db.query(Matter).filter(Matter.id == matter_uuid, Matter.is_deleted == False).first()
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")
    return matter_uuid


@app.post("/problem-formulation", response_model=dict)
async def problem_formulation(query: str = Body(..., embed=True, max_length=5000)):
    """Identify the legal issues / research questions implied by a query."""
    try:
        from backend.services.problem_formulation import identify_issues
    except ImportError:
        try:
            from services.problem_formulation import identify_issues
        except ImportError:
            from .services.problem_formulation import identify_issues
    result = await identify_issues(query)
    return result or {"issues": [], "missing_information": []}


@app.get("/citations/lookup", response_model=dict)
async def citation_lookup(
    citation: str = QueryParam(..., max_length=300),
    jurisdiction: str = QueryParam("US", max_length=8),
):
    """Look up a single legal citation in external databases (CourtListener etc.)."""
    try:
        from backend.services.citation_lookup import lookup_citation
    except ImportError:
        try:
            from services.citation_lookup import lookup_citation
        except ImportError:
            from .services.citation_lookup import lookup_citation
    result = await lookup_citation(citation, jurisdiction)
    if not result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Citation not found")
    return result


@app.post("/citations/lookup-batch", response_model=dict)
async def citation_lookup_batch(
    citations: list[str] = Body(..., embed=True),
    jurisdiction: str = Body("US", embed=True),
):
    """Look up multiple citations concurrently (max 50)."""
    if not citations:
        return {"results": {}}
    if len(citations) > 50:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Maximum 50 citations per batch")
    try:
        from backend.services.citation_lookup import lookup_citations_batch
    except ImportError:
        try:
            from services.citation_lookup import lookup_citations_batch
        except ImportError:
            from .services.citation_lookup import lookup_citations_batch
    payload = [
        {"raw_text": c, "jurisdiction": jurisdiction, "index": i}
        for i, c in enumerate(citations) if isinstance(c, str) and c.strip()
    ]
    results = await lookup_citations_batch(payload)
    # Dict[int, ...] -> map back to the original citation strings for clarity
    return {"results": {citations[i]: results.get(i) for i in range(len(citations))}}


@app.post("/matters/{matter_id}/conflicts", response_model=dict)
async def matter_conflicts(
    matter_id: str,
    query: str = Body(..., embed=True, max_length=5000),
    db: Session = Depends(get_db),
):
    """Detect cross-source conflicts among the chunks most relevant to a query."""
    matter_uuid = _guard_matter(db, matter_id)
    chunks = await _retrieve_chunks_for(matter_uuid, query)
    try:
        from backend.services.conflict_detector import detect_conflicts
    except ImportError:
        try:
            from services.conflict_detector import detect_conflicts
        except ImportError:
            from .services.conflict_detector import detect_conflicts
    result = await asyncio.to_thread(detect_conflicts, chunks, query)
    return result or {"conflicts": [], "summary": {"total_conflicts": 0, "high_severity": 0,
                                                    "medium_severity": 0, "recommended_action": ""}}


@app.post("/matters/{matter_id}/verify-citations", response_model=dict)
async def matter_verify_citations(
    matter_id: str,
    answer: str = Body(..., embed=True, max_length=20000),
    query: str = Body("", embed=True, max_length=5000),
    db: Session = Depends(get_db),
):
    """Verify the citations in a supplied answer against the matter's documents."""
    matter_uuid = _guard_matter(db, matter_id)
    chunks = await _retrieve_chunks_for(matter_uuid, query or answer)
    try:
        from backend.services.citation_agent import verify_response_citations
    except ImportError:
        try:
            from services.citation_agent import verify_response_citations
        except ImportError:
            from .services.citation_agent import verify_response_citations
    result = await verify_response_citations(answer, chunks)
    return result or {"verified_citations": [], "summary": {}}


@app.post("/matters/{matter_id}/verify-claims", response_model=dict)
async def matter_verify_claims(
    matter_id: str,
    answer: str = Body(..., embed=True, max_length=20000),
    query: str = Body("", embed=True, max_length=5000),
    db: Session = Depends(get_db),
):
    """Verify (ground) the factual claims in a supplied answer against the matter's documents."""
    matter_uuid = _guard_matter(db, matter_id)
    chunks = await _retrieve_chunks_for(matter_uuid, query or answer)
    try:
        from backend.services.claim_verifier import verify_claims
    except ImportError:
        try:
            from services.claim_verifier import verify_claims
        except ImportError:
            from .services.claim_verifier import verify_claims
    result = await verify_claims(answer, chunks)
    return result or {"verified_claims": [], "summary": {}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
