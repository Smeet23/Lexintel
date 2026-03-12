from typing import List
from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Body, Query as QueryParam, Depends
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
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

try:
    from backend.config import get_settings
    from backend.database import get_db
    from backend.models import Matter, Chunk, Query, Document
    from backend.services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
    from backend.services.rag_engine import query_matter
    from backend.validators import validate_filename, validate_matter_name, validate_question, validate_file_type
    from backend.services.progress import publish_uploaded
    from backend.exceptions import BlobDownloadException
except ImportError:
    try:
        from config import get_settings
        from database import get_db
        from models import Matter, Chunk, Query, Document
        from services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
        from services.rag_engine import query_matter
        from validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from services.progress import publish_uploaded
        from exceptions import BlobDownloadException
    except ImportError:
        from .config import get_settings
        from .database import get_db
        from .models import Matter, Chunk, Query, Document
        from .services.storage import upload_document_to_blob, download_document_from_blob, validate_file_format
        from .services.rag_engine import query_matter
        from .validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from .services.progress import publish_uploaded
        from .exceptions import BlobDownloadException

settings = get_settings()


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
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
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
            file_content = await file.read()

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
                queue='celery'
            )
            document.celery_task_id = task.id
            task_ids.append(task.id)
            logger.info(f"Queued processing task {task.id} for document {document_id} ({file.filename})")

        # Update matter blob_storage_path to first document's path
        matter.blob_storage_path = blob_path
        db.commit()
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

    try:
        file_content = await file.read()

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
            queue='celery'
        )
        document.celery_task_id = task.id
        db.commit()
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
# RAG QUERY ENDPOINTS
# ============================================

@app.post("/matters/{matter_id}/ask", response_model=dict)
async def ask_question(
    matter_id: str,
    question: str = Body(..., embed=True),
    db: Session = Depends(get_db)
):
    """Ask a question about a matter"""
    # Validate question input
    validate_question(question)

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

    # Fetch recent conversation history for follow-up context
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

    # Get RAG response
    try:
        rag_result = await query_matter(str(matter_uuid), question, db, conversation_history=conversation_history)

        # Only store if answer was generated successfully
        if rag_result.get("answer"):
            db_query = Query(
                id=uuid.uuid4(),
                matter_id=matter_uuid,
                question=question,
                answer=rag_result.get("answer", ""),
                citations=rag_result.get("sources", []),
                created_at=datetime.now(timezone.utc)
            )
            db.add(db_query)
            db.commit()

        return rag_result
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
        """Generate SSE events from Redis pub/sub."""
        redis_client = await aioredis.from_url(
            settings.celery_broker_url,
            decode_responses=True
        )
        pubsub = redis_client.pubsub()
        channel = f"lexintel:matter:{matter_id}:progress"

        try:
            await pubsub.subscribe(channel)
            logger.info(f"SSE client subscribed to {channel}")

            # Send initial connection event
            yield {
                "event": "connected",
                "data": json.dumps({"matter_id": matter_id, "status": "connected"})
            }

            # Listen for messages with heartbeat and timeout
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
