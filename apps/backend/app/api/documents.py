from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db
from shared.models import Document, DocumentType, ProcessingStatus
from app.schemas.document import DocumentResponse
from app.services.storage import storage_service
from app.celery_app import celery_app
from app.config import settings
from uuid import uuid4
import logging
import redis.asyncio as redis

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    case_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document to a case
    Starts async processing pipeline
    """
    try:
        # Validate file
        if not storage_service.validate_file(file.filename, file.size):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid file",
            )

        # Read file content
        file_content = await file.read()

        # Create document record
        document_id = str(uuid4())

        new_document = Document(
            id=document_id,
            case_id=case_id,
            title=file.filename,
            filename=file.filename,
            type=DocumentType.OTHER,  # TODO: Auto-detect type
            processing_status=ProcessingStatus.PENDING,
            file_size=file.size,
        )

        db.add(new_document)
        await db.commit()
        await db.refresh(new_document)

        # Save file
        file_path = storage_service.save_file(document_id, file.filename, file_content)
        new_document.file_path = file_path

        await db.commit()
        await db.refresh(new_document)

        # Queue extraction task
        job_payload = {
            "document_id": document_id,
            "case_id": case_id,
            "source": "upload",
        }

        celery_app.send_task(
            "workers.document_extraction.extract_text_from_document",
            args=[job_payload],
        )

        logger.info(f"[documents] Uploaded document: {document_id}")
        return new_document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[documents] Upload error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload document",
        )

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get document details"""
    try:
        stmt = select(Document).where(Document.id == document_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[documents] Get document error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get document",
        )

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete document"""
    try:
        stmt = select(Document).where(Document.id == document_id)
        result = await db.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found",
            )

        # Delete file
        if document.file_path:
            storage_service.delete_file(document.file_path)

        # Delete document
        await db.delete(document)
        await db.commit()

        logger.info(f"[documents] Deleted document: {document_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[documents] Delete document error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document",
        )

@router.get("/documents/{document_id}/progress")
async def stream_document_progress(document_id: str):
    """Stream document processing progress via SSE."""
    async def event_generator():
        redis_client = redis.from_url(settings.REDIS_URL)
        try:
            pubsub = redis_client.pubsub()
            await pubsub.subscribe(f"progress:{document_id}")

            async for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode()
                    yield f"data: {data}\n\n"
        finally:
            await redis_client.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
