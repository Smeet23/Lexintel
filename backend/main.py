from fastapi import FastAPI, HTTPException, status, File, UploadFile, Form, Body, Query as QueryParam, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from uuid import UUID
import uuid
import json
import logging
import asyncio
import redis.asyncio as aioredis
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

try:
    from backend.config import get_settings
    from backend.database import get_db
    from backend.models import Matter, Query
    from backend.services.storage import upload_document_to_blob, validate_file_format
    from backend.services.rag_engine import query_matter
    from backend.validators import validate_filename, validate_matter_name, validate_question, validate_file_type
    from backend.services.progress import publish_uploaded
except ImportError:
    try:
        from config import get_settings
        from database import get_db
        from models import Matter, Query
        from services.storage import upload_document_to_blob, validate_file_format
        from services.rag_engine import query_matter
        from validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from services.progress import publish_uploaded
    except ImportError:
        from .config import get_settings
        from .database import get_db
        from .models import Matter, Query
        from .services.storage import upload_document_to_blob, validate_file_format
        from .services.rag_engine import query_matter
        from .validators import validate_filename, validate_matter_name, validate_question, validate_file_type
        from .services.progress import publish_uploaded

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


# Import and register firm router
try:
    from backend.routers.firms import router as firms_router
except ImportError:
    try:
        from routers.firms import router as firms_router
    except ImportError:
        from .routers.firms import router as firms_router

app.include_router(firms_router)


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
    matters = db.query(Matter).filter(Matter.is_deleted == False).all()
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

    return {
        "id": str(matter.id),
        "name": matter.name,
        "status": matter.status,
        "file_type": matter.file_type,
        "blob_storage_path": matter.blob_storage_path,
        "documents_count": len(matter.chunks),
        "queries_count": len(matter.queries),
        "created_at": matter.created_at.isoformat(),
        "updated_at": matter.updated_at.isoformat() if matter.updated_at else None
    }


@app.post("/matters", response_model=dict)
async def upload_matter(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Upload a matter document (PDF, DOCX, or TXT)"""
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

    # Validate filename using dedicated validator
    if file.filename:
        validate_filename(file.filename)

    # Validate matter name using dedicated validator
    validate_matter_name(name)

    try:
        # Read file content early for validation
        file_content = await file.read()

        # Detect and validate file type
        try:
            file_type = validate_file_type(file.content_type, file.filename)
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(e)
            )

        # Validate file format matches declared type
        if not validate_file_format(file_content, file_type):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid {file_type.upper()} file format"
            )

        # Create matter record with status "processing"
        matter_id = uuid.uuid4()
        matter = Matter(
            id=matter_id,
            name=name,
            blob_storage_path="",
            file_type=file_type,
            status="processing"
        )
        db.add(matter)
        db.commit()

        # Upload file to blob storage
        blob_path = await upload_document_to_blob(file_content, str(matter_id), file.filename)

        # Update matter with blob path
        matter.blob_storage_path = blob_path
        db.commit()
        db.refresh(matter)

        # Publish "uploaded" progress event
        publish_uploaded(str(matter_id), file.filename or "document")

        # Send document processing task to Celery queue
        try:
            from .celery_app import celery_app
        except ImportError:
            from celery_app import celery_app

        task = celery_app.send_task(
            'backend.tasks.process_document_task',
            args=(str(matter_id),),
            queue='celery'
        )
        logger.info(f"Queued document processing task {task.id} for matter {matter_id}")

        return {
            "id": str(matter.id),
            "name": matter.name,
            "status": matter.status,
            "file_type": matter.file_type,
            "blob_storage_path": matter.blob_storage_path,
            "task_id": task.id,
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

    return {"id": str(matter.id), "deleted": True}


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

    # Check if matter exists
    matter = db.query(Matter).filter(Matter.id == matter_uuid).first()
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

    # Get RAG response
    try:
        rag_result = await query_matter(str(matter_uuid), question, db)

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

    matter = db.query(Matter).filter(Matter.id == matter_uuid).first()
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

    # Check matter exists
    matter = db.query(Matter).filter(Matter.id == matter_uuid).first()
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

            # Listen for messages
            async for message in pubsub.listen():
                if message["type"] == "message":
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
