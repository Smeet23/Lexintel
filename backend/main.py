from fastapi import FastAPI, Depends, HTTPException, status, Header, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timezone
from uuid import UUID
import uuid
import logging

logger = logging.getLogger(__name__)

try:
    from backend.config import get_settings
    from backend.database import get_db
    from backend.models import User, Case, Query
    from backend.schemas import UserCreate, UserResponse, TokenResponse
    from backend.auth import hash_password, verify_password, create_access_token, decode_token
    from backend.services.storage import upload_pdf_to_blob, validate_pdf
    from backend.services.rag_engine import query_case
    from backend.validators import validate_filename, validate_case_name, validate_question
except ImportError:
    try:
        from config import get_settings
        from database import get_db
        from models import User, Case, Query
        from schemas import UserCreate, UserResponse, TokenResponse
        from auth import hash_password, verify_password, create_access_token, decode_token
        from services.storage import upload_pdf_to_blob, validate_pdf
        from services.rag_engine import query_case
        from validators import validate_filename, validate_case_name, validate_question
    except ImportError:
        from .config import get_settings
        from .database import get_db
        from .models import User, Case, Query
        from .schemas import UserCreate, UserResponse, TokenResponse
        from .auth import hash_password, verify_password, create_access_token, decode_token
        from .services.storage import upload_pdf_to_blob, validate_pdf
        from .services.rag_engine import query_case
        from .validators import validate_filename, validate_case_name, validate_question

settings = get_settings()


def get_cors_origins() -> list:
    """Get allowed CORS origins from settings"""
    origins = settings.get_allowed_origins_list()

    # Validate no placeholder domains
    placeholder_domains = ["yourdomain.com", "example.com", "localhost.com"]
    for origin in origins:
        for placeholder in placeholder_domains:
            if placeholder in origin:
                raise ValueError(
                    f"Placeholder domain '{placeholder}' found in CORS configuration. "
                    f"Please set ALLOWED_ORIGINS environment variable to valid domain(s)."
                )

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
    allow_headers=["Content-Type", "Authorization"],
)


@app.on_event("startup")
async def startup_validation():
    """Validate configuration on application startup"""
    # Validate SECRET_KEY is not default in production
    if not settings.debug and settings.secret_key == "dev-secret-key-change-in-production":
        logger.warning(
            "WARNING: Using default SECRET_KEY in production. "
            "This is a security risk. Please set SECRET_KEY environment variable."
        )

    # Validate CORS configuration again
    try:
        cors_origins = get_cors_origins()
        logger.info(f"CORS origins configured: {cors_origins}")
    except ValueError as e:
        logger.error(f"CORS configuration error: {str(e)}")
        raise

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

async def get_current_user(
    authorization: str = Header(None),
    db: Session = Depends(get_db)
) -> UUID:
    """Extract and validate JWT token from Authorization header"""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header"
        )

    try:
        scheme, token = authorization.split(" ")
        if scheme.lower() != "bearer":
            raise ValueError()
    except (ValueError, IndexError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format"
        )

    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token"
        )

    # Verify user exists and not deleted
    user = db.query(User).filter(User.id == user_id, User.is_deleted == False).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return UUID(user_id)


def verify_case_ownership(case, user_id: UUID) -> None:
    """Verify that a user owns a case"""
    if case.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this case"
        )


@app.post("/auth/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register new user"""
    # Check if user already exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    new_user = User(
        id=uuid.uuid4(),
        email=user_data.email,
        password_hash=hash_password(user_data.password),
        created_at=datetime.now(timezone.utc)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@app.post("/auth/login", response_model=TokenResponse)
def login(user_data: UserCreate, db: Session = Depends(get_db)):
    """Authenticate user and return JWT token"""
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/user/profile", response_model=UserResponse)
def get_profile(
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user's profile (protected endpoint)"""
    user = db.query(User).filter(User.id == current_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ============================================
# CASE MANAGEMENT ENDPOINTS
# ============================================

@app.get("/cases", response_model=list)
async def list_cases(
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all cases for the current user (protected endpoint)"""
    cases = db.query(Case).filter(Case.user_id == current_user_id).all()
    return [
        {
            "id": str(case.id),
            "name": case.name,
            "status": case.status,
            "created_at": case.created_at.isoformat(),
            "updated_at": case.updated_at.isoformat() if case.updated_at else None
        }
        for case in cases
    ]

@app.post("/cases", response_model=dict)
async def upload_case(
    name: str = Form(...),
    file: UploadFile = File(...),
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload a case PDF document"""
    # Validate file is PDF
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files allowed"
        )

    # Validate filename using dedicated validator
    if file.filename:
        validate_filename(file.filename)

    # Validate case name using dedicated validator
    validate_case_name(name)

    try:
        # Read file content early for validation
        file_content = await file.read()

        # Validate PDF magic bytes
        if not validate_pdf(file_content):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid filename: must be a PDF"
            )

        # Create case record with status "processing"
        case_id = uuid.uuid4()
        case = Case(
            id=case_id,
            user_id=current_user_id,
            name=name,
            blob_storage_path="",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Upload file to blob storage
        blob_path = await upload_pdf_to_blob(file_content, str(case_id), file.filename)

        # Update case with blob path
        case.blob_storage_path = blob_path
        db.commit()
        db.refresh(case)

        # Send document processing task to Celery queue
        try:
            from .celery_app import celery_app
        except ImportError:
            from celery_app import celery_app

        task = celery_app.send_task(
            'backend.tasks.process_document_task',
            args=(str(case_id),),
            queue='celery'
        )
        logger.info(f"Queued document processing task {task.id} for case {case_id}")

        return {
            "id": str(case.id),
            "name": case.name,
            "status": case.status,
            "blob_storage_path": case.blob_storage_path,
            "task_id": task.id,
            "created_at": case.created_at.isoformat()
        }

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.exception(f"Failed to upload case: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload case. Please try again."
        )


# ============================================
# RAG QUERY ENDPOINTS
# ============================================

@app.post("/cases/{case_id}/ask", response_model=dict)
async def ask_question(
    case_id: str,
    question: str = Body(..., embed=True),
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ask a question about a case"""
    # Validate question input
    validate_question(question)

    try:
        from uuid import UUID
        case_uuid = UUID(case_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format"
        )

    # Check if case exists
    case = db.query(Case).filter(Case.id == case_uuid).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )

    # Verify user owns this case
    verify_case_ownership(case, current_user_id)

    # Check if case is ready for querying
    if case.status == "processing":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Case is still being processed. Please try again in a moment."
        )
    elif case.status == "error":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Case processing failed. Please re-upload the document."
        )

    # Get RAG response
    try:
        rag_result = await query_case(str(case_uuid), question, db)

        # Only store if answer was generated successfully
        if rag_result.get("answer"):
            db_query = Query(
                id=uuid.uuid4(),
                case_id=case_uuid,
                user_id=case.user_id,
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


@app.get("/cases/{case_id}/status", response_model=dict)
async def get_case_status(
    case_id: str,
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get the status of a case"""
    try:
        from uuid import UUID
        case_uuid = UUID(case_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid case ID format"
        )

    case = db.query(Case).filter(Case.id == case_uuid).first()
    if not case:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )

    # Verify user owns this case
    verify_case_ownership(case, current_user_id)

    return {
        "id": str(case.id),
        "name": case.name,
        "status": case.status,
        "created_at": case.created_at.isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
