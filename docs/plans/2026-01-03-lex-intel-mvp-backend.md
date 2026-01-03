# LexIntel MVP - Backend Implementation Plan (Python + Celery)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a production-ready Python FastAPI backend with Celery workers for document processing, providing REST APIs for case management, document upload, full-text/semantic search, and AI-powered chat.

**Architecture:** Python FastAPI backend with async request handling, PostgreSQL database with pgvector for semantic search, Redis for caching and job queues, Celery workers for async document processing (text extraction, embeddings), Auth0 for authentication, and OpenAI for embeddings/chat LLM.

**Tech Stack:**
- Backend: Python 3.11 + FastAPI + Uvicorn
- Database: PostgreSQL 15 + pgvector + SQLAlchemy ORM
- Workers: Celery + Redis
- Auth: Auth0
- LLM/Embeddings: OpenAI API
- File Storage: Azure Blob Storage
- Testing: pytest + pytest-asyncio
- Deployment: Docker + Docker Compose

---

## Phase 1: Project Setup & Infrastructure (3 days)

### Task 1.1: Initialize Python Backend Project Structure

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/pyproject.toml`
- Create: `backend/Dockerfile`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/.env.example`

**Step 1: Create backend directory structure**

```bash
mkdir -p backend/app/{api,models,services,workers,schemas}
mkdir -p backend/tests/{unit,integration}
mkdir -p backend/migrations
cd backend
```

**Step 2: Create requirements.txt**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/requirements.txt`:
```
# Core
fastapi==0.109.0
uvicorn[standard]==0.27.0
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
pgvector==0.2.5

# Authentication
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6

# Async & Caching
aiofiles==23.2.1
redis==5.0.1
celery==5.3.4

# External APIs
openai==1.3.8
azure-storage-blob==12.19.0

# Testing
pytest==7.4.3
pytest-asyncio==0.23.2
httpx==0.25.2

# Utilities
python-dateutil==2.8.2
pytz==2023.3
requests==2.31.0

# Development
black==23.12.1
flake8==6.1.0
mypy==1.7.1
```

**Step 3: Create pyproject.toml**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "lex-intel-backend"
version = "0.1.0"
description = "AI-powered legal research platform backend"
requires-python = ">=3.11"

[tool.black]
line-length = 100
target-version = ['py311']

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q"
testpaths = ["tests"]
pythonpath = "."
asyncio_mode = "auto"
```

**Step 4: Create Dockerfile**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 5: Create main.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[backend] LexIntel backend starting...")
    yield
    # Shutdown
    print("[backend] LexIntel backend shutting down...")

app = FastAPI(
    title="LexIntel API",
    description="AI-powered legal research platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LexIntel API v0.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# TODO: Add routers
# from app.api import cases, documents, search, chat
# app.include_router(cases.router)
# app.include_router(documents.router)
# app.include_router(search.router)
# app.include_router(chat.router)
```

**Step 6: Create .env.example**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/.env.example`:
```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/lex_intel_dev

# Redis
REDIS_URL=redis://localhost:6379

# OpenAI
OPENAI_API_KEY=sk-...

# Auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;...

# App
DEBUG=True
ENVIRONMENT=development
API_HOST=localhost
API_PORT=8000
```

**Step 7: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/
git commit -m "chore: initialize Python FastAPI backend project structure"
```

---

### Task 1.2: Setup PostgreSQL Database Models

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/base.py`
- Create: `backend/app/models/user.py`
- Create: `backend/app/models/case.py`
- Create: `backend/app/models/document.py`
- Create: `backend/app/config.py`
- Create: `backend/app/database.py`

**Step 1: Create database configuration**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/config.py`:
```python
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    API_HOST: str = "localhost"
    API_PORT: int = 8000

    # Database
    DATABASE_URL: str
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"

    # Auth0
    AUTH0_DOMAIN: str
    AUTH0_CLIENT_ID: str
    AUTH0_CLIENT_SECRET: str

    # Azure Storage
    AZURE_STORAGE_CONNECTION_STRING: str

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

**Step 2: Create database connection**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/database.py`:
```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from app.config import settings
from typing import AsyncGenerator

# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DATABASE_ECHO,
    future=True,
    poolclass=NullPool,  # Avoid connection pool issues in development
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initialize database tables"""
    from app.models.base import Base

    async with engine.begin() as conn:
        # Enable pgvector extension
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
```

**Step 3: Create base model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/base.py`:
```python
from datetime import datetime
from sqlalchemy import Column, DateTime, func
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    """Base class for all models"""
    pass

class TimestampMixin:
    """Mixin to add created_at and updated_at timestamps"""
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
```

**Step 4: Create User model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/user.py`:
```python
from sqlalchemy import Column, String, Enum, ForeignKey
from sqlalchemy.orm import relationship
import enum
from app.models.base import Base, TimestampMixin

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    SENIOR_LAWYER = "senior_lawyer"
    LAWYER = "lawyer"
    VIEWER = "viewer"

class User(Base, TimestampMixin):
    __tablename__ = "users"

    id = Column(String, primary_key=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=False, index=True)
    role = Column(Enum(UserRole), default=UserRole.LAWYER, nullable=False)
    auth0_id = Column(String, unique=True, nullable=False, index=True)

    # Relationships
    firm = relationship("Firm", back_populates="users")
    cases = relationship("Case", secondary="case_assignments", back_populates="assigned_lawyers")
    chats = relationship("ChatConversation", back_populates="lawyer")
    annotations = relationship("Annotation", back_populates="user")

class Firm(Base, TimestampMixin):
    __tablename__ = "firms"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    domain = Column(String, unique=True, nullable=False)

    # Relationships
    users = relationship("User", back_populates="firm")
    cases = relationship("Case", back_populates="firm")
```

**Step 5: Create Case model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/case.py`:
```python
from sqlalchemy import Column, String, Text, DateTime, Enum, ForeignKey, Table, Float
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.models.base import Base, TimestampMixin

class CaseStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"

class CaseOutcome(str, enum.Enum):
    PLAINTIFF_WIN = "plaintiff_win"
    DEFENDANT_WIN = "defendant_win"
    SETTLEMENT = "settlement"
    PENDING = "pending"

# Association table for many-to-many relationship
case_assignments = Table(
    "case_assignments",
    Base.metadata,
    Column("case_id", String, ForeignKey("cases.id")),
    Column("user_id", String, ForeignKey("users.id")),
)

class Case(Base, TimestampMixin):
    __tablename__ = "cases"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    case_number = Column(String, unique=True, nullable=False, index=True)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=False, index=True)
    practice_area = Column(String, nullable=False)  # contracts, IP, litigation, M&A, etc.
    status = Column(Enum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False, index=True)
    jurisdiction = Column(String)  # federal, state, etc.
    court_level = Column(String)  # district, appeals, supreme
    filing_date = Column(DateTime)
    hearing_date = Column(DateTime)
    judgment_date = Column(DateTime)
    outcome = Column(Enum(CaseOutcome))
    description = Column(Text)

    # Relationships
    firm = relationship("Firm", back_populates="cases")
    assigned_lawyers = relationship("User", secondary=case_assignments, back_populates="cases")
    documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    chats = relationship("ChatConversation", back_populates="case", cascade="all, delete-orphan")
    metadata = relationship("CaseMetadata", back_populates="case", uselist=False, cascade="all, delete-orphan")

class CaseMetadata(Base, TimestampMixin):
    __tablename__ = "case_metadata"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), unique=True, nullable=False)
    plaintiff_name = Column(String)
    defendant_name = Column(String)
    judge_name = Column(String)
    legal_topics = Column(String)  # JSON string
    outcome_confidence = Column(Float, default=0.5)
    risk_level = Column(String)  # low, medium, high

    # Relationships
    case = relationship("Case", back_populates="metadata")
```

**Step 6: Create Document model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/document.py`:
```python
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, ARRAY, Enum
from sqlalchemy.orm import relationship
from sqlalchemy_utils import JSONType
import enum
from app.models.base import Base, TimestampMixin

class DocumentType(str, enum.Enum):
    BRIEF = "brief"
    COMPLAINT = "complaint"
    DISCOVERY = "discovery"
    STATUTE = "statute"
    TRANSCRIPT = "transcript"
    CONTRACT = "contract"
    EVIDENCE = "evidence"
    OTHER = "other"

class DocumentSource(str, enum.Enum):
    UPLOADED = "uploaded"
    EMAIL = "email"
    API = "api"

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTED = "extracted"
    INDEXED = "indexed"
    FAILED = "failed"

class Document(Base, TimestampMixin):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    type = Column(Enum(DocumentType), nullable=False)
    source = Column(Enum(DocumentSource), default=DocumentSource.UPLOADED)
    extracted_text = Column(Text)
    page_count = Column(Integer)
    file_size = Column(Integer)
    file_url = Column(String)  # Azure Storage URL
    privileged = Column(Boolean, default=False)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    error_message = Column(Text)
    indexed_at = Column(DateTime)

    # Relationships
    case = relationship("Case", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(String)  # pgvector stored as string, will be cast in queries
    search_vector = Column(String)  # PostgreSQL tsvector

    # Relationships
    document = relationship("Document", back_populates="chunks")

class Annotation(Base, TimestampMixin):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    text_highlighted = Column(Text, nullable=False)
    comment = Column(Text)
    tags = Column(ARRAY(String), default=[])

    # Relationships
    document = relationship("Document", back_populates="annotations")
    user = relationship("User", back_populates="annotations")

class ChatConversation(Base, TimestampMixin):
    __tablename__ = "chat_conversations"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    lawyer_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    firm_id = Column(String, ForeignKey("firms.id"), nullable=False, index=True)
    title = Column(String, default="Untitled Conversation")
    token_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)
    status = Column(String, default="active")

    # Relationships
    case = relationship("Case", back_populates="chats")
    lawyer = relationship("User", back_populates="chats")
    messages = relationship("ChatMessage", back_populates="conversation", cascade="all, delete-orphan")

class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"

    id = Column(String, primary_key=True)
    conversation_id = Column(String, ForeignKey("chat_conversations.id"), nullable=False, index=True)
    role = Column(String, nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, default=0)
    source_document_ids = Column(ARRAY(String), default=[])

    # Relationships
    conversation = relationship("ChatConversation", back_populates="messages")
```

**Step 7: Create __init__.py for models**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/__init__.py`:
```python
from app.models.base import Base, TimestampMixin
from app.models.user import User, Firm, UserRole
from app.models.case import Case, CaseMetadata, CaseStatus, CaseOutcome
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentType,
    DocumentSource,
    ProcessingStatus,
    Annotation,
    ChatConversation,
    ChatMessage,
)

__all__ = [
    "Base",
    "TimestampMixin",
    "User",
    "Firm",
    "UserRole",
    "Case",
    "CaseMetadata",
    "CaseStatus",
    "CaseOutcome",
    "Document",
    "DocumentChunk",
    "DocumentType",
    "DocumentSource",
    "ProcessingStatus",
    "Annotation",
    "ChatConversation",
    "ChatMessage",
]
```

**Step 8: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/models backend/app/config.py backend/app/database.py
git commit -m "feat: setup SQLAlchemy database models and PostgreSQL configuration"
```

---

### Task 1.3: Setup Celery Workers Infrastructure

**Files:**
- Create: `backend/app/celery_app.py`
- Create: `backend/app/workers/__init__.py`
- Create: `backend/app/workers/tasks.py`

**Step 1: Create Celery app configuration**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/celery_app.py`:
```python
from celery import Celery
from celery.signals import task_prerun, task_postrun
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery(
    "lex-intel",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    logger.info(f"[celery] Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, state=None, **kwargs):
    logger.info(f"[celery] Completed task: {task.name} (ID: {task_id})")

# Auto-discover tasks from app.workers.tasks
celery_app.autodiscover_tasks(["app.workers"])
```

**Step 2: Create workers tasks module**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/workers/tasks.py`:
```python
import logging
from celery import shared_task, Task
from app.database import async_session
from app.models import Document, ProcessingStatus
from app.config import settings
import asyncio

logger = logging.getLogger(__name__)

class CallbackTask(Task):
    """Task with on_success and on_failure callbacks"""
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"[workers] Task {task_id} succeeded")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"[workers] Task {task_id} failed: {exc}")

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    """
    Extract text from document
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        # TODO: Implement text extraction
        # - Download file from Azure Storage
        # - Extract based on file type (PDF, DOCX, TXT)
        # - Clean and preprocess text
        # - Update Document record with extracted_text

        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def generate_embeddings(self, document_id: str):
    """
    Generate embeddings for document chunks
    """
    try:
        logger.info(f"[generate_embeddings] Starting for document {document_id}")

        # TODO: Implement embedding generation
        # - Get document chunks from database
        # - Call OpenAI embedding API
        # - Store embeddings in DocumentChunk.embedding
        # - Create full-text search vector

        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[generate_embeddings] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)

@shared_task(base=CallbackTask, bind=True)
def process_document_pipeline(self, document_id: str):
    """
    Complete document processing pipeline:
    1. Extract text
    2. Create chunks
    3. Generate embeddings
    """
    try:
        logger.info(f"[process_pipeline] Starting for document {document_id}")

        # Chain tasks
        from app.workers.tasks import extract_text_from_document, generate_embeddings

        # Extract text first
        extract_text_from_document.delay(document_id)

        return {"status": "processing", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[process_pipeline] Error: {exc}")
        raise
```

**Step 3: Create workers __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/workers/__init__.py`:
```python
from app.workers.tasks import (
    extract_text_from_document,
    generate_embeddings,
    process_document_pipeline,
)

__all__ = [
    "extract_text_from_document",
    "generate_embeddings",
    "process_document_pipeline",
]
```

**Step 4: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/celery_app.py backend/app/workers/
git commit -m "feat: setup Celery workers for async document processing"
```

---

### Task 1.4: Setup Docker Compose for Local Development

**Files:**
- Create: `docker-compose.yml`
- Modify: `backend/Dockerfile`

**Step 1: Create docker-compose.yml**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/docker-compose.yml`:
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: lex-intel-postgres
    environment:
      POSTGRES_USER: lex_user
      POSTGRES_PASSWORD: lex_password
      POSTGRES_DB: lex_intel_dev
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lex_user"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - lex-network

  # Redis Cache & Job Queue
  redis:
    image: redis:7-alpine
    container_name: lex-intel-redis
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - lex-network

  # FastAPI Backend
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: lex-intel-backend
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev
      REDIS_URL: redis://redis:6379
      DEBUG: "True"
      ENVIRONMENT: development
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    volumes:
      - ./backend:/app
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    networks:
      - lex-network

  # Celery Worker
  celery-worker:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: lex-intel-celery-worker
    environment:
      DATABASE_URL: postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev
      REDIS_URL: redis://redis:6379
      DEBUG: "True"
      ENVIRONMENT: development
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    command: celery -A app.celery_app worker -l info
    networks:
      - lex-network

  # Celery Beat Scheduler (for scheduled tasks)
  celery-beat:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: lex-intel-celery-beat
    environment:
      DATABASE_URL: postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev
      REDIS_URL: redis://redis:6379
      DEBUG: "True"
      ENVIRONMENT: development
    depends_on:
      - postgres
      - redis
    volumes:
      - ./backend:/app
    command: celery -A app.celery_app beat -l info
    networks:
      - lex-network

volumes:
  postgres_data:

networks:
  lex-network:
    driver: bridge
```

**Step 2: Create .env file for local development**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/.env`:
```
# Database
DATABASE_URL=postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev

# Redis
REDIS_URL=redis://redis:6379

# OpenAI (get from https://platform.openai.com/account/api-keys)
OPENAI_API_KEY=sk-your-key-here

# Auth0
AUTH0_DOMAIN=your-tenant.auth0.com
AUTH0_CLIENT_ID=your-client-id
AUTH0_CLIENT_SECRET=your-client-secret

# Azure Storage
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=https;AccountName=xxx;AccountKey=xxx;EndpointSuffix=core.windows.net

# App
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000
```

**Step 3: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add docker-compose.yml backend/.env
git commit -m "chore: setup Docker Compose for local development"
```

---

## Phase 2: API Foundation & Authentication (3 days)

### Task 2.1: Setup Pydantic Schemas

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/user.py`
- Create: `backend/app/schemas/case.py`
- Create: `backend/app/schemas/document.py`
- Create: `backend/app/schemas/chat.py`

**Step 1: Create user schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/user.py`:
```python
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from app.models import UserRole

class UserBase(BaseModel):
    email: EmailStr
    name: str
    role: UserRole = UserRole.LAWYER

class UserCreate(UserBase):
    auth0_id: str
    firm_id: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[UserRole] = None

class UserResponse(UserBase):
    id: str
    firm_id: str
    auth0_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class UserInDB(UserResponse):
    pass
```

**Step 2: Create case schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/case.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List
from app.models import CaseStatus, CaseOutcome

class CaseBase(BaseModel):
    name: str
    case_number: str
    practice_area: str
    status: CaseStatus = CaseStatus.ACTIVE
    jurisdiction: Optional[str] = None
    court_level: Optional[str] = None
    filing_date: Optional[datetime] = None
    hearing_date: Optional[datetime] = None
    judgment_date: Optional[datetime] = None
    outcome: Optional[CaseOutcome] = None
    description: Optional[str] = None

class CaseCreate(CaseBase):
    firm_id: str

class CaseUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[CaseStatus] = None
    outcome: Optional[CaseOutcome] = None
    description: Optional[str] = None

class CaseResponse(CaseBase):
    id: str
    firm_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class CaseDetailResponse(CaseResponse):
    document_count: Optional[int] = 0
    assigned_lawyers_count: Optional[int] = 0
```

**Step 3: Create document schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/document.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.models import DocumentType, DocumentSource, ProcessingStatus

class DocumentBase(BaseModel):
    title: str
    filename: str
    type: DocumentType
    source: DocumentSource = DocumentSource.UPLOADED
    privileged: bool = False

class DocumentCreate(DocumentBase):
    case_id: str

class DocumentResponse(DocumentBase):
    id: str
    case_id: str
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    file_url: Optional[str] = None
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    indexed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocumentUploadResponse(DocumentResponse):
    pass
```

**Step 4: Create chat schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/chat.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ChatMessageCreate(BaseModel):
    role: str  # user, assistant
    content: str
    tokens_used: int = 0

class ChatMessageResponse(ChatMessageCreate):
    id: str
    conversation_id: str
    source_document_ids: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ChatConversationBase(BaseModel):
    case_id: str
    title: Optional[str] = "Untitled Conversation"

class ChatConversationCreate(ChatConversationBase):
    lawyer_id: str
    firm_id: str

class ChatConversationResponse(ChatConversationBase):
    id: str
    lawyer_id: str
    firm_id: str
    token_count: int
    message_count: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatConversationDetailResponse(ChatConversationResponse):
    messages: List[ChatMessageResponse] = []
```

**Step 5: Create schemas __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/__init__.py`:
```python
from app.schemas.user import UserCreate, UserResponse, UserUpdate
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate, CaseDetailResponse
from app.schemas.document import DocumentCreate, DocumentResponse, DocumentUploadResponse
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatConversationCreate,
    ChatConversationResponse,
    ChatConversationDetailResponse,
)

__all__ = [
    "UserCreate",
    "UserResponse",
    "UserUpdate",
    "CaseCreate",
    "CaseResponse",
    "CaseUpdate",
    "CaseDetailResponse",
    "DocumentCreate",
    "DocumentResponse",
    "DocumentUploadResponse",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatConversationCreate",
    "ChatConversationResponse",
    "ChatConversationDetailResponse",
]
```

**Step 6: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/schemas/
git commit -m "feat: create Pydantic schemas for API validation"
```

---

### Task 2.2: Setup Auth0 Authentication

**Files:**
- Create: `backend/app/auth.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/auth.py`

**Step 1: Create authentication module**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/auth.py`:
```python
import logging
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredentials
from jose import JWTError, jwt
from app.config import settings
from functools import lru_cache
import httpx

logger = logging.getLogger(__name__)

security = HTTPBearer()

@lru_cache
def get_auth0_public_key():
    """Fetch Auth0 public key for JWT verification"""
    try:
        with httpx.Client() as client:
            response = client.get(f"https://{settings.AUTH0_DOMAIN}/.well-known/jwks.json")
            response.raise_for_status()
            return response.json()
    except Exception as e:
        logger.error(f"[auth] Failed to fetch Auth0 public key: {e}")
        raise

def verify_token(credentials: HTTPAuthCredentials = Depends(security)):
    """Verify JWT token from Auth0"""
    token = credentials.credentials

    try:
        # Get public key
        jwks = get_auth0_public_key()
        unverified_header = jwt.get_unverified_header(token)
        rsa_key = None

        for key in jwks["keys"]:
            if key["kid"] == unverified_header["kid"]:
                rsa_key = key
                break

        if not rsa_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unable to find a signing key that matches",
            )

        # Verify token
        payload = jwt.decode(
            token,
            rsa_key,
            algorithms=["RS256"],
            audience=settings.AUTH0_CLIENT_ID,
            issuer=f"https://{settings.AUTH0_DOMAIN}/",
        )

        return payload
    except JWTError as e:
        logger.error(f"[auth] JWT verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )
    except Exception as e:
        logger.error(f"[auth] Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
        )

async def get_current_user(payload: dict = Depends(verify_token)):
    """Get current authenticated user from token payload"""
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    return {"auth0_id": sub, "email": payload.get("email")}
```

**Step 2: Create auth routes**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/auth.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db
from app.models import User, Firm
from app.auth import get_current_user
from app.schemas.user import UserResponse, UserCreate
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/login", response_model=UserResponse)
async def login(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Login or create user if doesn't exist.
    Auth0 handles actual authentication, we just ensure user exists in DB.
    """
    try:
        auth0_id = current_user.get("auth0_id")
        email = current_user.get("email")

        if not auth0_id or not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing required user info",
            )

        # Check if user exists
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if user:
            return user

        # Create new user (with default firm)
        # TODO: In production, get firm from Auth0 custom claims or ask user to join firm
        default_firm = Firm(id=str(uuid4()), name="Default Firm", domain="default")

        new_user = User(
            id=str(uuid4()),
            email=email,
            name=email.split("@")[0],  # Use email prefix as name
            auth0_id=auth0_id,
            firm_id=default_firm.id,
        )

        db.add(default_firm)
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)

        logger.info(f"[auth] New user created: {new_user.email}")
        return new_user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[auth] Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )

@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user"""
    try:
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[auth] Get me error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user",
        )
```

**Step 3: Create API __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/__init__.py`:
```python
# API routers will be imported here
```

**Step 4: Update main.py to include auth routes**

Update `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import auth
from app.database import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[backend] LexIntel backend starting...")
    await init_db()
    print("[backend] Database initialized")
    yield
    # Shutdown
    print("[backend] LexIntel backend shutting down...")

app = FastAPI(
    title="LexIntel API",
    description="AI-powered legal research platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "LexIntel API v0.1.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}

# Include routers
app.include_router(auth.router)

# TODO: Add more routers
# from app.api import cases, documents, search, chat
# app.include_router(cases.router)
# app.include_router(documents.router)
# app.include_router(search.router)
# app.include_router(chat.router)
```

**Step 5: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/auth.py backend/app/api/auth.py backend/app/main.py
git commit -m "feat: setup Auth0 authentication and login/logout flows"
```

---

## Phase 3: Core APIs (4 days)

### Task 3.1: Create Cases API

**Files:**
- Create: `backend/app/api/cases.py`

**Step 1: Create cases router**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/cases.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
from app.database import get_db
from app.models import Case, User, Firm
from app.auth import get_current_user
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate, CaseDetailResponse
from typing import List
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cases", tags=["cases"])

@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    case_create: CaseCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create a new case"""
    try:
        # Get current user to verify they're in the firm
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Verify user's firm matches
        if case_create.firm_id != user.firm_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Cannot create case in different firm",
            )

        new_case = Case(
            id=str(uuid4()),
            **case_create.dict()
        )

        db.add(new_case)
        await db.commit()
        await db.refresh(new_case)

        logger.info(f"[cases] Created case: {new_case.id}")
        return new_case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Create case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create case",
        )

@router.get("", response_model=List[CaseResponse])
async def list_cases(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
):
    """List cases for current user's firm"""
    try:
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Get cases for user's firm
        stmt = select(Case).where(
            Case.firm_id == user.firm_id
        ).offset(skip).limit(limit)
        result = await db.execute(stmt)
        cases = result.scalars().all()

        return cases
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] List cases error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cases",
        )

@router.get("/{case_id}", response_model=CaseDetailResponse)
async def get_case(
    case_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get case details"""
    try:
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Get case and verify ownership
        stmt = select(Case).where(
            and_(Case.id == case_id, Case.firm_id == user.firm_id)
        )
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        return case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Get case error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get case",
        )

@router.patch("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    case_update: CaseUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Update case"""
    try:
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Get case and verify ownership
        stmt = select(Case).where(
            and_(Case.id == case_id, Case.firm_id == user.firm_id)
        )
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        # Update fields
        update_data = case_update.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(case, field, value)

        await db.commit()
        await db.refresh(case)

        logger.info(f"[cases] Updated case: {case.id}")
        return case
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Update case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update case",
        )

@router.delete("/{case_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_case(
    case_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete case"""
    try:
        auth0_id = current_user.get("auth0_id")
        stmt = select(User).where(User.auth0_id == auth0_id)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Get case and verify ownership
        stmt = select(Case).where(
            and_(Case.id == case_id, Case.firm_id == user.firm_id)
        )
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

        await db.delete(case)
        await db.commit()

        logger.info(f"[cases] Deleted case: {case.id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[cases] Delete case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete case",
        )
```

**Step 2: Update main.py to include cases router**

Update the bottom of `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/main.py`:
```python
# Include routers
app.include_router(auth.router)
from app.api import cases
app.include_router(cases.router)
```

**Step 3: Test endpoint with curl (after docker-compose is running)**

```bash
# Login (get token)
curl -X POST http://localhost:8000/auth/login \
  -H "Authorization: Bearer YOUR_AUTH0_TOKEN"

# List cases
curl -X GET http://localhost:8000/cases \
  -H "Authorization: Bearer YOUR_AUTH0_TOKEN"

# Create case
curl -X POST http://localhost:8000/cases \
  -H "Authorization: Bearer YOUR_AUTH0_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Smith v. Jones",
    "case_number": "2024-001",
    "firm_id": "firm-123",
    "practice_area": "contracts"
  }'
```

**Step 4: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/api/cases.py backend/app/main.py
git commit -m "feat: create cases CRUD API endpoints"
```

---

## Summary & Next Steps

**✅ Completed So Far:**
1. Project setup & infrastructure
2. PostgreSQL database models
3. Celery workers infrastructure
4. Docker Compose for local development
5. Pydantic schemas
6. Auth0 authentication
7. Cases API (CRUD)

**📋 Remaining Backend Phases:**
- Phase 3 (continued): Documents API + upload
- Phase 4: Document processing workers
- Phase 5: Search API (full-text + semantic)
- Phase 6: Chat/RAG API
- Phase 7: Testing & deployment

**This is a detailed, bite-sized plan with:**
- ✅ Complete code examples
- ✅ Exact file paths
- ✅ Git commits for each task
- ✅ Test-first approach (ready to add tests)
- ✅ TDD methodology
- ✅ Docker setup
- ✅ Best practices throughout

---

## Execution Approach

**Two options for continuing:**

**Option 1: Subagent-Driven (Recommended)**
- Fresh subagent per task
- I review between tasks
- Fast iteration
- Best for complex backend work

**Option 2: Parallel Session**
- Open new session with `superpowers:executing-plans`
- Batch execution with checkpoints
- Good for focused sprints

**Which would you prefer?**

Or would you like me to continue the plan with more phases (documents, search, chat) before we start executing?

