# LexIntel MVP - Backend Implementation Plan (RAG-First with Local Azurite)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a production-ready Python FastAPI backend focused on RAG (Retrieval-Augmented Generation) features: document upload, text extraction, embeddings generation, full-text/semantic search, and streaming AI chat - all with local development setup using Azurite for Azure Blob Storage emulation.

**Architecture:** Python FastAPI backend with async request handling, PostgreSQL database with pgvector for semantic search, Redis for caching and job queues, Celery workers for async document processing (text extraction, embeddings), OpenAI API for embeddings/chat LLM, and Azurite (Azure Storage emulator) for local Blob Storage.

**Tech Stack:**
- Backend: Python 3.11 + FastAPI + Uvicorn
- Database: PostgreSQL 15 + pgvector + SQLAlchemy ORM
- Workers: Celery + Redis
- LLM/Embeddings: OpenAI API
- File Storage: Azurite (local Azure Blob Storage emulator)
- Testing: pytest + pytest-asyncio
- Deployment: Docker + Docker Compose
- **NO Auth0 for MVP** (add later)

**Priority Order:**
1. Setup & infrastructure
2. **Document upload & storage** (Azurite)
3. **Text extraction** (workers)
4. **Embeddings** (OpenAI)
5. **Search** (full-text + semantic)
6. **Chat/RAG** (streaming)
7. *Later:* Auth0 integration

---

## Phase 1: Project Setup & Infrastructure (2-3 days)

### Task 1.1: Initialize Python Backend Project Structure

**Files:**
- Create: `backend/requirements.txt` (updated for Azurite)
- Create: `backend/pyproject.toml`
- Create: `backend/Dockerfile`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/.env.example`

**Step 1: Create backend directory structure**

```bash
mkdir -p backend/app/{api,models,services,workers,schemas}
mkdir -p backend/tests/{unit,integration}
mkdir -p backend/uploads
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

# File Operations
python-multipart==0.0.6
aiofiles==23.2.1

# Async & Caching
redis==5.0.1
celery==5.3.4

# External APIs
openai==1.3.8
azure-storage-blob==12.19.0

# Document Processing
pdf2image==1.17.0
PyPDF2==3.0.1
python-pptx==0.6.21
openpyxl==3.11.0

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

# Create uploads directory
RUN mkdir -p /app/uploads

# Run FastAPI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 5: Create main.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[backend] LexIntel backend starting...")
    # TODO: Initialize database
    yield
    # Shutdown
    logger.info("[backend] LexIntel backend shutting down...")

app = FastAPI(
    title="LexIntel API",
    description="AI-powered legal research platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
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

# TODO: Include routers
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
DATABASE_URL=postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev

# Redis
REDIS_URL=redis://redis:6379

# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Azurite (Local Azure Storage Emulator)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;

# App
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000

# Upload settings
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt,.doc,.pptx

# File processing
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
```

**Step 7: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/
git commit -m "chore: initialize Python FastAPI backend with Azurite support"
```

---

### Task 1.2: Setup PostgreSQL Database Models (Simplified for MVP)

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/base.py`
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

    # Azure Storage (Azurite locally)
    AZURE_STORAGE_CONNECTION_STRING: str

    # Upload settings
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.doc,.pptx"

    # Processing
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 400

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
    poolclass=NullPool,
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
        # Enable pg_trgm for full-text search
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
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

**Step 4: Create Case model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/case.py`:
```python
from sqlalchemy import Column, String, Text, DateTime, Enum
from app.models.base import Base, TimestampMixin
import enum

class CaseStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"

class Case(Base, TimestampMixin):
    __tablename__ = "cases"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    case_number = Column(String, unique=True, nullable=False, index=True)
    practice_area = Column(String, nullable=False)
    status = Column(Enum(CaseStatus), default=CaseStatus.ACTIVE, nullable=False, index=True)
    description = Column(Text)

    # Relationships
    # documents = relationship("Document", back_populates="case", cascade="all, delete-orphan")
    # chats = relationship("ChatConversation", back_populates="case", cascade="all, delete-orphan")
```

**Step 5: Create Document model**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/document.py`:
```python
from sqlalchemy import Column, String, Text, Integer, Boolean, DateTime, ForeignKey, ARRAY, Enum
from sqlalchemy.orm import relationship
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
    extracted_text = Column(Text)
    page_count = Column(Integer)
    file_size = Column(Integer)
    file_path = Column(String)  # Local file path
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    error_message = Column(Text)
    indexed_at = Column(DateTime)

    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True)
    document_id = Column(String, ForeignKey("documents.id"), nullable=False, index=True)
    chunk_text = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    embedding = Column(String)  # pgvector stored as string
    search_vector = Column(String)  # PostgreSQL tsvector

    # Relationships
    document = relationship("Document", back_populates="chunks")

class ChatConversation(Base, TimestampMixin):
    __tablename__ = "chat_conversations"

    id = Column(String, primary_key=True)
    case_id = Column(String, ForeignKey("cases.id"), nullable=False, index=True)
    title = Column(String, default="Untitled Conversation")
    token_count = Column(Integer, default=0)
    message_count = Column(Integer, default=0)

    # Relationships
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

**Step 6: Create __init__.py for models**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/models/__init__.py`:
```python
from app.models.base import Base, TimestampMixin
from app.models.case import Case, CaseStatus
from app.models.document import (
    Document,
    DocumentChunk,
    DocumentType,
    ProcessingStatus,
    ChatConversation,
    ChatMessage,
)

__all__ = [
    "Base",
    "TimestampMixin",
    "Case",
    "CaseStatus",
    "Document",
    "DocumentChunk",
    "DocumentType",
    "ProcessingStatus",
    "ChatConversation",
    "ChatMessage",
]
```

**Step 7: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/models backend/app/config.py backend/app/database.py
git commit -m "feat: setup SQLAlchemy database models for RAG MVP"
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

# Auto-discover tasks
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
import os

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
    Extract text from document (PDF, DOCX, TXT, etc.)
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        # TODO: Implement text extraction
        # 1. Get document from DB
        # 2. Download from local file path
        # 3. Extract text based on file type
        # 4. Update Document.extracted_text
        # 5. Queue embedding generation task

        return {"status": "success", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def generate_embeddings(self, document_id: str):
    """
    Generate embeddings for document chunks using OpenAI
    """
    try:
        logger.info(f"[generate_embeddings] Starting for document {document_id}")

        # TODO: Implement embedding generation
        # 1. Get document chunks from DB
        # 2. Call OpenAI embedding API
        # 3. Store embeddings in DocumentChunk.embedding
        # 4. Update Document status to INDEXED

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

        # Queue text extraction first
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

### Task 1.4: Setup Docker Compose with Azurite

**Files:**
- Create: `docker-compose.yml`
- Create: `backend/.env`

**Step 1: Create docker-compose.yml with Azurite**

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

  # Azurite (Azure Storage Emulator)
  azurite:
    image: mcr.microsoft.com/azure-storage/azurite:latest
    container_name: lex-intel-azurite
    ports:
      - "10000:10000"  # Blob Storage
      - "10001:10001"  # Queue Storage
      - "10002:10002"  # Table Storage
    volumes:
      - azurite_data:/data
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
      AZURE_STORAGE_CONNECTION_STRING: DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DEBUG: "True"
      ENVIRONMENT: development
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      azurite:
        condition: service_started
    volumes:
      - ./backend:/app
      - lex_uploads:/app/uploads
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
      AZURE_STORAGE_CONNECTION_STRING: DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      DEBUG: "True"
      ENVIRONMENT: development
    depends_on:
      - postgres
      - redis
      - azurite
    volumes:
      - ./backend:/app
      - lex_uploads:/app/uploads
    command: celery -A app.celery_app worker -l info
    networks:
      - lex-network

volumes:
  postgres_data:
  azurite_data:
  lex_uploads:

networks:
  lex-network:
    driver: bridge
```

**Step 2: Create .env file**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/.env`:
```
# Database
DATABASE_URL=postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev

# Redis
REDIS_URL=redis://redis:6379

# OpenAI (get from https://platform.openai.com/account/api-keys)
OPENAI_API_KEY=sk-your-key-here

# Azurite (local Azure Storage emulator - pre-configured)
AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;

# App
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000

# Upload settings
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# Processing
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
```

**Step 3: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add docker-compose.yml backend/.env
git commit -m "chore: setup Docker Compose with PostgreSQL, Redis, and Azurite"
```

---

## Phase 2: Document Upload & Storage (2-3 days)

### Task 2.1: Create Pydantic Schemas

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/case.py`
- Create: `backend/app/schemas/document.py`
- Create: `backend/app/schemas/chat.py`

**Step 1: Create case schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/case.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.models import CaseStatus

class CaseBase(BaseModel):
    name: str
    case_number: str
    practice_area: str
    status: CaseStatus = CaseStatus.ACTIVE
    description: Optional[str] = None

class CaseCreate(CaseBase):
    pass

class CaseUpdate(BaseModel):
    name: Optional[str] = None
    status: Optional[CaseStatus] = None
    description: Optional[str] = None

class CaseResponse(CaseBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
```

**Step 2: Create document schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/document.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.models import DocumentType, ProcessingStatus

class DocumentBase(BaseModel):
    title: str
    filename: str
    type: DocumentType

class DocumentResponse(DocumentBase):
    id: str
    case_id: str
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    file_path: Optional[str] = None
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    indexed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_text: str
    chunk_index: int

    class Config:
        from_attributes = True
```

**Step 3: Create chat schemas**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/chat.py`:
```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional, List

class ChatMessageCreate(BaseModel):
    role: str  # user, assistant
    content: str

class ChatMessageResponse(BaseModel):
    id: str
    conversation_id: str
    role: str
    content: str
    tokens_used: int
    source_document_ids: List[str]
    created_at: datetime

    class Config:
        from_attributes = True

class ChatConversationCreate(BaseModel):
    case_id: str
    title: Optional[str] = "Untitled Conversation"

class ChatConversationResponse(BaseModel):
    id: str
    case_id: str
    title: str
    token_count: int
    message_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class ChatConversationDetailResponse(ChatConversationResponse):
    messages: List[ChatMessageResponse] = []
```

**Step 4: Create __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/schemas/__init__.py`:
```python
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate
from app.schemas.document import DocumentResponse, DocumentChunkResponse
from app.schemas.chat import (
    ChatMessageCreate,
    ChatMessageResponse,
    ChatConversationCreate,
    ChatConversationResponse,
    ChatConversationDetailResponse,
)

__all__ = [
    "CaseCreate",
    "CaseResponse",
    "CaseUpdate",
    "DocumentResponse",
    "DocumentChunkResponse",
    "ChatMessageCreate",
    "ChatMessageResponse",
    "ChatConversationCreate",
    "ChatConversationResponse",
    "ChatConversationDetailResponse",
]
```

**Step 5: Commit**

```bash
git add backend/app/schemas/
git commit -m "feat: create Pydantic schemas for API validation"
```

---

### Task 2.2: Create File Upload Service

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/storage.py`

**Step 1: Create storage service**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/services/storage.py`:
```python
import os
import shutil
from pathlib import Path
from typing import Optional
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class StorageService:
    """Handle file uploads and storage"""

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    def save_file(self, document_id: str, filename: str, file_content: bytes) -> str:
        """
        Save uploaded file to disk
        Returns: file path
        """
        try:
            # Create document directory
            doc_dir = self.upload_dir / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            # Save file
            file_path = doc_dir / filename
            file_path.write_bytes(file_content)

            logger.info(f"[storage] Saved file: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"[storage] Failed to save file: {e}")
            raise

    def get_file(self, file_path: str) -> bytes:
        """Read file from disk"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            return path.read_bytes()
        except Exception as e:
            logger.error(f"[storage] Failed to read file: {e}")
            raise

    def delete_file(self, file_path: str) -> None:
        """Delete file from disk"""
        try:
            path = Path(file_path)
            if path.exists():
                path.unlink()
                logger.info(f"[storage] Deleted file: {file_path}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete file: {e}")
            raise

    def delete_document_files(self, document_id: str) -> None:
        """Delete all files for a document"""
        try:
            doc_dir = self.upload_dir / document_id
            if doc_dir.exists():
                shutil.rmtree(doc_dir)
                logger.info(f"[storage] Deleted document directory: {doc_dir}")
        except Exception as e:
            logger.error(f"[storage] Failed to delete document directory: {e}")
            raise

    def validate_file(self, filename: str, file_size: int) -> bool:
        """Validate file before upload"""
        # Check file extension
        allowed = settings.ALLOWED_EXTENSIONS.split(",")
        ext = Path(filename).suffix.lower()
        if ext not in allowed:
            logger.warning(f"[storage] Invalid file extension: {ext}")
            return False

        # Check file size
        if file_size > settings.MAX_UPLOAD_SIZE:
            logger.warning(f"[storage] File too large: {file_size} bytes")
            return False

        return True

storage_service = StorageService()
```

**Step 2: Create __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/services/__init__.py`:
```python
from app.services.storage import storage_service

__all__ = ["storage_service"]
```

**Step 3: Commit**

```bash
git add backend/app/services/
git commit -m "feat: create storage service for file upload/download"
```

---

### Task 2.3: Create Cases and Documents API Routes

**Files:**
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/cases.py`
- Create: `backend/app/api/documents.py`

**Step 1: Create cases router**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/cases.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import and_
from app.database import get_db
from app.models import Case
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate
from typing import List
from uuid import uuid4
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cases", tags=["cases"])

@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    case_create: CaseCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new case"""
    try:
        new_case = Case(id=str(uuid4()), **case_create.dict())
        db.add(new_case)
        await db.commit()
        await db.refresh(new_case)

        logger.info(f"[cases] Created case: {new_case.id}")
        return new_case
    except Exception as e:
        logger.error(f"[cases] Create case error: {e}")
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create case",
        )

@router.get("", response_model=List[CaseResponse])
async def list_cases(
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 50,
):
    """List all cases"""
    try:
        stmt = select(Case).offset(skip).limit(limit)
        result = await db.execute(stmt)
        cases = result.scalars().all()
        return cases
    except Exception as e:
        logger.error(f"[cases] List cases error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list cases",
        )

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get case details"""
    try:
        stmt = select(Case).where(Case.id == case_id)
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
    db: AsyncSession = Depends(get_db),
):
    """Update case"""
    try:
        stmt = select(Case).where(Case.id == case_id)
        result = await db.execute(stmt)
        case = result.scalar_one_or_none()

        if not case:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Case not found",
            )

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
    db: AsyncSession = Depends(get_db),
):
    """Delete case"""
    try:
        stmt = select(Case).where(Case.id == case_id)
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

**Step 2: Create documents router**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/documents.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.database import get_db
from app.models import Document, DocumentType, ProcessingStatus
from app.schemas.document import DocumentResponse
from app.services.storage import storage_service
from app.workers.tasks import process_document_pipeline
from pathlib import Path
from uuid import uuid4
import logging

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

        # Start async processing pipeline
        process_document_pipeline.delay(document_id)

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
```

**Step 3: Create API __init__.py**

Create `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/api/__init__.py`:
```python
# API routers
```

**Step 4: Update main.py to include routes**

Update `/Users/smeet/Documents/GitHub/Self-Learning/lex-intel/backend/app/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import cases, documents
from app.database import init_db
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[backend] LexIntel backend starting...")
    await init_db()
    logger.info("[backend] Database initialized")
    yield
    # Shutdown
    logger.info("[backend] LexIntel backend shutting down...")

app = FastAPI(
    title="LexIntel API",
    description="AI-powered legal research platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
app.include_router(cases.router)
app.include_router(documents.router)

# TODO: Add more routers
# from app.api import search, chat
# app.include_router(search.router)
# app.include_router(chat.router)
```

**Step 5: Commit**

```bash
git add backend/app/api/ backend/app/main.py
git commit -m "feat: create cases and documents CRUD APIs with file upload"
```

---

## Summary

**✅ Completed Phase 1 & 2:**
1. ✅ Project setup with Azurite
2. ✅ PostgreSQL models for RAG
3. ✅ Celery workers infrastructure
4. ✅ Docker Compose setup
5. ✅ Pydantic schemas
6. ✅ File upload service (local storage)
7. ✅ Cases and Documents APIs

**📋 Remaining Phases (to continue):**
- **Phase 3:** Text extraction workers (PyPDF2, python-pptx)
- **Phase 4:** Embedding generation (OpenAI API)
- **Phase 5:** Search API (full-text + semantic pgvector)
- **Phase 6:** Chat/RAG API (streaming with context)
- **Phase 7:** Testing & documentation

---

## Next Steps

**Would you like me to:**

1. **Continue the plan** with Phases 3-6 (text extraction, embeddings, search, chat)?
2. **Start implementing now** using superpowers:executing-plans?
3. **Review/adjust** the current plan?

What's your preference? 🚀