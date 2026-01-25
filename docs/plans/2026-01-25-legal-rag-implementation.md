# Legal RAG App - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a production-ready legal RAG app with FastAPI backend, Next.js frontend, PostgreSQL database, Qdrant vector store, and OpenAI integration for Q&A and case summarization.

**Architecture:** Lawyer uploads case PDFs → backend chunks and embeds documents → stores in vector DB and PostgreSQL → lawyer queries documents → retrieval + LLM reasoning → citations returned to frontend with PDF highlighting.

**Tech Stack:**
- Backend: FastAPI, Python 3.11, LangChain
- Frontend: Next.js 14, React, TypeScript, Tailwind CSS
- Databases: PostgreSQL, Qdrant
- Storage: Azure Blob Storage (local Azurite for dev)
- LLM: OpenAI GPT-4o + text-embedding-3-large
- Auth: JWT
- Deployment: Docker, Azure

---

## PHASE 1: PROJECT SETUP (Tiny Steps)

### Task 1: Create Project Directories

**Files:**
- Create: `backend/`
- Create: `frontend/`
- Create: `docs/plans/`
- Create: `.gitignore`

**Step 1: Create folder structure**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/LexIntel
mkdir -p backend frontend/app docs/plans
```

**Step 2: Create .gitignore**

Create `.gitignore`:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.env.local

# Node
node_modules/
.next/
dist/
build/
*.tsbuildinfo

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Temp
*.log
```

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: initialize project structure"
```

---

### Task 2: Setup Backend - Python Virtual Environment

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/.env.example`

**Step 1: Create Python virtual environment**

```bash
cd backend
python3.11 -m venv venv
source venv/bin/activate
```

**Step 2: Create requirements.txt with initial dependencies**

Create `backend/requirements.txt`:
```
fastapi==0.109.0
uvicorn==0.27.0
python-dotenv==1.0.0
pydantic==2.5.3
pydantic-settings==2.1.0
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
```

**Step 3: Create .env.example**

Create `backend/.env.example`:
```
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/legal_rag
POSTGRES_USER=legal_user
POSTGRES_PASSWORD=secure_password
POSTGRES_DB=legal_rag

# OpenAI
OPENAI_API_KEY=sk-xxx

# Qdrant
QDRANT_URL=http://localhost:6333

# Azure
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true

# JWT
SECRET_KEY=your-secret-key-change-in-production
ALGORITHM=HS256

# Environment
DEBUG=True
```

**Step 4: Commit**

```bash
git add backend/requirements.txt backend/.env.example
git commit -m "chore: add backend dependencies and env template"
```

---

### Task 3: Setup Backend - FastAPI Project Structure

**Files:**
- Create: `backend/main.py`
- Create: `backend/config.py`
- Create: `backend/pyproject.toml`

**Step 1: Create pyproject.toml**

Create `backend/pyproject.toml`:
```toml
[project]
name = "legal-rag-backend"
version = "0.1.0"
description = "Legal RAG App Backend"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**Step 2: Create config.py**

Create `backend/config.py`:
```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Database
    database_url: str

    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Azure Blob Storage
    azure_storage_connection_string: str

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # Environment
    debug: bool = False

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings():
    return Settings()
```

**Step 3: Create main.py**

Create `backend/main.py`:
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config import get_settings

settings = get_settings()

app = FastAPI(
    title="Legal RAG API",
    description="RAG system for legal document analysis",
    version="0.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
```

**Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

**Step 5: Test the app runs**

```bash
python main.py
# Expected: INFO:     Uvicorn running on http://0.0.0.0:8000
# Press Ctrl+C to stop
```

**Step 6: Commit**

```bash
git add backend/main.py backend/config.py backend/pyproject.toml
git commit -m "feat: initialize FastAPI project structure"
```

---

### Task 4: Setup Frontend - Next.js Project

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/.env.example`

**Step 1: Create package.json**

Create `frontend/package.json`:
```json
{
  "name": "legal-rag-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "jest"
  },
  "dependencies": {
    "next": "14.1.0",
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "typescript": "^5.3.0",
    "axios": "^1.6.2",
    "tailwindcss": "^3.4.1",
    "autoprefixer": "^10.4.17",
    "postcss": "^8.4.33"
  },
  "devDependencies": {
    "@types/node": "^20.10.0",
    "@types/react": "^18.2.42",
    "@types/react-dom": "^18.2.17",
    "eslint": "^8.55.0",
    "eslint-config-next": "14.1.0"
  }
}
```

**Step 2: Create tsconfig.json**

Create `frontend/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2020",
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "jsx": "preserve",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ]
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

**Step 3: Create .env.example**

Create `frontend/.env.example`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=LexIntel
```

**Step 4: Create next.config.js**

Create `frontend/next.config.js`:
```javascript
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
}

module.exports = nextConfig
```

**Step 5: Commit**

```bash
git add frontend/package.json frontend/tsconfig.json frontend/.env.example frontend/next.config.js
git commit -m "chore: initialize Next.js project configuration"
```

---

### Task 5: Setup Docker Compose for Local Development

**Files:**
- Create: `docker-compose.yml`

**Step 1: Create docker-compose.yml**

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  # PostgreSQL Database
  postgres:
    image: postgres:16-alpine
    container_name: legal_rag_postgres
    environment:
      POSTGRES_USER: legal_user
      POSTGRES_PASSWORD: secure_password
      POSTGRES_DB: legal_rag
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U legal_user -d legal_rag"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    container_name: legal_rag_qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Azure Blob Storage Emulator (Azurite)
  azurite:
    image: mcr.microsoft.com/azure-storage/azurite:latest
    container_name: legal_rag_azurite
    ports:
      - "10000:10000"
      - "10001:10001"
    volumes:
      - azurite_data:/data

volumes:
  postgres_data:
  qdrant_data:
  azurite_data:
```

**Step 2: Create docker-compose up command**

```bash
docker-compose up -d
# Expected output:
# Creating legal_rag_postgres ... done
# Creating legal_rag_qdrant ... done
# Creating legal_rag_azurite ... done
```

**Step 3: Verify services are running**

```bash
docker-compose ps
# Expected: All services showing healthy or running
```

**Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: add Docker Compose for local development"
```

---

## PHASE 2: DATABASE SETUP

### Task 6: Create PostgreSQL Schema & Models

**Files:**
- Create: `backend/models.py`
- Create: `backend/database.py`

**Step 1: Create database.py**

Create `backend/database.py`:
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import get_settings

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

**Step 2: Create models.py**

Create `backend/models.py`:
```python
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from database import Base
from datetime import datetime
import uuid

class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Case(Base):
    __tablename__ = "cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    blob_storage_path = Column(String(500), nullable=False)
    status = Column(String(50), default="processing")  # processing, ready, error
    created_at = Column(DateTime, default=datetime.utcnow)

class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    page_num = Column(String(50), nullable=True)
    section_name = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

**Step 3: Create database tables**

```bash
cd backend
python -c "from database import engine, Base; from models import *; Base.metadata.create_all(bind=engine)"
# Expected: Tables created silently (or see them in postgres)
```

**Step 4: Verify tables exist**

```bash
psql -U legal_user -d legal_rag -h localhost -c "\dt"
# Expected: List of tables (users, cases, chunks, queries)
```

**Step 5: Commit**

```bash
git add backend/models.py backend/database.py
git commit -m "feat: create PostgreSQL schema and SQLAlchemy models"
```

---

### Task 7: Create Alembic for Database Migrations

**Files:**
- Create: `backend/alembic/` (directory structure)

**Step 1: Add alembic to requirements**

Edit `backend/requirements.txt`, add:
```
alembic==1.13.0
```

**Step 2: Install alembic**

```bash
pip install alembic
```

**Step 3: Initialize alembic**

```bash
alembic init alembic
```

**Step 4: Configure alembic.ini**

Edit `backend/alembic/alembic.ini` and update sqlalchemy.url:
```
sqlalchemy.url = driver://user:password@localhost/dbname
```

To:
```
sqlalchemy.url =
```

(We'll use env.py instead)

**Step 5: Configure env.py**

Edit `backend/alembic/env.py`:
```python
from logging.config import fileConfig
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from alembic import context
from config import get_settings
from database import Base
from models import *

config = context.config
settings = get_settings()

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

def run_migrations_offline() -> None:
    url = settings.database_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online() -> None:
    configuration = config.get_section(config.config_ini_section)
    configuration["sqlalchemy.url"] = settings.database_url

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

**Step 6: Create initial migration**

```bash
alembic revision --autogenerate -m "Initial schema"
```

**Step 7: Apply migration**

```bash
alembic upgrade head
```

**Step 8: Commit**

```bash
git add backend/alembic/
git commit -m "chore: setup Alembic for database migrations"
```

---

## PHASE 3: AUTHENTICATION

### Task 8: Create JWT Authentication

**Files:**
- Create: `backend/auth.py`
- Create: `backend/schemas.py`

**Step 1: Add JWT dependencies to requirements.txt**

Edit `backend/requirements.txt`, add:
```
python-jose==3.3.0
passlib==1.7.4
bcrypt==4.1.1
```

**Step 2: Install dependencies**

```bash
pip install python-jose passlib bcrypt
```

**Step 3: Create auth.py**

Create `backend/auth.py`:
```python
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from config import get_settings

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

def decode_token(token: str):
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None
        return user_id
    except JWTError:
        return None
```

**Step 4: Create schemas.py**

Create `backend/schemas.py`:
```python
from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional
from uuid import UUID

class UserCreate(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: UUID
    email: str
    created_at: datetime

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

class CaseCreate(BaseModel):
    name: str

class CaseResponse(BaseModel):
    id: UUID
    name: str
    status: str
    uploaded_at: datetime

    class Config:
        from_attributes = True
```

**Step 5: Create auth endpoints in main.py**

Edit `backend/main.py`:
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import timedelta
from config import get_settings
from database import get_db, SessionLocal
from models import User
from schemas import UserCreate, TokenResponse, UserResponse
from auth import hash_password, verify_password, create_access_token

settings = get_settings()

app = FastAPI(
    title="Legal RAG API",
    description="RAG system for legal document analysis",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/auth/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    new_user = User(
        email=user_data.email,
        password_hash=hash_password(user_data.password)
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/auth/login", response_model=TokenResponse)
def login(user_data: UserCreate, db: Session = Depends(get_db)):
    # Find user
    user = db.query(User).filter(User.email == user_data.email).first()
    if not user or not verify_password(user_data.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Create token
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=settings.debug)
```

**Step 6: Test the endpoints with curl**

```bash
# Register
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "lawyer@example.com", "password": "securepass123"}'

# Expected: {"id": "...", "email": "lawyer@example.com", "created_at": "..."}

# Login
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "lawyer@example.com", "password": "securepass123"}'

# Expected: {"access_token": "...", "token_type": "bearer"}
```

**Step 7: Commit**

```bash
git add backend/auth.py backend/schemas.py backend/main.py backend/requirements.txt
git commit -m "feat: add JWT authentication with register and login"
```

---

## PHASE 4: CORE API ENDPOINTS

### Task 9: Create Case Upload Endpoint

**Files:**
- Modify: `backend/main.py`
- Create: `backend/services/storage.py`

**Step 1: Add file upload dependencies**

Edit `backend/requirements.txt`, add:
```
python-multipart==0.0.6
azure-storage-blob==12.19.0
```

**Step 2: Install dependencies**

```bash
pip install python-multipart azure-storage-blob
```

**Step 3: Create storage service**

Create `backend/services/storage.py`:
```python
from azure.storage.blob import BlobServiceClient
from config import get_settings
import io

settings = get_settings()

def get_blob_client():
    return BlobServiceClient.from_connection_string(
        settings.azure_storage_connection_string
    )

async def upload_pdf_to_blob(file_content: bytes, case_id: str, filename: str) -> str:
    """Upload PDF to Azure Blob Storage and return path"""
    blob_client = get_blob_client()
    container_client = blob_client.get_container_client("cases")

    # Create container if it doesn't exist
    try:
        container_client.get_container_properties()
    except:
        container_client = blob_client.create_container("cases")

    # Upload blob
    blob_name = f"{case_id}/{filename}"
    blob_client_ref = container_client.get_blob_client(blob_name)
    blob_client_ref.upload_blob(file_content, overwrite=True)

    return blob_name
```

**Step 4: Add upload endpoint to main.py**

Edit `backend/main.py`, add imports and endpoint:
```python
from fastapi import File, UploadFile
from services.storage import upload_pdf_to_blob
from uuid import uuid4

@app.post("/cases")
async def upload_case(
    name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # Validate file is PDF
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    # Create case record
    case_id = str(uuid4())
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
    file_content = await file.read()
    blob_path = await upload_pdf_to_blob(file_content, case_id, file.filename)

    # Update case with blob path
    case.blob_storage_path = blob_path
    db.commit()
    db.refresh(case)

    return {"id": case.id, "name": case.name, "status": case.status}
```

**Step 5: Add current user dependency**

Edit `backend/main.py`, add function:
```python
from fastapi import Header

async def get_current_user(authorization: str = Header(...), db: Session = Depends(get_db)):
    """Extract and validate JWT token from Authorization header"""
    try:
        token = authorization.split(" ")[1]
    except:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

    from auth import decode_token
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    return user_id
```

**Step 6: Test upload endpoint**

```bash
# Get token first
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "lawyer@example.com", "password": "securepass123"}' | jq -r '.access_token')

# Upload case (create dummy PDF first)
echo "%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
trailer
<< /Size 4 /Root 1 0 R >>
startxref
214
%%EOF" > test.pdf

curl -X POST http://localhost:8000/cases \
  -H "Authorization: Bearer $TOKEN" \
  -F "name=Test Case" \
  -F "file=@test.pdf"

# Expected: {"id": "...", "name": "Test Case", "status": "processing"}
```

**Step 7: Commit**

```bash
git add backend/services/storage.py backend/main.py backend/requirements.txt
git commit -m "feat: add case upload endpoint with Azure Blob Storage"
```

---

## PHASE 5: RAG PIPELINE (CHUNKING & EMBEDDINGS)

### Task 10: Create Document Chunking Service

**Files:**
- Create: `backend/services/chunking.py`

**Step 1: Add LangChain dependencies**

Edit `backend/requirements.txt`, add:
```
langchain==0.1.7
pymupdf==1.24.0
openai==1.3.5
```

**Step 2: Install dependencies**

```bash
pip install langchain pymupdf openai
```

**Step 3: Create chunking service**

Create `backend/services/chunking.py`:
```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
import tempfile
import os

def chunk_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Chunk a PDF into semantic pieces with metadata
    Returns list of dicts: {content, page_num, section_name}
    """
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Chunk with overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for page in pages:
        split_docs = splitter.split_documents([page])

        for i, doc in enumerate(split_docs):
            # Extract page number from metadata
            page_num = doc.metadata.get("page", "unknown")

            chunk_dict = {
                "content": doc.page_content,
                "page_num": str(page_num),
                "section_name": f"Chunk {i+1}"  # Simple naming for MVP
            }
            chunks.append(chunk_dict)

    return chunks

async def chunk_pdf_from_blob(blob_content: bytes) -> List[Dict[str, str]]:
    """
    Chunk PDF from blob storage content
    """
    # Write to temp file
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(blob_content)
        tmp_path = tmp.name

    try:
        chunks = chunk_pdf(tmp_path)
        return chunks
    finally:
        os.unlink(tmp_path)
```

**Step 4: Test chunking**

```bash
python -c "
from services.chunking import chunk_pdf
# Test with the dummy PDF we created
chunks = chunk_pdf('test.pdf')
for chunk in chunks:
    print(f'Page {chunk[\"page_num\"]}: {chunk[\"content\"][:50]}...')
"
```

**Step 5: Commit**

```bash
git add backend/services/chunking.py backend/requirements.txt
git commit -m "feat: add PDF chunking service with LangChain"
```

---

### Task 11: Create Embeddings Service

**Files:**
- Create: `backend/services/embeddings.py`

**Step 1: Create embeddings service**

Create `backend/services/embeddings.py`:
```python
from langchain.embeddings.openai import OpenAIEmbeddings
from config import get_settings
from typing import List

settings = get_settings()

def get_embeddings_client():
    """Get OpenAI embeddings client"""
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model="text-embedding-3-large"
    )

async def embed_text(text: str) -> List[float]:
    """Embed a single piece of text"""
    embeddings = get_embeddings_client()
    embedding = embeddings.embed_query(text)
    return embedding

async def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Embed multiple chunks of text"""
    embeddings = get_embeddings_client()
    embeddings_list = embeddings.embed_documents(chunks)
    return embeddings_list
```

**Step 2: Test embedding**

```bash
# Set OpenAI API key first
export OPENAI_API_KEY="sk-..."

python -c "
import asyncio
from services.embeddings import embed_text

async def test():
    embedding = await embed_text('What is the court judgment?')
    print(f'Embedding shape: {len(embedding)} dimensions')
    print(f'First 5 values: {embedding[:5]}')

asyncio.run(test())
"
```

**Step 3: Commit**

```bash
git add backend/services/embeddings.py
git commit -m "feat: add OpenAI embeddings service"
```

---

### Task 12: Create Qdrant Vector Store Service

**Files:**
- Create: `backend/services/vector_store.py`

**Step 1: Add Qdrant dependency**

Edit `backend/requirements.txt`, add:
```
qdrant-client==2.7.1
```

**Step 2: Install**

```bash
pip install qdrant-client
```

**Step 3: Create vector store service**

Create `backend/services/vector_store.py`:
```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
from config import get_settings
from typing import List, Dict
import uuid

settings = get_settings()

def get_qdrant_client():
    """Get Qdrant client"""
    return QdrantClient(url=settings.qdrant_url)

async def create_collection(collection_name: str, vector_size: int = 3072):
    """Create a new collection in Qdrant"""
    client = get_qdrant_client()

    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

async def upsert_vectors(
    collection_name: str,
    vectors: List[List[float]],
    metadata: List[Dict],
    case_id: str
):
    """Insert vectors with metadata into Qdrant"""
    client = get_qdrant_client()

    # Ensure collection exists
    await create_collection(collection_name)

    # Create points
    points = []
    for i, (vector, meta) in enumerate(zip(vectors, metadata)):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "case_id": case_id,
                "page_num": meta.get("page_num"),
                "section_name": meta.get("section_name"),
                "content": meta.get("content")
            }
        )
        points.append(point)

    # Upsert points
    client.upsert(
        collection_name=collection_name,
        points=points
    )

async def search_vectors(
    collection_name: str,
    query_vector: List[float],
    case_id: str,
    top_k: int = 4
) -> List[Dict]:
    """Search for similar vectors"""
    client = get_qdrant_client()

    results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter={
            "must": [
                {
                    "key": "case_id",
                    "match": {"value": case_id}
                }
            ]
        },
        limit=top_k
    )

    retrieved = []
    for result in results:
        retrieved.append({
            "content": result.payload.get("content"),
            "page_num": result.payload.get("page_num"),
            "section_name": result.payload.get("section_name"),
            "score": result.score
        })

    return retrieved
```

**Step 4: Test Qdrant**

```bash
python -c "
import asyncio
from services.vector_store import create_collection, upsert_vectors

async def test():
    await create_collection('legal_rag')
    print('Collection created')

asyncio.run(test())
"
```

**Step 5: Commit**

```bash
git add backend/services/vector_store.py backend/requirements.txt
git commit -m "feat: add Qdrant vector store service"
```

---

## PHASE 6: DOCUMENT PROCESSING PIPELINE

### Task 13: Create Background Job for Document Processing

**Files:**
- Create: `backend/services/document_processor.py`

**Step 1: Create document processor**

Create `backend/services/document_processor.py`:
```python
from services.chunking import chunk_pdf_from_blob
from services.embeddings import embed_chunks
from services.vector_store import upsert_vectors
from database import SessionLocal
from models import Case, Chunk
from azure.storage.blob import BlobServiceClient
from config import get_settings

settings = get_settings()

async def process_case_document(case_id: str):
    """
    Full pipeline: download PDF → chunk → embed → store
    """
    db = SessionLocal()
    try:
        # Get case
        case = db.query(Case).filter(Case.id == case_id).first()
        if not case:
            return {"error": "Case not found"}

        # Download PDF from blob storage
        blob_client = BlobServiceClient.from_connection_string(
            settings.azure_storage_connection_string
        )
        blob_client_ref = blob_client.get_blob_client(
            container="cases",
            blob=case.blob_storage_path
        )
        pdf_content = blob_client_ref.download_blob().readall()

        # Chunk PDF
        chunks = await chunk_pdf_from_blob(pdf_content)

        # Extract content for embeddings
        chunk_contents = [chunk["content"] for chunk in chunks]

        # Create embeddings
        embeddings = await embed_chunks(chunk_contents)

        # Store vectors in Qdrant
        await upsert_vectors(
            collection_name="legal_rag",
            vectors=embeddings,
            metadata=chunks,
            case_id=case_id
        )

        # Store chunk metadata in PostgreSQL
        for chunk in chunks:
            db_chunk = Chunk(
                case_id=case_id,
                page_num=chunk["page_num"],
                section_name=chunk["section_name"],
                content=chunk["content"]
            )
            db.add(db_chunk)

        # Update case status
        case.status = "ready"
        db.commit()

        return {"status": "success", "chunks_processed": len(chunks)}

    except Exception as e:
        case = db.query(Case).filter(Case.id == case_id).first()
        if case:
            case.status = "error"
            db.commit()
        return {"error": str(e)}
    finally:
        db.close()
```

**Step 2: Add celery dependency for background jobs**

Edit `backend/requirements.txt`, add:
```
celery==5.3.4
redis==5.0.1
```

**Step 3: Create celery configuration**

Create `backend/celery_app.py`:
```python
from celery import Celery
from config import get_settings

settings = get_settings()

celery_app = Celery(
    "legal_rag",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)

@celery_app.task
def process_document_task(case_id: str):
    from services.document_processor import process_case_document
    import asyncio
    return asyncio.run(process_case_document(case_id))
```

**Step 4: Update upload endpoint to use celery**

Edit `backend/main.py`:
```python
@app.post("/cases")
async def upload_case(
    name: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    # ... existing code ...

    # Queue background job
    from celery_app import celery_app
    celery_app.send_task('celery_app.process_document_task', args=[case_id])

    return {"id": case.id, "name": case.name, "status": case.status}
```

**Step 5: Commit**

```bash
git add backend/services/document_processor.py backend/celery_app.py backend/main.py backend/requirements.txt
git commit -m "feat: add background job for document processing pipeline"
```

---

## PHASE 7: RAG QUERY ENDPOINT

### Task 14: Create RAG Query Service

**Files:**
- Create: `backend/services/rag_engine.py`

**Step 1: Create RAG engine**

Create `backend/services/rag_engine.py`:
```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from services.embeddings import embed_text
from services.vector_store import search_vectors
from config import get_settings
from typing import Dict, List

settings = get_settings()

async def query_rag(case_id: str, question: str) -> Dict:
    """
    Full RAG flow: embed question → retrieve chunks → ask LLM → return answer
    """
    try:
        # 1. Embed the question
        question_embedding = await embed_text(question)

        # 2. Retrieve relevant chunks from Qdrant
        retrieved_chunks = await search_vectors(
            collection_name="legal_rag",
            query_vector=question_embedding,
            case_id=case_id,
            top_k=4
        )

        if not retrieved_chunks:
            return {
                "answer": "No relevant information found in the case documents.",
                "citations": []
            }

        # 3. Build context
        context = "\n\n".join([
            f"[Page {chunk['page_num']}] {chunk['content']}"
            for chunk in retrieved_chunks
        ])

        # 4. Create prompt
        prompt_template = ChatPromptTemplate.from_template("""
You are a legal assistant analyzing court documents. Answer the following question ONLY based on the provided context.

If the answer is not found in the context, respond with: "This information is not available in the provided documents."

Always cite the page number when referencing the document.

Context:
{context}

Question: {question}

Answer:
""")

        # 5. Call LLM
        llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model="gpt-4-1106-preview",
            temperature=0.3
        )

        prompt = prompt_template.format(context=context, question=question)
        response = llm.predict(prompt)

        # 6. Extract citations
        citations = [
            {
                "page": chunk["page_num"],
                "section": chunk["section_name"],
                "content_snippet": chunk["content"][:100]
            }
            for chunk in retrieved_chunks
        ]

        return {
            "answer": response,
            "citations": citations
        }

    except Exception as e:
        return {
            "error": str(e),
            "answer": "Error processing query"
        }
```

**Step 2: Test RAG query**

```bash
python -c "
import asyncio
from services.rag_engine import query_rag

async def test():
    result = await query_rag('test-case-id', 'What is the judgment?')
    print(result)

asyncio.run(test())
"
```

**Step 3: Commit**

```bash
git add backend/services/rag_engine.py
git commit -m "feat: add RAG query service with LLM integration"
```

---

### Task 15: Create Query Endpoint

**Files:**
- Modify: `backend/main.py`

**Step 1: Add query endpoint**

Edit `backend/main.py`, add:
```python
from services.rag_engine import query_rag
from models import Query

@app.post("/cases/{case_id}/ask")
async def ask_question(
    case_id: str,
    question: str,
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user)
):
    """Ask a question about a specific case"""
    # Verify case belongs to user
    case = db.query(Case).filter(
        Case.id == case_id,
        Case.user_id == current_user_id
    ).first()

    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    if case.status != "ready":
        raise HTTPException(
            status_code=400,
            detail=f"Case is {case.status}. Please wait for processing to complete."
        )

    # Get RAG response
    rag_result = await query_rag(case_id, question)

    # Store query in database
    db_query = Query(
        case_id=case_id,
        user_id=current_user_id,
        question=question,
        answer=rag_result.get("answer", ""),
        citations=rag_result.get("citations", [])
    )
    db.add(db_query)
    db.commit()

    return rag_result
```

**Step 2: Test endpoint**

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "lawyer@example.com", "password": "securepass123"}' | jq -r '.access_token')

curl -X POST http://localhost:8000/cases/{case_id}/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the judgment?"}'
```

**Step 3: Commit**

```bash
git add backend/main.py
git commit -m "feat: add case query endpoint with RAG"
```

---

## PHASE 8: FRONTEND SETUP

### Task 16: Initialize Next.js and Create Auth Pages

**Files:**
- Create: `frontend/app/layout.tsx`
- Create: `frontend/app/page.tsx`
- Create: `frontend/app/auth/login/page.tsx`
- Create: `frontend/app/auth/register/page.tsx`

**Step 1: Install frontend dependencies**

```bash
cd frontend
npm install
```

**Step 2: Create app layout**

Create `frontend/app/layout.tsx`:
```typescript
import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LexIntel - Legal RAG',
  description: 'RAG system for legal document analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="bg-gray-50">
        {children}
      </body>
    </html>
  )
}
```

**Step 3: Create globals.css**

Create `frontend/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  font-family: system-ui, -apple-system, sans-serif;
}
```

**Step 4: Create home page**

Create `frontend/app/page.tsx`:
```typescript
'use client'

import Link from 'next/link'

export default function Home() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="text-center">
        <h1 className="text-4xl font-bold mb-4">LexIntel</h1>
        <p className="text-xl text-gray-600 mb-8">Legal Document Analysis with RAG</p>
        <div className="space-x-4">
          <Link href="/auth/login" className="bg-blue-600 text-white px-6 py-3 rounded">
            Login
          </Link>
          <Link href="/auth/register" className="bg-gray-600 text-white px-6 py-3 rounded">
            Register
          </Link>
        </div>
      </div>
    </div>
  )
}
```

**Step 5: Create login page**

Create `frontend/app/auth/login/page.tsx`:
```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      if (!res.ok) throw new Error('Login failed')

      const data = await res.json()
      localStorage.setItem('token', data.access_token)
      router.push('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white p-8 rounded shadow-lg w-96">
        <h2 className="text-2xl font-bold mb-6">Login</h2>
        {error && <div className="bg-red-100 text-red-700 p-3 mb-4 rounded">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border p-2 rounded"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border p-2 rounded"
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white p-2 rounded disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Login'}
          </button>
        </form>
        <p className="text-center mt-4">
          Don't have an account? <Link href="/auth/register" className="text-blue-600">Register</Link>
        </p>
      </div>
    </div>
  )
}
```

**Step 6: Create register page**

Create `frontend/app/auth/register/page.tsx`:
```typescript
'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import Link from 'next/link'

export default function RegisterPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)
  const router = useRouter()

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError('')

    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/auth/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      })

      if (!res.ok) throw new Error('Registration failed')

      router.push('/auth/login')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-100">
      <div className="bg-white p-8 rounded shadow-lg w-96">
        <h2 className="text-2xl font-bold mb-6">Register</h2>
        {error && <div className="bg-red-100 text-red-700 p-3 mb-4 rounded">{error}</div>}
        <form onSubmit={handleSubmit} className="space-y-4">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full border p-2 rounded"
            required
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full border p-2 rounded"
            required
          />
          <button
            type="submit"
            disabled={loading}
            className="w-full bg-blue-600 text-white p-2 rounded disabled:opacity-50"
          >
            {loading ? 'Loading...' : 'Register'}
          </button>
        </form>
        <p className="text-center mt-4">
          Already have an account? <Link href="/auth/login" className="text-blue-600">Login</Link>
        </p>
      </div>
    </div>
  )
}
```

**Step 7: Setup Tailwind**

Create `frontend/tailwind.config.ts`:
```typescript
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
export default config
```

**Step 8: Test frontend**

```bash
npm run dev
# Expected: http://localhost:3000 running
```

**Step 9: Commit**

```bash
git add frontend/
git commit -m "feat: setup Next.js with authentication pages"
```

---

## REMAINING PHASES (Brief Descriptions)

### PHASE 9: Dashboard & Case Management
- Create `/dashboard` showing user's cases
- Create `/upload` page for new case uploads
- Create GET /cases endpoint

### PHASE 10: Q&A & Summarization UI
- Create `/cases/[id]` detail page
- Add Q&A form and answer display
- Add summary generation tab
- Create GET /cases/{id}/summarize endpoint

### PHASE 11: PDF Viewer Integration
- Integrate react-pdf library
- Add citation highlighting
- Add page navigation

### PHASE 12: Testing & Deployment
- Write unit tests (backend + frontend)
- Create E2E tests
- Setup CI/CD pipeline
- Docker images for deployment

---

## Summary

This plan breaks the entire legal RAG app into bite-sized, executable steps. Each task is 2-5 minutes of focused work.

**After PHASE 7 (completed):**
- ✅ Backend fully functional with RAG pipeline
- ✅ Authentication working
- ✅ Basic frontend auth pages
- ✅ Can upload cases and query them via API

**Continue with PHASE 8+ for:**
- Dashboard UI
- Case management
- PDF viewer
- Testing & deployment

---

**Ready to execute?**
