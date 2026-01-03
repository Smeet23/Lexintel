# LexIntel - AI-Powered Legal Research Platform

> Intelligent document management and RAG-powered case research platform for law firms

![Status](https://img.shields.io/badge/status-MVP-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/fastapi-0.109.0-green)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Overview

**LexIntel** is an AI-powered legal research platform designed for law firms to efficiently manage cases, upload documents, and leverage Retrieval-Augmented Generation (RAG) to conduct intelligent case research. The platform combines full-text search, semantic search using vector embeddings, and streaming AI-powered chat to help lawyers make better decisions faster.

### Key Capabilities
- 📄 **Document Management**: Upload and organize case documents (PDFs, Word, TXT)
- 🔍 **Dual Search**: Full-text search + semantic search with vector embeddings
- 🤖 **RAG Chat**: AI-powered streaming chat with document context awareness
- ⚡ **Async Processing**: Background document processing with Celery workers
- 🏢 **Multi-Case Management**: Organize documents by cases with custom tagging
- 📊 **Local Development**: Complete Docker Compose setup with PostgreSQL, Redis, and Azurite

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│        FastAPI Backend (Port 8000)      │
│  - Cases CRUD                           │
│  - Document Upload & Management         │
│  - Search APIs (full-text + semantic)   │
│  - Chat/RAG APIs (TODO)                 │
└─────────────────┬───────────────────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
┌─────▼──┐  ┌────▼──┐  ┌────▼─────┐
│PostgreSQL│  │ Redis │  │ Azurite  │
│ +pgvector│  │(Queue)│  │(Storage) │
└────────┘  └───────┘  └──────────┘

┌─────────────────────────────┐
│  Celery Workers (Background)│
│  - Text Extraction          │
│  - Embedding Generation     │
│  - Document Processing      │
└─────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** (recommended for local development)
- **Node.js 20+** (for OpenAI API key generation)
- **Python 3.11+** (if running without Docker)
- **OpenAI API Key** (get from https://platform.openai.com/account/api-keys)

### Installation

#### 1. Clone Repository
```bash
git clone git@github.com-personalwork:Smeet23/Lexintel.git
cd Lexintel
```

#### 2. Set Up Environment Variables
```bash
cp backend/.env.example backend/.env

# Edit backend/.env and add your OpenAI API key
OPENAI_API_KEY=sk-your-key-here
```

#### 3. Start Services with Docker Compose
```bash
docker-compose up -d
```

This starts:
- **PostgreSQL** on port 5432
- **Redis** on port 6379
- **Azurite** (Azure Storage emulator) on ports 10000-10002
- **FastAPI Backend** on port 8000
- **Celery Worker** for async tasks

#### 4. Verify Services
```bash
# Check all containers are running
docker-compose ps

# Check API is responding
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

---

## 📚 API Documentation

### Base URL
```
http://localhost:8000
```

### Auto-Generated Docs
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Core Endpoints

#### Cases API

**Create Case**
```bash
curl -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Smith v. Jones",
    "case_number": "2024-001",
    "practice_area": "contracts",
    "status": "active",
    "description": "Contract dispute case"
  }'
```

**List Cases**
```bash
curl http://localhost:8000/cases?skip=0&limit=50
```

**Get Case Details**
```bash
curl http://localhost:8000/cases/{case_id}
```

**Update Case**
```bash
curl -X PATCH http://localhost:8000/cases/{case_id} \
  -H "Content-Type: application/json" \
  -d '{"status": "closed"}'
```

**Delete Case**
```bash
curl -X DELETE http://localhost:8000/cases/{case_id}
```

#### Documents API

**Upload Document**
```bash
curl -X POST "http://localhost:8000/documents/upload?case_id={case_id}" \
  -F "file=@/path/to/document.pdf"
```

**Get Document Details**
```bash
curl http://localhost:8000/documents/{document_id}
```

**Delete Document**
```bash
curl -X DELETE http://localhost:8000/documents/{document_id}
```

---

## 🗂️ Project Structure

```
lex-intel/
├── backend/                          # FastAPI application
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI app entry point
│   │   ├── config.py                 # Configuration & settings
│   │   ├── database.py               # Database connection setup
│   │   ├── celery_app.py             # Celery configuration
│   │   │
│   │   ├── models/                   # SQLAlchemy ORM models
│   │   │   ├── __init__.py
│   │   │   ├── base.py               # Base model & mixins
│   │   │   ├── case.py               # Case model
│   │   │   └── document.py           # Document & Chat models
│   │   │
│   │   ├── schemas/                  # Pydantic validation schemas
│   │   │   ├── __init__.py
│   │   │   ├── case.py
│   │   │   ├── document.py
│   │   │   └── chat.py
│   │   │
│   │   ├── api/                      # API routers
│   │   │   ├── __init__.py
│   │   │   ├── cases.py              # Cases endpoints
│   │   │   ├── documents.py          # Documents endpoints
│   │   │   ├── search.py             # Search endpoints (TODO)
│   │   │   └── chat.py               # Chat/RAG endpoints (TODO)
│   │   │
│   │   ├── services/                 # Business logic services
│   │   │   ├── __init__.py
│   │   │   ├── storage.py            # File storage service
│   │   │   ├── search.py             # Search service (TODO)
│   │   │   └── embeddings.py         # Embeddings service (TODO)
│   │   │
│   │   └── workers/                  # Celery async tasks
│   │       ├── __init__.py
│   │       └── tasks.py              # Document processing tasks
│   │
│   ├── tests/                        # Test suite
│   │   ├── unit/
│   │   └── integration/
│   │
│   ├── uploads/                      # Document storage (local)
│   │
│   ├── requirements.txt               # Python dependencies
│   ├── pyproject.toml                # Project configuration
│   ├── Dockerfile                    # Docker image for backend
│   └── .env.example                  # Environment variables template
│
├── docker-compose.yml                # Docker Compose orchestration
├── .gitignore
├── README.md                         # This file
└── docs/
    ├── plans/                        # Implementation plans
    └── architecture/                 # Architecture diagrams
```

---

## 🛠️ Development Workflow

### Running Locally (with Docker)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f backend

# Stop services
docker-compose down

# Rebuild after dependency changes
docker-compose up --build
```

### Running Without Docker

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# Create PostgreSQL database
createdb -U postgres lex_intel_dev

# Run migrations
# (Currently handled by SQLAlchemy init_db)

# Start FastAPI backend
cd backend
uvicorn app.main:app --reload --port 8000

# In another terminal, start Celery worker
celery -A app.celery_app worker -l info
```

### Code Quality

```bash
# Format code with Black
black backend/

# Check types with mypy
mypy backend/

# Lint with flake8
flake8 backend/

# Run tests
pytest backend/tests/
```

---

## 📊 Database Schema

### Core Models

#### Cases
- `id`: Unique identifier
- `name`: Case name (e.g., "Smith v. Jones")
- `case_number`: Case number
- `practice_area`: Legal practice area
- `status`: active/closed/archived
- `description`: Case description
- `created_at`, `updated_at`: Timestamps

#### Documents
- `id`: Unique identifier
- `case_id`: Foreign key to Case
- `title`: Document title
- `filename`: Original filename
- `type`: brief/complaint/discovery/statute/transcript/contract
- `extracted_text`: Full text (after processing)
- `processing_status`: pending/extracted/indexed/failed
- `file_path`: Local storage path
- `created_at`, `updated_at`: Timestamps

#### DocumentChunks
- `id`: Unique identifier
- `document_id`: Foreign key to Document
- `chunk_text`: Text content (4000 char chunks with overlap)
- `chunk_index`: Chunk sequence number
- `embedding`: pgvector embedding (1536 dimensions)
- `search_vector`: PostgreSQL tsvector for full-text search

#### ChatConversations & ChatMessages
- Stores conversation history per case
- Tracks token usage
- Links to source documents

---

## 🔄 Document Processing Pipeline

```
1. Document Upload
   ↓
2. Store in local filesystem
   ↓
3. Queue async processing task
   ↓
4. Extract Text (Celery worker)
   ├─ PDF → pdf-parse
   ├─ DOCX → python-pptx
   └─ TXT → direct read
   ↓
5. Split into Chunks (4000 chars, 400 char overlap)
   ↓
6. Generate Embeddings (OpenAI API)
   ↓
7. Store in PostgreSQL with pgvector
   ↓
8. Document ready for search & chat
```

---

## 🔍 Search Capabilities

### Full-Text Search
- PostgreSQL `tsvector` with `pg_trgm` trigram matching
- Fast keyword search across documents
- Phrase matching support
- Fuzzy matching for typos

### Semantic Search
- OpenAI embeddings (text-embedding-3-small)
- pgvector cosine similarity matching
- Find conceptually similar cases
- Handles synonyms and semantic variations

### Combined Search
- Ranks full-text + semantic results
- Returns hybrid results with confidence scores

---

## 🤖 Async Processing with Celery

### Tasks
- `extract_text_from_document`: Parse PDF/DOCX/TXT files
- `generate_embeddings`: Create OpenAI embeddings
- `process_document_pipeline`: Orchestrate end-to-end pipeline

### Monitoring
```bash
# Check Celery worker logs
docker-compose logs celery-worker

# List active tasks
celery -A app.celery_app inspect active
```

---

## 🔐 Environment Variables

Required in `backend/.env`:

```bash
# Database
DATABASE_URL=postgresql://lex_user:lex_password@postgres:5432/lex_intel_dev

# Redis
REDIS_URL=redis://redis:6379

# OpenAI (REQUIRED - get from https://platform.openai.com)
OPENAI_API_KEY=sk-your-key-here

# Azurite (Local Azure Storage - pre-configured for Docker)
AZURE_STORAGE_CONNECTION_STRING=...

# App Settings
DEBUG=True
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8000

# Upload Settings
UPLOAD_DIR=/app/uploads
MAX_UPLOAD_SIZE=104857600  # 100MB
ALLOWED_EXTENSIONS=.pdf,.docx,.txt

# Processing
CHUNK_SIZE=4000
CHUNK_OVERLAP=400
```

---

## 📈 Performance Considerations

### Database
- **Connection Pooling**: Configured via SQLAlchemy
- **Indexes**: On frequently queried fields (case_id, processing_status)
- **Partitioning**: Consider for very large document_chunks table

### Search
- **pgvector**: IVFFlat indexes on embedding column (coming soon)
- **Full-text**: GIN indexes on tsvector column (coming soon)
- **Query Optimization**: Limit chunk retrieval to top-K similar items

### Async Processing
- **Worker Concurrency**: Configured for 4 workers
- **Retry Logic**: Up to 3 retries with exponential backoff
- **Task Timeouts**: 30 min soft, 25 min hard limits

---

## 🗓️ Roadmap

### Current Status: MVP
- ✅ Cases CRUD
- ✅ Document upload & storage
- ✅ Pydantic validation
- ✅ Docker Compose setup
- ⏳ Text extraction workers
- ⏳ Embedding generation
- ⏳ Search APIs
- ⏳ Chat/RAG APIs

### Coming Soon (Phase 2)
- Authentication with Auth0
- Full-text search API
- Semantic search API
- Document tagging & filtering

### Future (Phase 3+)
- Streaming chat/RAG
- Citation extraction & precedent linking
- Brief/memo generation
- Advanced analytics
- Frontend UI (React)

---

## 🐛 Troubleshooting

### Docker Issues

**Port already in use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in docker-compose.yml
```

**Database connection errors**
```bash
# Check PostgreSQL is running
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

### API Issues

**500 errors**
```bash
# Check backend logs
docker-compose logs backend

# Check environment variables
docker-compose exec backend env | grep DATABASE
```

**Files not uploading**
```bash
# Check directory permissions
docker-compose exec backend ls -la /app/uploads

# Check max file size (default 100MB)
# Edit backend/.env MAX_UPLOAD_SIZE
```

---

## 📝 Logging

Logs are configured to show:
- `[backend]` - Main API logs
- `[celery]` - Celery worker logs
- `[extract_text]` - Text extraction task logs
- `[storage]` - File storage operations

Example log format:
```
[backend] LexIntel backend starting...
[celery] Starting task: extract_text_from_document (ID: abc123)
[storage] Saved file: /app/uploads/doc-id/document.pdf
```

---

## 🧪 Testing

```bash
# Run all tests
pytest backend/tests/

# Run specific test
pytest backend/tests/unit/test_cases.py::test_create_case

# Run with coverage
pytest backend/tests/ --cov=app --cov-report=html
```

---

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make changes and test: `pytest backend/tests/`
3. Commit with clear messages: `git commit -m "feat: add feature"`
4. Push to branch: `git push origin feature/your-feature`
5. Create Pull Request on GitHub

### Code Style
- Python: Follow PEP 8 (enforced by Black)
- Type hints: Always use type annotations
- Docstrings: Use Google-style docstrings
- Tests: Write tests for new features

---

## 📄 License

MIT License - See LICENSE file for details

---

## 📧 Contact & Support

For questions or issues:
- 📧 Email: smeetagrawal23@gmail.com
- 🐙 GitHub: https://github.com/Smeet23/Lexintel
- 📋 Issues: Create an issue on GitHub

---

## 🙏 Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [SQLAlchemy](https://www.sqlalchemy.org/) - SQL toolkit and ORM
- [Celery](https://docs.celeryproject.org/) - Distributed task queue
- [pgvector](https://github.com/pgvector/pgvector) - Vector search in PostgreSQL
- [OpenAI](https://openai.com/) - LLM & embedding models
- [Docker](https://www.docker.com/) - Containerization

---

## 📊 Stats

- **Backend**: ~1,000 lines of Python code
- **Models**: 8 core SQLAlchemy models
- **APIs**: 7 REST endpoints (cases + documents)
- **Services**: 2 services (storage, embeddings)
- **Workers**: 3 Celery tasks
- **Tests**: Unit + Integration tests (coming)

---

**Version**: 0.1.0 (MVP)
**Last Updated**: January 3, 2026
**Status**: 🚀 In Development
