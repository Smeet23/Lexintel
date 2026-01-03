# LexIntel Architecture

## Overview

LexIntel is an AI-powered legal research platform built with a modern, scalable microservice architecture. The system employs a monorepo structure with separate backend and worker services, a shared database layer, and real-time progress tracking capabilities.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client (Browser)                         │
└────────┬────────────────────────────────────────────────────┘
         │
         │ HTTP / SSE
         ▼
┌────────────────────────────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                    │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Cases API    │ Documents API │ Progress SSE │ Search │   │
│  └─────────────────────────────────────────────────────┘   │
└────────┬────────────────────────────────────────────────────┘
         │
         │ Queue Job / Get Results
         ▼
┌────────────────────────────────────────────────────────────┐
│                      Redis                                  │
│     (Job Queue, Session Cache, Pub/Sub Progress)           │
└────────┬────────────────────────────────────────────────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌─────────┐  ┌──────────────────────────┐
│PostgreSQL  │  Celery Workers          │
│           │  (Text Extraction,        │
│ Models:   │   Embeddings,             │
│ - Cases   │   Chunking)               │
│ - Docs    └──────────────────────────┘
│ - Chunks        │
│ - Embeddings    │ Update / Publish
│                 │ Progress
└─────────────────┘
```

## Directory Structure

```
lex-intel/
├── apps/
│   ├── backend/
│   │   ├── app/
│   │   │   ├── main.py              # FastAPI application
│   │   │   ├── config.py            # Configuration
│   │   │   ├── database.py          # Re-exports from shared
│   │   │   ├── models.py            # Re-exports from shared
│   │   │   ├── schemas.py           # Pydantic validation schemas
│   │   │   ├── api/                 # API route handlers
│   │   │   │   ├── cases.py
│   │   │   │   ├── documents.py
│   │   │   │   └── progress.py
│   │   │   └── services/            # Business logic
│   │   ├── requirements.txt
│   │   ├── Dockerfile
│   │   └── tests/
│   │
│   └── workers/
│       ├── src/
│       │   ├── __main__.py          # Worker entry point
│       │   ├── config.py
│       │   ├── celery_app.py        # Celery configuration
│       │   ├── lib/
│       │   │   ├── redis.py
│       │   │   ├── progress.py
│       │   │   └── logging.py
│       │   └── workers/             # Task implementations
│       │       ├── document_extraction.py
│       │       └── embeddings.py
│       ├── requirements.txt
│       ├── Dockerfile
│       └── tests/
│
├── packages/
│   └── shared/
│       └── src/shared/
│           ├── models.py            # SQLAlchemy ORM models
│           ├── database.py          # AsyncSession, engine setup
│           ├── schemas.py           # Shared Pydantic schemas
│           ├── errors.py            # PermanentError, RetryableError
│           └── utils/
│
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── BACKEND.md                   # Backend API patterns
│   ├── WORKERS.md                   # Worker documentation
│   ├── DATABASE.md                  # Database schema
│   ├── SETUP.md                     # Development setup
│   └── MIGRATION_GUIDE.md           # Refactoring guide
│
├── docker-compose.yml
├── claude.md                        # Project overview
└── README.md                        # Public documentation
```

## Key Design Decisions

### 1. Monorepo with Apps and Packages

The project uses a monorepo structure with:
- **apps/** - Independent services (backend, workers)
- **packages/shared/** - Reusable code (models, database, schemas)

**Why**: Single source of truth for database models, easy dependency management, simpler deployment coordination.

### 2. Real-Time Progress with SSE + Redis Pub/Sub

Instead of polling (2000ms latency), workers publish progress to Redis Pub/Sub, and the backend streams updates via Server-Sent Events (25ms latency).

**Why**: 80x faster feedback, reduced server load, better UX for long-running tasks.

### 3. Pure Async/Await with SQLAlchemy Async

Both backend and workers use SQLAlchemy's async engine with `async_sessionmaker`.

**Why**: Non-blocking I/O, efficient connection pooling, compatible with Celery async tasks.

### 4. Shared Database Layer

`packages/shared` contains SQLAlchemy models and AsyncSession configuration used by both backend and workers.

**Why**: Consistency across services, no model duplication, single schema source of truth.

### 5. Graceful Shutdown Handlers

Signal handlers (SIGTERM, SIGINT) ensure:
- In-flight tasks complete before worker shutdown
- Database connections are properly closed
- Redis connections are cleaned up

**Why**: Data consistency, zero dropped tasks, clean logs.

## Communication Patterns

### Backend → Workers
1. Backend API creates task: `extract_text_from_document(job_payload)`
2. FastAPI queues to Redis: Job ID, document path, parameters
3. Celery dequeues from Redis and processes

### Workers → Backend (Progress)
1. Worker publishes: `redis.publish(f"progress:{doc_id}", json.dumps(progress_data))`
2. Backend SSE endpoint subscribes to channel
3. Browser EventSource receives streaming updates

### Workers → Database
1. Worker processes document
2. Updates database: `Document.extraction_status = "completed"`
3. Commits session: triggers webhook/event if configured

## Testing Strategy

| Level | Scope | Tools | Target |
|-------|-------|-------|--------|
| **Unit** | Isolated functions | pytest + mocks | 80% coverage |
| **Integration** | Full workflows | pytest + real DB | Document extraction pipeline |
| **System** | Docker Compose | docker-compose test | End-to-end scenarios |
| **E2E** | Browser → API → Worker → DB | Playwright + Python server | Critical paths |

**Coverage Target**: >85% across all services.

## Deployment Information

### Development
```bash
docker-compose up
# Brings up: PostgreSQL, Redis, Backend, Workers
# Backend: http://localhost:8000
```

### Production
- **Backend**: Kubernetes deployment, managed PostgreSQL, Redis cluster
- **Workers**: Horizontal scaling (CPU-based), managed Redis
- **Database**: Separate managed service (e.g., AWS RDS)
- **Monitoring**: Structured JSON logs, Celery metrics, APM integration

### Environment Variables
```bash
# Shared
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/lex_intel
REDIS_URL=redis://localhost:6379

# Backend
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000

# Workers
CELERY_BROKER_URL=${REDIS_URL}/0
CELERY_RESULT_BACKEND=${REDIS_URL}/1
WORKER_PREFETCH_MULTIPLIER=1
TASK_TIME_LIMIT=1800
```

## Key Technology Choices

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| API Framework | FastAPI | Modern async, auto-docs, validation |
| Async ORM | SQLAlchemy 2.0 | Pure async, pgvector support, type hints |
| Job Queue | Celery + Redis | Scalable, distributed, retry logic |
| Database | PostgreSQL | Relational + pgvector + full-text search |
| Cache | Redis | Fast, Pub/Sub, session management |
| Container | Docker | Consistent environments |
| Language | Python 3.10+ | Data science ecosystem, type hints |

---

**Last Updated**: January 3, 2026
**Current Phase**: Phase 8 (Documentation & Finalization)
