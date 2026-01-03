# LexIntel - Claude Project Context

> AI-Powered Legal Research Platform with RAG Capabilities

**Status**: MVP Development (Phases 1-3 + Refactoring Complete, Phase 8 Documentation)
**Last Updated**: January 3, 2026

## 📚 Documentation Structure

This project uses modular documentation for different concerns:

### Core Documentation
- **[claude.md](./claude.md)** - This file. Project overview & navigation
- **[docs/BACKEND.md](./docs/BACKEND.md)** - Backend API architecture & patterns
- **[docs/WORKERS.md](./docs/WORKERS.md)** - Celery workers & async processing
- **[docs/DATABASE.md](./docs/DATABASE.md)** - Database models & schema
- **[docs/SETUP.md](./docs/SETUP.md)** - Development setup & deployment
- **[docs/IMPLEMENTATION_PLAN.md](./docs/IMPLEMENTATION_PLAN.md)** - Phase-by-phase breakdown

### Project Documentation
- **[README.md](./README.md)** - Public project overview
- **[docs/plans/](./docs/plans/)** - Implementation plans with detailed tasks

---

## 🎯 Quick Status

### ✅ Completed (Phase 1-3 + Refactoring)
- [x] Project structure & FastAPI setup
- [x] PostgreSQL models + SQLAlchemy ORM
- [x] Celery infrastructure + Redis
- [x] Docker Compose (PostgreSQL, Redis)
- [x] **[NEW]** Worker refactoring to microservice architecture
- [x] **[NEW]** Real-time progress tracking (SSE + Redis Pub/Sub)
- [x] **[NEW]** Pure async/await patterns throughout
- [x] **[NEW]** Graceful shutdown handlers
- [x] **[NEW]** Comprehensive error handling (PermanentError vs RetryableError)

### Architecture Changes
- Moved from `backend/` to `apps/backend/` + `apps/workers/` monorepo
- Created `packages/shared/` for database models and schemas
- Implemented real-time progress with Server-Sent Events
- Enhanced error handling with retry logic and classification
- Added structured JSON logging across services

### ⏳ TODO (Phase 4+)
- [ ] PDF extraction (PyPDF2)
- [ ] DOCX extraction (python-docx)
- [ ] Embeddings generation (OpenAI API)
- [ ] Search APIs (full-text + semantic)
- [ ] Chat/RAG APIs (streaming)

---

## 🏗️ Architecture at a Glance

```
FastAPI Backend (Port 8000)
├── Cases API
├── Documents API (upload)
├── Search API (TODO)
└── Chat/RAG API (TODO)
    ↓
PostgreSQL (pgvector, tsvector)
Redis (caching, job queue)
Azurite (local Azure Storage)
    ↓
Celery Workers
├── Text extraction
├── Embeddings generation
└── Document processing pipeline
```

---

## 📖 Documentation Guide

**Getting Started?** → See [docs/SETUP.md](./docs/SETUP.md)

**Building APIs?** → See [docs/BACKEND.md](./docs/BACKEND.md)

**Working with Workers?** → See [docs/WORKERS.md](./docs/WORKERS.md)

**Database Questions?** → See [docs/DATABASE.md](./docs/DATABASE.md)

**Implementation Details?** → See [docs/IMPLEMENTATION_PLAN.md](./docs/IMPLEMENTATION_PLAN.md)

---

## 🔑 Key Files

| File | Purpose |
|------|---------|
| `app/main.py` | FastAPI app initialization |
| `app/config.py` | Settings from environment |
| `app/database.py` | SQLAlchemy setup |
| `app/celery_app.py` | Celery configuration |
| `docker-compose.yml` | Service orchestration |
| `requirements.txt` | Python dependencies |

---

## 📚 Recent Documentation Updates

**Phase 8 Completion**: Comprehensive documentation added
- [docs/ARCHITECTURE.md](./docs/ARCHITECTURE.md) - System design and diagrams
- Updated [docs/WORKERS.md](./docs/WORKERS.md) - Microservice worker patterns
- [docs/MIGRATION_GUIDE.md](./docs/MIGRATION_GUIDE.md) - Refactoring guide for developers

## 🚀 Next Phase

**Phase 4: PDF and DOCX Text Extraction**
- PDF extraction using PyPDF2 or pdfplumber
- DOCX extraction using python-docx
- Unified extraction interface with fallback handling
- Extended chunk creation with metadata preservation

See [docs/IMPLEMENTATION_PLAN.md](./docs/IMPLEMENTATION_PLAN.md) for details.

---

**For Claude**: This modular structure keeps documentation organized and focused. Each file handles one concern for easier navigation and maintenance.
