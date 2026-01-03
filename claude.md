# LexIntel - Claude Project Context

> AI-Powered Legal Research Platform with RAG Capabilities

**Status**: MVP Development (Phases 1-3 Complete)
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

### ✅ Completed (Phase 1-3)
- [x] Project structure & FastAPI setup
- [x] PostgreSQL models + SQLAlchemy ORM
- [x] Celery infrastructure + Redis
- [x] Docker Compose (PostgreSQL, Redis, Azurite)
- [x] Pydantic schemas for validation
- [x] File storage service (local filesystem)
- [x] Cases CRUD API
- [x] Documents upload/management API
- [x] Text extraction workers (TXT files)

### ⏳ TODO (Phase 3+ Enhancement & Phase 4-6)
- [ ] PDF extraction (PyPDF2)
- [ ] DOCX extraction (python-pptx)
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

## 🚀 Next Phase

**Phase 3: Text Extraction Workers**
- Extract text from PDF/DOCX/TXT files
- Create document chunks (4000 chars, 400 overlap)
- Update document processing status

See [docs/IMPLEMENTATION_PLAN.md](./docs/IMPLEMENTATION_PLAN.md) for details.

---

**For Claude**: This modular structure keeps documentation organized and focused. Each file handles one concern for easier navigation and maintenance.
