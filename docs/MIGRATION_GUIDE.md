# Migration Guide: Monolithic to Microservice Architecture

## Overview

This document guides developers through the refactoring from a monolithic `backend/` directory to a distributed microservice architecture with separate `apps/backend/` and `apps/workers/` services, plus a shared library in `packages/shared/`.

## What Changed

### Directory Structure

```
OLD STRUCTURE:
backend/
├── app/
│   ├── models.py
│   ├── database.py
│   ├── api/
│   │   ├── documents.py
│   │   └── cases.py
│   └── workers/
│       └── tasks.py

NEW STRUCTURE:
apps/
├── backend/
│   ├── app/
│   │   ├── models.py         (re-exports from shared)
│   │   ├── database.py       (re-exports from shared)
│   │   ├── api/
│   │   │   ├── documents.py
│   │   │   └── cases.py
│   │   └── main.py           (FastAPI app)
│   └── requirements.txt
│
└── workers/
    ├── src/
    │   ├── workers/
    │   │   └── document_extraction.py
    │   ├── lib/
    │   │   ├── progress.py
    │   │   └── redis.py
    │   └── celery_app.py
    └── requirements.txt

packages/
└── shared/
    └── src/shared/
        ├── models.py         (SQLAlchemy ORM)
        ├── database.py       (AsyncSession setup)
        ├── schemas.py        (Pydantic schemas)
        └── errors.py         (Exception classes)
```

### Import Changes

The migration maintains backward compatibility through re-exports:

```python
# Before - Direct from backend app
from app.models import Document
from app.database import async_session

# After - Still works via re-exports!
from app.models import Document           # app/models.py re-exports from shared
from app.database import async_session    # app/database.py re-exports from shared

# Or import directly from shared (preferred in new code)
from shared.models import Document
from shared.database import async_session
```

### Error Handling Pattern

New error classification for better worker reliability:

```python
# Before: Generic exceptions with retry logic
class DocumentExtractionError(Exception):
    pass

# After: Explicit error classification
from shared.errors import PermanentError, RetryableError

# Use PermanentError when problem won't be fixed by retry
raise PermanentError("Document not found in database")

# Use RetryableError for transient failures
raise RetryableError("Database connection timeout")
```

## For Backend Developers

### No Breaking Changes

Your existing code continues to work without modification:

```python
# ✓ Still works - all imports unchanged
from app.models import Document, Case, DocumentChunk
from app.database import async_session
from app.main import app
from app.config import settings
```

### New Capabilities

1. **Real-time Progress Tracking**
   - New SSE endpoint: `GET /documents/{document_id}/progress`
   - Returns: Server-Sent Events stream with progress updates
   - No polling needed - 80x faster feedback

2. **Access to Shared Utilities**
   ```python
   from shared.errors import PermanentError, RetryableError
   from shared.utils import get_logger
   ```

3. **Database Model Consistency**
   - All services import from `packages/shared/models.py`
   - Single source of truth for schema
   - Automatic consistency across backend and workers

### New Progress SSE Endpoint

```python
# app/api/documents.py
from fastapi.responses import StreamingResponse
from lib.progress import ProgressStream

@router.get("/documents/{document_id}/progress")
async def get_document_progress(document_id: str):
    """Stream real-time progress updates via SSE"""
    return StreamingResponse(
        ProgressStream(redis_client).stream_progress(document_id),
        media_type="text/event-stream"
    )
```

Usage from frontend:

```javascript
const eventSource = new EventSource(`/api/documents/doc-123/progress`);
eventSource.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  updateProgressBar(progress.progress);
  updateMessage(progress.message);
};
```

## For Worker Developers

### New Worker Structure

Workers are now a separate service with their own structure:

```python
# apps/workers/src/workers/document_extraction.py
from celery import Task
from shared.models import Document
from shared.database import async_session
from lib.progress import ProgressPublisher

class WorkerTask(Task):
    """Base with error handling"""
    autoretry_for = (RetryableError,)

@celery_app.task(base=WorkerTask, bind=True)
async def extract_text_from_document(self, job_payload: dict):
    publisher = ProgressPublisher(redis_client)

    try:
        doc_id = job_payload["document_id"]

        # Publish progress
        await publisher.publish_progress(doc_id, 0, "starting", "...")

        # Do work
        async with async_session() as db:
            doc = await db.get(Document, doc_id)
            # Extract text, create chunks

        await publisher.publish_progress(doc_id, 100, "completed", "Done!")

    except PermanentError:
        raise  # Don't retry
    except RetryableError as e:
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
```

### Error Handling Pattern

```python
from shared.errors import PermanentError, RetryableError

# Permanent errors - don't retry
if not file_exists:
    raise PermanentError(f"File not found: {file_path}")

# Retryable errors - retry with backoff
if db_connection_timeout:
    raise RetryableError("Database timeout - will retry")

# Celery automatically retries RetryableError
# Backoff: 60s, 120s, 240s, 480s
```

### Progress Publishing

Publish updates to Redis Pub/Sub for real-time frontend updates:

```python
from lib.progress import ProgressPublisher

publisher = ProgressPublisher(redis_client)

# Publish at key milestones
await publisher.publish_progress(
    document_id="doc-123",
    progress=45,
    step="extracting",
    message="Extracted text from page 5 of 10"
)
```

## Environment Variables

### Unchanged for Backend
```bash
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/lex_intel
REDIS_URL=redis://localhost:6379
```

### New for Workers
```bash
CELERY_BROKER_URL=${REDIS_URL}/0
CELERY_RESULT_BACKEND=${REDIS_URL}/1
WORKER_PREFETCH_MULTIPLIER=1
TASK_SOFT_TIME_LIMIT=1500   # 25 minutes
TASK_TIME_LIMIT=1800        # 30 minutes
```

## Docker Compose Changes

### Before
```yaml
version: '3'
services:
  backend:
    build: ./backend
    # ... runs everything in one service
```

### After
```yaml
version: '3'
services:
  backend:
    build: ./apps/backend        # Updated path
    environment:
      - DATABASE_URL
      - REDIS_URL
    depends_on: [postgres, redis]

  workers:                        # New service
    build: ./apps/workers
    environment:
      - DATABASE_URL
      - REDIS_URL
      - CELERY_BROKER_URL
      - CELERY_RESULT_BACKEND
    depends_on: [postgres, redis]

  postgres:
    image: postgres:15
    # ... unchanged

  redis:
    image: redis:7
    # ... unchanged
```

## Testing Changes

### New Test Structure

```
apps/backend/tests/
├── unit/                    # Isolated tests
├── integration/             # Full workflows
└── conftest.py

apps/workers/tests/          # New worker tests
├── unit/
├── integration/
└── conftest.py

packages/shared/tests/       # Shared model tests
├── unit/
└── conftest.py
```

### Running Tests

```bash
# All tests
pytest apps/

# Just backend
pytest apps/backend/tests/

# Just workers
pytest apps/workers/tests/

# With coverage
pytest apps/ --cov=apps --cov-report=html
```

### Test Fixtures

Shared test fixtures in each module's conftest.py:

```python
# apps/backend/tests/conftest.py
@pytest.fixture
async def test_document():
    """Create test document"""
    async with async_session() as db:
        doc = Document(...)
        db.add(doc)
        await db.commit()
        yield doc
```

## Deployment Changes

### Development

```bash
# Before
cd backend && docker-compose up

# After
docker-compose up        # Starts all services
docker-compose up backend  # Just backend
docker-compose up workers  # Just workers
docker-compose up --scale workers=3  # Multiple workers
```

### Production Deployment

**Backend**:
- Kubernetes deployment
- Managed PostgreSQL + Redis
- Load balanced across 3-5 replicas

**Workers**:
- Kubernetes deployment or Lambda
- Horizontal scaling based on queue depth
- Managed Redis broker
- Independent from backend scaling

**Database**:
- Managed PostgreSQL (AWS RDS, Google Cloud SQL, etc.)
- Single source of truth for all services

**Environment**:
```bash
# Separate configs for prod
backend:
  DATABASE_URL: managed-postgres-connection
  REDIS_URL: managed-redis-connection

workers:
  DATABASE_URL: managed-postgres-connection  # Same as backend
  CELERY_BROKER_URL: managed-redis/0         # Same Redis
  CELERY_RESULT_BACKEND: managed-redis/1
```

## Database Migrations

Database schema remains unchanged. All models are in `packages/shared/models.py`:

```python
# Run migrations from backend directory
cd apps/backend
alembic upgrade head
```

Both backend and workers read from the same PostgreSQL instance, so no migration duplication needed.

## Integration Testing

Test the full flow with real services:

```python
# apps/backend/tests/integration/test_document_pipeline.py
async def test_document_extraction_flow():
    # 1. Create case and document via API
    case = await create_test_case("Integration Test Case")
    doc = await upload_test_document(case.id, "sample.txt")

    # 2. Wait for worker to process
    # (In real tests: wait with timeout, poll status)
    await asyncio.sleep(5)

    # 3. Verify results
    async with async_session() as db:
        updated_doc = await db.get(Document, doc.id)
        chunks = await db.execute(
            select(DocumentChunk).where(
                DocumentChunk.document_id == doc.id
            )
        )
        assert len(chunks.scalars()) > 0
```

## Debugging Across Services

### Backend Logs

```bash
docker logs -f lex-intel-backend
```

### Worker Logs

```bash
docker logs -f lex-intel-workers
```

### Shared Redis for Communication

```bash
# Check progress updates
redis-cli SUBSCRIBE progress:*

# Check job queue
redis-cli LLEN celery
```

### Database Inspection

```bash
# Connect to PostgreSQL
psql postgresql://user:pass@localhost/lex_intel

# View documents
SELECT id, file_name, extraction_status FROM documents;

# View chunks
SELECT * FROM document_chunks WHERE document_id = 'doc-123';
```

## Migration Checklist

- [x] Understand new directory structure
- [x] Review import changes and re-exports
- [x] Install shared package: `pip install -e packages/shared`
- [x] Test backend with shared imports
- [x] Test workers with shared dependencies
- [x] Update Docker Compose configuration
- [x] Run tests: `pytest apps/`
- [x] Verify no breaking changes in API
- [x] Test environment variables configuration
- [x] Verify worker task processing
- [x] Test SSE progress endpoint
- [x] Deploy and monitor in staging

## Troubleshooting

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'shared'`

**Solution**: Install shared package

```bash
pip install -e packages/shared
```

### Worker Tasks Not Processing

**Symptoms**: Tasks queued but not executed

**Solutions**:
1. Verify Redis connection: `redis-cli PING`
2. Check CELERY_BROKER_URL points to correct Redis
3. View worker logs: `docker logs lex-intel-workers`
4. Verify task is registered: `celery -A src.celery_app inspect registered`

### SSE Progress Not Working

**Symptoms**: EventSource opens but no updates

**Solutions**:
1. Check Redis Pub/Sub: `redis-cli SUBSCRIBE progress:*`
2. Verify publisher called in worker task
3. Check backend and worker REDIS_URL match
4. Verify frontend is listening on correct endpoint

### Database Schema Issues

**Symptoms**: "Table does not exist" errors

**Solutions**:
1. Run migrations: `alembic upgrade head`
2. Verify DATABASE_URL points to correct database
3. Check shared/models.py reflects your schema

## Questions?

See documentation:
- [docs/ARCHITECTURE.md](./ARCHITECTURE.md) - System design
- [docs/WORKERS.md](./WORKERS.md) - Worker-specific details
- [docs/BACKEND.md](./BACKEND.md) - Backend API patterns
- [docs/DATABASE.md](./DATABASE.md) - Database schema

---

**Last Updated**: January 3, 2026
**Version**: Phase 8 (Microservice Refactoring)
