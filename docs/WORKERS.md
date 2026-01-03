# Celery Workers & Async Processing

> Asynchronous task queue for document processing

---

## 🎯 Overview

**Purpose**: Handle long-running tasks asynchronously
- Text extraction from documents
- Embeddings generation
- Document indexing
- Any CPU/I/O intensive work

**Tech Stack**:
- **Celery**: Distributed task queue
- **Redis**: Message broker & result backend
- **Worker**: Processes tasks in background

---

## 🏗️ Architecture

```
FastAPI Request
    ↓
Task queued to Redis
    ↓
Celery Worker picks up task
    ↓
Task execution (async, retryable)
    ↓
Result stored in Redis
    ↓
API can poll status or receive callback
```

---

## 📍 File Locations

```
backend/app/
├── celery_app.py          # Celery configuration & initialization
└── workers/
    ├── __init__.py        # Task exports
    └── tasks.py           # Task implementations
```

---

## ⚙️ Configuration (app/celery_app.py)

```python
from celery import Celery
from celery.signals import task_prerun, task_postrun
from app.config import settings

celery_app = Celery(
    "lex-intel",
    broker=settings.REDIS_URL,      # Where to queue tasks
    backend=settings.REDIS_URL,     # Where to store results
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,         # 30 minutes hard limit
    task_soft_time_limit=25 * 60,    # 25 minutes soft limit
    worker_prefetch_multiplier=1,    # Don't prefetch tasks
    worker_max_tasks_per_child=1000, # Restart worker periodically
)

# Task lifecycle hooks
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    logger.info(f"[celery] Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, state=None, **kwargs):
    logger.info(f"[celery] Completed task: {task.name} (ID: {task_id})")

# Auto-discover tasks
celery_app.autodiscover_tasks(["app.workers"])
```

---

## 🎯 Task Pattern

### Standard Task Template

```python
from celery import shared_task, Task
from app.database import async_session
from app.models import Document, ProcessingStatus
import logging

logger = logging.getLogger(__name__)

class CallbackTask(Task):
    """Task with success/failure callbacks"""
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"[workers] Task {task_id} succeeded: {retval}")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"[workers] Task {task_id} failed: {exc}")

@shared_task(base=CallbackTask, bind=True, max_retries=3)
def my_task(self, document_id: str):
    """
    Description of what this task does

    Args:
        document_id: UUID of document to process

    Returns:
        dict with status and result
    """
    try:
        logger.info(f"[task_prefix] Starting for document {document_id}")

        # Use async_session to access database
        async def run():
            async with async_session() as db:
                # Get document
                from sqlalchemy.future import select
                stmt = select(Document).where(Document.id == document_id)
                result = await db.execute(stmt)
                doc = result.scalar_one_or_none()

                if not doc:
                    raise ValueError(f"Document {document_id} not found")

                # Do work
                # ...

                # Update status
                doc.processing_status = ProcessingStatus.INDEXED
                await db.commit()

        # Run async code
        import asyncio
        asyncio.run(run())

        return {"status": "success", "document_id": document_id}

    except Exception as exc:
        logger.error(f"[task_prefix] Error: {exc}")
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=60)
```

---

## 📋 Current Tasks

### 1. extract_text_from_document
**Status**: TODO (Phase 3)

**Purpose**: Extract text from PDF/DOCX/TXT files

**Input**: `document_id`

**Output**:
```json
{
  "status": "success",
  "document_id": "uuid",
  "text_length": 5000,
  "chunks_created": 5
}
```

**Implementation**:
1. Get document from DB
2. Read file from disk
3. Extract text based on file type (PDF2, python-pptx, native read)
4. Split into chunks (4000 chars, 400 overlap)
5. Create DocumentChunk records
6. Update status to EXTRACTED
7. Queue `generate_embeddings` task

---

### 2. generate_embeddings
**Status**: TODO (Phase 4)

**Purpose**: Generate OpenAI embeddings for document chunks

**Input**: `document_id`

**Output**:
```json
{
  "status": "success",
  "document_id": "uuid",
  "embeddings_generated": 5
}
```

**Implementation**:
1. Get document chunks with null embeddings
2. Batch call OpenAI embedding API
3. Store embeddings in DocumentChunk.embedding (pgvector)
4. Update Document status to INDEXED
5. Log completion

---

### 3. process_document_pipeline
**Status**: ✅ Implemented (Phase 2)

**Purpose**: Orchestrate full document processing pipeline

**Input**: `document_id`

**Output**:
```json
{
  "status": "processing",
  "document_id": "uuid"
}
```

**Implementation**:
```python
@shared_task(base=CallbackTask, bind=True)
def process_document_pipeline(self, document_id: str):
    try:
        logger.info(f"[process_pipeline] Starting for {document_id}")

        # Queue text extraction first
        extract_text_from_document.delay(document_id)

        return {"status": "processing", "document_id": document_id}
    except Exception as exc:
        logger.error(f"[process_pipeline] Error: {exc}")
        raise
```

---

## 🔄 Task Chaining

### Sequential Execution
```python
# Task A → Task B → Task C
task_a.apply_async(
    args=(arg1,),
    link=task_b.s(arg2),  # Run after task_a
)
```

### Current Pipeline
```python
# In upload_document endpoint:
process_document_pipeline.delay(document_id)

# In process_document_pipeline task:
extract_text_from_document.delay(document_id)

# At end of extract_text_from_document:
generate_embeddings.delay(document_id)
```

---

## 📊 Monitoring Tasks

### Check Worker Status
```bash
# Connected workers
celery -A app.celery_app inspect active

# List registered tasks
celery -A app.celery_app inspect registered

# Check queue length
celery -A app.celery_app inspect reserved
```

### View Logs
```bash
# Docker
docker-compose logs -f celery-worker

# Local
celery -A app.celery_app worker -l info
```

### Task State Tracking
```python
from app.celery_app import celery_app

# Get task state
result = celery_app.AsyncResult(task_id)
print(result.state)      # PENDING, STARTED, SUCCESS, FAILURE, RETRY
print(result.result)     # Task output
```

---

## ⚙️ Configuration Options

### Retries
```python
@shared_task(max_retries=3)
def my_task(self, arg):
    try:
        # work
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)  # Retry after 60 seconds
```

### Timeouts
```python
# Hard limit (kills task)
task_time_limit = 30 * 60  # 30 minutes

# Soft limit (allows cleanup)
task_soft_time_limit = 25 * 60  # 25 minutes

# Handle soft limit
from celery.exceptions import SoftTimeLimitExceeded

@shared_task
def my_task():
    try:
        # work
    except SoftTimeLimitExceeded:
        # cleanup before hard kill
        logger.warning("Task timeout - cleaning up")
```

### Rate Limiting
```python
@shared_task(rate_limit='10/m')  # 10 tasks per minute
def rate_limited_task():
    pass
```

---

## 🐛 Debugging Tasks

### Test Task Locally
```python
# Don't use .delay() for testing, call directly
from app.workers.tasks import my_task

result = my_task(arg1, arg2)  # Synchronous execution
print(result)
```

### Enable Eager Mode (for testing)
```python
# In tests or development
celery_app.conf.task_always_eager = True  # Execute tasks synchronously
celery_app.conf.task_eager_propagates = True  # Propagate exceptions
```

### Task State Persistence
```python
from app.celery_app import celery_app
from celery.result import AsyncResult

task_id = "..."
result = AsyncResult(task_id, app=celery_app)

print(f"State: {result.state}")
print(f"Result: {result.result}")
print(f"Ready: {result.ready()}")
print(f"Successful: {result.successful()}")
print(f"Failed: {result.failed()}")
```

---

## 🚀 Adding New Tasks

### Step 1: Create Task Function
```python
# app/workers/tasks.py
@shared_task(base=CallbackTask, bind=True, max_retries=3)
def new_task(self, arg1: str, arg2: int):
    try:
        logger.info(f"[new_task] Processing {arg1}")
        # implementation
        return {"status": "success"}
    except Exception as exc:
        logger.error(f"[new_task] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)
```

### Step 2: Export Task
```python
# app/workers/__init__.py
from app.workers.tasks import new_task

__all__ = ["new_task"]
```

### Step 3: Queue Task from API
```python
# app/api/resource.py
from app.workers.tasks import new_task

# Queue task
new_task.delay(arg1, arg2)

# Or with result tracking
task = new_task.apply_async(args=(arg1, arg2), task_id=custom_id)
return {"task_id": task.id}
```

### Step 4: Monitor Task
```python
# app/api/resource.py
from celery.result import AsyncResult

@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    result = AsyncResult(task_id)
    return {
        "task_id": task_id,
        "state": result.state,
        "result": result.result if result.ready() else None,
    }
```

---

## ✅ Checklist for New Tasks

- [ ] Task function in `app/workers/tasks.py`
- [ ] Exported in `app/workers/__init__.py`
- [ ] Uses `@shared_task` decorator
- [ ] Has `CallbackTask` base class
- [ ] Has `max_retries=N` set
- [ ] Logging with `[prefix]` format
- [ ] Error handling with `.retry()`
- [ ] Uses `async_session()` for DB access
- [ ] Tested with synchronous call first
- [ ] Queued from API endpoint
- [ ] Status tracking available

---

## 📚 References

- **Celery Docs**: https://docs.celeryproject.org/
- **Redis**: https://redis.io/
- **Task Queue Patterns**: https://docs.celeryproject.org/en/stable/userguide/tasks.html
