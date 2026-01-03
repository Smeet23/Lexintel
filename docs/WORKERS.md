# Celery Workers Documentation

## Overview

Celery workers handle asynchronous document processing tasks including text extraction, chunking, and embedding generation. The workers service is a standalone microservice that integrates with the backend via Redis and PostgreSQL.

## Architecture

### Worker Service Structure

```
apps/workers/
├── src/
│   ├── __main__.py              # Entry point - starts Celery worker
│   ├── config.py                # Configuration from environment
│   ├── celery_app.py            # Celery initialization and setup
│   ├── lib/
│   │   ├── redis.py             # Redis client factory
│   │   ├── progress.py          # ProgressPublisher for real-time updates
│   │   └── logging.py           # Structured JSON logging
│   ├── workers/
│   │   ├── __init__.py
│   │   ├── document_extraction.py    # Text extraction from files
│   │   └── embeddings.py            # OpenAI embedding generation
│   └── utils/
│       └── errors.py            # PermanentError, RetryableError
├── tests/
│   ├── unit/                    # Isolated unit tests
│   └── integration/             # Full workflow tests
├── requirements.txt
├── Dockerfile
└── pytest.ini
```

### Task Definition Pattern

All worker tasks follow a consistent pattern:

```python
# apps/workers/src/workers/document_extraction.py
from celery import Task
from shared.models import Document
from shared.database import async_session
from lib.progress import ProgressPublisher
from utils.errors import PermanentError, RetryableError

class WorkerTask(Task):
    """Base task class with error handling"""
    autoretry_for = (RetryableError,)
    retry_kwargs = {"max_retries": 3}

    def retry(self, args=None, kwargs=None, exc=None, **options):
        if isinstance(exc, RetryableError):
            countdown = 60 * (2 ** self.request.retries)
            return super().retry(*args, **options, countdown=countdown)
        return super().retry(*args, **options)

@celery_app.task(base=WorkerTask, bind=True)
async def extract_text_from_document(self, job_payload: dict):
    """Extract text from document and create chunks"""
    publisher = ProgressPublisher(redis_client)

    try:
        # Parse input
        doc_id = job_payload["document_id"]
        file_path = job_payload["file_path"]

        # Publish progress: starting
        await publisher.publish_progress(
            doc_id, 0, "starting", "Initializing text extraction..."
        )

        # Get document from DB
        async with async_session() as db:
            doc = await db.get(Document, doc_id)
            if not doc:
                raise PermanentError(f"Document {doc_id} not found")

        # Extract text
        text = extract_text_by_type(file_path)
        await publisher.publish_progress(doc_id, 30, "extracting", "Text extracted")

        # Create chunks
        chunks = create_chunks(text, size=4000, overlap=400)
        await publisher.publish_progress(doc_id, 60, "chunking", f"Created {len(chunks)} chunks")

        # Save chunks to DB
        async with async_session() as db:
            doc = await db.get(Document, doc_id)
            # ... save chunks logic
            await db.commit()

        await publisher.publish_progress(doc_id, 100, "completed", "Done!")

    except PermanentError:
        # Don't retry - fail immediately
        raise
    except RetryableError as e:
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))
```

## Error Handling

### Permanent Errors (Don't Retry)

These errors indicate issues that won't be fixed by retrying:

- File not found: Document already deleted
- Document not in database: Invalid input
- Invalid file format: Can't extract from this format
- Invalid configuration: Missing OpenAI key

```python
raise PermanentError("Document not found in database")
```

### Retryable Errors (Retry with Backoff)

These errors are transient and may succeed on retry:

- Database connection timeout
- Redis connection error
- OpenAI API rate limit (429)
- Network failures
- Temporary service unavailability

```python
raise RetryableError("Database connection timeout")
```

**Backoff Schedule**:
- Attempt 1: Immediate
- Attempt 2: 60 seconds
- Attempt 3: 120 seconds (2^1 * 60)
- Attempt 4: 240 seconds (2^2 * 60)
- Final attempt: 480 seconds (2^3 * 60)

## Progress Tracking

### Publishing Progress

Workers publish progress updates to Redis Pub/Sub in real-time:

```python
from lib.progress import ProgressPublisher

publisher = ProgressPublisher(redis_client)

await publisher.publish_progress(
    document_id="doc-123",
    progress=45,                    # 0-100
    step="extracting",              # Current step
    message="Processing page 5 of 10"  # Human-readable message
)
```

### Frontend Subscription

The backend exposes an SSE endpoint that clients subscribe to:

```javascript
// Frontend
const eventSource = new EventSource(`/api/documents/doc-123/progress`);
eventSource.onmessage = (event) => {
  const progress = JSON.parse(event.data);
  console.log(`Progress: ${progress.progress}% - ${progress.message}`);
};
```

### Progress Data Format

```json
{
  "document_id": "doc-123",
  "progress": 45,
  "step": "extracting",
  "message": "Processing page 5 of 10",
  "timestamp": "2026-01-03T10:30:45Z"
}
```

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/lex_intel

# Redis
REDIS_URL=redis://localhost:6379

# Celery
CELERY_BROKER_URL=${REDIS_URL}/0
CELERY_RESULT_BACKEND=${REDIS_URL}/1

# Worker Configuration
WORKER_PREFETCH_MULTIPLIER=1      # Process one task at a time
TASK_SOFT_TIME_LIMIT=1500         # 25 minutes soft limit
TASK_TIME_LIMIT=1800              # 30 minutes hard limit

# Optional: External Services
OPENAI_API_KEY=sk-...
```

### Celery Configuration

```python
# apps/workers/src/celery_app.py
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,            # Only ack after task completes
    worker_prefetch_multiplier=int(os.getenv("WORKER_PREFETCH_MULTIPLIER", 1)),
    worker_max_tasks_per_child=1000, # Restart periodically for memory cleanup
    task_time_limit=int(os.getenv("TASK_TIME_LIMIT", 1800)),
    task_soft_time_limit=int(os.getenv("TASK_SOFT_TIME_LIMIT", 1500)),
)
```

## Running Workers

### Local Development

```bash
# Install dependencies
cd apps/workers
pip install -r requirements.txt

# Start worker
python -m src
```

### Docker

```bash
# Run single worker
docker-compose up workers

# Scale to multiple workers
docker-compose up --scale workers=3
```

### Kubernetes

```bash
kubectl apply -f k8s/workers-deployment.yaml
kubectl scale deployment workers --replicas=3
```

## Monitoring

### Active Tasks

```bash
celery -A src.celery_app inspect active
```

Output:
```
[{
  'name': 'extract_text_from_document',
  'id': 'abc-123',
  'args': ["{'document_id': 'doc-123', ...}"],
  'worker': 'celery@worker1',
  'time_start': 1234567890,
}]
```

### Queue Depth

```bash
celery -A src.celery_app inspect reserved
```

Shows tasks waiting to be processed by workers.

### Worker Stats

```bash
celery -A src.celery_app inspect stats
```

Includes:
- Pool size
- Max concurrency
- Prefetch multiplier
- Total task count
- Uptime

### Logs

```bash
# Docker
docker logs -f lex-intel-workers

# Kubernetes
kubectl logs -f deployment/workers
```

Logs use structured JSON format for easy parsing:

```json
{
  "timestamp": "2026-01-03T10:30:45Z",
  "level": "INFO",
  "logger": "celery.worker",
  "message": "Started processing task",
  "task_id": "abc-123",
  "task_name": "extract_text_from_document"
}
```

## Testing

### Unit Tests

Test individual functions with mocks:

```bash
pytest apps/workers/tests/unit/ -v
```

Example:
```python
# tests/unit/test_extract.py
def test_extract_text_from_txt():
    text = extract_text("sample.txt")
    assert len(text) > 0
    assert "sample" in text.lower()
```

### Integration Tests

Test full workflows with real database:

```bash
pytest apps/workers/tests/integration/ -v
```

Example:
```python
# tests/integration/test_document_extraction.py
async def test_extract_creates_chunks():
    # Create test document
    doc = await create_test_document("sample.pdf")

    # Run extraction task
    result = await extract_text_from_document({
        "document_id": doc.id,
        "file_path": doc.file_path
    })

    # Verify chunks created
    chunks = await db.query(DocumentChunk).filter_by(document_id=doc.id)
    assert len(chunks) > 0
```

### Coverage

```bash
pytest apps/workers/tests/ --cov=src --cov-report=html
# Opens htmlcov/index.html with coverage report
```

## Troubleshooting

### Worker Not Picking Up Tasks

**Symptoms**: Tasks remain in queue, not processed

**Solutions**:
1. Check Redis connection: `redis-cli PING`
2. Verify CELERY_BROKER_URL: Should be Redis URL
3. Check worker logs: `docker logs lex-intel-workers`
4. Ensure tasks are registered: `celery -A src.celery_app inspect registered`

### Progress Not Updating

**Symptoms**: SSE endpoint opens but no updates received

**Solutions**:
1. Verify Redis Pub/Sub: `redis-cli SUBSCRIBE progress:*`
2. Check ProgressPublisher is being called in task
3. Verify REDIS_URL is same for backend and workers
4. Check frontend EventSource is listening on correct URL

### High Memory Usage

**Symptoms**: Worker process grows to 1GB+

**Solutions**:
1. Reduce WORKER_PREFETCH_MULTIPLIER to 1
2. Set worker_max_tasks_per_child=1000
3. Profile memory: `python -m memory_profiler worker_task.py`
4. Check for memory leaks in custom code

### Task Timeout

**Symptoms**: "Task timed out" errors

**Solutions**:
1. Increase TASK_TIME_LIMIT (default 1800 = 30 min)
2. Check if task is actually stuck or slow
3. Profile with timing: Add logging at each step
4. Consider breaking task into subtasks

---

**Last Updated**: January 3, 2026
**Current Version**: Phase 8 (Microservice Architecture)
