# SSE Progress Tracking: Quick Implementation Guide

**Get real-time document progress in 4 hours**

---

## Overview

This guide implements **Redis Pub/Sub + Server-Sent Events (SSE)** for tracking document processing progress. Users will see:

- Real-time percentage completion
- Current processing step (extraction, chunking, embedding)
- Human-readable status messages
- Error notifications

---

## What You're Building

```
User uploads document
  ↓
Backend queues extraction task
  ↓
Worker publishes progress to Redis
  └─→ "document:123:progress" channel
      {"progress": 25, "step": "extraction", "message": "..."}
      ↓
      SSE endpoint streams to client
      ↓
      Browser UI updates in real-time
```

---

## Step 1: Update Dependencies (0 minutes)

**No new dependencies needed!** Redis is already required for Celery.

Check `backend/requirements.txt` - you already have:
- `redis==5.0.1` ✅
- `fastapi==0.109.0` ✅
- `celery==5.3.4` ✅

---

## Step 2: Create Progress API Endpoint (30 minutes)

**Create file:** `backend/app/api/progress.py`

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
import logging
from redis import Redis
from app.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/progress", tags=["progress"])

# Redis client for Pub/Sub
redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

@router.get("/documents/{document_id}/stream")
async def stream_document_progress(document_id: str):
    """
    Server-Sent Events endpoint for document progress.

    Example usage:
    ```javascript
    const es = new EventSource('/progress/documents/abc-123/stream');
    es.onmessage = (e) => {
        const progress = JSON.parse(e.data);
        console.log(`${progress.progress}% - ${progress.message}`);
    };
    ```
    """

    def event_generator():
        pubsub = redis_client.pubsub()
        channel = f"document:{document_id}:progress"

        logger.info(f"[progress] SSE client connected for {document_id}")

        try:
            pubsub.subscribe(channel)

            # Send connection confirmation
            yield f"data: {json.dumps({'status': 'connected'})}\n\n"

            # Stream progress from worker
            for message in pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    yield f"data: {data}\n\n"

                    # Stop when complete
                    try:
                        parsed = json.loads(data)
                        if parsed.get("progress") == 100:
                            break
                    except json.JSONDecodeError:
                        pass

        except GeneratorExit:
            logger.info(f"[progress] Client disconnected for {document_id}")
        finally:
            pubsub.close()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

@router.get("/documents/{document_id}/status")
async def get_document_status(document_id: str, db=None):
    """
    Fallback endpoint: get current status from database.
    Used if SSE connection drops (mobile network switch).
    """
    from app.database import async_session
    from sqlalchemy.future import select
    from app.models import Document
    from fastapi import HTTPException

    async with async_session() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return {
            "document_id": document_id,
            "status": document.processing_status.value if document.processing_status else "pending",
            "error_message": document.error_message,
        }
```

---

## Step 3: Update Workers with Progress Publishing (30 minutes)

**File:** `backend/app/workers/tasks.py`

Add this helper function at the top:

```python
import json
from datetime import datetime
from redis import Redis
from app.config import settings

redis_client = Redis.from_url(settings.REDIS_URL, decode_responses=True)

def publish_progress(document_id: str, step: str, progress: int, message: str):
    """
    Publish progress update to Redis.

    Args:
        document_id: Document being processed
        step: Current step (e.g., "extraction", "chunking", "embedding")
        progress: Progress 0-100
        message: Human-readable message
    """
    data = {
        "document_id": document_id,
        "status": "processing",
        "step": step,
        "progress": progress,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }
    redis_client.publish(f"document:{document_id}:progress", json.dumps(data))
    logger.info(f"[progress] {document_id}: {step} {progress}% - {message}")
```

Then update `extract_text_from_document` function:

```python
@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    """Extract text from document with progress updates"""
    try:
        logger.info(f"[extract_text] Starting for {document_id}")
        publish_progress(document_id, "extraction", 5, "Starting text extraction...")

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            _extract_and_chunk_document(document_id)
        )

        logger.info(f"[extract_text] Completed for {document_id}")
        return result
    except Exception as exc:
        logger.error(f"[extract_text] Error: {exc}")
        publish_progress(document_id, "extraction", 0, f"Error: {str(exc)}")
        raise self.retry(exc=exc, countdown=60)
```

Update `_extract_and_chunk_document` to add progress calls:

```python
async def _extract_and_chunk_document(document_id: str) -> dict:
    """Extract text and chunks with progress updates"""
    from sqlalchemy import select

    async with async_session() as session:
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        document = result.scalars().first()

        if not document:
            raise ValueError(f"Document not found")

        try:
            # Extraction
            logger.info(f"[extract_text] Extracting {document.filename}")
            publish_progress(document_id, "extraction", 10,
                f"Extracting text from {document.filename}...")

            text = await extract_file(document.file_path)

            # Chunking
            publish_progress(document_id, "chunking", 40, "Creating text chunks...")
            logger.info(f"[extract_text] Creating chunks")

            chunk_ids = await create_text_chunks(
                document_id=document_id,
                text=text,
                session=session,
            )

            # Finalizing
            publish_progress(document_id, "finalizing", 70,
                f"Finalizing {len(chunk_ids)} chunks...")

            document.extracted_text = text
            document.processing_status = ProcessingStatus.EXTRACTED
            await session.commit()

            # Queue next task
            publish_progress(document_id, "queuing", 90,
                "Queuing embeddings task...")

            from app.workers.tasks import generate_embeddings
            generate_embeddings.delay(document_id)

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": len(chunk_ids),
                "text_length": len(text),
            }

        except Exception as e:
            logger.error(f"[extract_text] Error: {e}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
            raise
```

---

## Step 4: Register Progress Router (15 minutes)

**File:** `backend/app/main.py`

Add imports:
```python
from app.api import progress
```

Register router:
```python
# Include routers
app.include_router(cases.router)
app.include_router(documents.router)
app.include_router(progress.router)  # ← Add this line
```

---

## Step 5: Frontend Hook (45 minutes)

**Create file:** `frontend/hooks/useDocumentProgress.ts` (or `.js`)

```typescript
import { useState, useEffect } from 'react';

interface DocumentProgress {
  document_id: string;
  status: 'connected' | 'processing' | 'completed' | 'failed';
  step: string;
  progress: number;
  message: string;
  timestamp?: string;
  error_message?: string;
}

export function useDocumentProgress(documentId: string | null) {
  const [progress, setProgress] = useState<DocumentProgress>({
    document_id: documentId || '',
    status: 'connected',
    step: 'waiting',
    progress: 0,
    message: 'Waiting to start...',
  });

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!documentId) return;

    // Create SSE connection
    const eventSource = new EventSource(
      `/progress/documents/${documentId}/stream`
    );

    // Listen for progress updates
    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);

        // Skip connection confirmation
        if (data.status === 'connected') {
          return;
        }

        setProgress(data);
        setError(null);

        // Close when done
        if (data.progress === 100 || data.status === 'completed') {
          eventSource.close();
        }
      } catch (e) {
        console.error('Failed to parse progress:', e);
      }
    };

    // Handle connection errors
    eventSource.onerror = (error) => {
      console.error('Progress stream error:', error);
      setError('Connection lost - attempting to reconnect...');
      eventSource.close();

      // Fallback to polling
      startPolling(documentId);
    };

    return () => {
      eventSource.close();
    };
  }, [documentId]);

  function startPolling(docId: string) {
    const pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/progress/documents/${docId}/status`);
        if (!res.ok) return;

        const data = await res.json();
        setProgress((prev) => ({
          ...prev,
          status: data.status === 'pending' ? 'processing' : data.status,
          error_message: data.error_message,
        }));

        // Stop polling when done
        if (data.status !== 'pending') {
          clearInterval(pollInterval);
        }
      } catch (e) {
        console.error('Polling error:', e);
      }
    }, 3000); // Poll every 3 seconds
  }

  return { progress, error };
}
```

---

## Step 6: Use in Component (30 minutes)

**Example React component:**

```typescript
import { useState } from 'react';
import { useDocumentProgress } from '../hooks/useDocumentProgress';

interface UploadResult {
  id: string;
  filename: string;
}

export function DocumentUploader() {
  const [documentId, setDocumentId] = useState<string | null>(null);
  const { progress, error } = useDocumentProgress(documentId);

  async function handleFileSelect(file: File) {
    // Upload document
    const formData = new FormData();
    formData.append('file', file);
    formData.append('case_id', 'case-123'); // Get from context

    try {
      const res = await fetch('/documents/upload', {
        method: 'POST',
        body: formData,
      });

      const doc: UploadResult = await res.json();
      setDocumentId(doc.id); // Starts progress tracking
    } catch (e) {
      console.error('Upload failed:', e);
    }
  }

  return (
    <div className="upload-container">
      <input
        type="file"
        onChange={(e) => {
          if (e.target.files) {
            handleFileSelect(e.target.files[0]);
          }
        }}
        disabled={!!documentId}
      />

      {documentId && (
        <ProgressTracker progress={progress} error={error} />
      )}
    </div>
  );
}

function ProgressTracker({ progress, error }) {
  return (
    <div className="progress-tracker">
      <div className="progress-header">
        <span className="step">{progress.step}</span>
        <span className="percentage">{progress.progress}%</span>
      </div>

      <progress value={progress.progress} max="100" />

      <div className="progress-message">
        {progress.message}
      </div>

      {error && (
        <div className="error-message">
          ⚠️ {error}
        </div>
      )}

      {progress.progress === 100 && (
        <div className="success-message">
          ✓ Processing complete!
        </div>
      )}
    </div>
  );
}
```

---

## Step 7: Add Styles (15 minutes)

**CSS:**

```css
.progress-tracker {
  padding: 20px;
  border: 1px solid #e0e0e0;
  border-radius: 8px;
  background: #f9f9f9;
}

.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 12px;
  font-size: 14px;
  font-weight: 500;
}

progress {
  width: 100%;
  height: 8px;
  border-radius: 4px;
  margin: 12px 0;
}

.progress-message {
  margin: 12px 0;
  font-size: 13px;
  color: #666;
  min-height: 20px;
}

.error-message {
  margin-top: 12px;
  padding: 8px;
  background: #ffebee;
  color: #c62828;
  border-radius: 4px;
  font-size: 13px;
}

.success-message {
  margin-top: 12px;
  padding: 8px;
  background: #e8f5e9;
  color: #2e7d32;
  border-radius: 4px;
  font-size: 13px;
}
```

---

## Step 8: Testing (30 minutes)

### Test SSE Endpoint

```bash
# Terminal 1: Start backend
docker-compose up backend

# Terminal 2: Connect to SSE
curl -N http://localhost:8000/progress/documents/test-doc/stream

# Terminal 3: Publish progress (in Python shell)
import redis
r = redis.from_url('redis://localhost:6379', decode_responses=True)
import json
from datetime import datetime

# Publish progress
for progress in [10, 25, 50, 75, 100]:
    r.publish('document:test-doc:progress', json.dumps({
        'document_id': 'test-doc',
        'status': 'processing',
        'step': 'test',
        'progress': progress,
        'message': f'Test {progress}%',
        'timestamp': datetime.utcnow().isoformat(),
    }))
    time.sleep(1)
```

### Test Upload Flow

```bash
# Upload a document
curl -X POST \
  -F "file=@test.pdf" \
  -F "case_id=case-123" \
  http://localhost:8000/documents/upload

# Watch progress in browser
# Open: http://localhost:3000/cases/case-123/documents
# You should see progress bar update in real-time
```

### Browser Console Test

```javascript
// Open browser console
// Run in your application's context

const es = new EventSource('/progress/documents/test-doc/stream');

es.onmessage = (e) => {
  const progress = JSON.parse(e.data);
  console.log(`Progress: ${progress.progress}%`);
};

es.onerror = (error) => {
  console.error('Connection error:', error);
};
```

---

## Step 9: Error Handling (20 minutes)

Update `publish_progress` to handle errors:

```python
def publish_progress(document_id: str, step: str, progress: int, message: str,
                    status: str = "processing", error: str = None):
    """Publish progress, handling any Redis issues gracefully"""
    try:
        data = {
            "document_id": document_id,
            "status": status,
            "step": step,
            "progress": progress,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if error:
            data["error"] = error

        redis_client.publish(f"document:{document_id}:progress", json.dumps(data))
        logger.info(f"[progress] {document_id}: {message}")
    except Exception as e:
        # If Redis fails, still log progress
        # Client can fallback to polling
        logger.warning(f"[progress] Failed to publish: {e}")
```

Update worker task error handling:

```python
@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    try:
        publish_progress(document_id, "extraction", 5, "Starting...")

        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(_extract_and_chunk_document(document_id))

        return result

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        publish_progress(document_id, "extraction", 0,
            f"File not found: {e}", status="failed", error=str(e))
        raise

    except Exception as exc:
        logger.error(f"Error: {exc}")
        publish_progress(document_id, "extraction", 0,
            f"Error: {exc}", status="failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)
```

---

## Step 10: Deployment Checklist

Before going to production:

- [ ] Redis is accessible from all worker instances
- [ ] SSE endpoint has proper CORS headers (check `main.py`)
- [ ] Nginx configured to not buffer SSE responses
- [ ] Health check doesn't interfere with progress streams
- [ ] Error messages are user-friendly
- [ ] Fallback polling works (test by killing SSE)
- [ ] Progress persists across worker restarts
- [ ] Mobile testing done (network switch scenario)

**Nginx configuration (if using):**

```nginx
location /progress/ {
    proxy_pass http://backend:8000;
    proxy_buffering off;
    proxy_cache off;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
}
```

---

## Troubleshooting

### SSE Connection Not Sending Updates

**Check 1:** Is Redis running?
```bash
redis-cli ping
# Should return: PONG
```

**Check 2:** Is worker publishing?
```bash
redis-cli
> SUBSCRIBE document:*:progress
# Upload a document, you should see messages
```

**Check 3:** Browser console
```javascript
// Check for errors
const es = new EventSource('/progress/docs/xxx/stream');
es.onerror = (e) => console.log('Error:', e);
```

### High Latency (>100ms)

**Likely causes:**
1. Redis instance is slow (check `redis-cli --latency`)
2. Network latency between services
3. Worker is synchronous (should be async)

**Fix:** Check worker code is calling `publish_progress` frequently

### Memory Leak in SSE

**Symptom:** Memory grows indefinitely

**Cause:** Not closing SSE connections or Redis pubsub

**Fix:** Ensure `finally` block in `event_generator()` closes pubsub

```python
finally:
    pubsub.close()  # ← Must be here
```

---

## Performance Monitoring

Add basic metrics:

```python
# In worker task
import time

start = time.time()
publish_progress(document_id, "extraction", 10, "Starting...")

# ... do work ...

duration = time.time() - start
logger.info(f"[metrics] extraction for {document_id} took {duration:.2f}s")
```

Monitor Redis Pub/Sub:

```bash
# In another terminal
redis-cli
> MONITOR

# Upload a document in browser
# Watch all Redis commands in real-time
```

---

## Next Steps

1. Implement SSE (this guide)
2. Test with single document upload
3. Test with 10 concurrent uploads
4. Add progress to embedding task (Phase 4)
5. Monitor latency with 100+ concurrent users
6. Consider upgrading to WebSocket if needed (Phase 5+)

---

## Timeline

- **Today:** Steps 1-4 (45 minutes)
- **Tomorrow morning:** Steps 5-7 (1.5 hours)
- **Tomorrow afternoon:** Steps 8-10 (1.5 hours)
- **Wednesday:** Testing and iteration (1 hour)

**Total: ~4 hours to production-ready SSE progress tracking**

---

## Questions?

Refer to detailed analysis in `/docs/REAL_TIME_PROGRESS_ANALYSIS.md`
