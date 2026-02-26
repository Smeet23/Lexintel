# LexIntel RAG - Quick Start Guide

This guide will help you get the core RAG system running: **upload PDF → chunk & embed → query**.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+
- Google AI API key (from https://aistudio.google.com/apikey)
- Azure Storage connection string (or use Azurite for local development)

## Setup

### 1. Clone and Navigate

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/LexIntel
```

### 2. Configure Environment

Copy `.env.example` to `.env` and update values:

```bash
cp .env.example .env
```

Edit `.env`:
```
DATABASE_URL=postgresql://legal_user:secure_password@localhost:5432/legal_rag
QDRANT_URL=http://localhost:6333
GOOGLE_API_KEY=your-google-api-key-here
AZURE_STORAGE_CONNECTION_STRING=UseDevelopmentStorage=true
SECRET_KEY=your-secret-key
DEBUG=True
```

### 3. Start Infrastructure (Docker)

```bash
docker-compose up -d
```

**Wait 30 seconds for services to start**, then verify:

```bash
docker-compose ps
```

All services should show `healthy` or `running`.

### 4. Initialize Database

```bash
cd backend
python -m alembic upgrade head
cd ..
```

## Running the System

### All Services at Once

```bash
# Start all services (API, Celery workers, databases)
docker-compose up -d

# Wait 30 seconds for services to be healthy
sleep 30

# Initialize database
cd backend
python -m alembic upgrade head
cd ..

# Run the test
python test_workflow.py
```

### OR Run Services Separately (for development)

**Terminal 1: Start FastAPI Server**
```bash
cd backend
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Terminal 2: Start Celery Worker(s)**
```bash
cd backend
celery -A celery_app worker -l info
```

Expected output:
```
[Tasks]
  . backend.tasks.process_document_task

[2025-01-26 10:00:00,000: INFO/MainProcess] celery@hostname ready.
```

**Terminal 3: Run the Test**
```bash
python test_workflow.py
```

Expected output:
```
============================================================
LexIntel RAG System - Workflow Test
============================================================

[1] Testing API health...
✓ API is healthy: {'status': 'ok'}

[2] Uploading test PDF...
✓ PDF uploaded successfully!
  Case ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
  Status: processing

[3] Checking case status...
✓ Case status: processing

[4] Waiting for document processing (max 60s)...
  Status: processing, waiting...
  ...
✓ Document processing complete!

[5] Querying the document...
✓ Query successful!
  Question: What is this document about?
  Answer: This is a test document...
  Confidence: high
  Sources: 1 documents

============================================================
✓ Complete workflow test PASSED!
============================================================
```

## API Endpoints

### Upload PDF
```bash
curl -X POST http://localhost:8000/cases \
  -F "name=My Case" \
  -F "file=@document.pdf"
```

Response:
```json
{
  "id": "case-uuid",
  "name": "My Case",
  "status": "processing",
  "created_at": "2025-01-26T..."
}
```

### Check Status
```bash
curl http://localhost:8000/cases/{case_id}/status
```

Response:
```json
{
  "id": "case-uuid",
  "name": "My Case",
  "status": "ready",
  "created_at": "2025-01-26T..."
}
```

### Query Document
```bash
curl -X POST http://localhost:8000/cases/{case_id}/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main issue?"}'
```

Response:
```json
{
  "question": "What is the main issue?",
  "answer": "Based on the documents provided...",
  "sources": [
    {
      "page_num": "1",
      "relevance_score": 0.95,
      "content_preview": "..."
    }
  ],
  "case_id": "case-uuid",
  "model": "gemini-2.5-flash-lite",
  "tokens_used": 234,
  "confidence": "high",
  "error": null
}
```

## How It Works

### 1. **Upload** (`POST /cases`)
   - Validates PDF file
   - Uploads to Azure Blob Storage
   - **Sends `process_document_task` to Celery queue** (via Redis)
   - Returns immediately with `status: processing` + `task_id`

### 2. **Process** (Event-Driven via Celery)
   - Celery worker(s) listen to Redis queue
   - When task arrives:
     - Downloads PDF from Blob Storage
     - **Chunks** document using hybrid semantic chunking (markdown headers → SemanticChunker → fallback)
     - **Embeds** chunks using Google `gemini-embedding-001` (768-dim)
     - **Stores** vectors in Qdrant + metadata in PostgreSQL
     - Updates case `status: ready`
   - On error:
     - Automatically retries (3 attempts max)
     - Exponential backoff: 5s, 10s, 15s
     - Updates case `status: error` if all retries fail

### 3. **Query** (`POST /cases/{case_id}/ask`)
   - Validates case is `ready`
   - **Embeds** question with same model
   - **Searches** Qdrant for top-10 similar chunks
   - **Filters** by confidence (≥0.7 similarity score)
   - **Generates** answer using Gemini 2.5 Flash Lite with legal prompt
   - **Extracts** citations from answer
   - Stores query in database
   - Returns answer + sources

## Why Celery Instead of Polling?

✅ **Event-Driven**: Tasks are processed immediately when they arrive
✅ **No Overhead**: No constant polling of the database
✅ **Scalable**: Run multiple Celery workers for parallel processing
✅ **Reliable**: Built-in retry logic with exponential backoff
✅ **Monitoring**: Can inspect task status, queue length, worker health
✅ **Standard**: Industry-standard for Python async task processing

## Troubleshooting

### Celery Worker Not Processing Tasks

Check Celery logs:
```bash
# If running via docker-compose:
docker-compose logs celery-worker

# If running locally:
# Look at the terminal where you ran 'celery -A celery_app worker'
```

Verify Redis is connected:
```bash
redis-cli ping
# Should return: PONG
```

Verify tasks in queue:
```bash
# Check pending tasks
celery -A celery_app inspect active

# Check queue length
redis-cli LLEN celery  # Default queue
```

### API Errors

Check FastAPI logs for exceptions.

If you see "Qdrant not available":
```bash
curl http://localhost:6333/health
```

If you see "Google API key" error:
```bash
# Verify .env file
cat .env | grep GOOGLE_API_KEY
```

### Chunking Issues

If chunks are empty, the PDF might have extraction issues. Try with a different PDF or check chunking service logs.

## Next Steps

Once the core workflow is working:

1. **Build Frontend** (`frontend/app/`)
   - Dashboard to list cases
   - Upload interface
   - Q&A chat interface
   - PDF viewer with highlighting

2. **Add Features**
   - Case summarization
   - Search across multiple cases
   - Multi-language support
   - Authentication UI

3. **Deploy**
   - Build Docker images
   - Deploy to Azure Container Registry
   - Setup CI/CD pipeline

## Files Modified

- `backend/main.py` - Added `/cases/{id}/ask` and `/cases/{id}/status` endpoints
- `backend/worker.py` - Created background job processor
- `backend/services/vector_store.py` - Fixed content mapping for RAG
- `.env.example` - Already configured

## Performance Notes

- **Upload**: < 1 second
- **Chunking**: 1-3 seconds (hybrid semantic, depends on document size)
- **Embedding**: 1-5 seconds (Google API + batch size, with caching)
- **Query**: 2-5 seconds (Qdrant search + Gemini generation)
- **Total**: ~10-20 seconds from upload to first query

## Cost Estimates

Using Google Gemini + Qdrant:
- Upload (100 page PDF): minimal cost (Google embeddings)
- Query: minimal cost per query (Gemini 2.5 Flash Lite)
- Storage: ~$10/month (Qdrant Pro)

## Common Issues

### Issue: "Collection not found" when querying
**Cause**: Processing hasn't completed yet
**Fix**: Wait for case status to show "ready"

### Issue: Empty answer from Gemini
**Cause**: No relevant chunks found (low similarity scores)
**Fix**: Try different search terms or verify PDF has searchable text

### Issue: Worker crashes on embedding
**Cause**: Google API key invalid
**Fix**: Check `.env` file and verify GOOGLE_API_KEY is active

---

**Status**: Core RAG pipeline working. Ready for frontend development.
