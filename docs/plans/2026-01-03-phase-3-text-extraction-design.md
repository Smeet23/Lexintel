# Phase 3: Text Extraction Workers - Detailed Design

> Comprehensive design for extracting text from uploaded documents and creating searchable chunks

**Status**: Design Phase (awaiting approval)
**Last Updated**: January 3, 2026

---

## 📋 Table of Contents

1. [Data Storage](#data-storage)
2. [Process Flow](#process-flow)
3. [Step-by-Step Algorithm](#step-by-step-algorithm)
4. [State Management](#state-management)
5. [Database Operations](#database-operations)
6. [Error Handling](#error-handling)
7. [Edge Cases](#edge-cases)
8. [File Organization](#file-organization)

---

## 📦 Data Storage

### What Gets Stored

#### **Documents Table** (`documents`)
After Phase 2 (upload), a Document record exists with:
```
id:                 UUID (primary key)
case_id:            UUID (foreign key → cases)
title:              String (from API)
filename:           String (original filename)
type:               Enum (brief, complaint, discovery, etc.)
file_size:          Integer (bytes)
file_path:          String (local filesystem path: /app/uploads/{doc_id}/filename)
processing_status:  Enum = "pending"
extracted_text:     NULL (Phase 3 will populate)
page_count:         NULL (optional, for PDFs)
error_message:      NULL
indexed_at:         NULL
created_at:         DateTime (auto)
updated_at:         DateTime (auto)
```

#### **DocumentChunk Table** (`document_chunks`)
Phase 3 creates these records:
```
id:             UUID (primary key, generated)
document_id:    UUID (foreign key → documents)
chunk_text:     Text (4000 chars max)
chunk_index:    Integer (0-indexed, order within document)
embedding:      NULL (Phase 4 will populate)
search_vector:  NULL (PostgreSQL auto-generates from chunk_text)
created_at:     DateTime (auto)
updated_at:     DateTime (auto)
```

### Where Data Gets Stored

```
Database: lex_intel (PostgreSQL)
Tables involved:
  ├── documents (UPDATE: extracted_text, processing_status)
  └── document_chunks (INSERT: new chunks)

Filesystem:
  └── /app/uploads/{document_id}/{filename}
      (Already exists from Phase 2, just being read)
```

---

## 🔄 Process Flow

### High-Level Flow

```
1. Document uploaded via API (Phase 2) → Document.processing_status = "pending"
                                        → File stored on filesystem
                                        → Document record created
                                ↓
2. API calls process_document_pipeline(document_id) → Celery task queued
                                ↓
3. Celery worker receives extract_text_from_document(document_id)
                                ↓
4. Worker executes _extract_and_chunk_document(document_id):
   a. Load Document from DB
   b. Validate file exists
   c. Read file from filesystem
   d. Extract text (Phase 3: TXT only)
   e. Clean text
   f. Split into chunks (4000 chars, 400 overlap)
   g. Create DocumentChunk records
   h. Update Document.extracted_text
   i. Update Document.processing_status → "extracted"
                                ↓
5. Worker queues generate_embeddings(document_id) task (Phase 4)
                                ↓
6. Task completes with success response
```

---

## 🔍 Step-by-Step Algorithm

### **Step 1: Task Entry Point**

**Function**: `extract_text_from_document(document_id: str)`
**Location**: `app/workers/tasks.py`
**Celery Config**:
- `@shared_task(base=CallbackTask, bind=True, max_retries=3)`
- Retry backoff: 60 seconds (exponential)
- Timeout: 25 minutes

**Input validation**:
```python
if not document_id:
    raise ValueError("document_id required")

# Async execution
loop = asyncio.get_event_loop()
result = loop.run_until_complete(
    _extract_and_chunk_document(document_id)
)
```

**Output**:
```json
{
  "status": "success",
  "document_id": "uuid",
  "chunks_created": 15,
  "text_length": 45000
}
```

---

### **Step 2: Load Document from Database**

**Function**: `_extract_and_chunk_document(document_id: str)`
**Location**: `app/workers/tasks.py`

```python
async with async_session() as session:
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalars().first()

    if not document:
        raise ValueError(f"Document {document_id} not found")
```

**What we validate**:
- Document exists in DB
- `document.file_path` is set
- `document.case_id` is set (for tracking)
- `document.processing_status == "pending"`

---

### **Step 3: Validate File Exists**

```python
if not document.file_path:
    raise FileNotFoundError(
        f"Document {document_id} has no file_path"
    )

if not os.path.exists(document.file_path):
    raise FileNotFoundError(
        f"File not found: {document.file_path}"
    )
```

**Where file is located**:
- Path format: `/app/uploads/{document_id}/{original_filename}`
- Created by Phase 2 upload handler
- Persists on container filesystem (or mounted volume)

---

### **Step 4: Extract Text from File**

**Function**: `extract_file(file_path: str) -> str`
**Location**: `app/services/extraction.py`

```python
async def extract_file(file_path: str) -> str:
    """Extract text from file based on file type"""

    # Phase 3: TXT files only
    if file_path.lower().endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
```

**What happens**:
- Opens file with UTF-8 encoding
- Reads entire file into memory
- Returns raw text (no processing yet)
- Encoding errors: use `errors='replace'` to skip bad chars

**Phase 3 limitation**: TXT only
**Phase 3+**: Add PDF (PyPDF2) and DOCX (python-pptx) support

---

### **Step 5: Clean Text**

**Function**: `_extract_and_chunk_document()` → text cleaning section
**Location**: `app/workers/tasks.py`

```python
text = text
    # Remove NULL bytes (PostgreSQL incompatibility)
    .replace('\x00', '')
    # Remove control characters (except \n, \r, \t)
    .replace('\x01-\x08\x0B\x0C\x0E-\x1F\x7F', '')
    # Remove leading/trailing whitespace
    .strip()
    # Normalize multiple newlines to double newlines
    .replace('\n{3,}', '\n\n')
    # Remove page numbers (common OCR artifact)
    .replace(r'^\s*\d+\s*$', '', flags=MULTILINE)
    # Remove page headers (PDF extraction artifact)
    .replace(r'^Page \d+.*$', '', flags=MULTILINE)
    # Convert form feeds to newlines
    .replace('\f', '\n')
    # Normalize line endings
    .replace('\r\n', '\n')
    # Normalize multiple spaces to single space
    .replace('[ \t]{2,}', ' ')
    # Final trim
    .strip()
```

**Why clean**:
- Remove encoding artifacts from PDF/scanned documents
- Normalize whitespace for better chunking
- PostgreSQL can't store NULL bytes in UTF-8 text
- Control characters confuse embeddings

**Result**: Clean text ready for chunking

---

### **Step 6: Split Text into Chunks**

**Function**: `create_text_chunks(document_id, text, session)`
**Location**: `app/services/extraction.py`

```python
async def create_text_chunks(
    document_id: str,
    text: str,
    session: AsyncSession,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[str]:
    """
    Split text into overlapping chunks and create DB records

    Algorithm:
    1. Initialize: start=0, chunk_index=0, chunk_ids=[]
    2. Loop while start < len(text):
        a. Calculate end = min(start + chunk_size, len(text))
        b. Extract slice: text[start:end]
        c. Trim trailing whitespace from slice
        d. Check if slice length >= 200 chars (minimum):
           - If yes: Create DocumentChunk record
           - If no: Skip (too small/empty)
        e. Update progress
        f. Calculate next_start = end - overlap
        g. If next_start <= start: break (avoid infinite loop)
        h. Set start = next_start
    3. Return list of created chunk IDs
    """
```

**Chunking Example**:
```
Text: "This is a test document with content....[4000 chars]....more content...[remaining]"

Chunk 0: chars 0-4000
Chunk 1: chars 3600-7600 (400 char overlap with chunk 0)
Chunk 2: chars 7200-11200 (400 char overlap with chunk 1)
Chunk 3: chars 10800-14800
...until end of text

If chunk < 200 chars: skip (too small)
```

**Why 4000 chars?**
- ≈ 1000 tokens (OpenAI embedding model expects 500-1000 tokens)
- Good balance between context preservation and granularity
- Matches your production system

**Why 400 char overlap?**
- Ensures information spanning chunk boundaries is preserved
- User searches won't miss results because of artificial boundaries
- ≈ 100 tokens overlap = good semantic continuity

---

### **Step 7: Create DocumentChunk Records**

**Function**: Inside `create_text_chunks()` loop
**Location**: `app/services/extraction.py`

```python
chunk = DocumentChunk(
    id=str(uuid4()),
    document_id=document_id,
    chunk_text=slice.trim(),  # The actual chunk content
    chunk_index=chunk_index,  # 0, 1, 2, 3, ...
)

session.add(chunk)
chunk_index += 1
chunk_ids.append(chunk.id)
```

**What gets stored**:
```
DocumentChunk record:
  id:            "550e8400-e29b-41d4-a716-446655440000"
  document_id:   "parent-doc-uuid"
  chunk_text:    "This is chunk content, up to 4000 characters..."
  chunk_index:   0
  embedding:     NULL (Phase 4 will generate)
  search_vector: NULL (PostgreSQL trigger auto-generates)
  created_at:    "2026-01-03T10:30:00Z"
  updated_at:    "2026-01-03T10:30:00Z"
```

**Database indices**:
- `document_id` (foreign key, for fast lookups)
- `chunk_index` (for ordering chunks)
- Future: vector index for `embedding`

---

### **Step 8: Update Document Record**

**Function**: After all chunks created
**Location**: `app/workers/tasks.py`

```python
document.extracted_text = text  # Store full text
document.processing_status = ProcessingStatus.EXTRACTED
document.updated_at = datetime.now()

await session.commit()
```

**What changes in Document**:
```
Before:
  extracted_text:     NULL
  processing_status:  "pending"
  updated_at:         "2026-01-03T10:00:00Z"

After:
  extracted_text:     "Full text of document with all content..."
  processing_status:  "extracted"
  updated_at:         "2026-01-03T10:30:00Z"
```

**Why store full text**:
- Faster re-processing if needed
- Debugging and inspection
- Fall back if chunks are corrupted
- Audit trail

---

### **Step 9: Queue Next Task**

**Function**: End of `_extract_and_chunk_document()`
**Location**: `app/workers/tasks.py`

```python
# Queue embeddings generation for Phase 4
from app.workers.tasks import generate_embeddings
generate_embeddings.delay(document_id)

logger.info(
    f"Queued embeddings for document {document_id} "
    f"with {len(chunk_ids)} chunks"
)
```

**What happens**:
- Task added to Celery queue
- Celery worker picks it up (async, independent)
- Phase 4 will process chunks and generate embeddings

---

## 📊 State Management

### Document Status Lifecycle

```
Phase 2 (Upload)
    |
    v
Document created with:
  processing_status = "pending"
  extracted_text = NULL
    |
    v
Phase 3 (Extraction) - THIS PHASE
    |
    +-- Task starts
    |    └─> Progress: "processing"
    |
    +-- Chunks created successfully
    |    └─> Update: processing_status = "extracted"
    |    └─> Update: extracted_text = full text
    |    └─> Create: DocumentChunk records
    |
    +-- Task completes
    |    └─> Queue: generate_embeddings task
    |    └─> Return: success response
    |
    v
Phase 4 (Embeddings)
    |
    └─> Query DocumentChunk records
    └─> Generate embeddings for each chunk
    └─> Update: processing_status = "indexed"
```

### State Transitions

```
Valid transitions:
  pending       → extracted  (success)
  pending       → failed     (error)

  extracted     → indexed    (Phase 4 success)
  extracted     → failed     (Phase 4 error)
```

**No backwards transitions**: Once failed, must manually restart

---

## 🗄️ Database Operations

### Session Management

```python
from app.database import async_session
from sqlalchemy import select

async with async_session() as session:
    # All DB operations within this context

    # Read
    stmt = select(Document).where(Document.id == document_id)
    result = await session.execute(stmt)
    document = result.scalars().first()

    # Create
    chunk = DocumentChunk(...)
    session.add(chunk)

    # Update
    document.processing_status = ProcessingStatus.EXTRACTED

    # Commit
    await session.commit()  # Atomic transaction
```

### Atomic Operations

```python
# All-or-nothing: If anything fails, entire transaction rolls back
try:
    async with async_session() as session:
        # 1. Load document
        # 2. Create chunks (multiple inserts)
        # 3. Update document status
        await session.commit()  # Only if all succeed
except Exception as e:
    # Transaction auto-rolled back
    # Document still in "pending" state
    raise
```

### Transaction Isolation

- **Isolation level**: READ_COMMITTED (default PostgreSQL)
- **Risk**: While processing, another process could read partial updates
- **Mitigation**: Status is updated last, ensuring atomicity from client perspective

---

## ⚠️ Error Handling

### Error Scenarios & Handling

#### **Scenario 1: Document Not Found**

```
Trigger: Document.id doesn't exist in DB

Handling:
  1. Raise ValueError("Document X not found")
  2. Celery catches exception
  3. Task marked as FAILED
  4. No retry (data error, not transient)
  5. DB remains unchanged (no updates)

User sees: Document not found error in API
Database: Document.processing_status still = "pending"
```

#### **Scenario 2: File Not Found**

```
Trigger: document.file_path doesn't exist on filesystem

Handling:
  1. Raise FileNotFoundError("File not found: /path/to/file")
  2. Catch in except block
  3. Set: document.processing_status = "failed"
  4. Set: document.error_message = "File not found: /path/to/file"
  5. Commit to DB
  6. Raise exception (triggers task failure)
  7. No retry

User sees: Document status = "failed", can see error message
Database: Document.error_message populated
```

#### **Scenario 3: Encoding Error**

```
Trigger: File has invalid UTF-8 bytes

Handling:
  1. Open file with encoding='utf-8', errors='replace'
  2. Invalid bytes replaced with U+FFFD (replacement char)
  3. Continue processing
  4. Log warning: "Encoding error in document X, used replacement chars"
  5. Task completes successfully

User sees: Document extracted (status = "extracted")
Database: extracted_text may contain replacement chars (rare)
Result: Embeddings slightly affected but acceptable
```

#### **Scenario 4: DB Connection Error**

```
Trigger: PostgreSQL connection drops, query fails

Handling:
  1. Exception raised by SQLAlchemy
  2. Caught by Celery retry mechanism
  3. Task retried with 60-second backoff
  4. Max 3 retries = up to 3 minutes of waiting
  5. On final failure: task marked as FAILED

User sees: Document status = "pending" (no update)
Database: Unchanged (no commit occurred)
Action: Manual retry needed or investigate connection
```

#### **Scenario 5: Task Timeout (>25 min)**

```
Trigger: Celery task runs longer than timeout

Handling:
  1. Task forcibly killed by Celery
  2. Marked as FAILED
  3. No retry (timeout would recur)
  4. Document.processing_status remains "pending"

User sees: Document still "pending", task failed in logs
Database: Partially created chunks may exist
Action: Delete partial chunks, retry with smaller doc or investigate
```

### Error Recovery

```python
# In _extract_and_chunk_document():

try:
    # Main processing
    text = extract_file(document.file_path)
    chunks = await create_text_chunks(...)
    document.extracted_text = text
    document.processing_status = ProcessingStatus.EXTRACTED
    await session.commit()

except FileNotFoundError as e:
    # File errors: permanent, no retry
    document.processing_status = ProcessingStatus.FAILED
    document.error_message = str(e)
    await session.commit()
    raise  # Task marked FAILED

except Exception as e:
    # All other errors: transient, retry
    document.processing_status = ProcessingStatus.FAILED
    document.error_message = str(e)
    await session.commit()
    raise self.retry(exc=e, countdown=60)  # Retry in 60s
```

---

## 🔧 Edge Cases

### Edge Case 1: Empty Document

```
Input: File with 0 bytes or only whitespace

Handling:
  1. extract_file() returns "" or "   "
  2. After cleaning: text = ""
  3. Chunking loop: start=0, len(text)=0
  4. Condition: start < len(text) is False
  5. No chunks created
  6. DocumentChunk table: 0 records for this document
  7. Document.extracted_text = ""
  8. Document.processing_status = "extracted"

Result: Valid state, embeddings phase will handle empty chunks gracefully
User sees: Document extracted with 0 chunks
```

### Edge Case 2: Very Large Document

```
Input: 10MB text file

Handling:
  1. extract_file() reads entire 10MB into memory
  2. Text cleaning: ~10MB
  3. Chunking: ~2500 chunks (10MB / 4KB per chunk)
  4. Database: Create 2500 DocumentChunk inserts
  5. Progress tracking: Update job progress % every chunk
  6. Commit: All at once at end (single transaction)

Performance:
  - Memory: Python process uses ~20MB peak
  - DB inserts: ~1-2 seconds
  - Total time: ~30 seconds to 2 minutes
  - Fits well within 25-minute timeout

Result: Valid, supported scenario
```

### Edge Case 3: Chunk Boundary Edge Case

```
Input: Text where boundary falls in middle of word

Example:
  Text: "...contractual obligations are binding..."
         ^--- chunk boundary here

  Chunk 1: "...contractual ob"
  Chunk 2: "obligations are binding..."

Handling:
  Current: As-is (may break words)
  Future: Could trim to word boundaries

Impact: Minimal (word boundaries preserved across overlap)
```

### Edge Case 4: Special Characters

```
Input: Text with emojis, math symbols, special unicode

Example: "Patent © 2024 | Royalty: €500 | Status: ✓"

Handling:
  1. UTF-8 encoding preserves all unicode
  2. Special chars included in chunks as-is
  3. Embeddings handle unicode naturally
  4. Search queries work correctly

Result: Full unicode support, no data loss
```

### Edge Case 5: Duplicate Content

```
Scenario: Text with 400-char overlap between chunks

Chunk A: [0-4000]      = "...Lorem ipsum dolor sit amet..."
Chunk B: [3600-7600]   = "...dolor sit amet...[new content]..."
                         ^--- 400 chars duplicated

Impact:
  - Embeddings generated for both chunks (includes overlap)
  - Search results may return duplicate text
  - Acceptable trade-off for semantic continuity

Solution (Phase 5+): Can deduplicate at search time
```

---

## 📁 File Organization

### Code Structure

```
backend/app/
├── workers/
│   └── tasks.py
│       ├── extract_text_from_document()  ← Main task
│       ├── _extract_and_chunk_document()  ← Async helper
│       ├── generate_embeddings()  ← Phase 4 (placeholder)
│       └── process_document_pipeline()  ← Orchestrator
│
└── services/
    └── extraction.py
        ├── extract_file()  ← Text extraction by type
        └── create_text_chunks()  ← Chunking algorithm
```

### Dependencies to Add

```
# requirements.txt additions:
# Phase 3 (TXT only):
# None! Built-in Python features only

# Phase 3+ (PDF/DOCX support):
# PyPDF2==3.0.1       # PDF parsing
# python-pptx==0.6.21 # DOCX parsing
```

### Environment Variables

```
# .env (no new vars needed for Phase 3)
# But useful to add:

CELERY_MAX_RETRIES=3          # Task retries
CELERY_RETRY_BACKOFF=60       # Seconds between retries
EXTRACT_MAX_TEXT_CHARS=1000000 # Limit doc size (optional)
PDF_MAX_PAGES=50              # Limit PDF pages (Phase 3+)
```

---

## ✅ Implementation Checklist

- [ ] Create `app/services/extraction.py`
  - [ ] `extract_file(file_path: str) -> str`
  - [ ] `create_text_chunks(...) -> List[str]`
  - [ ] Text cleaning logic
  - [ ] Chunking algorithm

- [ ] Update `app/workers/tasks.py`
  - [ ] Import extraction service
  - [ ] Implement `_extract_and_chunk_document()` async helper
  - [ ] Error handling in extraction task
  - [ ] Queue embeddings task at end

- [ ] Testing
  - [ ] Create test TXT file
  - [ ] Test end-to-end with API
  - [ ] Verify DocumentChunk records
  - [ ] Verify status updates
  - [ ] Test error scenarios

- [ ] Documentation
  - [ ] Update claude.md with Phase 3 status
  - [ ] Document extraction service API
  - [ ] Add to WORKERS.md

---

## 📚 References

- **Text Encoding**: Python `str` type handles UTF-8 natively
- **Asyncio**: `asyncio.get_event_loop()` for mixing sync/async
- **SQLAlchemy**: Async session for concurrent DB access
- **Celery**: Task queuing, retries, timeouts

