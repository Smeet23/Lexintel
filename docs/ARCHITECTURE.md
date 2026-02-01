# LexIntel System Architecture Documentation

**Version:** 1.0
**Last Updated:** February 2026
**Status:** Production-Ready

## Table of Contents

1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Details](#component-details)
4. [Data Models](#data-models)
5. [Data Flow](#data-flow)
6. [API Layer Design](#api-layer-design)
7. [Component Relationships](#component-relationships)
8. [Component Lifecycle](#component-lifecycle)
9. [Integration Points](#integration-points)
10. [Cross-References](#cross-references)

---

## System Overview

**LexIntel** is a production-ready **Retrieval-Augmented Generation (RAG)** system designed for intelligent legal document analysis and case management. The system enables lawyers and legal professionals to:

- Upload legal documents (PDFs, DOCX, TXT) with automatic semantic processing
- Ask natural language questions about their cases
- Receive AI-generated answers with precise source citations
- Manage multiple cases with background job processing
- Maintain comprehensive audit trails for compliance

### Core Architecture Principles

- **Separation of Concerns:** Each component has a single, well-defined responsibility
- **Async-First Design:** Background processing prevents blocking operations
- **Data Isolation:** Multi-user support with row-level security
- **Error Resilience:** Comprehensive retry logic and error handling
- **Auditability:** All actions logged for legal compliance

---

## High-Level Architecture

### System Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Layer                                  │
│                  (Frontend: Next.js + React + TypeScript)               │
└────────────────────────┬────────────────────────────────────────────────┘
                         │ HTTPS REST API
                         ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      API Layer (FastAPI)                                │
├──────────────┬──────────────┬──────────────────┬──────────────────┬────┤
│  Auth        │  Case        │  RAG Query       │  Job Status      │    │
│  Service     │  Management  │  Engine          │  Monitoring      │    │
│  - Register  │  - Upload    │  - Query Case    │  - Health Check  │    │
│  - Login     │  - List      │  - Get Answer    │                  │    │
│  - Validate  │  - Delete    │  - Citations     │                  │    │
└──────────────┴──────────────┴──────────────────┴──────────────────┴────┘
                    │              │                    │
        ┌───────────┼──────────────┼────────────────────┘
        ↓           ↓              ↓
    ┌──────────┐ ┌──────────┐  ┌──────────┐
    │PostgreSQL│ │  Redis   │  │  Azure   │
    │(Metadata)│ │(Cache)   │  │  Blob    │
    └──────────┘ └──────────┘  └──────────┘
        │           │              │
        └───────────┼──────────────┘
                    ↓
    ┌─────────────────────────────────┐
    │    Background Job Processor      │
    ├─────────────────────────────────┤
    │ - Document Chunking             │
    │ - Embedding Generation          │
    │ - Vector Store Upsert           │
    │ - Status Tracking               │
    └───────────┬─────────────────────┘
                ↓
    ┌─────────────────────────────────┐
    │       External Services         │
    ├─────────────────────────────────┤
    │ - Qdrant (Vector DB)            │
    │ - OpenAI (Embeddings + LLM)     │
    │ - Azure Blob Storage            │
    └─────────────────────────────────┘
```

### Logical Layers

```
┌─────────────────────────────────────────────────────────────┐
│               Presentation Layer (FastAPI)                  │
│  - REST API Endpoints                                       │
│  - Input Validation (Pydantic Schemas)                      │
│  - JWT Authentication                                       │
│  - CORS Configuration                                       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│               Application Layer (Services)                   │
│  - Authentication Service (auth.py)                         │
│  - Storage Service (storage.py)                             │
│  - Chunking Service (chunking.py)                           │
│  - Embeddings Service (embeddings.py)                       │
│  - Vector Store Service (vector_store.py)                   │
│  - RAG Query Engine (rag_engine.py)                         │
│  - Job Processor (job_processor.py)                         │
│  - Cache Manager (cache_manager.py)                         │
│  - Text Extraction (text_extraction.py)                     │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│               Data Access Layer (SQLAlchemy ORM)             │
│  - Database Models (User, Case, Chunk, Query, ProcessingJob)│
│  - Database Transactions                                    │
│  - Connection Pooling                                       │
└──────────────────────┬──────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────────┐
│               Data Layer                                     │
│  - PostgreSQL (Structured Data)                             │
│  - Redis (Caching & Queue)                                  │
│  - Qdrant (Vector Embeddings)                               │
│  - Azure Blob Storage (Raw Documents)                       │
│  - OpenAI APIs (Embeddings + LLM)                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. API Layer

**Location:** `backend/main.py`

**Responsibilities:**
- Route HTTP requests to appropriate handlers
- Validate incoming requests using Pydantic schemas
- Handle JWT authentication and authorization
- Manage CORS configuration
- Return standardized HTTP responses
- Log all API calls for audit trails

**Key Endpoints:**
```
Authentication
  POST   /auth/register              Register new user
  POST   /auth/login                 Login and get JWT token
  GET    /user/profile               Get current user profile (protected)

Case Management
  POST   /cases                       Upload new document (PDF/DOCX/TXT)
  GET    /cases                       List user's cases (protected)
  GET    /cases/{case_id}             Get case details (protected)
  DELETE /cases/{case_id}             Delete case (protected)

Query & Analysis
  POST   /cases/{case_id}/ask         Query case with RAG (protected)
  GET    /cases/{case_id}/queries     Get query history (protected)

Health & Monitoring
  GET    /health                      Health check endpoint
```

**Error Handling:**
- 400: Invalid input
- 401: Unauthorized (missing/invalid JWT)
- 403: Forbidden (user lacks access)
- 404: Resource not found
- 429: Rate limited
- 500: Server error (with generic message, detailed logs)

---

### 2. Authentication Service

**Location:** `backend/auth.py`

**Responsibilities:**
- User registration with email/password
- Secure password hashing using bcrypt
- JWT token generation and validation
- Token expiry management (1440 minutes default)
- HS256 signing algorithm

**Key Functions:**
- `hash_password(password: str) -> str` - Generate bcrypt hash
- `verify_password(password: str, hash: str) -> bool` - Verify password against hash
- `create_access_token(user_id: UUID, email: str) -> str` - Generate JWT
- `decode_token(token: str) -> dict` - Decode and validate JWT

**Security Features:**
- Passwords hashed with bcrypt salt (cost=12)
- JWT tokens signed with secret key
- Token expiry prevents indefinite access
- Refresh token mechanism available for extended sessions

---

### 3. Configuration Management

**Location:** `backend/config.py`

**Responsibilities:**
- Load environment variables from `.env`
- Validate required configuration
- Provide settings to all components
- Handle different environments (dev/prod)

**Configuration Variables:**
```
DATABASE_URL              PostgreSQL connection string
OPENAI_API_KEY           OpenAI API key for embeddings/LLM
QDRANT_URL               Qdrant vector DB endpoint
AZURE_STORAGE_CONNECTION Azure Blob Storage connection
SECRET_KEY               JWT signing key
ALLOWED_ORIGINS          CORS allowed origins (comma-separated)
DEBUG                    Debug mode flag
CACHE_ENABLED            Enable/disable query caching
CACHE_TTL_SECONDS        Cache time-to-live (default: 86400)
REDIS_URL                Redis connection for cache/queue
CELERY_BROKER_URL        Celery task broker URL
CELERY_RESULT_BACKEND    Celery result backend URL
```

**Validation:**
- Required fields validated on startup
- CORS origins checked for placeholder domains
- API keys validated before use
- Database URL tested on connection

---

### 4. Database Layer

**Location:** `backend/database.py`, `backend/models.py`

**Responsibilities:**
- Create and manage PostgreSQL connections
- Define ORM models using SQLAlchemy 2.0
- Manage database sessions and transactions
- Implement relationship constraints
- Create indices for query optimization

**Database Models:**

#### User Model
```
id: UUID (PK)
email: String (unique, indexed)
password_hash: String
is_deleted: Boolean (soft delete)
created_at: DateTime
updated_at: DateTime

Relationships:
  - cases: One-to-Many (User -> Case)
```

#### Case Model
```
id: UUID (PK)
user_id: UUID (FK -> User)
name: String
blob_storage_path: String
file_type: String (pdf, docx, txt)
status: String (processing, ready, error)
is_deleted: Boolean (soft delete)
created_at: DateTime
updated_at: DateTime

Relationships:
  - user: Many-to-One (Case -> User)
  - chunks: One-to-Many (Case -> Chunk)
  - queries: One-to-Many (Case -> Query)

Indices:
  - (user_id, status) - Fast status queries
```

#### Chunk Model
```
id: UUID (PK)
case_id: UUID (FK -> Case)
page_num: String
section_name: String
content: Text
embedding_hash: String (SHA256 for deduplication)
chunk_sequence: Integer (order within case)
created_at: DateTime

Relationships:
  - case: Many-to-One (Chunk -> Case)

Indices:
  - case_id (FK lookup)
  - (case_id, chunk_sequence) - Document ordering
```

#### Query Model
```
id: UUID (PK)
case_id: UUID (FK -> Case)
user_id: UUID (FK -> User)
question: Text
answer: Text
citations: JSON (list of citation dicts)
created_at: DateTime

Relationships:
  - case: Many-to-One (Query -> Case)

Indices:
  - case_id (FK lookup)
  - (case_id, created_at) - Query history
```

#### ProcessingJob Model
```
id: UUID (PK)
case_id: UUID (FK -> Case)
status: String (pending, processing, completed, failed)
error_message: String (nullable)
attempts: Integer (retry count)
max_attempts: Integer (default: 3)
created_at: DateTime
started_at: DateTime (nullable)
completed_at: DateTime (nullable)
next_retry_at: DateTime (nullable)

Indices:
  - case_id (FK lookup)
  - status (job queue queries)
```

---

### 5. Storage Service

**Location:** `backend/services/storage.py`

**Responsibilities:**
- Upload documents to Azure Blob Storage
- Download documents from blob storage
- Validate file format (PDF magic bytes)
- Manage blob storage paths
- Handle authentication with Azure

**Key Functions:**
- `upload_document_to_blob(file: UploadFile, case_name: str) -> str` - Upload and return path
- `download_document_from_blob(blob_path: str) -> bytes` - Download document
- `validate_file_format(file_bytes: bytes, file_type: str) -> bool` - Validate file format
- `delete_document_from_blob(blob_path: str) -> bool` - Delete document

**File Format Validation:**
- PDF: Magic bytes `%PDF` (0x25 0x50 0x44 0x46)
- DOCX: ZIP format with `[Content_Types].xml`
- TXT: No magic bytes validation (plain text)

**Error Handling:**
- Connection failures with retry logic
- Invalid file format rejection
- Storage quota exceeded
- Missing blob paths

---

### 6. Document Processing Pipeline

**Location:** `backend/services/chunking.py`, `backend/services/text_extraction.py`

**Responsibilities:**
- Extract text from uploaded documents
- Perform semantic chunking of documents
- Preserve metadata (page numbers, section names)
- Handle multi-format documents (PDF, DOCX, TXT)
- Maintain chunk coherence for legal context

**Chunking Strategy:**
- **Chunk Size:** 800 characters (balance between context and specificity)
- **Overlap:** 150 characters (preserve argument continuity)
- **Separators:** Hierarchical (paragraph → sentence → word)
- **Metadata:** Page number, section name, sequence

**Text Extraction Methods:**
- **PDF:** PyMuPDF (fitz) with page-level extraction
- **DOCX:** python-docx library with paragraph extraction
- **TXT:** Direct line-by-line reading

**Key Functions:**
- `chunk_document_from_blob(blob_path: str, file_type: str) -> List[Chunk]` - Chunk uploaded doc
- `extract_text_from_pdf(pdf_bytes: bytes) -> str` - Extract PDF text
- `extract_text_from_docx(docx_bytes: bytes) -> str` - Extract DOCX text
- `extract_text_from_txt(txt_bytes: bytes) -> str` - Extract TXT text

**Output:**
```
List of Chunk objects:
  - content: "full chunk text..."
  - page_num: "5" or "2-3" (for ranges)
  - section_name: "Introduction" or "Arguments"
  - chunk_sequence: 0 (order within document)
```

---

### 7. Embeddings Service

**Location:** `backend/services/embeddings.py`, `backend/services/embedding_cache.py`

**Responsibilities:**
- Generate vector embeddings for text chunks
- Batch embed multiple chunks for efficiency
- Cache embeddings to prevent redundant API calls
- Track embedding costs
- Handle API rate limits and failures

**Embedding Model:**
- **Model:** OpenAI text-embedding-3-large
- **Dimensions:** 3072
- **Cost:** $0.02 per 1M tokens
- **Batch Size:** 20 chunks (for efficiency)

**Caching Strategy:**
- Hash chunk content (SHA256) to create cache keys
- Store in local cache (in-memory during session)
- Check cache before API calls
- Cost tracking prevents unnecessary calls

**Key Functions:**
- `embed_text(text: str) -> List[float]` - Embed single text
- `embed_chunks(chunks: List[str]) -> List[List[float]]` - Batch embed
- `get_cached_embedding(content_hash: str) -> Optional[List[float]]` - Get from cache
- `cache_embedding(content_hash: str, embedding: List[float]) -> None` - Store in cache

**Error Handling:**
- API rate limit errors (429) with exponential backoff
- Timeout errors with retry logic
- Invalid token counts validation
- Graceful degradation on failure

---

### 8. Vector Store Service

**Location:** `backend/services/vector_store.py`

**Responsibilities:**
- Manage Qdrant vector database
- Create collections for each case
- Upsert vectors with metadata
- Search vectors by similarity
- Delete vectors on case removal

**Collection Structure:**
```
Collection Name: case_{case_id}
Point Schema:
  id: hash(chunk_id) % 2^32 (deterministic)
  vector: [3072 dimensions]
  payload: {
    chunk_id: UUID,
    case_id: UUID,
    page_num: String,
    section_name: String,
    content: String (for display),
    sequence: Integer
  }
```

**Search Configuration:**
- **Similarity Metric:** Cosine distance
- **Default K:** 10 (retrieve top 10)
- **Confidence Filter:** ≥0.6 (60% similarity minimum)
- **Hybrid Ordering:** Relevance score + document order

**Key Functions:**
- `create_collection(case_id: UUID) -> bool` - Create collection
- `upsert_vectors(case_id: UUID, chunks: List[Chunk], embeddings: List[List[float]]) -> bool` - Store vectors
- `search_vectors(case_id: UUID, query_embedding: List[float], top_k: int) -> List[Dict]` - Search vectors
- `delete_collection(case_id: UUID) -> bool` - Delete collection

**Error Handling:**
- Network failures with retry
- Collection creation idempotency
- Vector dimension mismatch validation
- Payload size limits

---

### 9. RAG Query Engine

**Location:** `backend/services/rag_engine.py`

**Responsibilities:**
- Orchestrate complete RAG pipeline
- Query embedding and semantic search
- Context window management
- LLM answer generation
- Citation extraction and validation
- Token budgeting and cost tracking

**RAG Pipeline:**
```
1. Query Embedding
   Input: User question
   Output: 3072-dimensional embedding

2. Vector Similarity Search
   Input: Query embedding
   Config: Top 10, filter ≥0.6 similarity
   Output: List of relevant chunks with scores

3. Context Window Management
   Input: Retrieved chunks
   Budget: ~12,800 tokens (~50% of GPT-4o context)
   Logic: Select top 4 chunks, validate token count
   Output: Formatted context string

4. LLM Answer Generation
   Model: OpenAI GPT-4o
   Temperature: 0.2 (high precision for legal docs)
   System Prompt: Legal assistant role
   Input: Context + Query
   Output: Answer text

5. Citation Extraction
   Input: Answer text
   Method: Regex pattern matching for [Page X] references
   Output: List of citations with page numbers

6. Source Attribution
   Input: Citations + Retrieved chunks
   Validation: Verify citations reference actual chunks
   Output: Standardized citations
```

**Configuration:**
```
CONTEXT_TOKEN_BUDGET = 12,800 tokens
MIN_QUERY_LENGTH = 3 characters
MIN_CONFIDENCE_SCORE = 0.6 (60% similarity)
RETRIEVAL_TOP_K = 10 chunks
FINAL_CHUNK_COUNT = 4 chunks (for context window)
GPT_MODEL = "gpt-4o"
TEMPERATURE = 0.2 (low for legal precision)
```

**System Prompt:**
```
You are an expert legal assistant specialized in analyzing court documents,
case law, and legal statutes. Your role is to:
1. Answer questions ONLY based on the provided document excerpts
2. Provide precise, factually accurate responses
3. Always cite the exact location in square brackets:
   - For PDFs: [Page X]
   - For Word documents: [Paragraph X]
   - For text files: [Lines X-Y]
4. Distinguish between facts, arguments, and judgments
5. Flag any ambiguities or gaps in the source material
6. Never speculate beyond what the documents state
```

**Key Functions:**
- `query_case(case_id: UUID, question: str, db: Session) -> Dict` - Execute RAG pipeline
- `count_tokens_gpt4o(text: str) -> int` - Token counting
- `search_and_rank_chunks(...)` - Retrieval and ranking
- `format_context(chunks: List[Chunk]) -> str` - Context formatting
- `extract_citations(answer: str) -> List[Dict]` - Citation extraction

**Error Handling:**
- Insufficient context for query
- OpenAI API timeout/rate limits
- Citation mismatch (hallucination detection)
- Empty or invalid queries
- Vector search failures
- Database connection errors

---

### 10. Background Job Processor

**Location:** `backend/services/job_processor.py`, `backend/tasks.py`, `backend/celery_app.py`

**Responsibilities:**
- Process pending document upload jobs asynchronously
- Orchestrate chunking, embedding, and vector store operations
- Manage retry logic with exponential backoff
- Track job status and case status
- Handle errors gracefully with logging
- Provide idempotent processing

**Job Lifecycle:**
```
1. User uploads document
   ↓
2. Case created with status="processing"
3. ProcessingJob created with status="pending"
   ↓
4. Background worker picks up job
5. Job status → "processing"
   ↓
6. Pipeline Execution:
   a. Download PDF from blob storage
   b. Extract and chunk document
   c. Generate embeddings for chunks
   d. Create Qdrant collection for case
   e. Upsert vectors with metadata
   f. Store chunks in PostgreSQL
   ↓
7. On Success:
   - Case status → "ready"
   - Job status → "completed"
   - Frontend notified (via polling or WebSocket)
   ↓
8. On Failure (with Retries):
   - Job attempts incremented
   - Next retry scheduled (0s → 5s → 10s)
   - After max attempts (3): Case status → "error"
   - Error message stored for debugging
```

**Retry Strategy:**
- **Max Attempts:** 3
- **Backoff Delays:** 0s, 5s, 10s
- **Trigger Conditions:** Temporary failures (network, timeouts)
- **No Retry On:** Validation errors, file format errors

**Key Functions:**
- `get_pending_jobs(db: Session, limit: int) -> List[ProcessingJob]` - Fetch pending jobs
- `process_case(case_id: UUID, db: Session) -> Dict` - Execute processing pipeline
- `mark_job_complete(case_id: UUID, db: Session) -> bool` - Mark successful
- `mark_job_failed(case_id: UUID, db: Session, error_msg: str) -> bool` - Mark failed
- `run_worker()` - Main worker loop (polls every 10 seconds)

**Worker Configuration:**
- **Poll Interval:** 10 seconds
- **Batch Size:** 5 jobs per cycle
- **Graceful Shutdown:** Finish current job before stopping
- **Logging:** All operations logged with timestamps

---

### 11. Cache Manager

**Location:** `backend/services/cache_manager.py`

**Responsibilities:**
- Cache query results to Redis
- Manage cache invalidation
- TTL-based expiration (default: 24 hours)
- Track cache hits/misses
- Reduce redundant API calls

**Cache Key Strategy:**
```
Format: "lex_query:{case_id}:{question_hash}"
Example: "lex_query:550e8400-e29b-41d4-a716-446655440000:a1b2c3d4e5f6"

Key Components:
  - case_id: UUID of the case
  - question_hash: SHA256(question) - prevents long keys
```

**TTL Configuration:**
- **Default:** 86400 seconds (24 hours)
- **Min:** 3600 seconds (1 hour)
- **Max:** 2592000 seconds (30 days)
- **Per-Query Override:** Allowed in request

**Key Functions:**
- `get_cached_query(case_id: UUID, question: str) -> Optional[Dict]` - Retrieve from cache
- `cache_query(case_id: UUID, question: str, result: Dict, ttl: int) -> bool` - Store in cache
- `invalidate_case_cache(case_id: UUID) -> bool` - Clear case queries
- `clear_all_cache() -> bool` - Clear all cache

---

## Data Models

### User Lifecycle
```
1. Registration
   - Email + Password → hash
   - Create User record
   - Return confirmation message

2. Login
   - Email + Password → verify
   - Generate JWT token
   - Return token to client

3. Access Control
   - JWT token validates on each request
   - User ID extracted from token
   - Query filter: WHERE user_id = current_user_id

4. Soft Delete (Deletion)
   - Set is_deleted = True
   - Don't remove from database
   - Queries filter out deleted users
```

### Case Lifecycle
```
1. Creation
   - User uploads document
   - Case created with status="processing"
   - File stored in Blob Storage
   - ProcessingJob enqueued

2. Processing
   - Background worker starts processing
   - Chunks created and stored
   - Embeddings generated
   - Vectors stored in Qdrant
   - Case status → "ready"

3. Query
   - User asks question
   - Query embedded and searched
   - LLM generates answer with citations
   - Query record stored with results

4. Deletion
   - Set is_deleted = True (soft delete)
   - Remove from Blob Storage
   - Delete Qdrant collection
   - Clean up chunks and queries
```

### Status Values

**Case Status:**
- `processing` - Document being processed
- `ready` - Ready for queries
- `error` - Processing failed, user notified

**ProcessingJob Status:**
- `pending` - Waiting to be processed
- `processing` - Currently being processed
- `completed` - Successfully processed
- `failed` - Failed after max retries

**File Type:**
- `pdf` - PDF documents
- `docx` - Microsoft Word documents
- `txt` - Plain text files

---

## Data Flow

### Complete Document Upload & Query Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENT UPLOAD FLOW                              │
└─────────────────────────────────────────────────────────────────────┘

1. Frontend Upload
   User selects PDF/DOCX/TXT file
        ↓
2. HTTP POST /cases
   - Validate JWT token
   - Validate file format (magic bytes)
   - Validate file size
   - Extract metadata (filename, size)
        ↓
3. Upload to Blob Storage
   - Upload file to Azure Blob Storage
   - Generate blob path: "cases/{case_id}/{timestamp}"
   - Return blob URL
        ↓
4. Create Case Record
   - Store Case in PostgreSQL:
     {user_id, name, blob_storage_path, status="processing"}
   - Return case_id to frontend
        ↓
5. Enqueue Processing Job
   - Create ProcessingJob: {case_id, status="pending"}
   - Job added to work queue
   - Return status to frontend
        ↓
6. Frontend Polling
   - Poll GET /cases/{case_id} every 2 seconds
   - Display "Processing..." status
        ↓
7. Background Job Processing (Async)
   - Worker picks up pending job
   - Mark as status="processing"
   - Download document from Blob Storage
   - Extract text (PDF/DOCX/TXT specific logic)
   - Chunk document semantically:
     * 800 char chunks with 150 char overlap
     * Preserve page numbers & section names
   - Generate embeddings (OpenAI text-embedding-3-large)
   - Create Qdrant collection: case_{case_id}
   - Upsert vectors with metadata
   - Store chunks in PostgreSQL
   - Mark Case status="ready"
   - Mark Job status="completed"
   - Log completion
        ↓
8. Frontend Detects Ready
   - Poll returns status="ready"
   - Display "Ready for queries"
   - Enable query input
   - Show case details


┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY & RAG FLOW                                  │
└─────────────────────────────────────────────────────────────────────┘

1. User Asks Question
   Input: "What are the plaintiff's key arguments?"
        ↓
2. HTTP POST /cases/{case_id}/ask
   - Validate JWT
   - Validate case_id belongs to user
   - Validate question (min 3 chars)
   - Check cache (Redis key lookup)
        ↓
3. Cache Hit? → Return Cached Answer
        ↓ (if miss)
4. Embed Question
   - Convert question to 3072-dim vector
   - Use same embedding model as documents
        ↓
5. Vector Similarity Search (Qdrant)
   - Query collection: case_{case_id}
   - Cosine similarity search
   - Filter: score ≥ 0.6
   - Retrieve: top 10 chunks
        ↓
6. Context Window Management
   - Sort by relevance + document order
   - Select top 4 chunks (highest scores)
   - Count tokens in selected chunks
   - Validate total ≤ 12,800 tokens
   - Format as context string with metadata
        ↓
7. LLM Answer Generation (OpenAI GPT-4o)
   System Prompt: "You are a legal assistant..."
   Context: [4 chunks with page numbers]
   Query: "What are the plaintiff's key arguments?"
   Temperature: 0.2 (high precision)
        ↓
8. LLM Response
   Answer: "The plaintiff argues that... [Page 5]
            Additionally... [Page 7]"
        ↓
9. Citation Extraction
   - Parse answer for [Page X] references
   - Validate citations match retrieved chunks
   - Extract structured citations
   - Detect hallucinations (citations not in chunks)
        ↓
10. Store Query Record
    - Create Query entry:
      {case_id, user_id, question, answer, citations}
    - Store in PostgreSQL
    - Log to audit trail
        ↓
11. Cache Answer
    - Store in Redis with 24-hour TTL
    - Key: "lex_query:{case_id}:{question_hash}"
        ↓
12. Return to Frontend
    {
      "answer": "The plaintiff argues that...",
      "citations": [
        {"page": 5, "content": "..."},
        {"page": 7, "content": "..."}
      ],
      "sources": [
        {"chunk_id": "...", "page": 5, "section": "Arguments"}
      ]
    }
        ↓
13. Frontend Display
    - Show answer text
    - Highlight citation references
    - Display source PDF pages
    - Show query history
```

### Error Recovery Flows

**Document Processing Failure:**
```
Processing starts
    ↓
Exception occurs (e.g., network timeout)
    ↓
Error logged with full traceback
    ↓
Mark job failed with attempt count
    ↓
Attempt < max_attempts (3)?
    ↓ YES: Schedule retry
    - Next retry: 0s (1st), 5s (2nd), 10s (3rd)
    - Job status remains "pending"
    - Worker will re-pick this job
    ↓ NO: Mark job failed permanently
    - Case status → "error"
    - Error message stored
    - User notified in frontend
```

**Query Processing Failure:**
```
Query embedding fails (OpenAI API timeout)
    ↓
Return error to user
    ↓ (cached result available)
    - Return last known good answer
    ↓ (no cache)
    - Return: "Service temporarily unavailable"

Vector search fails (Qdrant down)
    ↓
Fallback: Return error message
    (Note: Could add keyword search fallback)
```

---

## API Layer Design

### Request/Response Flow

```
Client Request
    ↓
FastAPI Route Handler
    ↓
Pydantic Validation
    └─→ Invalid? Return 400 with error details
    ↓
Authentication Check (JWT)
    └─→ Invalid token? Return 401
    └─→ Expired? Return 401 with refresh hint
    ↓
Authorization Check (User ID match)
    └─→ Access denied? Return 403
    ↓
Service Layer Call
    ├─ Logging (input + params)
    ├─ Database transaction
    ├─ Error handling
    └─ Logging (output + results)
    ↓
Response Serialization
    └─ Convert ORM models to Pydantic schemas
    ↓
HTTP Response (200, 400, 401, 403, 404, 500)
    ↓
Client
```

### Authentication Flow

```
1. Registration
   POST /auth/register
   {email: "user@example.com", password: "secure"}
   ↓
   Validate email format
   Validate password strength
   Hash password with bcrypt
   Create User record
   Return: {user_id, email, created_at}

2. Login
   POST /auth/login
   {email: "user@example.com", password: "secure"}
   ↓
   Lookup User by email
   Verify password against hash
   Generate JWT token (HS256)
   Token payload: {user_id, email, exp: now+1440min}
   Return: {access_token, token_type: "bearer"}

3. Protected Endpoint
   GET /user/profile
   Header: Authorization: Bearer {token}
   ↓
   Extract token from header
   Decode token with SECRET_KEY
   Validate signature
   Validate expiry (not past)
   Extract user_id from token
   Fetch user from database
   Return user profile
```

### Pagination & Filtering

**Cases List:**
```
GET /cases?skip=0&limit=10&status=ready

Query Parameters:
  skip: int = 0 (offset)
  limit: int = 10 (max results)
  status: str = None (filter: processing/ready/error)

Returns:
{
  "total": 25,
  "items": [Case, ...],
  "skip": 0,
  "limit": 10
}
```

**Query History:**
```
GET /cases/{case_id}/queries?skip=0&limit=20

Returns:
{
  "total": 45,
  "items": [Query, ...],
  "skip": 0,
  "limit": 20
}
```

---

## Component Relationships

### Dependency Graph

```
FastAPI (main.py)
├── Auth Service
│   └── Config
├── Storage Service
│   ├── Config
│   └── Azure Blob Storage API
├── Chunking Service
│   ├── Text Extraction Service
│   └── LangChain RecursiveCharacterTextSplitter
├── Embeddings Service
│   ├── Config
│   ├── OpenAI API
│   └── Embedding Cache Manager
├── Vector Store Service
│   ├── Qdrant API
│   └── Config
├── RAG Query Engine
│   ├── Embeddings Service
│   ├── Vector Store Service
│   ├── OpenAI API (GPT-4o)
│   └── Database Models
├── Job Processor
│   ├── Storage Service
│   ├── Chunking Service
│   ├── Embeddings Service
│   ├── Vector Store Service
│   └── Database Models
└── Cache Manager
    └── Redis
```

### Service Communication

**Synchronous (Direct Calls):**
- API Layer → Auth Service
- API Layer → RAG Query Engine
- RAG Query Engine → Embeddings Service
- RAG Query Engine → Vector Store Service

**Asynchronous (via Job Queue):**
- API Layer enqueues → Job Processor worker
- Job Processor → Storage Service
- Job Processor → Chunking Service
- Job Processor → Embeddings Service
- Job Processor → Vector Store Service

**External APIs:**
- Any Service → OpenAI (embeddings, LLM)
- Any Service → Azure Blob Storage
- Any Service → Qdrant (vector DB)
- Cache Manager → Redis

---

## Component Lifecycle

### Startup Sequence

```
1. Application Start
   ↓
2. Load Configuration (config.py)
   - Read .env file
   - Validate required variables
   - Parse CORS origins
   ↓
3. Database Connection (database.py)
   - Create SQLAlchemy engine
   - Configure connection pooling
   - Test connection to PostgreSQL
   ↓
4. Alembic Migrations (if needed)
   - Check schema version
   - Run pending migrations
   - Update to latest schema
   ↓
5. FastAPI Startup Events
   - Initialize CORS middleware
   - Validate SECRET_KEY
   - Register route handlers
   ↓
6. Background Worker Start (job_processor.py)
   - Connect to database
   - Connect to Redis (if using Celery)
   - Start worker loop
   - Poll for pending jobs every 10 seconds
   ↓
7. System Ready
   - API accepting requests on port 8000
   - Worker processing jobs
   - All external services connected
```

### Shutdown Sequence

```
1. Signal Received (SIGTERM/SIGINT)
   ↓
2. Stop Accepting New Requests
   ↓
3. Finish Current Operations
   - Complete active request
   - Finish current job (if processing)
   ↓
4. Close Connections
   - Close database connections
   - Close Redis connection
   - Flush any pending logs
   ↓
5. Save State
   - No unsaved data in memory
   - All transactions committed/rolled back
   ↓
6. Exit Gracefully
```

---

## Integration Points

### External Service Integrations

**1. Azure Blob Storage**
- **Purpose:** Store raw PDF/DOCX/TXT documents
- **Authentication:** Connection string in config
- **Operations:** Upload, download, delete
- **Failure Handling:** Retry with exponential backoff
- **Integration Point:** `backend/services/storage.py`

**2. OpenAI APIs**
- **Embeddings:** text-embedding-3-large (3072 dims)
  - Purpose: Convert text → vectors
  - Cost: $0.02 per 1M tokens
  - Batching: 20 chunks per request
  - Integration: `backend/services/embeddings.py`
- **LLM:** GPT-4o for answer generation
  - Purpose: Generate contextual answers
  - Temperature: 0.2 (legal precision)
  - Context: ~12.8K tokens
  - Integration: `backend/services/rag_engine.py`

**3. Qdrant Vector Database**
- **Purpose:** Store and search embeddings
- **Collections:** One per case (case_{case_id})
- **Search:** Cosine similarity, top-K retrieval
- **Operations:** Create, upsert, search, delete
- **Integration:** `backend/services/vector_store.py`

**4. PostgreSQL Database**
- **Purpose:** Store metadata and audit trail
- **Models:** User, Case, Chunk, Query, ProcessingJob
- **Connection:** SQLAlchemy ORM
- **Transactions:** ACID compliance for consistency
- **Integration:** `backend/database.py`, `backend/models.py`

**5. Redis Cache**
- **Purpose:** Cache query results, async task queue
- **Query Cache:** 24-hour TTL
- **Task Queue:** Celery broker for job processing
- **Integration:** `backend/services/cache_manager.py`, `backend/celery_app.py`

### Frontend Integration Points

**Frontend:** Next.js + React + TypeScript
**Backend API:** FastAPI on port 8000

**API Communication:**
```
Frontend                    Backend
   │                          │
   ├─ POST /auth/register ────→
   ├─ POST /auth/login ───────→
   ├─ POST /cases (upload) ───→
   ├─ GET /cases ─────────────→
   ├─ GET /cases/{id} ────────→
   ├─ POST /cases/{id}/ask ───→
   └─ GET /health ────────────→
```

---

## Cross-References

### Related Documentation

1. **TECH_STACK.md**
   - Detailed technology choices and alternatives
   - Dependency versions and compatibility matrix
   - Infrastructure requirements

2. **FLOWCHARTS.md**
   - Mermaid diagrams for all major flows
   - State machines for job processing
   - Sequence diagrams for component interaction

3. **RAG_PIPELINE.md**
   - Deep-dive into RAG mechanics
   - Retrieval ranking algorithms
   - LLM prompt engineering
   - Citation validation strategies

4. **FILE_REFERENCE.md**
   - Complete function reference guide
   - Parameter and return type documentation
   - Error handling specifications
   - Code location index

### Implementation Notes

**Backend Structure:**
```
backend/
├── main.py                    # FastAPI app, route handlers
├── auth.py                    # Password hashing, JWT
├── config.py                  # Settings management
├── database.py                # SQLAlchemy setup
├── models.py                  # ORM models
├── schemas.py                 # Pydantic schemas
├── exceptions.py              # Custom exceptions
├── validators.py              # Input validation
├── celery_app.py              # Celery configuration
├── tasks.py                   # Celery task definitions
├── services/
│   ├── storage.py             # Azure Blob Storage
│   ├── chunking.py            # Document chunking
│   ├── embeddings.py          # OpenAI embeddings
│   ├── embedding_cache.py     # Embedding cache
│   ├── vector_store.py        # Qdrant integration
│   ├── rag_engine.py          # RAG pipeline
│   ├── job_processor.py       # Background jobs
│   ├── cache_manager.py       # Redis caching
│   └── text_extraction.py     # Text extraction
└── alembic/
    └── versions/              # Database migrations
```

### Performance Metrics

- **Document Chunking:** ~2-3 seconds for average legal doc
- **Embedding Generation:** ~100ms per chunk (batched)
- **Vector Search:** <100ms for top-K retrieval
- **LLM Answer Generation:** 1-3 seconds
- **Total Query Latency:** ~3-5 seconds end-to-end
- **Cache Hit:** <50ms (return cached result)

### Scalability Considerations

**Horizontal Scaling:**
- Multiple backend instances behind load balancer
- Shared PostgreSQL database
- Shared Qdrant vector DB
- Shared Redis cache
- Multiple job processor workers

**Vertical Scaling:**
- Increase chunk size (8KB limit)
- Increase batch embedding size
- Optimize query timeout
- Connection pooling tuning

**Future Enhancements:**
- Document reranking with cross-encoders
- Query expansion techniques
- Semantic caching (embed cache results)
- Hybrid search (vector + keyword)
- Case similarity matching
- Multi-query decomposition

---

## Summary

**LexIntel's** architecture is designed for:

✅ **Production-Ready:** Comprehensive error handling, retry logic, audit logging
✅ **Scalable:** Async processing, background jobs, horizontal scaling support
✅ **Secure:** JWT auth, data isolation, password hashing, input validation
✅ **Maintainable:** Clear separation of concerns, documented components, consistent patterns
✅ **Legal-Grade:** Compliance-focused, audit trails, data retention
✅ **RAG-Optimized:** Semantic chunking, intelligent retrieval, citation validation

The system successfully combines FastAPI backend, PostgreSQL database, Qdrant vector store, and OpenAI APIs to provide intelligent legal document analysis with proper source attribution and comprehensive audit trails.

---

**Document Version:** 1.0
**Last Updated:** February 2026
**Status:** Complete and Production-Ready
