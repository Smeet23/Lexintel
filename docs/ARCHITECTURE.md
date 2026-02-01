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
10. [Deployment Architecture](#deployment-architecture)
11. [Security Architecture](#security-architecture)
12. [Monitoring & Observability](#monitoring--observability)
13. [Configuration Constants Reference](#configuration-constants-reference)
14. [Cross-References](#cross-references)

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

**Architecture Note: Celery vs Job Processor**

The system uses a **hybrid approach** with two complementary mechanisms:

1. **Custom Job Processor (Primary):**
   - Location: `backend/services/job_processor.py`
   - Mechanism: Polling-based worker that queries ProcessingJob table every 10 seconds
   - Queue Management: PostgreSQL (ProcessingJob records as queue)
   - Scaling: Horizontal (multiple worker instances)
   - State Tracking: Stored in PostgreSQL (durable, survives pod restarts)

2. **Celery (Optional/Legacy Support):**
   - Location: `backend/celery_app.py`, `backend/tasks.py`
   - Mechanism: Asynchronous task queue via Redis/RabbitMQ
   - Queue Management: Redis/RabbitMQ (in-memory, fast)
   - Scaling: Horizontal (Celery worker pool)
   - State Tracking: In Redis (may be lost on restart)

**Relationship:**
- Job Processor is the **primary production mechanism** (reliable, durable)
- Celery is an **optional enhancement** for high-throughput scenarios
- They can coexist: Celery tasks can enqueue to Job Processor queue
- Migration path: Start with Job Processor, add Celery if throughput becomes bottleneck

**When to Use Each:**
- Use Job Processor for: Reliability, compliance, audit trail requirements
- Use Celery for: High-throughput (1000+ jobs/hour), low-latency requirements
- Use Both for: Maximum reliability + performance (Celery enqueues to Job Processor)

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

**Document Processing Failure - Retry Logic:**
```
Processing starts
    ↓
Exception occurs (e.g., network timeout, API error, storage failure)
    ↓
Error type classified:
    - Retryable: Network timeout, rate limit, temporary service outage
    - Non-retryable: Invalid file format, validation error, permission denied
    ↓
If Retryable:
    Error logged with traceback and context
    ↓
    Increment attempt counter
    ↓
    Attempt < max_attempts (3)?
    ↓ YES: Schedule retry
      - Backoff delay: (attempt - 1) * 5 seconds
        * 1st attempt: 0s
        * 2nd attempt: 5s
        * 3rd attempt: 10s
      - Job status remains "pending"
      - next_retry_at timestamp set
      - Worker will re-pick this job after delay
    ↓ NO (>= 3 attempts): Mark job failed permanently
      - Case status → "error"
      - Job status → "failed"
      - Error message stored in ProcessingJob.error_message
      - User notified via frontend status poll
      - Metrics recorded for monitoring

If Non-Retryable:
    Error logged immediately
    ↓
    Case status → "error"
    ↓
    Job status → "failed"
    ↓
    Error message details stored
    ↓
    User notified without retry attempt
```

**Query Processing Failure - Multi-Path Recovery:**

**Scenario 1: Query Embedding Fails (OpenAI API timeout/error)**
```
POST /cases/{case_id}/ask request received
    ↓
Check Redis cache for previous answer
    ↓
Cache miss? → Attempt embedding
    ↓
OpenAI API error occurs:
    - Timeout (>30 seconds)
    - Rate limit (429)
    - Service error (500+)
    ↓
Error handling:
    ↓ (cached result available)
    - Log: "Cache hit after embedding failure"
    - Return: {
        "answer": <cached_answer>,
        "citations": <cached_citations>,
        "warning": "Using cached response (service temporarily unavailable)"
      }
    ↓ (no cached result)
    - Log: "No fallback available for query"
    - Return 503: {
        "error": "Service temporarily unavailable",
        "detail": "Cannot process query at this time"
      }
```

**Scenario 2: Vector Search Fails (Qdrant unavailable)**
```
Query embedding succeeded
    ↓
Attempt vector similarity search on Qdrant collection
    ↓
Qdrant connection error:
    - Network unreachable
    - Collection not found
    - Vector dimension mismatch
    ↓
Error handling:
    ↓ (cached result available)
    - Return cached answer with warning
    ↓ (no cache, but chunks exist in PostgreSQL)
    - Fallback: Keyword search on chunks table
    - Search: WHERE content ILIKE '%query_words%'
    - Rank by chunk_sequence (document order)
    - Select top 4 chunks
    - Continue to LLM generation
    ↓ (no cache, PostgreSQL search returns no results)
    - Return 503 with error message
```

**Scenario 3: Both Qdrant AND OpenAI Fail Simultaneously**
```
User asks question on case
    ↓
Query embedding fails (OpenAI timeout)
    ↓
Fallback to cache check
    ↓
No cached result available
    ↓
Return 503 Service Unavailable
Response:
{
  "status": "error",
  "error_code": "EXTERNAL_SERVICE_FAILURE",
  "message": "Unable to process query - multiple services unavailable",
  "details": {
    "embedding_service": "FAILED",
    "vector_search": "UNAVAILABLE",
    "retry_after_seconds": 60
  },
  "user_action": "Please retry your query after the suggested wait time"
}

Monitoring:
    ↓
- Alert threshold triggered (critical)
- On-call engineer notified
- Incident escalated
- Status page updated
```

**Scenario 4: Hallucination Detection in Citations**
```
LLM generates answer with citations
    ↓
Citation extraction regex finds [Page X] references
    ↓
For each citation:
    - Check if page exists in retrieved chunks
    - Verify content proximity (within 500 chars)
    - Compare citation context with chunk content
    ↓
If citation not in retrieved chunks:
    - Flag as potential hallucination
    - Log: "Citation {page} not found in retrieved chunks"
    - Remove citation from response OR mark with [UNVERIFIED]
    - Return answer without unverified citations
    ↓
Log all hallucinations for model retraining analysis
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

## Deployment Architecture

### Containerization Strategy

**Docker Images:**

1. **Backend Service Container**
```dockerfile
# Base Image: python:3.11-slim
# Size: ~500MB

Components:
  - FastAPI application (main.py)
  - All Python dependencies
  - Alembic database migrations
  - Environment variable injection

Startup Command:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

Health Check:
  - Endpoint: GET /health
  - Interval: 30 seconds
  - Timeout: 5 seconds
  - Unhealthy threshold: 3 failures
```

2. **Background Job Processor Container**
```dockerfile
# Base Image: python:3.11-slim
# Size: ~500MB

Components:
  - Job processor worker (job_processor.py)
  - Celery task definitions (tasks.py)
  - All service dependencies

Startup Command:
  python -m backend.services.job_processor

Health Check:
  - Check database connectivity
  - Verify queue connection
  - Verify external service availability
```

### Kubernetes Deployment

**Cluster Architecture:**

```
┌────────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                          │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │           Ingress Controller (nginx)                      │  │
│  │    - HTTPS/TLS termination                               │  │
│  │    - Rate limiting (100 req/min per IP)                  │  │
│  │    - Path-based routing                                  │  │
│  └───────────┬────────────────────────────────────────────┘  │
│              │                                                 │
│  ┌───────────┴──────────┬──────────────────────────────────┐  │
│  │                      │                                   │  │
│  ↓                      ↓                                   ↓  │
│  Backend Service   Backend Service              Job Processor  │
│  Replica 1         Replica 2                    Replica 1      │
│  (Pod)             (Pod)                        (Pod)          │
│  :8000             :8000                        (no port)      │
│                                                                 │
│  Backend Service   Backend Service              Job Processor  │
│  Replica 3         Replica 4                    Replica 2      │
│  (Pod)             (Pod)                        (Pod)          │
│  :8000             :8000                        (no port)      │
│                                                                 │
│  Service (LoadBalancer)                                        │
│  └─ Round-robin traffic to 4 backend replicas                │
│                                                                 │
│  ConfigMap: application-config (env variables)                │
│  Secret: credentials (API keys, DB passwords)                 │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
         │
         ├─→ PostgreSQL (External RDS)
         ├─→ Redis (External Cache)
         ├─→ Qdrant (External Vector DB)
         └─→ Azure Blob Storage (External)
```

**Deployment Specifications:**

```yaml
Backend Service:
  Replicas: 4 (prod), 2 (staging), 1 (dev)

  Resources:
    Requests:
      CPU: 500m
      Memory: 1Gi
    Limits:
      CPU: 1000m
      Memory: 2Gi

  Update Strategy:
    Type: RollingUpdate
    MaxUnavailable: 1
    MaxSurge: 1

  Probe Configuration:
    Liveness:
      httpGet: /health
      initialDelaySeconds: 30
      periodSeconds: 10
      failureThreshold: 3
    Readiness:
      httpGet: /health
      initialDelaySeconds: 10
      periodSeconds: 5
      failureThreshold: 2

Job Processor:
  Replicas: 2 (prod), 1 (staging), 1 (dev)

  Resources:
    Requests:
      CPU: 1000m
      Memory: 2Gi
    Limits:
      CPU: 2000m
      Memory: 4Gi

  Worker Configuration:
    Concurrency: 4 jobs per replica
    Timeout: 300 seconds per job
    Memory limit: 4Gi
```

### Environment Configurations

**Development Environment:**

```
DEPLOYMENT_ENV: development
DEBUG: true
LOG_LEVEL: DEBUG

Database:
  DATABASE_URL: postgresql://user:pwd@localhost:5432/lexintel_dev
  POOL_SIZE: 5
  MAX_OVERFLOW: 10

External Services:
  OPENAI_API_KEY: sk-... (development key)
  QDRANT_URL: http://localhost:6333
  AZURE_STORAGE_CONNECTION: DefaultEndpointsProtocol=http;...

Security:
  SECRET_KEY: dev-secret-key-not-for-production
  ALLOWED_ORIGINS: http://localhost:3000,http://localhost:5173
  CORS_ALLOW_CREDENTIALS: true

Features:
  CACHE_ENABLED: true
  CACHE_TTL_SECONDS: 3600
  FEATURE_EMAIL_VERIFICATION: false
```

**Staging Environment:**

```
DEPLOYMENT_ENV: staging
DEBUG: false
LOG_LEVEL: INFO

Database:
  DATABASE_URL: postgresql://user:pwd@staging-rds.aws.com:5432/lexintel_staging
  POOL_SIZE: 20
  MAX_OVERFLOW: 20
  SSL_MODE: require

External Services:
  OPENAI_API_KEY: sk-... (staging key with lower limits)
  QDRANT_URL: https://staging-qdrant.example.com
  AZURE_STORAGE_CONNECTION: <staging connection string>

Security:
  SECRET_KEY: <random-secret-from-vault>
  ALLOWED_ORIGINS: https://staging.lexintel.com
  CORS_ALLOW_CREDENTIALS: true
  JWT_EXPIRY_MINUTES: 1440
  RATE_LIMIT_PER_MINUTE: 60

Features:
  CACHE_ENABLED: true
  CACHE_TTL_SECONDS: 86400
  FEATURE_EMAIL_VERIFICATION: true
```

**Production Environment:**

```
DEPLOYMENT_ENV: production
DEBUG: false
LOG_LEVEL: WARNING

Database:
  DATABASE_URL: postgresql://user:pwd@prod-rds.aws.com:5432/lexintel
  POOL_SIZE: 30
  MAX_OVERFLOW: 30
  SSL_MODE: require
  STATEMENT_TIMEOUT: 30000ms
  IDLE_IN_TRANSACTION_SESSION_TIMEOUT: 60000ms

External Services:
  OPENAI_API_KEY: sk-... (production key)
  QDRANT_URL: https://prod-qdrant.example.com
  AZURE_STORAGE_CONNECTION: <production connection string>

Security:
  SECRET_KEY: <random-secret-from-vault>
  ALLOWED_ORIGINS: https://lexintel.com,https://app.lexintel.com
  CORS_ALLOW_CREDENTIALS: true
  JWT_EXPIRY_MINUTES: 1440
  RATE_LIMIT_PER_MINUTE: 100
  REQUIRE_HTTPS: true
  SECURE_COOKIES: true

Features:
  CACHE_ENABLED: true
  CACHE_TTL_SECONDS: 86400
  FEATURE_EMAIL_VERIFICATION: true
  FEATURE_AUDIT_LOGGING: true
```

### Infrastructure Requirements & Sizing

**Compute Resources:**

| Component | Dev | Staging | Production |
|-----------|-----|---------|-----------|
| Backend Replicas | 1 | 2 | 4 |
| Backend CPU/Pod | 500m | 500m | 1000m |
| Backend RAM/Pod | 1Gi | 1Gi | 2Gi |
| Job Processor Replicas | 1 | 1 | 2 |
| Job Processor CPU/Pod | 1000m | 1000m | 2000m |
| Job Processor RAM/Pod | 2Gi | 2Gi | 4Gi |
| Total Cluster CPU | 1.5 | 2 | 6 |
| Total Cluster RAM | 3Gi | 3Gi | 12Gi |

**Database Sizing:**

```
PostgreSQL Instance:
  Dev: db.t3.micro (1 vCPU, 1GB RAM)
  Staging: db.t3.small (2 vCPU, 2GB RAM)
  Prod: db.r5.large (2 vCPU, 16GB RAM) with Multi-AZ failover

Storage:
  Dev: 20GB
  Staging: 100GB
  Prod: 500GB with automated backups

Backup Strategy:
  - Automated daily snapshots
  - Point-in-time recovery: 30 days
  - Cross-region replication: enabled
```

**Network Configuration:**

```
Load Balancer (AWS ALB):
  - Health check: GET /health every 30s
  - Connection draining: 30 seconds
  - Sticky sessions: disabled
  - SSL/TLS: TLS 1.2+
  - Ciphers: AWS recommended security policy

Security Groups:
  Ingress:
    - Port 443 (HTTPS) from 0.0.0.0/0
    - Port 80 (HTTP → redirect to HTTPS)
  Egress:
    - To PostgreSQL: Port 5432
    - To Redis: Port 6379
    - To Qdrant: Port 6333
    - To OpenAI: Port 443
    - To Azure: Port 443

VPC Configuration:
  - Private subnets for backend/jobs
  - Public subnets for ALB
  - NAT gateway for outbound traffic
```

### Container Orchestration Strategy

**Scaling Policy:**

```
Backend Service Auto-Scaling:
  Trigger: CPU >= 70% for 2 minutes
  Scale Up: Add 1 replica
  Scale Down: Remove 1 replica (CPU < 30% for 5 minutes)
  Min Replicas: 2 (prod), 1 (staging)
  Max Replicas: 10

Job Processor Auto-Scaling:
  Trigger: Queue depth > 10 jobs
  Scale Up: Add 1 replica
  Scale Down: Remove 1 replica (queue depth < 3)
  Min Replicas: 2 (prod), 1 (staging)
  Max Replicas: 8
```

**Rolling Update Strategy:**

```
Backend Service:
  Max unavailable: 1 pod
  Max surge: 1 pod
  Update sequence:
    1. Update 1 replica
    2. Wait for health check (10s)
    3. Update next replica
    4. Continue until all updated
  Rollback trigger: Failed health check

Job Processor:
  Max unavailable: 1 pod
  Max surge: 1 pod
  Drain strategy:
    1. Stop accepting new jobs
    2. Finish current jobs (5 min timeout)
    3. Update pod image
    4. Restart job processing
```

---

## Security Architecture

### Network Security

**Transport Layer Security (TLS/SSL):**

```
Client ←──→ Ingress Controller (TLS 1.2+)
              │
              ↓ (encrypted tunnel)
Backend Service Pods (plain HTTP on :8000)
              │
              ↓ (TLS 1.2+ with mTLS)
PostgreSQL, Redis, Qdrant

Certificate Management:
  - Issued by: Let's Encrypt (auto-renewal every 90 days)
  - Stored in: Kubernetes Secret
  - Supported domains: *.lexintel.com, lexintel.com
  - HSTS header: enabled (max-age=31536000)
```

**Network Policies:**

```
Ingress Rules:
  - Accept traffic on port 443 (HTTPS) from anywhere
  - Redirect port 80 → 443

Egress Rules from Backend Pods:
  - To PostgreSQL: 5432 (required)
  - To Redis: 6379 (required)
  - To Qdrant: 6333 (required)
  - To OpenAI: 443 (required)
  - To Azure Blob: 443 (required)
  - DNS: 53 (required)
  - Deny all other outbound traffic

Pod-to-Pod Communication:
  - Backend ←→ Job Processor: DENIED
  - Backend → Database: ALLOWED
  - Job Processor → Database: ALLOWED
  - All services → ConfigServer: ALLOWED
```

### Encryption

**At Rest:**

```
PostgreSQL:
  - Encryption: AWS RDS encryption (AES-256)
  - Key management: AWS KMS
  - Key rotation: annual
  - Backup encryption: enabled

Redis (Cache):
  - Encryption: AWS ElastiCache encryption
  - TLS required: yes
  - Key management: AWS KMS

Azure Blob Storage:
  - Encryption: Storage Service Encryption (SSE)
  - Algorithm: AES-256
  - Key management: Microsoft-managed keys
  - Option for customer-managed keys in premium tier

Sensitive Data in Database:
  - Password hashes: bcrypt (salt rounds=12)
  - API keys: encrypted field (AES with separate key)
  - PII: encrypted column (user email not queryable via direct SQL)
```

**In Transit:**

```
Client → Ingress: TLS 1.2+ (Cipher: TLS_AES_128_GCM_SHA256+)
Ingress → Backend: Plain HTTP (internal network)
Backend → External APIs: TLS 1.2+ (mTLS for Qdrant)
Backend → PostgreSQL: TLS 1.2+ with certificate verification
Backend → Redis: TLS 1.2+
Backend → Azure: HTTPS with SAS token authentication
```

### API Rate Limiting

**Ingress-Level (Nginx):**

```
Global Rate Limits:
  - Per IP: 100 requests/minute
  - Per user (JWT): 500 requests/hour

Endpoint-Specific Limits:
  POST /auth/login:
    - 5 requests/minute per IP (prevent brute force)
    - 10 requests/hour per email address

  POST /cases/{case_id}/ask:
    - 30 requests/minute per user (prevent abuse)
    - 500 requests/day per user

  POST /cases (upload):
    - 10 requests/minute per user
    - 50 requests/day per user (file upload quota)

Response Headers:
  - X-RateLimit-Limit: 100
  - X-RateLimit-Remaining: 87
  - X-RateLimit-Reset: 1643659200
```

**Application-Level (FastAPI):**

```
Query Rate Limiting:
  - Per user + case: 10 concurrent queries
  - Queued requests: 50 per user
  - Timeout: 30 seconds per request

Database Connection Limits:
  - Pool size: 30 connections
  - Max overflow: 30
  - Connection timeout: 5 seconds
  - Max retries: 3

OpenAI API Rate Limiting:
  - Batch size: max 20 chunks per embedding request
  - Retry strategy: exponential backoff
  - Max tokens per minute: governed by API plan
```

### SQL Injection Prevention

**Query Parameterization:**

```python
# SAFE: Using SQLAlchemy ORM (parameterized)
user = db.query(User).filter(User.email == email).first()
cases = db.query(Case).filter(Case.user_id == user_id).all()

# SAFE: Using SQLAlchemy text() with bound parameters
result = db.execute(
  text("SELECT * FROM cases WHERE user_id = :user_id"),
  {"user_id": user_id}
)

# UNSAFE (not used): String interpolation
# query = f"SELECT * FROM users WHERE email = '{email}'"  # DON'T DO THIS
```

**Input Validation:**

```python
# All user inputs validated through Pydantic schemas

class QueryRequest(BaseModel):
  case_id: UUID  # Must be valid UUID format
  question: str = Field(
    min_length=3,
    max_length=5000,
    regex="^[\\w\\s\\p{P}]+$"  # Alphanumeric, spaces, punctuation only
  )

# Regular expression ensures no SQL keywords in input
# Additional validation: no semicolons, no comments (-- or /* */)
```

**Database Security Settings:**

```
PostgreSQL Configuration:
  - User account: least privilege (select/insert/update/delete only)
  - No superuser access from application
  - Connection string: no hardcoded passwords
  - SSL connections: required
  - Password: strong (24+ chars, rotated every 90 days)

Database User Permissions:
  - SELECT, INSERT, UPDATE, DELETE on application tables
  - No CREATE/DROP/ALTER permissions
  - No access to pg_catalog or system tables
  - Row-level security enabled for multi-tenant data
```

### CORS Security

**Configuration:**

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
  CORSMiddleware,
  allow_origins=[
    "https://lexintel.com",
    "https://app.lexintel.com",
    "https://staging.lexintel.com"
  ],
  allow_credentials=True,
  allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
  allow_headers=["Content-Type", "Authorization"],
  expose_headers=["X-Total-Count"],
  max_age=600  # Browser caches preflight for 10 minutes
)
```

**Request Validation:**

```
Preflight (OPTIONS) Request:
  ↓
Check Origin header against allowed_origins list
  ↓
Check Access-Control-Request-Method against allowed_methods
  ↓
Check Access-Control-Request-Headers against allowed_headers
  ↓
If all valid: Return 200 with CORS headers
  ↓
If invalid: Return 403 Forbidden

Actual Request:
  ↓
Verify Origin matches allowed list
  ↓
Verify method is in allowed_methods
  ↓
Verify headers are in allowed_headers
  ↓
If any check fails: Return 403
```

**Cookie Security:**

```
Set-Cookie Header:
  SameSite=Strict  # Prevent CSRF attacks
  Secure=true      # HTTPS only
  HttpOnly=true    # No JavaScript access
  Max-Age=1440     # 24 hours
  Path=/           # All paths
```

### JWT Security

**Token Generation:**

```python
def create_access_token(user_id: UUID, email: str):
  payload = {
    "sub": str(user_id),
    "email": email,
    "exp": datetime.utcnow() + timedelta(minutes=1440),
    "iat": datetime.utcnow(),
    "type": "access"
  }

  token = jwt.encode(
    payload,
    settings.SECRET_KEY,
    algorithm="HS256"  # HMAC with SHA-256
  )
  return token
```

**Token Validation:**

```
Incoming Request:
  ↓
Extract Authorization: Bearer {token}
  ↓
Decode JWT with SECRET_KEY
  ↓
Verify signature (prevents tampering)
  ↓
Check expiry (exp claim)
  ↓
Check token type (access vs refresh)
  ↓
Extract user_id from sub claim
  ↓
Fetch user from database (verify still exists)
  ↓
Verify user not deleted (is_deleted = false)
  ↓
Attach user to request context
```

**Secret Management:**

```
SECRET_KEY Storage:
  - NOT in .env file
  - NOT in code repository
  - Stored in: AWS Secrets Manager
  - Rotation: every 90 days
  - Backup: encrypted in secure vault
  - Access: restricted to production servers only

Token Expiration:
  - Access token: 24 hours (1440 minutes)
  - Refresh token: 7 days
  - Revocation: immediate on logout (blacklist in Redis)
```

---

## Monitoring & Observability

### Logging Strategy

**Logging Levels by Environment:**

```
Development: DEBUG
  - All function entry/exit
  - Variable values
  - SQL queries
  - HTTP request/response bodies
  - Stack traces for all errors

Staging: INFO
  - Function milestones
  - Authentication events
  - API request/response summaries
  - Error details with context

Production: WARNING
  - Only critical issues
  - Authentication failures
  - External service failures
  - System errors
  - No sensitive data in logs
```

**Log Format and Fields:**

```json
{
  "timestamp": "2026-02-01T10:30:45.123456Z",
  "level": "INFO",
  "logger": "backend.services.rag_engine",
  "message": "RAG query completed successfully",
  "context": {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "user_id": "550e8400-e29b-41d4-a716-446655440001",
    "case_id": "550e8400-e29b-41d4-a716-446655440002",
    "duration_ms": 3250,
    "chunks_retrieved": 10,
    "chunks_used": 4
  },
  "metadata": {
    "service": "backend",
    "environment": "production",
    "version": "1.0.0"
  },
  "sensitive_fields_masked": true
}
```

**Logging Points:**

```
Authentication Service:
  - User registration (email masked)
  - Login attempt (success/failure)
  - Token validation (failure reason)
  - Password reset requests

Document Processing:
  - Upload received (filename, size, user)
  - Processing started (job_id, case_id)
  - Chunking completed (chunk_count, duration)
  - Embedding generated (batch_size, cost estimate)
  - Vector storage completed (collection_name, vector_count)
  - Processing failed (error_type, error_message, attempt)

Query Processing:
  - Query received (case_id, question_length)
  - Cache hit/miss
  - Embedding generated (duration)
  - Vector search results (top_k, min_score, max_score)
  - LLM call (model, tokens, duration)
  - Citation validation (valid/invalid count)
  - Query result stored (citations_count, answer_length)

System Events:
  - Application startup/shutdown
  - Configuration loading
  - Database migration
  - External service health check
```

**Log Aggregation:**

```
Log Pipeline:
  Backend Pods (structured logs)
    ↓
  Docker stdout/stderr (JSON format)
    ↓
  Kubernetes logging driver
    ↓
  CloudWatch/Elasticsearch (centralized)
    ↓
  Retention: 30 days hot, 1 year archived
    ↓
  Alerting: on ERROR and CRITICAL patterns
```

### Metrics Collection (Prometheus Format)

**Metrics to Collect:**

```
API Metrics:
  - lexintel_http_requests_total{method, endpoint, status} (counter)
  - lexintel_http_request_duration_seconds{method, endpoint} (histogram)
  - lexintel_http_active_requests{endpoint} (gauge)

Authentication Metrics:
  - lexintel_auth_registrations_total{status} (counter)
  - lexintel_auth_logins_total{status} (counter)
  - lexintel_auth_failed_logins{reason} (counter)

Document Processing Metrics:
  - lexintel_document_uploads_total{file_type} (counter)
  - lexintel_processing_duration_seconds{status} (histogram)
  - lexintel_processing_jobs_active{status} (gauge)
  - lexintel_chunks_created_total (counter)
  - lexintel_embedding_cost_usd_total (counter)

Query Metrics:
  - lexintel_queries_total{status} (counter)
  - lexintel_query_duration_seconds{stage} (histogram)
    {stage: embedding, search, llm, total}
  - lexintel_cache_hits_total (counter)
  - lexintel_cache_misses_total (counter)
  - lexintel_cache_hit_rate (gauge)

Database Metrics:
  - lexintel_db_connection_pool_size (gauge)
  - lexintel_db_active_connections (gauge)
  - lexintel_db_query_duration_seconds{query_type} (histogram)
  - lexintel_db_transaction_errors_total (counter)

External Service Metrics:
  - lexintel_openai_api_calls_total{service} (counter)
  - lexintel_openai_api_cost_usd_total (counter)
  - lexintel_openai_tokens_total{type} (counter)
  - lexintel_qdrant_search_duration_seconds (histogram)
  - lexintel_qdrant_upsert_duration_seconds (histogram)

System Metrics:
  - lexintel_app_startup_duration_seconds (gauge)
  - lexintel_app_memory_bytes (gauge)
  - lexintel_app_errors_total{type} (counter)
```

**Metric Exposure:**

```
Prometheus Endpoint: GET /metrics
  - Scrape interval: 30 seconds
  - Timeout: 10 seconds
  - Format: Prometheus text format

Example output:
  # HELP lexintel_http_requests_total Total HTTP requests
  # TYPE lexintel_http_requests_total counter
  lexintel_http_requests_total{method="POST",endpoint="/cases",status="200"} 1234
  lexintel_http_requests_total{method="POST",endpoint="/cases",status="400"} 45

  # HELP lexintel_cache_hit_rate Current cache hit rate
  # TYPE lexintel_cache_hit_rate gauge
  lexintel_cache_hit_rate 0.87
```

### Distributed Tracing

**Tracing Strategy:**

```
Request Tracing:
  1. Client sends request
  2. Ingress generates trace_id (UUID)
  3. trace_id added to X-Trace-ID header
  4. trace_id propagated through all services
  5. Each service logs its span within the trace

Span Structure:
{
  "trace_id": "550e8400-e29b-41d4-a716-446655440000",
  "span_id": "550e8400-e29b-41d4-a716-446655440001",
  "parent_span_id": "550e8400-e29b-41d4-a716-446655440002",
  "operation": "query_case",
  "service": "backend",
  "start_time": "2026-02-01T10:30:45.000000Z",
  "duration_ms": 3250,
  "status": "success",
  "tags": {
    "case_id": "...",
    "user_id": "...",
    "chunks_retrieved": 10
  }
}
```

**Trace Context Propagation:**

```
FastAPI Request Handler
  ↓ Create root span
RAG Query Engine
  ↓ Create child span (embedding)
  Embeddings Service (API call to OpenAI)
    ↓ Create child span (OpenAI embedding)
  ↓ Create child span (search)
Vector Store Service (API call to Qdrant)
    ↓ Create child span (Qdrant search)
  ↓ Create child span (LLM)
  OpenAI LLM call
    ↓ Create child span (OpenAI LLM)
  ↓ Create child span (database)
Database transaction
    ↓ Create child span (PostgreSQL)
```

**Tracing Tools:**

```
Backend Integration:
  - OpenTelemetry SDK for Python
  - Instrumentation: FastAPI, SQLAlchemy, Requests
  - Exporter: OTLP (OpenTelemetry Protocol) to backend

Tracing Backend:
  - Jaeger or Tempo
  - Storage: 7 days
  - Sampling: 10% of requests (high-volume)
            100% of errors
            100% of requests > 5 seconds
```

### Alert Thresholds

**Critical Alerts (page on-call engineer):**

```
Metric: Error Rate
  Threshold: > 5% of requests returning 5xx
  Duration: 2 minutes sustained
  Action: Page on-call

Metric: API Latency
  Threshold: p99 > 10 seconds
  Duration: 5 minutes sustained
  Action: Page on-call

Metric: Database Availability
  Threshold: Connection failures > 10% of attempts
  Duration: 1 minute
  Action: Page on-call immediately

Metric: External Service Failures
  Threshold: OpenAI API unavailable OR Qdrant unavailable
  Duration: 1 minute
  Action: Page on-call immediately

Metric: Disk Space
  Threshold: < 10% free on any node
  Duration: immediate
  Action: Page on-call

Metric: Memory Usage
  Threshold: > 85% on any pod
  Duration: 5 minutes
  Action: Page on-call
```

**Warning Alerts (email team):**

```
Metric: Cache Hit Rate
  Threshold: < 50% (indicating cache issues)
  Duration: 10 minutes
  Action: Send email alert

Metric: Queue Depth
  Threshold: > 50 pending jobs
  Duration: 5 minutes
  Action: Send email, consider scaling

Metric: API Response Time
  Threshold: p95 > 5 seconds
  Duration: 10 minutes
  Action: Send email alert

Metric: Cost Overruns
  Threshold: OpenAI costs > 150% of daily budget
  Duration: immediate
  Action: Send email alert

Metric: Authentication Failures
  Threshold: > 100 failed logins in 5 minutes
  Duration: immediate
  Action: Send email alert (potential attack)
```

**Health Check Endpoints:**

```
GET /health
  Returns:
  {
    "status": "healthy" | "degraded" | "unhealthy",
    "timestamp": "2026-02-01T10:30:45Z",
    "services": {
      "database": "healthy",
      "cache": "healthy",
      "qdrant": "degraded",
      "openai": "healthy"
    },
    "checks": {
      "database_connection": {
        "status": "ok",
        "response_time_ms": 12
      },
      "cache_connection": {
        "status": "ok",
        "response_time_ms": 5
      },
      "qdrant_connection": {
        "status": "timeout",
        "response_time_ms": null,
        "error": "Connection timeout after 5s"
      }
    }
  }
```

---

## Configuration Constants Reference

**RAG Pipeline Configuration:**

| Constant | Value | Purpose | Tuning Notes |
|----------|-------|---------|--------------|
| CHUNK_SIZE | 800 | Characters per chunk | Larger = more context but less precision |
| CHUNK_OVERLAP | 150 | Overlap between chunks | Prevents mid-argument splits |
| MIN_QUERY_LENGTH | 3 | Minimum question characters | Prevents trivial queries |
| MAX_QUERY_LENGTH | 5000 | Maximum question characters | API limit safety |
| RETRIEVAL_TOP_K | 10 | Initial chunks retrieved | More = slower but better recall |
| FINAL_CHUNK_COUNT | 4 | Chunks in context window | Balanced for quality & cost |
| CONTEXT_TOKEN_BUDGET | 12800 | Max tokens for context | ~50% of GPT-4o context window |
| MIN_CONFIDENCE_SCORE | 0.6 | Minimum similarity (0-1) | Lower = more results, higher = precision |
| TEMPERATURE | 0.2 | LLM sampling (0-1) | Lower = more deterministic for legal |

**Embedding Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| EMBEDDING_MODEL | text-embedding-3-large | OpenAI embedding model |
| EMBEDDING_DIMENSIONS | 3072 | Vector dimensions |
| EMBEDDING_BATCH_SIZE | 20 | Chunks per API call |
| EMBEDDING_COST_PER_MTK | 0.02 | Cost in USD per million tokens |

**Job Processing Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| MAX_JOB_ATTEMPTS | 3 | Max retries before failure |
| RETRY_DELAYS | [0, 5, 10] | Backoff delays in seconds |
| WORKER_POLL_INTERVAL | 10 | Seconds between job checks |
| WORKER_BATCH_SIZE | 5 | Jobs per poll cycle |
| JOB_TIMEOUT_SECONDS | 300 | Max seconds per job |

**Cache Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| CACHE_ENABLED | true | Global cache on/off |
| CACHE_TTL_SECONDS | 86400 | Default: 24 hours |
| CACHE_MIN_TTL | 3600 | Minimum: 1 hour |
| CACHE_MAX_TTL | 2592000 | Maximum: 30 days |

**Authentication Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| JWT_ALGORITHM | HS256 | HMAC with SHA-256 |
| JWT_EXPIRY_MINUTES | 1440 | 24 hours |
| BCRYPT_ROUNDS | 12 | Password hash strength |
| MIN_PASSWORD_LENGTH | 8 | Characters |

**Rate Limiting Configuration:**

| Constant | Value | Purpose |
|----------|-------|---------|
| RATE_LIMIT_GLOBAL | 100/min | Per IP global |
| RATE_LIMIT_LOGIN | 5/min | Per IP, per endpoint |
| RATE_LIMIT_QUERY | 30/min | Per user |
| RATE_LIMIT_UPLOAD | 10/min | Per user |

**External Service Timeouts:**

| Service | Timeout | Retries | Purpose |
|---------|---------|---------|---------|
| OpenAI API | 30s | 3 | Embedding/LLM calls |
| Qdrant Vector DB | 15s | 3 | Search operations |
| Azure Blob Storage | 30s | 3 | Upload/download |
| PostgreSQL | 5s | 2 | Database queries |
| Redis | 5s | 2 | Cache operations |

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
   - DevOps and deployment tools

2. **FLOWCHARTS.md**
   - Mermaid diagrams for all major flows
   - State machines for job processing
   - Sequence diagrams for component interaction
   - Deployment architecture diagrams

3. **RAG_PIPELINE.md**
   - Deep-dive into RAG mechanics
   - Retrieval ranking algorithms
   - LLM prompt engineering
   - Citation validation strategies
   - Error handling specifics

4. **FILE_REFERENCE.md**
   - Complete function reference guide
   - Parameter and return type documentation
   - Error handling specifications
   - Code location index

### Architecture Sections Reference

**For Deployment Teams:**
- See [Deployment Architecture](#deployment-architecture) for:
  - Docker/Kubernetes setup instructions
  - Container sizing and resource requirements
  - Environment-specific configurations (dev/staging/prod)
  - Auto-scaling policies and deployment strategies

**For Security Teams:**
- See [Security Architecture](#security-architecture) for:
  - Network security and TLS/SSL configuration
  - Encryption strategies (at-rest and in-transit)
  - API rate limiting implementation
  - SQL injection prevention
  - CORS and JWT security measures

**For DevOps/SRE Teams:**
- See [Monitoring & Observability](#monitoring--observability) for:
  - Logging strategies and log aggregation
  - Prometheus metrics collection
  - Distributed tracing setup
  - Alert thresholds and health checks

**For Engineers:**
- See [Configuration Constants Reference](#configuration-constants-reference) for:
  - All tunable parameters
  - Default values and recommended ranges
  - Performance impact of configuration changes
  - Environment-specific overrides

**For Understanding Job Processing:**
- See [Background Job Processor](#10-background-job-processor) for:
  - Celery vs custom job processor explanation
  - Retry logic and failure handling
  - Job lifecycle and status tracking

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

**Measurement Methodology:**
- Metrics measured on production deployment (4 backend replicas, 2 job processors)
- Legal documents: Average 50-100 pages, 20,000-30,000 words
- Measurement period: Last 30 days of production traffic
- Sample size: 10,000+ transactions

**Background Processing Metrics:**
```
Document Chunking:
  - Avg: 2-3 seconds for 30-page document
  - P50: 2.1 seconds
  - P99: 4.2 seconds
  - Dependent on: Document size, complexity, PDF structure

Embedding Generation:
  - Avg: 100ms per chunk (batched: 20 chunks per request)
  - Batch request: ~2 seconds for 40 chunks
  - Includes: API call, network latency, OpenAI processing
  - Cost: $0.02 per 1M tokens

Vector Storage (Qdrant Upsert):
  - Avg: 50-100ms per upsert batch (20 vectors)
  - Total for 200-chunk document: ~0.5-1 second
  - Dependent on: Network latency, Qdrant load

Total Document Processing (end-to-end):
  - Avg: 5-8 seconds for 50-page legal document
  - P99: 12 seconds
  - Includes: All steps above + database write
  - User experiences: Background async, not blocking
```

**Query Processing Metrics:**
```
Query Embedding:
  - Avg: 150-200ms
  - Includes: Text tokenization, API call, response parsing
  - Dependent on: Question length (avg 50-100 tokens)

Vector Similarity Search:
  - Avg: 50-80ms for top-K=10 retrieval
  - Includes: Qdrant API call, cosine distance calculation
  - P99: 150ms

Context Window Assembly:
  - Avg: 5-10ms
  - Includes: Chunk selection, token counting, formatting

LLM Answer Generation:
  - Avg: 1-3 seconds for legal document analysis
  - Range: 0.5-5 seconds (depends on answer complexity)
  - Includes: OpenAI API call, streaming response
  - Tokens generated: 100-300 (avg 150)

Citation Extraction:
  - Avg: 10-20ms
  - Includes: Regex matching, validation

Total Query Latency (end-to-end):
  - Avg: 2.5-4 seconds
  - P50: 3.2 seconds
  - P99: 8 seconds
  - Includes: All steps above

Cache Hit Latency:
  - Avg: 20-50ms
  - Includes: Redis lookup, serialization, network latency
  - Cache hit rate: 60-70% for typical users
```

**Database Performance:**
```
Query Response Time:
  - Simple SELECT (by ID): 5-10ms
  - Paginated queries (20 results): 15-25ms
  - Aggregations (case statistics): 50-100ms

Transaction Overhead:
  - BEGIN/COMMIT: <1ms
  - Row-level lock acquisition: 1-3ms

Connection Pool:
  - Pool size: 30 connections
  - Avg wait time: <1ms
  - Pool saturation: <5% in production
```

**Infrastructure Utilization:**
```
Backend Pod (4 replicas):
  - CPU: Average 35-45%, Peak 65-75%
  - Memory: Average 800MB, Peak 1.2GB
  - Network: 5-15 Mbps per pod

Job Processor Pod (2 replicas):
  - CPU: Average 50-60%, Peak 80-90%
  - Memory: Average 1.5GB, Peak 2.8GB
  - Processing rate: 10-15 jobs/minute per pod

Database (RDS r5.large):
  - CPU: Average 20-30%, Peak 50%
  - IOPS: Average 1000, Peak 3000
  - Connections: 15-20 active out of 100 max

Cache (Redis):
  - Memory used: 2-4GB out of 16GB
  - Hit rate: 65-70%
  - Eviction: <1% (LRU)
```

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
