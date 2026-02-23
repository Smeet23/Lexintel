# LexIntel - Architecture Documentation

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Component Details](#component-details)
4. [Data Models](#data-models)
5. [API Design](#api-design)
6. [Data Flow](#data-flow)
7. [Integration Points](#integration-points)
8. [Security Architecture](#security-architecture)
9. [Performance Characteristics](#performance-characteristics)
10. [Scalability & Future Considerations](#scalability--future-considerations)

---

## System Overview

**LexIntel** is a production-ready **Retrieval-Augmented Generation (RAG)** system designed for intelligent legal document analysis and case management. It enables lawyers and legal professionals to:

- Upload legal documents (PDFs) with automatic semantic processing
- Ask natural language questions about their cases
- Receive AI-generated answers with precise source citations
- Manage multiple cases with background job processing

### Core Value Propositions

1. **Semantic Understanding**: Documents are chunked intelligently and converted to vector embeddings for context-aware retrieval
2. **Source Attribution**: All answers include citations with specific page numbers from source documents
3. **Async Processing**: Background workers handle time-consuming document processing without blocking the API
4. **Enterprise-Grade**: Production-ready with comprehensive error handling, retry logic, and monitoring

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
│                    (Frontend/Mobile Apps)                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTPS REST API
                         ↓
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                           │
├──────────────┬──────────────────┬──────────────┬────────────────┤
│ Auth Service │ Case Management  │ RAG Engine   │ Health/Status  │
│ - Register   │ - Upload PDF     │ - Query Case │ - Monitoring   │
│ - Login      │ - List Cases     │ - Get Answer │                │
│ - JWT Auth   │ - Get Status     │ - Citations  │                │
└──────────────┴──────────────────┴──────────────┴────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  PostgreSQL  │  │ Async Queue  │  │ Cloud Storage│
│  (Metadata)  │  │  (Redis/Cel) │  │ (Azure Blob) │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │
        └────────────────┼────────────────┐
                         ↓                ↓
                  ┌──────────────┐  ┌──────────────┐
                  │   Qdrant     │  │ Google AI API│
                  │(Vector Store)│  │(Embeddings+  │
                  │              │  │ LLM Answer)  │
                  └──────────────┘  └──────────────┘
```

### Logical Layers

#### 1. **Presentation Layer**
- FastAPI REST API endpoints
- Input validation via Pydantic schemas
- JWT-based authentication

#### 2. **Application Layer**
- Business logic service classes
- Orchestration of external services
- Error handling and recovery

#### 3. **Data Layer**
- PostgreSQL for structured metadata
- Qdrant for vector embeddings
- Azure Blob Storage for PDF files

#### 4. **External Services**
- Google AI APIs (embeddings + LLM)
- Azure Blob Storage
- Redis for async task queue

---

## Component Details

### 1. Authentication Service

**Location**: `backend/auth.py`

**Responsibilities**:
- User registration with email/password
- Password hashing using bcrypt
- JWT token generation and validation
- Secure token expiry management

**Key Functions**:
```python
hash_password(password: str) → str              # Bcrypt hashing
verify_password(plain: str, hashed: str) → bool # Verification
create_access_token(data: dict) → str           # JWT generation
decode_token(token: str) → Optional[str]        # JWT validation
```

**Security Features**:
- 1440-minute (24-hour) token expiry
- HS256 signing algorithm
- Password complexity validation (8+ chars, uppercase, digit)
- Constant-time token comparison

---

### 2. Database Layer

**Location**: `backend/database.py`, `backend/models.py`

#### Connection Management
- SQLAlchemy engine with connection pooling (`pool_pre_ping=True`)
- Async session support for non-blocking I/O
- Automatic session cleanup via FastAPI dependency injection

#### Data Models

**User Model**
```
id (UUID, PK)
├─ email (String, unique, indexed)
├─ password_hash (String)
├─ is_deleted (Boolean) → soft delete flag
├─ created_at (DateTime)
├─ updated_at (DateTime)
└─ relationships
   └─ cases: One-to-Many → Case
```

**Case Model**
```
id (UUID, PK)
├─ user_id (UUID, FK) → User
├─ name (String[255])
├─ blob_storage_path (String) → Azure path
├─ status (Enum: processing|ready|error)
├─ created_at (DateTime)
├─ updated_at (DateTime)
├─ indexes
│  └─ (user_id, status) composite
└─ relationships
   ├─ user: Many-to-One → User
   ├─ chunks: One-to-Many → Chunk
   ├─ queries: One-to-Many → Query
   └─ processing_jobs: One-to-Many → ProcessingJob
```

**Chunk Model** (Document segments)
```
id (UUID, PK)
├─ case_id (UUID, FK) → Case
├─ page_num (Integer)
├─ section_name (String[255])
├─ content (Text) → full chunk text
├─ embedding_hash (String[64]) → SHA256 deduplication
├─ chunk_sequence (Integer) → order in document
├─ created_at (DateTime)
├─ indexes
│  └─ (case_id, chunk_sequence) composite
└─ relationship
   └─ case: Many-to-One → Case
```

**Query Model** (Q&A history)
```
id (UUID, PK)
├─ case_id (UUID, FK) → Case
├─ user_id (UUID, FK) → User
├─ question (String)
├─ answer (Text)
├─ citations (JSON) → list of [Page X] references
├─ created_at (DateTime)
├─ indexes
│  └─ (case_id, created_at) composite
└─ relationships
   ├─ case: Many-to-One → Case
   └─ user: Many-to-One → User
```

**ProcessingJob Model** (Async task tracking)
```
id (UUID, PK)
├─ case_id (UUID, FK) → Case
├─ status (Enum: pending|processing|completed|failed)
├─ error_message (String, nullable)
├─ attempts (Integer) → retry counter
├─ max_attempts (Integer, default=3)
├─ created_at (DateTime)
├─ started_at (DateTime, nullable)
├─ completed_at (DateTime, nullable)
├─ next_retry_at (DateTime, nullable)
└─ relationship
   └─ case: Many-to-One → Case
```

---

### 3. Storage Service

**Location**: `backend/services/storage.py`

**Responsibilities**:
- PDF file uploads to Azure Blob Storage
- Secure PDF validation
- File download and deletion

**Key Functions**:
```python
validate_pdf(file_content: bytes) → bool
upload_pdf_to_blob(content, case_id, filename) → str
download_pdf_from_blob(blob_path) → bytes
delete_blob(blob_path) → bool
```

**Storage Path Structure**:
```
cases/
└─ {case_id}/
   └─ {filename}.pdf
```

**Security**:
- Magic byte validation (`%PDF` header)
- Filename validation (no path traversal)
- Azure authentication via connection string

---

### 4. Document Chunking Service

**Location**: `backend/services/chunking.py`

**Responsibilities**:
- Extract text from PDF documents
- Intelligently split text into semantic chunks
- Preserve document structure metadata

**Configuration**:
```python
CHUNK_SIZE = 1500 characters      # ~200-250 words
CHUNK_OVERLAP = 300 characters     # Context preservation
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Priority order
```

**Key Functions**:
```python
chunk_pdf(pdf_path: str) → List[Dict]
chunk_pdf_from_blob(blob_content: bytes) → List[Dict]
estimate_tokens(content: str) → int  # Token approximation
```

**Chunking Strategy**:

1. Extract text from PDF using PyMuPDF (fitz)
2. Apply LangChain's `RecursiveCharacterTextSplitter`
3. Split hierarchically: paragraph → sentence → word
4. Preserve metadata: page number, section name
5. Estimate token count for cost tracking

**Output Format**:
```python
{
    "content": "Chunk text content...",
    "page_num": 5,
    "section_name": "Section 2.1"
}
```

**Why This Approach**:
- Semantic chunking preserves legal argument continuity
- Overlap prevents important context loss at boundaries
- Hierarchical splitting maintains natural language structure
- Token estimation enables cost prediction

---

### 5. Embeddings Service

**Location**: `backend/services/embeddings.py`

**Responsibilities**:
- Generate vector embeddings for text chunks
- Manage Google AI API interactions
- Cost estimation and caching

**Configuration**:
```python
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768           # Vector size
EMBEDDING_COST = Free tier (Google AI)
```

**Key Functions**:
```python
get_embeddings_client() → GoogleGenerativeAIEmbeddings  # Cached client
embed_text(text: str) → List[float]         # Single embedding
embed_chunks(chunks: List[str]) → List[List[float]]  # Batch
estimate_embedding_cost(text_length: int) → float
```

**Why gemini-embedding-001**:
- 768-dimensional vectors capture semantic nuance
- Superior performance on legal/technical documents
- Free tier (Google AI) for cost-effective usage
- Strong performance in domain-specific retrieval

**Error Handling**:
- Retry on rate limits
- Graceful degradation on API failures
- Input validation (no empty strings)
- Batch size validation

---

### 6. Vector Store Service

**Location**: `backend/services/vector_store.py`

**Responsibilities**:
- Manage Qdrant vector database collections
- Store and retrieve document embeddings
- Semantic similarity search

**Configuration**:
```python
VECTOR_SIZE = 768                   # Matches embedding model
DISTANCE_METRIC = "Cosine"          # Similarity measure
Collection naming: case_{case_id}   # One collection per case
```

**Key Functions**:
```python
create_collection(case_id: str) → bool
upsert_vectors(case_id, chunks, embeddings) → int
search_vectors(case_id, query_embedding, limit=5) → List[Dict]
delete_collection(case_id: str) → bool
```

**Deterministic Point ID Generation**:
```python
point_id = int(MD5("{case_id}:{chunk_id}").hexdigest(), 16) % 2^63
```
- Ensures same chunk always gets same ID
- Enables idempotent operations
- Prevents duplicate vector storage

**Metadata Payload**:
```python
{
    "chunk_id": "uuid",
    "page_num": 5,
    "section_name": "Section 2.1",
    "content_preview": "First 200 characters..."
}
```

**Search Response**:
```python
{
    "score": 0.92,              # Cosine similarity
    "chunk_id": "uuid",
    "page_num": 5,
    "content": "Full chunk text...",
    "section_name": "Section 2.1"
}
```

---

### 7. RAG Engine

**Location**: `backend/services/rag_engine.py`

**Responsibilities**:
- Orchestrate complete RAG pipeline
- Query embedding and semantic search
- Context window management
- LLM integration and answer generation

**Configuration**:
```python
CONTEXT_TOKEN_BUDGET = 12,800      # Max context for Gemini 2.5 Flash Lite
MIN_QUERY_LENGTH = 3               # Minimum query length
MIN_CONFIDENCE_SCORE = 0.15        # Retrieval threshold
RETRIEVAL_TOP_K = 10               # Initial retrieval count
FINAL_CHUNK_COUNT = 4              # Chunks in final context
TEMPERATURE = 0.2                  # Low randomness for legal docs
```

**8-Step RAG Pipeline**:

```
1. Query Validation
   └─ Check length (≥3 chars)

2. Query Embedding
   └─ Convert question to 768-d vector

3. Semantic Search
   └─ Search Qdrant for top 10 similar chunks

4. Confidence Filtering
   └─ Keep only chunks with score ≥ 0.15

5. Context Formatting
   └─ Sort by relevance
   └─ Format with metadata (pages, scores)
   └─ Count tokens to fit in budget

6. Token Budget Management
   └─ If context exceeds 12,800 tokens
      └─ Trim to 2 chunks
      └─ If still exceeds, return error

7. LLM Answer Generation
   └─ Call Google AI Gemini 2.5 Flash Lite with context
   └─ Temperature: 0.2 for legal precision
   └─ Timeout: 30 seconds

8. Citation Extraction
   └─ Parse [Page X] references
   └─ Validate citations match sources
   └─ Flag potential hallucinations
```

**System Prompt** (Legal Assistant):
```
You are an expert legal analyst. Your role is to:
1. Answer ONLY based on provided document excerpts
2. Always cite specific page numbers [Page X]
3. Distinguish facts from arguments from judgments
4. Flag ambiguities and information gaps
5. Never speculate beyond provided content
6. Provide structured, precise responses
```

**Key Functions**:
```python
count_tokens(text: str) → int
format_legal_context(chunks, case_name) → str
embed_query(query: str) → List[float]
retrieve_chunks(case_id, query_embedding) → List[Dict]
extract_citations(answer: str, chunks) → List[Dict]
generate_answer(query, context, temperature) → Tuple[str, int]
query_case(case_id, query, db, top_k=4) → Dict
```

**Response Structure**:
```python
{
    "answer": "Generated legal answer...",
    "sources": [
        {
            "chunk_id": "uuid",
            "page_num": 5,
            "relevance_score": 0.92,
            "content_preview": "..."
        }
    ],
    "case_id": "uuid",
    "query": "Original question",
    "model": "gemini-2.5-flash-lite",
    "tokens_used": 1234,
    "confidence": "high|medium|low|none",
    "error": null
}
```

**Confidence Levels**:
- "high": Average retrieval score ≥ 0.9
- "medium": Average retrieval score ≥ 0.8
- "low": Average retrieval score < 0.8
- "none": Retrieval or processing error

---

### 8. Background Job Processor

**Location**: `backend/services/job_processor.py`, `backend/tasks.py`

**Responsibilities**:
- Queue and process long-running document tasks
- Manage retry logic and error recovery
- Coordinate status updates across system

**Architecture**:
- **Queue**: Redis via Celery
- **Workers**: Background processes
- **Batch Processing**: 5 jobs per cycle, 10s sleep

**Job Lifecycle**:
```
User uploads PDF
    ↓
ProcessingJob created (status: pending)
    ↓
Worker picks up job (status: pending → processing)
    ↓
Full pipeline execution:
    ├─ Download PDF from Blob
    ├─ Chunk document
    ├─ Generate embeddings
    ├─ Create Qdrant collection
    ├─ Upsert vectors
    └─ Store chunks in DB
    ↓
Success branch:
    ├─ Case status: processing → ready
    ├─ Job status: processing → completed
    └─ User notified

Failure branch:
    ├─ Attempt counter incremented
    ├─ Job status: processing → pending (for retry)
    ├─ Next retry scheduled: 0s → 5s → 10s
    ├─ If max_attempts exceeded
    │  └─ Job status: pending → failed
    │  └─ Case status: processing → error
    └─ User notified of error
```

**Retry Strategy**:
```python
Attempt 1: Immediate retry (0s delay)
Attempt 2: 5 second delay
Attempt 3: 10 second delay
Failure:   Mark job as failed, update case status
```

**Idempotency**:
- Existing chunks deleted before reprocessing
- Same chunk IDs generated (deterministic)
- Safe to retry without data duplication

---

### 9. API Endpoints

**Location**: `backend/main.py`

#### Authentication Endpoints

```
POST /auth/register
├─ Request: { email, password }
├─ Response: { access_token, token_type }
└─ Validation: Email format, password complexity

POST /auth/login
├─ Request: { email, password }
├─ Response: { access_token, token_type }
└─ Validation: User exists, password correct

GET /user/profile (protected)
├─ Headers: Authorization: Bearer {token}
├─ Response: { id, email, created_at }
└─ Validation: Valid JWT token
```

#### Case Management Endpoints

```
POST /cases (protected)
├─ Request: multipart/form-data
│  ├─ file: PDF (max size: configurable)
│  └─ name: Case name (string)
├─ Response: { id, name, status, blob_storage_path }
├─ Validations:
│  ├─ MIME type is application/pdf
│  ├─ PDF magic bytes (%PDF)
│  ├─ Filename no path traversal
│  ├─ Filename max 255 chars
│  └─ Case name 1-255 chars
└─ Status: "processing" (async job queued)

GET /cases (protected)
├─ Query: user_id (implicit from JWT)
├─ Response: [{ id, name, status, created_at }]
└─ Sorted: created_at DESC

GET /cases/{case_id} (protected)
├─ Response: Case details + processing progress
├─ Validation: User ownership
└─ Status: processing|ready|error

GET /cases/{case_id}/status (protected)
├─ Response: { status, chunks_count, error_message }
└─ Updated: Real-time from ProcessingJob
```

#### RAG Query Endpoints

```
POST /cases/{case_id}/ask (protected)
├─ Request: { question }
├─ Response: {
│  "answer": string,
│  "sources": [{ chunk_id, page_num, score, preview }],
│  "confidence": "high|medium|low|none",
│  "tokens_used": int,
│  "error": string or null
│ }
├─ Validations:
│  ├─ User ownership of case
│  ├─ Case status is "ready"
│  ├─ Question length 1-5000 chars
│  └─ Case has chunks
└─ Processing: Synchronous (3-5s typical)
```

#### Administrative Endpoints

```
GET /health
├─ Response: { status: "ok|degraded|error" }
└─ Checks: Database, Redis, Qdrant, Google AI API
```

---

## Data Flow

### Document Upload & Processing Flow

```
1. Client → POST /cases
   ├─ Upload PDF file
   ├─ Validate: MIME type, magic bytes, filename
   └─ Validate: Case name length

2. API Layer
   ├─ Create Case record (status: "processing")
   ├─ Upload PDF to Azure Blob (cases/{case_id}/{filename})
   └─ Queue Celery task (process_document_task)

3. Redis Queue
   └─ Task waiting for worker

4. Celery Worker
   ├─ Retrieve: PDF from Blob Storage
   ├─ Process: Chunk PDF (1500 char, 300 overlap)
   ├─ Embed: Chunks via Google AI API (768-d vectors)
   ├─ Store: Chunks in PostgreSQL
   ├─ Vector: Create Qdrant collection
   ├─ Upsert: Vectors with metadata to Qdrant
   ├─ Update: Case status to "ready"
   └─ Update: ProcessingJob status to "completed"

5. Client (polling GET /cases/{case_id}/status)
   ├─ Status: "processing" → "ready"
   └─ Notification: Document ready for queries
```

**Error Handling in Processing**:
```
Failure → Increment attempts
   ↓
attempt < max_attempts?
   ├─ YES: Schedule next retry (exponential backoff)
   │       status: pending
   │       next_retry_at: now + delay
   │
   └─ NO:  Mark job failed
          status: failed
          Case status: error
          User notification
```

### Query & Answer Generation Flow

```
1. Client → POST /cases/{case_id}/ask
   └─ Question: "What are the key obligations?"

2. RAG Engine - Query Processing
   ├─ Validate: Question length ≥ 3 chars
   ├─ Embed: Question → 768-d vector (Google AI)
   └─ Search: Qdrant vector similarity (top 10)

3. Qdrant - Semantic Search
   ├─ Query: Find top 10 most similar chunks
   ├─ Score: Cosine similarity (0.0-1.0)
   └─ Return: Chunks with scores

4. RAG Engine - Filtering & Formatting
   ├─ Filter: Keep score ≥ 0.15
   ├─ Sort: By relevance (highest first)
   ├─ Format: Add page numbers, scores, context
   ├─ Count: Token budget calculation
   └─ Trim: If exceeds 12,800 tokens, reduce chunks

5. LLM - Answer Generation
   ├─ Prompt: System (legal assistant) + Question + Context
   ├─ Model: gemini-2.5-flash-lite with temperature=0.2
   ├─ Token limit: 2000 output tokens
   └─ Timeout: 30 seconds

6. RAG Engine - Citation Extraction
   ├─ Parse: [Page X] references from answer
   ├─ Match: Citations to source chunks
   ├─ Flag: Unmatched citations (hallucination warning)
   └─ Calculate: Confidence level (high/medium/low/none)

7. Response to Client
   ├─ answer: Generated response with citations
   ├─ sources: List of chunks used
   ├─ confidence: high|medium|low|none
   └─ tokens_used: Total tokens consumed
```

**Token Budget Management Example**:
```
Context calculation:
├─ Query tokens: 50
├─ System prompt: 500
├─ Context chunks (4): 8000
├─ Buffer: 500
└─ Total: 9050 tokens (within 12,800 budget ✓)

If total > 12,800:
├─ Trim to 2 chunks: Now ~5,200 tokens ✓
└─ If still > 12,800: Return error (rare)
```

---

## Integration Points

### 1. Google AI Integration

**Services Used**:
- `gemini-embedding-001`: Document and query embeddings
- `gemini-2.5-flash-lite`: Answer generation

**API Calls**:
```python
# Embeddings
POST https://generativelanguage.googleapis.com/v1/models/gemini-embedding-001:embedContent
├─ Model: gemini-embedding-001
├─ Input: Text or batch of texts
└─ Output: 768-dimensional vectors

# Answer Generation
POST https://generativelanguage.googleapis.com/v1/models/gemini-2.5-flash-lite:generateContent
├─ Model: gemini-2.5-flash-lite
├─ Messages: System prompt + User query + Context
├─ Temperature: 0.2
└─ Max tokens: 2000
```

**Error Handling**:
- Rate limit retries with exponential backoff
- Token limit validation before API calls
- Timeout protection (30 seconds)

### 2. Azure Blob Storage Integration

**Container Structure**:
```
container: cases/
├─ {case_id_1}/
│  └─ document.pdf
├─ {case_id_2}/
│  └─ contract.pdf
└─ {case_id_3}/
   └─ legal_brief.pdf
```

**Operations**:
```python
# Upload
BlobServiceClient.get_blob_client(container, path).upload_blob(data)

# Download
BlobServiceClient.get_blob_client(container, path).download_blob().readall()

# Delete
BlobServiceClient.get_blob_client(container, path).delete_blob()
```

### 3. Qdrant Vector Database Integration

**Collection per Case**:
```
Collections:
├─ case_uuid_1
├─ case_uuid_2
└─ case_uuid_3
```

**REST API Endpoints**:
```
POST   /collections/{collection_name}/points        # Upsert vectors
GET    /collections/{collection_name}/points/search # Semantic search
DELETE /collections/{collection_name}               # Delete collection
```

**Search Query Format**:
```json
{
  "vector": [float × 768],
  "limit": 10,
  "score_threshold": 0.15
}
```

### 4. Redis Integration

**Purpose**: Message broker for Celery task queue

**Usage**:
- Task queue (default: 0)
- Result backend (default: 1)

**Celery Configuration**:
```python
broker_url = "redis://localhost:6379/0"
result_backend = "redis://localhost:6379/1"
task_serializer = "json"
result_serializer = "json"
```

---

## Security Architecture

### Authentication & Authorization

**Flow**:
```
1. User registration
   ├─ Email validation (valid format)
   ├─ Password validation (8+ chars, uppercase, digit)
   ├─ Hash password with bcrypt (salt included)
   └─ Store in database

2. User login
   ├─ Verify email exists
   ├─ Verify password with bcrypt
   ├─ Generate JWT token (HS256)
   ├─ Token includes: user_id, expiry (24 hours)
   └─ Return to client

3. Protected requests
   ├─ Client sends Authorization: Bearer {token}
   ├─ API extracts token from header
   ├─ Verify JWT signature with SECRET_KEY
   ├─ Verify token not expired
   ├─ Retrieve user from database
   ├─ Verify user not deleted
   └─ Grant access to protected resource
```

### Input Validation

**Layers**:
```
1. Content-Type validation
   └─ Only accept application/pdf for file uploads

2. MIME type validation
   └─ Verify Content-Type header matches file

3. Magic byte validation (PDF)
   └─ Check first bytes are "%PDF"

4. Filename validation
   ├─ Block path traversal: ../, ~/, /
   ├─ Max length: 255 characters
   └─ Must end with .pdf

5. Pydantic schema validation
   └─ Email format, password requirements, etc.

6. Business logic validation
   ├─ Case name length: 1-255 chars
   ├─ Question length: 1-5000 chars
   └─ Case ownership verification
```

### Data Security

**Password Storage**:
```python
# Bcrypt hashing
password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12))

# Verification
bcrypt.checkpw(plain_password.encode(), stored_hash)
```

**JWT Token Security**:
```python
# Creation (HS256)
token = jwt.encode(
    {"sub": user_id, "exp": expiry_timestamp},
    SECRET_KEY,
    algorithm="HS256"
)

# Validation
decoded = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
```

**Soft Deletes**:
- Users/Cases marked `is_deleted=True` instead of removal
- Preserves referential integrity and audit trail

### API Security

**CORS Configuration**:
```python
allowed_origins = ["http://localhost:3000", "https://yourdomain.com"]
# Check for unsafe patterns (wildcards, placeholders)
```

**Error Masking**:
- Internal errors logged with full details
- Users receive generic error messages
- Stack traces not exposed in responses

### Infrastructure Security

**Environment Variables**:
- Never committed to version control
- `.env` file in `.gitignore`
- Different values for dev/production
- Regular rotation of secrets

**Database Security**:
- Connection pooling (`pool_pre_ping=True`)
- Parameterized queries (SQLAlchemy ORM)
- No raw SQL strings
- Transaction rollback on errors

---

## Performance Characteristics

### Latency Benchmarks

| Operation | Typical Time | Factors |
|-----------|--------------|---------|
| PDF Upload | 500ms - 2s | File size, network |
| PDF Chunking | 2-3s | Document length, complexity |
| Embedding Generation | 100ms/chunk | Batch size, Google AI API |
| Vector Search | <100ms | Collection size, network |
| LLM Answer Generation | 1-3s | Answer length, model |
| **Full Query (end-to-end)** | **3-5s** | All of above |

### Throughput Metrics

**Document Processing**:
- 5 jobs per batch cycle
- 10 second sleep between batches
- ~30 documents/minute sustained
- ~2000 documents/hour capacity

**Query Processing**:
- Synchronous (no queuing)
- Limited by Google AI API rate limits
- Typical: 100+ queries/minute

### Storage Requirements

**Per Case (Example: 50-page legal document)**:
- PDF file: 2-10 MB (Azure Blob)
- Chunks: ~50-200 chunks
- Vectors: 50-200 × 768 floats × 4 bytes = 150 KB - 600 KB (Qdrant)
- Metadata: ~100 KB (PostgreSQL)
- **Total per case**: ~3-13 MB

**Scaling**:
- 1000 cases: 3-13 GB storage
- 10000 cases: 30-130 GB storage

### Cost Model

**Per Document (50-page legal document)**:
- Google AI embeddings: Free tier (Google AI)
- Google AI LLM queries: Free tier (Google AI) per query (context + answer)
- Azure storage: ~$0.01-0.05 per month (highly variable)
- **Total per case**: ~$0.02-0.07 (one-time) + query costs

**Per Query**:
- Google AI: Free tier (Google AI)
- Vector search: Free (self-hosted Qdrant)
- Database: Negligible
- **Total per query**: ~$0.01-0.05

---

## Scalability & Future Considerations

### Current Limitations

1. **Single Worker Architecture**: Currently one background job processor
   - **Solution**: Scale Celery workers horizontally

2. **Local Qdrant**: Single-instance vector database
   - **Solution**: Deploy Qdrant cluster for HA

3. **PostgreSQL Scaling**: Single database instance
   - **Solution**: Read replicas, sharding by user_id

4. **Google AI Rate Limits**: API rate limiting on embeddings/LLM
   - **Solution**: Queue management, caching layer

### Horizontal Scaling Strategy

```
┌─────────────────────────────────────────────────┐
│          Load Balancer (nginx/HAProxy)          │
└──────────────┬──────────────────────────────────┘
               │
    ┌──────────┼──────────┐
    ↓          ↓          ↓
┌─────────┐┌─────────┐┌─────────┐
│ FastAPI ││ FastAPI ││ FastAPI │  API Layer (scale: 3-10)
│ Pod 1   ││ Pod 2   ││ Pod 3   │
└────┬────┘└────┬────┘└────┬────┘
     │          │          │
     └──────────┼──────────┘
                │
         ┌──────┴───────┐
         ↓              ↓
    ┌──────────┐   ┌──────────┐
    │PostgreSQL│   │PostgreSQL│   Database (primary-replica)
    │Primary   │   │Replica   │
    └──────────┘   └──────────┘
         ↓
    ┌──────────────────┐
    │  Redis Cluster   │   Message broker (HA)
    └────────────────┬─┘
                     │
        ┌────────────┼────────────┐
        ↓            ↓            ↓
    ┌───────────┐┌───────────┐┌───────────┐
    │ Celery    ││ Celery    ││ Celery    │  Workers (scale: 5-20)
    │Worker 1   ││Worker 2   ││Worker 3   │
    └───────────┘└───────────┘└───────────┘
         ↓            ↓            ↓
    ┌──────────────────────────────────┐
    │    Qdrant Cluster (HA)           │   Vector DB (HA)
    │  Node 1 | Node 2 | Node 3       │
    └──────────────────────────────────┘
```

### Caching Strategy

**Current**: In-memory LRU cache
```python
@lru_cache(maxsize=128)
def get_settings(): ...

@lru_cache(maxsize=1)
def get_embeddings_client(): ...
```

**Future**: Distributed caching
- Redis for embedding cache (avoid re-computing common phrases)
- Query result caching (identical questions → cached answers)
- TTL-based invalidation

### Async Improvements

**Current**: Sequential Celery task processing
- One task at a time per worker

**Future**: Parallel pipeline stages
```
Task 1: Download PDF → Task 2: Chunk → Task 3: Embed → Task 4: Upsert
                                                              (parallel)
```

### Advanced Features (Roadmap)

1. **Hybrid Search**: Combine BM25 (keyword) with vector search
2. **Incremental Updates**: Update only new/changed chunks
3. **Query Result Caching**: Cache identical questions
4. **Multi-language Support**: Translate documents automatically
5. **Document Summarization**: Auto-generate case summaries
6. **Version Control**: Track document changes over time
7. **Collaborative Features**: Share cases between users
8. **Advanced Analytics**: Query patterns, popular sections

### Monitoring & Observability

**Recommended Tools**:
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Metrics**: Prometheus + Grafana
- **Tracing**: Jaeger or Zipkin
- **APM**: DataDog or New Relic

**Key Metrics**:
```
API Layer:
├─ Request latency (p50, p95, p99)
├─ Error rate by endpoint
├─ Request volume
└─ Active connections

Database:
├─ Query latency
├─ Connection pool usage
├─ Transaction duration
└─ Slow query log

Celery Workers:
├─ Task latency (processing duration)
├─ Task failure rate
├─ Queue depth (pending tasks)
└─ Worker utilization

Qdrant:
├─ Search latency
├─ Collection size
├─ Disk usage
└─ Error rate

Google AI API:
├─ Token usage
├─ Cost tracking
├─ Rate limit remaining
└─ Error rate
```

---

## Deployment Architecture

### Docker Containerization

```dockerfile
# backend/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0"]
```

### Docker Compose (Development)

```yaml
services:
  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: lexintel
      POSTGRES_USER: legal_user
      POSTGRES_PASSWORD: dev_password_change_in_prod
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_storage:/qdrant/storage

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  azurite:
    image: mcr.microsoft.com/azure-storage/azurite
    ports:
      - "10000:10000"
    command: azurite-blob --blobHost 0.0.0.0

  backend:
    build: .
    environment:
      - DATABASE_URL=postgresql://legal_user:dev_password_change_in_prod@postgres:5432/lexintel
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - qdrant
      - redis
      - azurite

  worker:
    build: .
    command: python -m backend.services.job_processor run_worker
    environment:
      - DATABASE_URL=postgresql://legal_user:dev_password_change_in_prod@postgres:5432/lexintel
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379/0
      - AZURE_STORAGE_CONNECTION_STRING=DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://azurite:10000/devstoreaccount1;
      - CELERY_BROKER_URL=redis://redis:6379/0
      - CELERY_RESULT_BACKEND=redis://redis:6379/1
    depends_on:
      - postgres
      - qdrant
      - redis
      - azurite

volumes:
  postgres_data:
  qdrant_storage:
```

### Production Deployment Considerations

**Infrastructure**:
- Kubernetes cluster with auto-scaling
- Managed PostgreSQL (AWS RDS, Azure Database for PostgreSQL)
- Managed Qdrant (Qdrant Cloud)
- Managed Redis (AWS ElastiCache, Azure Cache)
- CDN for static assets
- DNS with SSL/TLS termination

**Security**:
- Network policies restricting traffic
- Secrets management (AWS Secrets Manager, Azure Key Vault)
- WAF (Web Application Firewall)
- DDoS protection
- VPN for backend services

**Reliability**:
- Database backup strategy (daily snapshots)
- Disaster recovery plan
- Monitoring and alerting
- Load testing and capacity planning
- Incident response procedures

---

## Summary

LexIntel is a production-ready RAG system with:

✅ **Modular Architecture**: Clear separation of concerns
✅ **Scalable Design**: Horizontal scaling for all components
✅ **Robust Processing**: Async job queue with retry logic
✅ **Comprehensive Security**: Authentication, validation, error masking
✅ **Enterprise Features**: Citation tracking, confidence levels, token management
✅ **Observable System**: Structured logging and monitoring hooks
✅ **Legal-Grade Accuracy**: Temperature 0.2, source attribution, hallucination detection

This architecture supports both current MVP requirements and future scaling to enterprise deployments.
