# Implementation Plan - Phases 1-6

> Detailed breakdown of LexIntel MVP development phases

**Status**: Phase 1 & 2 Complete | Phases 3-6 In Progress

---

## ✅ Phase 1: Project Setup & Infrastructure (COMPLETE)

### Deliverables
- [x] Python FastAPI project structure
- [x] Pydantic settings configuration
- [x] SQLAlchemy database models
- [x] Celery workers setup
- [x] Docker Compose orchestration

### Key Files
```
backend/requirements.txt    # Python dependencies
backend/pyproject.toml      # Project metadata
backend/app/config.py       # Settings
backend/app/database.py     # Database connection
backend/app/celery_app.py   # Celery configuration
docker-compose.yml          # Service orchestration
```

### Commits
- `chore: initialize Python FastAPI backend with Azurite support`
- `feat: setup SQLAlchemy database models for RAG MVP`
- `feat: setup Celery workers for async document processing`
- `chore: setup Docker Compose with PostgreSQL, Redis, and Azurite`

---

## ✅ Phase 2: Document Upload & Storage (COMPLETE)

### Deliverables
- [x] Pydantic request/response schemas
- [x] File storage service
- [x] Cases CRUD API endpoints
- [x] Documents upload & management endpoints
- [x] PostgreSQL schema initialization

### Key Files
```
backend/app/schemas/          # Pydantic models
backend/app/services/storage.py
backend/app/api/cases.py      # Cases endpoints
backend/app/api/documents.py  # Document endpoints
backend/app/main.py           # Router registration
```

### API Endpoints Implemented
- `POST /cases` - Create case
- `GET /cases` - List cases
- `GET /cases/{case_id}` - Get case details
- `PATCH /cases/{case_id}` - Update case
- `DELETE /cases/{case_id}` - Delete case
- `POST /documents/upload?case_id=...` - Upload document
- `GET /documents/{document_id}` - Get document details
- `DELETE /documents/{document_id}` - Delete document

### Commits
- `feat: create Pydantic schemas for API validation`
- `feat: create storage service for file upload/download`
- `feat: create cases and documents CRUD APIs with file upload`

---

## ⏳ Phase 3: Text Extraction Workers (TODO)

### Goal
Extract text from various document formats and split into chunks for processing

### Deliverables
- [ ] Extract text from PDF files (PyPDF2)
- [ ] Extract text from DOCX files (python-pptx)
- [ ] Extract text from TXT files (native read)
- [ ] Split text into chunks (4000 chars, 400 char overlap)
- [ ] Create DocumentChunk records in database
- [ ] Update Document.processing_status → EXTRACTED

### Files to Create/Modify
```
backend/app/services/extraction.py    # Text extraction logic
backend/app/workers/tasks.py          # extract_text_from_document task
```

### Task: extract_text_from_document

**Input**: `document_id` (UUID)

**Steps**:
1. Get Document from database
2. Read file from storage using file_path
3. Extract text based on file type:
   - PDF: Use PyPDF2 or pdf2image
   - DOCX: Use python-pptx
   - TXT: Direct file read
4. Store extracted text in Document.extracted_text
5. Split text into chunks (4000 chars, 400 overlap)
6. Create DocumentChunk records with chunk_text and chunk_index
7. Update Document:
   - processing_status → EXTRACTED
   - page_count (if available)
8. Queue generate_embeddings task
9. Log completion

**Error Handling**:
- If file not found: log error, set Document.error_message
- If extraction fails: retry up to 3 times with 60-second backoff
- Max timeout: 25 minutes

**Success Response**:
```json
{
  "status": "success",
  "document_id": "uuid",
  "text_length": 50000,
  "chunks_created": 13
}
```

### Dependencies to Add
```
PyPDF2==3.0.1         # PDF parsing
python-pptx==0.6.21   # DOCX parsing (already in requirements)
```

### Testing
```bash
# Create test document
echo "Test content" > test.txt

# Upload via API
curl -X POST "http://localhost:8000/documents/upload?case_id=test-case" \
  -F "file=@test.txt"

# Monitor task
docker-compose logs -f celery-worker | grep extract_text
```

---

## ⏳ Phase 4: Embeddings Generation (TODO)

### Goal
Generate OpenAI embeddings for document chunks and store in pgvector

### Deliverables
- [ ] Create embeddings service
- [ ] Call OpenAI embedding API
- [ ] Batch embeddings for efficiency
- [ ] Store embeddings in DocumentChunk.embedding (pgvector)
- [ ] Update Document.processing_status → INDEXED
- [ ] Handle API rate limits and retries

### Files to Create/Modify
```
backend/app/services/embeddings.py    # OpenAI API integration
backend/app/workers/tasks.py          # generate_embeddings task
```

### Task: generate_embeddings

**Input**: `document_id` (UUID)

**Steps**:
1. Get all DocumentChunks for document (where embedding IS NULL)
2. Batch chunks (max 20-25 per batch)
3. For each batch:
   - Call OpenAI API: `text-embedding-3-small`
   - Get 1536-dimension vectors
   - Store in DocumentChunk.embedding (as JSON string)
4. Update Document:
   - processing_status → INDEXED
   - indexed_at → current timestamp
5. Log progress (e.g., "Generated embeddings for 13/13 chunks")
6. Return success status

**Error Handling**:
- If API rate limited: retry with exponential backoff
- If API key invalid: fail with error message
- If timeout: retry with smaller batch size
- Max retries: 3

**Batch Processing**:
```python
# Efficient batching
BATCH_SIZE = 20
for i in range(0, len(chunks), BATCH_SIZE):
    batch = chunks[i:i+BATCH_SIZE]
    embeddings = await openai_api.embed(batch)
    # Store embeddings
```

**Success Response**:
```json
{
  "status": "success",
  "document_id": "uuid",
  "embeddings_generated": 13,
  "total_tokens": 5120
}
```

### Service: EmbeddingsService

```python
class EmbeddingsService:
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API for embeddings"""
        # Call with batching and rate limit handling
        pass

    async def store_embeddings(self, chunks: List[DocumentChunk]):
        """Store embeddings in pgvector"""
        # Save to database
        pass
```

---

## ⏳ Phase 5: Search APIs (TODO)

### Goal
Implement full-text and semantic search capabilities

### Deliverables
- [ ] Full-text search (PostgreSQL tsvector)
- [ ] Semantic search (pgvector similarity)
- [ ] Hybrid search (combined ranking)
- [ ] Search endpoints with filtering
- [ ] Result ranking and relevance scoring

### Files to Create/Modify
```
backend/app/api/search.py          # Search endpoints
backend/app/services/search.py     # Search logic
backend/app/schemas/search.py      # Request/response schemas
```

### Endpoints

#### 1. Hybrid Search
```
POST /search/hybrid
{
  "case_id": "uuid",
  "query": "contract terms",
  "limit": 10,
  "threshold": 0.5
}

Response:
{
  "results": [
    {
      "document_id": "uuid",
      "document_title": "Contract Agreement",
      "chunk_id": "uuid",
      "chunk_text": "...",
      "relevance_score": 0.85,
      "search_type": "hybrid"
    }
  ],
  "total": 5,
  "execution_time_ms": 45
}
```

#### 2. Full-Text Search
```
POST /search/full-text
{
  "case_id": "uuid",
  "query": "contract terms",
  "limit": 10
}
```

#### 3. Semantic Search
```
POST /search/semantic
{
  "case_id": "uuid",
  "query": "What are the main contract obligations?",
  "limit": 10,
  "threshold": 0.5
}
```

### Search Implementation

**Full-Text (tsvector)**:
```python
# Create search vector on DocumentChunk
stmt = select(DocumentChunk).where(
    DocumentChunk.search_vector.match(query)  # tsvector matching
).order_by(
    DocumentChunk.document_id
)
```

**Semantic (pgvector)**:
```python
# Generate query embedding
query_embedding = await embeddings_service.embed(query)

# Find similar chunks
stmt = select(DocumentChunk).order_by(
    DocumentChunk.embedding.cosine_distance(query_embedding)  # Vector similarity
).limit(10)
```

**Hybrid Ranking**:
```python
# Combine scores: 0.5 * full_text + 0.5 * semantic
combined_score = 0.5 * bm25_score + 0.5 * cosine_similarity
```

---

## ⏳ Phase 6: Chat/RAG APIs (TODO)

### Goal
Implement streaming chat with document context (RAG)

### Deliverables
- [ ] Chat conversation management
- [ ] Message history persistence
- [ ] Document context retrieval
- [ ] Streaming responses
- [ ] Token usage tracking
- [ ] Citation/source tracking

### Files to Create/Modify
```
backend/app/api/chat.py            # Chat endpoints
backend/app/services/rag.py        # RAG orchestration
backend/app/schemas/chat.py        # Chat schemas
```

### Endpoints

#### 1. Create Conversation
```
POST /chat/conversations
{
  "case_id": "uuid",
  "title": "Contract Review"
}

Response:
{
  "id": "uuid",
  "case_id": "uuid",
  "title": "Contract Review",
  "created_at": "2024-01-03T10:00:00Z"
}
```

#### 2. Send Message (Streaming)
```
POST /chat/conversations/{conversation_id}/messages
{
  "content": "What are the key terms in this contract?",
  "top_k_documents": 5
}

Response: (Server-Sent Events)
data: {"type": "start", "timestamp": "..."}
data: {"type": "token", "content": "The"}
data: {"type": "token", "content": " key"}
data: {"type": "token", "content": " terms"}
...
data: {"type": "end", "tokens_used": 250, "sources": ["doc-1", "doc-2"]}
```

#### 3. Get Conversation History
```
GET /chat/conversations/{conversation_id}/messages

Response:
{
  "messages": [
    {
      "id": "uuid",
      "role": "user",
      "content": "What are the key terms?",
      "created_at": "..."
    },
    {
      "id": "uuid",
      "role": "assistant",
      "content": "The key terms are...",
      "tokens_used": 250,
      "source_document_ids": ["doc-1"]
    }
  ]
}
```

### RAG Pipeline

```
User Message
    ↓
Generate embedding of user query
    ↓
Search for relevant document chunks (semantic search)
    ↓
Rank by relevance, take top-K
    ↓
Build context: "Based on these documents: [chunks]"
    ↓
Prompt: "{context}\n\nUser question: {query}"
    ↓
Stream OpenAI response
    ↓
Save message to ChatMessage table
    ↓
Track tokens used
    ↓
Store source_document_ids
```

### Service: RAGService

```python
class RAGService:
    async def search_context(
        self,
        query: str,
        case_id: str,
        top_k: int = 5
    ) -> List[DocumentChunk]:
        """Search for relevant document chunks"""
        pass

    async def build_prompt(
        self,
        query: str,
        context_chunks: List[DocumentChunk]
    ) -> str:
        """Build system prompt with context"""
        pass

    async def stream_response(
        self,
        prompt: str
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response"""
        pass

    async def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens_used: int,
        source_ids: List[str]
    ) -> ChatMessage:
        """Save message to database"""
        pass
```

---

## 📊 Summary & Timeline

| Phase | Name | Status | Files | API Endpoints |
|-------|------|--------|-------|---------------|
| 1 | Setup & Infrastructure | ✅ DONE | 8 | - |
| 2 | Upload & Storage | ✅ DONE | 8 | 7 |
| 3 | Text Extraction | ⏳ TODO | 2 | - |
| 4 | Embeddings | ⏳ TODO | 2 | - |
| 5 | Search APIs | ⏳ TODO | 3 | 3 |
| 6 | Chat/RAG APIs | ⏳ TODO | 3 | 3 |

---

## 🎯 Success Criteria per Phase

### Phase 3: Text Extraction ✓
- [ ] PDF files parsed correctly
- [ ] DOCX files parsed correctly
- [ ] TXT files read correctly
- [ ] Text split into proper chunks
- [ ] DocumentChunk records created
- [ ] Processing status updated correctly
- [ ] Worker logs show task completion

### Phase 4: Embeddings ✓
- [ ] OpenAI API integration working
- [ ] Embeddings stored in pgvector
- [ ] Document status → INDEXED
- [ ] Batch processing efficient
- [ ] Retries handle rate limits
- [ ] Tokens tracked and logged

### Phase 5: Search ✓
- [ ] Full-text search returns results
- [ ] Semantic search returns ranked results
- [ ] Hybrid search combines both methods
- [ ] Performance is acceptable (<200ms)
- [ ] Filtering by case_id works
- [ ] Results ranked by relevance

### Phase 6: Chat/RAG ✓
- [ ] Conversations created and retrieved
- [ ] Messages streamed correctly
- [ ] Context chunks retrieved
- [ ] OpenAI responses generated
- [ ] Token usage tracked
- [ ] Source documents linked
- [ ] Messages persisted

---

## 📚 Documentation References

- **Phase 1 Details**: See original plan file
- **Phase 2 Details**: See original plan file
- **API Patterns**: [docs/BACKEND.md](./BACKEND.md)
- **Worker Patterns**: [docs/WORKERS.md](./WORKERS.md)
- **Database Schema**: [docs/DATABASE.md](./DATABASE.md)
- **Development Setup**: [docs/SETUP.md](./SETUP.md)

---

## 🚀 Next Steps

1. **Phase 3 Implementation**:
   - Create `app/services/extraction.py`
   - Implement `extract_text_from_document` task
   - Test with PDF, DOCX, TXT files

2. **Phase 4 Implementation**:
   - Create `app/services/embeddings.py`
   - Implement `generate_embeddings` task
   - Batch processing and rate limit handling

3. **Phase 5 Implementation**:
   - Create `app/api/search.py` and `app/services/search.py`
   - Implement hybrid search with ranking
   - Performance optimization

4. **Phase 6 Implementation**:
   - Create `app/api/chat.py` and `app/services/rag.py`
   - Streaming response implementation
   - Token usage tracking

---

**For Claude**: Use this document as a detailed blueprint for each phase implementation. Each phase builds on the previous one, so ensure all tests pass before moving to the next phase.
