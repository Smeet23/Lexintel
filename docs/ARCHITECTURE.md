# Legal RAG App - Architecture Design

**Date:** 2026-01-25
**Scope:** Minimum viable but production-robust legal RAG application
**Target Users:** 1-5 lawyers (MVP)
**Primary Use Cases:** Case document Q&A + Auto-summarization

---

## 1. System Architecture Overview

### Three-Layer Architecture

**Frontend (Next.js + React + TypeScript)**
- Upload case PDFs
- Ask questions about cases with Q&A interface
- View auto-generated case summaries
- Case management dashboard
- PDF viewer with citation highlighting

**Backend (FastAPI + Python)**
- REST APIs for document upload, queries, summarization
- Document processing pipeline (chunking, embedding)
- RAG orchestration (retrieval + LLM prompting)
- User authentication & case isolation
- Audit logging for compliance

**Data Layer**
- **PostgreSQL:** Users, cases, metadata, query history, audit logs
- **Qdrant:** Vector embeddings for semantic similarity search
- **Azure Blob Storage:** Raw case PDF documents
- **OpenAI APIs:** Text embeddings + GPT-4o for answer generation

### High-Level Data Flow

```
Upload PDF → Store in Blob Storage → Chunk & Embed → Store in Qdrant
                                                    → Store metadata in PostgreSQL

Ask Question → Create question embedding → Search Qdrant (top-4 chunks)
            → Send context to GPT-4o → Get answer with citations
            → Log to PostgreSQL → Return to frontend
```

---

## 2. Backend Components & API Design

### Core Components

**Authentication & User Management**
- JWT token-based authentication
- User registration/login endpoints
- Case-level access control (isolation)

**Document Processing Pipeline (Background Job)**
- Receives PDF from Blob Storage
- Chunks document semantically (by sections, not fixed size)
- Creates metadata for each chunk (page_num, section_name, case_id)
- Generates embeddings via OpenAI text-embedding-3-large
- Stores vectors + metadata in Qdrant
- Stores chunk info in PostgreSQL for audit trail

**RAG Query Engine**
- Retrieves relevant chunks from Qdrant (top-4 similar)
- Builds context-aware prompts
- Calls GPT-4o with context + citation requirements
- Returns answer with source paragraph references

**Summarization Engine**
- Same retrieval flow as Q&A
- Uses different prompt template for structured summaries
- Returns Facts, Arguments, Judgment with citations

**Case Management**
- CRUD operations for cases
- Query history per case
- Delete cases (removes from Blob, Qdrant, PostgreSQL)

### REST API Endpoints

```
POST   /auth/register              Create user account
POST   /auth/login                 Get JWT token
POST   /cases                       Create case (upload PDF)
GET    /cases                       List user's cases
GET    /cases/{id}                  Get case details
DELETE /cases/{id}                  Delete case
POST   /cases/{id}/ask              Ask question about case
POST   /cases/{id}/summarize        Get auto-generated summary
GET    /cases/{id}/queries          Get query history
```

### Database Schema

**users**
```sql
id (UUID primary key)
email (unique)
password_hash
created_at
```

**cases**
```sql
id (UUID primary key)
user_id (foreign key → users)
name (case title)
uploaded_at
blob_storage_path
status (processing/ready/error)
```

**chunks**
```sql
id (UUID primary key)
case_id (foreign key → cases)
page_num
section_name
content (text snippet, for audit only)
created_at
```

**queries**
```sql
id (UUID primary key)
case_id (foreign key → cases)
user_id (foreign key → users)
question
answer
citations (JSON: [{page: X, paragraph: Y}, ...])
created_at
```

---

## 3. Frontend Components & User Flow

### Pages & Components

**Dashboard** (`/dashboard`)
- List all uploaded cases with metadata
- "Upload New Case" button
- Search/filter cases

**Case Detail** (`/cases/[id]`)
- Two tabs: Q&A | Summary
- Q&A Tab:
  - Question input field
  - Answer display with highlighted citations
  - Link to PDF viewer for cited sections
- Summary Tab:
  - Auto-generated structured summary
  - Regenerate button

**Upload Flow** (`/upload`)
- Drag-drop or file picker
- Progress indicator (uploading → processing → ready)
- Redirect to case detail when complete

**PDF Viewer** (embedded in case detail)
- Display PDF with page numbers
- Highlight cited paragraphs on click
- Navigate to specific page

**Auth Pages** (`/auth`)
- Login page
- Register page

### Component Structure
```
app/
  ├─ auth/
  │   ├─ login/page.tsx
  │   └─ register/page.tsx
  ├─ dashboard/
  │   └─ page.tsx
  ├─ cases/
  │   └─ [id]/
  │       ├─ page.tsx
  │       ├─ query-form.tsx
  │       ├─ pdf-viewer.tsx
  │       └─ summary-view.tsx
  └─ upload/
      └─ page.tsx
```

---

## 4. Data Processing Pipeline (Detailed)

### Document Upload & Processing

1. Lawyer uploads PDF via frontend
2. FastAPI validates file (is it a valid PDF?)
3. Stores in Azure Blob Storage
4. Creates case record in PostgreSQL
5. Enqueues background job (Celery/RQ)

**Background Job Steps:**
1. Retrieve PDF from Blob Storage
2. Extract text using LangChain PDFLoader
3. Semantic chunking via LangChain TextSplitter
   - Split by sections (headings, paragraphs)
   - Preserve page numbers & paragraph IDs
   - Maintain semantic coherence
4. Create metadata for each chunk (case_id, page_num, section)
5. Generate embeddings via OpenAI (LangChain wrapper)
6. Store vectors + metadata in Qdrant
7. Store chunk metadata in PostgreSQL
8. Update case status to "Ready"
9. Frontend detects status change → shows "Ready to query"

### Query Processing

1. Lawyer asks question (e.g., "What are the plaintiff's key arguments?")
2. Frontend sends question to POST /cases/{id}/ask
3. FastAPI creates embedding for question
4. LangChain Retriever queries Qdrant (cosine similarity, top-4)
5. Retrieves chunks with metadata (page numbers, section names)
6. Builds prompt:
   ```
   Context from case documents:
   [chunk 1 - page X]
   [chunk 2 - page Y]
   [chunk 3 - page Z]
   [chunk 4 - page W]

   Question: {user_question}

   Instructions:
   - Answer ONLY using the context above
   - If answer not in context, say "Not found in documents"
   - Always cite the page number and section
   ```
7. Sends to GPT-4o
8. Receives answer with citations
9. Logs query to PostgreSQL (audit trail)
10. Returns answer + citation metadata to frontend
11. Frontend highlights PDF pages mentioned in citations

### Summarization Processing

- Same retrieval flow, but retrieves ALL chunks (not top-4)
- Different prompt template:
  ```
  Summarize the case in these sections:
  1. Facts (with page citations)
  2. Plaintiff's Arguments (with citations)
  3. Defendant's Arguments (with citations)
  4. Court's Judgment (with citations)
  ```
- Returns structured summary

---

## 5. Technology Stack

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Backend | FastAPI (Python) | RAG ecosystem, async support, scalable |
| Frontend | Next.js (React + TypeScript) | Professional UI, type safety, scalable |
| Vector DB | Qdrant | Self-hosted, excellent filtering, free |
| SQL DB | PostgreSQL | Structured data, audit requirements, reliable |
| File Storage | Azure Blob Storage | Integrates with Azure ecosystem, secure |
| Embeddings | OpenAI text-embedding-3-large | High quality for legal text |
| LLM | OpenAI GPT-4o | Best reasoning for complex legal analysis |
| Document Processing | LangChain | Simplifies RAG pipeline, proven library |
| Authentication | JWT | Stateless, scalable |
| Deployment | Azure (Container Instances / App Service) | Native integration with Blob Storage, PostgreSQL |

---

## 6. Security & Compliance

### Authentication & Authorization
- JWT tokens with 24-hour expiry + refresh tokens
- Bcrypt password hashing
- Row-level security: users can only access their own cases
- Database-level constraints enforce isolation

### Data Protection
- PostgreSQL passwords in `.env` (never in code)
- OpenAI API keys in `.env`
- HTTPS only in production
- Azure Blob Storage encryption at rest (default)

### Audit Logging (Required for Legal Use)
Every action logged to PostgreSQL:
- User uploads case X at timestamp Y
- User queries case X with question "..." at timestamp Y
- User views case X at timestamp Y

Lawyers can request: "Show all queries on Case ABC from Dec 2024"

### Access Control
- No "view as user" backdoors
- No shared cases in MVP (each user isolated)
- Rate limiting: 100 queries/day per user (prevent abuse)

### What NOT to do
- Never log full API responses (contains PII)
- Never expose error details to users
- Never allow direct database access
- Never share case data between users

---

## 7. Error Handling & Edge Cases

### Document Processing Errors
| Error | User Message |
|-------|--------------|
| Scanned PDF (no OCR) | "PDF contains images. OCR not yet supported." |
| Corrupted file | "File is corrupted. Please re-upload." |
| Empty PDF | "PDF contains no text." |
| File > 500 pages | Process in background, show progress |

### Query/RAG Errors
| Error | Behavior |
|-------|----------|
| No relevant chunks | "Answer not found in uploaded documents." |
| OpenAI API timeout | "Service temporarily unavailable. Try again in 30s." |
| Poor retrieval quality | (Internal: flag for prompt tuning) |

### Infrastructure Errors
| Failure | Response |
|---------|----------|
| PostgreSQL down | "Database service unavailable" |
| Qdrant down | "Search service unavailable" |
| Blob Storage unavailable | "File storage service unavailable" |

### Retry Logic
- Failed OpenAI calls: Retry 2x with exponential backoff
- Failed Qdrant queries: Retry once, fallback to keyword search

---

## 8. Testing Strategy

### Backend Testing
- Unit tests: RAG logic (retrieval, citation accuracy)
- API endpoint tests: upload, query, summarize flows
- Integration tests: full upload → query flow with real PDFs
- Target: 70% code coverage minimum

### Frontend Testing
- Component tests: Q&A form, PDF viewer, upload
- Integration tests: upload → query → results flow
- E2E tests: complete user journey (login → upload → query → view citations)

### RAG Quality Testing (Critical)
- Create test dataset: 5-10 real judgments
- For each: define expected Q&A with citations
- Test hallucination prevention: ask out-of-document questions, verify "Not found"
- Manually verify citations match the answer

### Deployment Testing
- Staging environment: full system test before production
- Test with real lawyers on staging (if possible)
- Load test: can system handle 5 concurrent users?

---

## 9. Deployment Architecture

### Local Development
```
Next.js (localhost:3000)
  ↓
FastAPI (localhost:8000)
  ↓
PostgreSQL (Docker)
Qdrant (Docker)
Azure Blob Storage (local Azurite or real account)
```

### Production
```
Frontend: Vercel (Next.js deployment)
Backend: Azure Container Instances or App Service (FastAPI)
Database: Azure Database for PostgreSQL
Vector DB: Qdrant (Azure Container Instances)
Storage: Azure Blob Storage
```

### CI/CD
- GitHub Actions: run tests on every commit
- Auto-deploy to staging if tests pass
- Manual approval for production deployment

### Monitoring
- API response times (target: <5 seconds for queries)
- OpenAI API costs (track embeddings + LLM usage)
- Infrastructure uptime alerts
- Failed queries logged for debugging

---

## 10. MVP Scope & Future Scaling

### MVP Features (Must Have)
- Single user authentication
- PDF upload + processing
- Q&A with citations
- Auto-summarization
- Audit logs
- Clean, functional UI

### Not in MVP (Do Later)
- Team/multi-user collaboration
- Case sharing between lawyers
- Advanced filtering (by date, judge, court)
- Contradiction detection
- Case similarity search
- Voice queries
- Fine-tuned legal LLM

### Scaling Path (After MVP)
1. Optimize costs: switch to Approach 3 (local LLM + cloud embeddings)
2. Multi-user features: team workspaces, case sharing
3. Advanced RAG: LangGraph for multi-step reasoning
4. Deployment: Kubernetes for auto-scaling

---

## Summary

This architecture provides:
- ✅ **Minimum complexity:** Clean separation of concerns, proven technologies
- ✅ **Production-ready:** Security, audit logging, error handling from day one
- ✅ **Scalable:** Can grow from 1-5 users to 100+ without major rearchitecture
- ✅ **Legal-grade:** Compliance-focused, data isolation, full audit trail
- ✅ **RAG-optimized:** Semantic chunking, metadata filtering, citation enforcement

**Ready to start implementation?** 👇
