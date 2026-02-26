# Veritas AI (Lexintel) - Comprehensive Issues & Upgrade Plan

> **Generated:** 2026-02-15
> **Scope:** Ingestion Pipeline, RAG Query Pipeline, Semantic Chunking Upgrade, Multi-Jurisdiction Support
> **Severity Levels:** CRITICAL (data loss / broken functionality), HIGH (significant degradation), MEDIUM (quality/performance impact), LOW (cosmetic / minor)

---

## Table of Contents

1. [Critical Issues](#1-critical-issues)
2. [Ingestion Pipeline Issues](#2-ingestion-pipeline-issues)
3. [RAG Query Pipeline Issues](#3-rag-query-pipeline-issues)
4. [Documentation vs Code Mismatches](#4-documentation-vs-code-mismatches)
5. [Multi-Jurisdiction & Legal Document Considerations](#5-multi-jurisdiction--legal-document-considerations)
6. [Semantic Chunking Upgrade Plan](#6-semantic-chunking-upgrade-plan)
7. [Unused / Dead Code](#7-unused--dead-code)
8. [Summary Table](#8-summary-table)

---

## 1. Critical Issues

These issues cause broken functionality in production. Must be fixed before any feature work.

### CRIT-1: Chunk ID Type Mismatch — Full Content Retrieval Always Fails

- **Severity:** CRITICAL
- **Files:**
  - `backend/tasks.py:115` — assigns `chunk["id"] = f"{case_id}:{idx}"` (string like `"abc-123:0"`)
  - `backend/services/vector_store.py:196` — stores `"chunk_id": chunk_id` in Qdrant payload as this string
  - `backend/services/rag_engine.py:1001` — queries `db.query(Chunk).filter(Chunk.id == chunk_id)` where `Chunk.id` is a PostgreSQL UUID
  - `backend/models.py` — `Chunk.id` is `Column(UUID(as_uuid=True))`
- **Impact:** Every RAG query falls through to the `except` block at `rag_engine.py:1007`, using the 200-character `content_preview` from Qdrant instead of full chunk text. This means the LLM generates answers from truncated snippets, causing:
  - Inaccurate answers
  - Missing context
  - Poor citation grounding
  - Reduced confidence scores
- **Fix:**
  ```python
  # Option A: Store chunk_sequence in Qdrant and query by (case_id, chunk_sequence)
  # In tasks.py:
  chunk["id"] = idx  # integer sequence
  # In vector_store.py payload:
  metadata["chunk_sequence"] = idx
  # In rag_engine.py:
  db_chunk = db.query(Chunk).filter(
      Chunk.case_id == UUID(case_id),
      Chunk.chunk_sequence == chunk_sequence
  ).first()

  # Option B: Store the actual Chunk UUID in Qdrant after DB insert
  # In tasks.py, insert chunks to DB first, get UUIDs, then upsert to Qdrant
  ```

### CRIT-2: `recreate_collection()` Destroys Existing Data on Re-process

- **Severity:** CRITICAL
- **File:** `backend/services/vector_store.py:122`
- **Impact:** If a document is re-processed (e.g., after a retry that progressed past chunking), `recreate_collection()` drops the entire Qdrant collection and creates a new one. All previously indexed vectors for that case are lost.
- **Fix:**
  ```python
  # Use collection_exists check + create_collection instead of recreate
  from qdrant_client.http.models import CollectionInfo

  collections = client.get_collections().collections
  if not any(c.name == collection_name for c in collections):
      client.create_collection(...)
  # For re-processing: delete points by filter, not the whole collection
  ```

### CRIT-3: Race Condition — Case Committed Before Blob Upload

- **Severity:** CRITICAL
- **File:** `backend/main.py:149-157`
- **Impact:** Case is committed to DB with `blob_storage_path=""` at line 150. If blob upload fails at line 153, the case exists in DB with an empty path. The Celery task will then try to download from an empty blob path and fail. The case is stuck in "processing" state with no way to recover.
- **Fix:**
  ```python
  # Upload blob first, then create case record in a single transaction
  blob_path = await upload_document_to_blob(file_content, str(case_id), file.filename)
  case = Case(
      id=case_id, name=name,
      blob_storage_path=blob_path,
      file_type=file_type, status="processing"
  )
  db.add(case)
  db.commit()
  ```

### CRIT-4: Error Response Shape Inconsistency (API Contract Break)

- **Severity:** CRITICAL
- **File:** `backend/services/rag_engine.py:837-846` vs `rag_engine.py:1042-1054`
- **Impact:** On success, `confidence` is a dict `{"level": str, "score": float, "factors": dict}`. On error, `confidence` is the string `"none"`. Any frontend code parsing `response.confidence.level` will crash on error responses.
- **Fix:**
  ```python
  # In error_response (line 844):
  "confidence": {
      "level": "none",
      "score": 0.0,
      "factors": {}
  },
  ```

---

## 2. Ingestion Pipeline Issues

Issues in the document upload → processing → indexing flow.

### ING-1: No File Size Limit on Upload

- **Severity:** HIGH
- **File:** `backend/main.py:122`
- **Impact:** `await file.read()` loads the entire file into memory with no cap. A 2GB PDF upload will crash the API server with OOM.
- **Fix:**
  ```python
  MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB (matches frontend expectation)

  file_content = await file.read()
  if len(file_content) > MAX_FILE_SIZE:
      raise HTTPException(
          status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
          detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024*1024)}MB"
      )
  ```

### ING-2: No Streaming Upload for Large Files

- **Severity:** MEDIUM
- **File:** `backend/main.py:122`
- **Impact:** Even with a size limit, reading 50MB entirely into memory is wasteful. Should stream to blob storage.
- **Fix:** Use `file.read(chunk_size)` in a loop, streaming directly to Azure Blob with `upload_blob(stream)`.

### ING-3: Retry Restarts Entire Pipeline from Scratch

- **Severity:** HIGH
- **File:** `backend/tasks.py:167`
- **Impact:** If embedding fails at chunk 190/200, the retry re-downloads the document, re-chunks, and re-embeds from chunk 0. Wastes time and API credits.
- **Fix:** Implement checkpoint-based retry:
  ```python
  # Store checkpoint in Redis:
  # {"stage": "embedding", "last_batch": 9, "chunks_file": "/tmp/case_id_chunks.json"}
  # On retry, resume from last successful batch
  ```

### ING-4: No Per-Batch Retry for Embeddings

- **Severity:** HIGH
- **File:** `backend/tasks.py:96-104`
- **Impact:** If batch 5 of 10 fails due to rate limiting, the entire task fails and retries from zero. Individual batch retry with backoff would prevent this.
- **Fix:**
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential

  @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=30))
  def embed_batch_with_retry(batch):
      return embed_chunks(batch)
  ```

### ING-5: Single Upsert Call for All Vectors (No Batching)

- **Severity:** MEDIUM
- **File:** `backend/services/vector_store.py:183-210`
- **Impact:** For a 500-page document with 2000+ chunks, a single upsert call sends all 2000 vectors at once. Qdrant recommends batch sizes of 100-256 points for optimal performance.
- **Fix:**
  ```python
  UPSERT_BATCH_SIZE = 100
  for i in range(0, len(points), UPSERT_BATCH_SIZE):
      batch = points[i:i + UPSERT_BATCH_SIZE]
      client.upsert(collection_name=collection_name, points=batch)
  ```

### ING-6: All Chunks Stored in Single DB Transaction

- **Severity:** MEDIUM
- **File:** `backend/tasks.py:131-144`
- **Impact:** If storing chunk 1999/2000 fails, the entire transaction rolls back and no chunks are persisted. Combined with ING-3, this means a retry processes everything again.
- **Fix:** Batch DB inserts with intermediate commits:
  ```python
  DB_BATCH_SIZE = 100
  for i in range(0, len(chunks), DB_BATCH_SIZE):
      batch = chunks[i:i + DB_BATCH_SIZE]
      for idx_in_batch, chunk in enumerate(batch):
          db.add(Chunk(...))
      db.commit()
  ```

### ING-7: Content Preview Too Short (200 chars)

- **Severity:** MEDIUM
- **File:** `backend/services/vector_store.py:199`
- **Impact:** `content_preview` stores only first 200 characters in Qdrant payload. When full content fetch fails (see CRIT-1), the LLM gets 200-char snippets. Even after CRIT-1 is fixed, the fallback preview is inadequate. Legal paragraphs often need 500-1000 chars for meaningful context.
- **Fix:** Increase to 500-800 chars or store full content in Qdrant payload (Qdrant handles arbitrary payload sizes).

### ING-8: No HNSW Index Configuration

- **Severity:** MEDIUM
- **File:** `backend/services/vector_store.py:122-128`
- **Impact:** Using default HNSW parameters. For 768-dimensional legal embeddings, custom `m`, `ef_construct`, and `ef` values can significantly improve search quality.
- **Fix:**
  ```python
  from qdrant_client.models import HnswConfigDiff

  client.create_collection(
      collection_name=collection_name,
      vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
      hnsw_config=HnswConfigDiff(m=16, ef_construct=200)
  )
  ```

### ING-9: No Payload Indexing on Qdrant Collections

- **Severity:** LOW
- **File:** `backend/services/vector_store.py`
- **Impact:** Without payload indexes on `page_num` or `section_name`, filtered searches are slower. Not critical now but matters at scale.
- **Fix:**
  ```python
  client.create_payload_index(
      collection_name=collection_name,
      field_name="page_num",
      field_schema="keyword"
  )
  ```

### ING-10: `download_document_from_blob()` Is Synchronous in Async Task

- **Severity:** MEDIUM
- **File:** `backend/services/storage.py` (download is sync) vs `upload_document_to_blob` (async)
- **Impact:** Download blocks the Celery worker thread. For large documents this ties up the worker.
- **Fix:** Either make download async or use `run_in_executor()` in the Celery task.

### ING-11: Deprecated `chunk_pdf_from_blob()` Still Exists

- **Severity:** LOW
- **File:** `backend/services/chunking.py:165-201`
- **Impact:** Dead code path. The function is marked deprecated but not removed. Could confuse developers.
- **Fix:** Remove the function entirely; it's not called anywhere.

### ING-12: No Duplicate Document Detection

- **Severity:** MEDIUM
- **File:** `backend/main.py:140-150`
- **Impact:** The same file can be uploaded multiple times for the same case, creating duplicate chunks and wasting embedding API credits.
- **Fix:** Compute file hash (SHA-256) on upload, check against existing cases:
  ```python
  file_hash = hashlib.sha256(file_content).hexdigest()
  existing = db.query(Case).filter(Case.file_hash == file_hash).first()
  if existing:
      raise HTTPException(409, "This document has already been uploaded")
  ```

### ING-13: Temp File Cleanup Not Guaranteed on Process Kill

- **Severity:** LOW
- **Files:** `backend/services/text_extraction.py:37-39`, `chunking.py:186-188`
- **Impact:** If the Celery worker is killed (SIGKILL, OOM), temp files remain on disk. Over time this fills up `/tmp`.
- **Fix:** Use a dedicated temp directory with periodic cleanup, or use `tempfile.TemporaryDirectory` context manager.

### ING-14: `section_name` Always "Chunk N" — Not Meaningful

- **Severity:** MEDIUM
- **File:** `backend/services/chunking.py:92,154`
- **Impact:** Every chunk gets a generic `section_name` like "Chunk 1", "Chunk 2". This provides no semantic information about the content. For legal docs, detecting actual section headers (e.g., "Article IV - Representations and Warranties") would dramatically improve search and citation quality.
- **Fix:** Part of the semantic chunking upgrade (see Section 6).

---

## 3. RAG Query Pipeline Issues

Issues in the query → retrieve → generate → cite flow.

### RAG-1: No Retry Logic on LLM API Calls

- **Severity:** HIGH
- **File:** `backend/services/rag_engine.py:764-796`
- **Impact:** `generate_answer()` makes a single Google AI API call. If it gets a rate limit or server error, the query fails immediately. The `RAG_PIPELINE.md` documents `call_with_retry()` with exponential backoff, but this function does not exist.
- **Fix:**
  ```python
  from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

  @retry(
      stop=stop_after_attempt(3),
      wait=wait_exponential(min=1, max=15),
      retry=retry_if_exception_type((RateLimitError, APIError))
  )
  async def generate_answer(query, context, temperature=0.2):
      ...
  ```

### RAG-2: No Retry Logic on Embedding API Calls

- **Severity:** HIGH
- **Files:** `backend/services/embeddings.py:62-103` (`embed_text`), `embeddings.py:106-157` (`embed_chunks`)
- **Impact:** Same as RAG-1 but for embedding generation. Rate limits on `gemini-embedding-001` will cause immediate failure.
- **Fix:** Add tenacity retry decorator to `embed_text()` and `embed_chunks()`.

### RAG-3: Missing `[Section "X"]` Citation Pattern

- **Severity:** MEDIUM
- **File:** `backend/services/rag_engine.py:360-364`
- **Impact:** `extract_citations()` only handles `[Page X]`, `[Paragraph X]`, and `[Lines X-Y]`. The `RAG_PIPELINE.md` documents a `[Section "X"]` pattern for named sections. If the LLM generates `[Section "Definitions"]`, it's treated as a hallucination and removed.
- **Fix:**
  ```python
  citation_patterns = [
      (r'\[Page\s+(\d+)\]', 'page'),
      (r'\[Paragraph\s+(\d+)\]', 'paragraph'),
      (r'\[Lines\s+(\d+-\d+)\]', 'line_range'),
      (r'\[Section\s+"([^"]+)"\]', 'section'),  # Add this
  ]
  ```

### RAG-4: Confidence Thresholds Differ from Documentation

- **Severity:** MEDIUM
- **File:** `backend/services/rag_engine.py:729-736`
- **Impact:** Code uses thresholds `0.75/0.60/0.40` for high/medium/low. Documentation says `0.70/0.50/0.30`. This means fewer queries get "high" confidence than the doc describes.
- **Fix:** Align code with documentation, or update documentation to reflect actual behavior. Recommend the code thresholds (0.75/0.60/0.40) as they are more conservative, appropriate for legal contexts.

### RAG-5: Citation Coverage Uses Crude `[` and `]` Heuristic

- **Severity:** MEDIUM
- **File:** `backend/services/rag_engine.py:684`
- **Impact:** `cited_sentences = sum(1 for s in sentences if "[" in s and "]" in s)` counts any sentence containing brackets as "cited". This inflates coverage for sentences with content like `"The defendant [Company A] argued..."` which aren't citations.
- **Fix:**
  ```python
  # Use proper citation pattern matching
  citation_re = re.compile(r'\[(Page|Paragraph|Lines|Section)\s+[^\]]+\]')
  cited_sentences = sum(1 for s in sentences if citation_re.search(s))
  ```

### RAG-6: Query Cache Not Integrated into Pipeline

- **Severity:** MEDIUM
- **File:** `backend/services/cache_manager.py` (exists, complete) vs `backend/services/rag_engine.py` (not imported)
- **Impact:** `QueryCache` is fully implemented but never used. Every identical query re-embeds, re-searches, and re-generates. Wasted API costs.
- **Fix:**
  ```python
  # In query_case():
  cache_key = generate_cache_key(query, case_id)
  cached = await query_cache.get(cache_key)
  if cached:
      return cached
  # ... run pipeline ...
  await query_cache.set(cache_key, response)
  ```

### RAG-7: Embedding Cache Not Used in Pipeline

- **Severity:** LOW
- **File:** `backend/services/embedding_cache.py` (exists) vs `backend/services/embeddings.py` (not imported)
- **Impact:** `EmbeddingCache` with LRU exists but is not used. Repeated queries with the same text re-call the Google AI API.
- **Fix:** Integrate `EmbeddingCache` into `embed_text()` for query embeddings.

### RAG-8: `embed_query()` Is Synchronous but Called in Async Context

- **Severity:** LOW
- **File:** `backend/services/rag_engine.py:862` — calls `embed_query(query)` which calls `embed_text()` (sync) from `embeddings.py`
- **Impact:** Blocks the async event loop during embedding. For single queries this is brief, but under concurrent load it reduces throughput.
- **Fix:** Either make `embed_text()` async or wrap with `asyncio.to_thread()`.

### RAG-9: Cross-Encoder Reranking Has No Error Handling

- **Severity:** MEDIUM
- **File:** `backend/services/rag_engine.py` (reranking section)
- **Impact:** If the cross-encoder model fails to load or throws an exception, the entire query fails. Should gracefully fall back to vector-only ranking.
- **Fix:** Wrap reranking in try/except, fall back to original ranking on failure.

### RAG-10: No Concurrent Retrieval Optimization

- **Severity:** LOW
- **File:** `backend/services/rag_engine.py`
- **Impact:** The doc describes async parallelization of embedding + retrieval. The code runs them sequentially. Under load, this adds latency.
- **Fix:** Use `asyncio.gather()` for independent async operations.

---

## 4. Documentation vs Code Mismatches

Discrepancies between `docs/RAG_PIPELINE.md` and actual implementation.

| # | Doc Claim | Actual Code | Severity | File |
|---|-----------|-------------|----------|------|
| DOC-1 | `call_with_retry()` with exponential backoff exists | No retry utility exists; errors raised immediately | HIGH | `rag_engine.py` |
| DOC-2 | Confidence thresholds: 0.70/0.50/0.30 | Code uses 0.75/0.60/0.40 | MEDIUM | `rag_engine.py:729-736` |
| DOC-3 | Error response has `confidence: {level, score, factors}` | Error response has `confidence: "none"` (string) | CRITICAL | `rag_engine.py:844` |
| DOC-4 | Document summary returns `case_name, case_number, court, filing_date, chunk_count` | Returns `filename, file_type, key_concepts, legal_significance, total_pages` | MEDIUM | `document_summary.py` |
| DOC-5 | `[Section "X"]` citation pattern supported | Not in `citation_patterns` list | MEDIUM | `rag_engine.py:360-364` |
| DOC-6 | Dual citation coverage: `_calculate_citation_coverage()` function | Uses inline `"[" in s and "]" in s` heuristic | MEDIUM | `rag_engine.py:684` |
| DOC-7 | Query cache integrated into pipeline | Cache exists but not imported/used | MEDIUM | `cache_manager.py` |
| DOC-8 | Content preview is 500 chars | Stores first 200 chars | MEDIUM | `vector_store.py:199` |
| DOC-9 | `embed_query()` is async | `embed_text()` is synchronous | LOW | `embeddings.py:62` |
| DOC-10 | Async parallelization of retrieval operations | Sequential execution | LOW | `rag_engine.py` |
| DOC-11 | HNSW config with `m=16, ef_construct=200` | No HNSW config, uses defaults | MEDIUM | `vector_store.py:122` |
| DOC-12 | Doc references `rag_pipeline.py` | File is `rag_engine.py` | LOW | `RAG_PIPELINE.md` |
| DOC-13 | Embedding cost: $0.13 per 1M tokens | Code uses $0.02 per 1M tokens | LOW | `embeddings.py:177` |
| DOC-14 | Token budget system with graceful degradation | Implemented correctly | OK | `rag_engine.py` |
| DOC-15 | Cross-encoder reranking with 40/60 weighting | Implemented correctly | OK | `rag_engine.py` |

---

## 5. Multi-Jurisdiction & Legal Document Considerations

Based on frontend analysis (`frontend/app/matters/page.tsx`, `frontend/lib/types.ts`, `frontend/app/settings/page.tsx`), Veritas AI handles legal matters across multiple jurisdictions:

### 5.1 Supported Jurisdictions

| Jurisdiction | Code | Document Languages | Legal Systems |
|---|---|---|---|
| US - Federal | `us-federal` | English | Common Law |
| US - California | `us-california` | English | Common Law (state) |
| US - New York | `us-new-york` | English | Common Law (state) |
| US - Delaware | `us-delaware` | English | Common Law (corporate) |
| US - Florida | `us-florida` | English | Common Law (state) |
| United Kingdom | `uk` | English | Common Law |
| European Union | `eu` | English, French, German, etc. | Civil Law |
| EU - GDPR Specific | `eu-gdpr` | English + EU languages | Regulatory |

### 5.2 Legal Document Types by Practice Area

| Practice Area | Document Types | Structural Features |
|---|---|---|
| **Litigation** | Complaints, Motions, Briefs, Orders, Depositions | Numbered paragraphs, captions, exhibits, case citations (e.g., *Brown v. Board* 347 U.S. 483) |
| **Corporate/M&A** | Merger Agreements, Stock Purchase Agreements, Due Diligence Reports | Articles/Sections, Schedules, Exhibits, Defined Terms, Representations & Warranties |
| **Real Estate** | Leases, Purchase Agreements, Title Reports, Deeds | Clauses, Sections, Legal Descriptions, Schedules |
| **IP/Patent** | Patent Applications, Office Actions, Prior Art, Licensing Agreements | Claims, Specifications, Drawings References, Classification Codes |
| **Data Privacy (GDPR)** | Privacy Policies, DPIAs, Processing Records, Consent Forms | Articles (referencing GDPR Articles 1-99), Recitals, Annexes |
| **Insurance** | Policies, Claims, Coverage Opinions, Reservation of Rights Letters | Sections, Endorsements, Exclusions, Conditions |
| **Estate Planning** | Wills, Trusts, Powers of Attorney, Estate Tax Returns | Articles, Sections, Schedules, Beneficiary Designations |

### 5.3 Jurisdiction-Aware Processing Gaps

The current backend has **zero jurisdiction awareness**:

1. **No `jurisdiction` field on Case model** — `backend/models.py` has no jurisdiction column. The frontend's `Matter.jurisdiction` is purely mock data.
2. **No jurisdiction in chunking** — Chunks don't carry jurisdiction metadata. Legal citation formats differ by jurisdiction (e.g., Bluebook in US, OSCOLA in UK, EU Official Journal format).
3. **No jurisdiction in system prompt** — `LEGAL_SYSTEM_PROMPT` in `rag_engine.py` doesn't reference the matter's jurisdiction. The LLM doesn't know if it's analyzing a UK lease or a US federal brief.
4. **No multi-language support** — EU documents may be in French, German, or other EU languages. Current text extraction assumes English.
5. **No jurisdiction-specific citation parsing** — US cases cite `*Smith v. Jones*, 123 F.3d 456 (9th Cir. 2024)`. UK cases cite `[2024] UKSC 1`. EU cites `Case C-123/24`. None of these are handled.

### 5.4 Required Backend Changes for Multi-Jurisdiction

```
Priority 1: Add jurisdiction to Case model + API
Priority 2: Pass jurisdiction to LEGAL_SYSTEM_PROMPT
Priority 3: Jurisdiction-specific citation patterns in extract_citations()
Priority 4: Jurisdiction metadata in Qdrant payload for filtered search
Priority 5: Multi-language text extraction (consider langdetect + per-language tokenizer)
```

---

## 6. Semantic Chunking Upgrade Plan

### 6.1 Current State

The chunking pipeline uses `RecursiveCharacterTextSplitter` with fixed parameters:
- `chunk_size=1500` characters
- `chunk_overlap=300` characters
- `separators=["\n\n", "\n", ". ", " ", ""]`
- `section_name` is always generic "Chunk N"

**Problems with this approach for legal documents:**
- Splits mid-paragraph, mid-sentence, or mid-clause
- No awareness of legal document structure (Articles, Sections, Definitions)
- No section header detection
- All chunks are equal — no differentiation between a "Definitions" section and a "Termination" clause
- Tables and structured data are flattened into text

### 6.2 Proposed Architecture: Hybrid Semantic Chunking

```
                    Document Bytes
                         │
                         ▼
              ┌──────────────────────┐
              │  Structure Extractor  │
              │  (pymupdf4llm / docx)│
              └──────────┬───────────┘
                         │ Markdown with headers
                         ▼
              ┌──────────────────────┐
              │  Section Detector     │
              │  (font-based / regex) │
              └──────────┬───────────┘
                         │ Sections with headers
                         ▼
              ┌──────────────────────┐
              │  Semantic Splitter    │
              │  (LangChain Semantic  │
              │   + legal-aware)      │
              └──────────┬───────────┘
                         │ Semantic chunks
                         ▼
              ┌──────────────────────┐
              │  Metadata Enricher    │
              │  (headers, page_num,  │
              │   jurisdiction, type) │
              └──────────┬───────────┘
                         │ Enriched chunks
                         ▼
                   Embed & Index
```

### 6.3 Implementation Steps

#### Step 1: Add `pymupdf4llm` for PDF Structure Extraction

```python
# backend/services/text_extraction.py — Enhanced PDF extraction
import pymupdf4llm

def extract_pdf_text_structured(file_bytes: bytes) -> List[Dict]:
    """Extract PDF text as Markdown with structure preservation."""
    temp_file = write_temp(file_bytes, ".pdf")
    try:
        # pymupdf4llm converts PDF to Markdown preserving:
        # - Headers (from font size analysis)
        # - Tables (as Markdown tables)
        # - Lists
        # - Bold/italic formatting
        md_text = pymupdf4llm.to_markdown(temp_file, page_chunks=True)

        sections = []
        for page_data in md_text:
            sections.append({
                "content": page_data["text"],
                "location": str(page_data["metadata"]["page"] + 1),
                "location_type": "page",
                "format": "markdown"
            })
        return sections
    finally:
        cleanup_temp(temp_file)
```

#### Step 2: Add Legal Section Header Detection

```python
# backend/services/section_detector.py — NEW FILE
import re
from typing import List, Dict, Tuple

# Patterns for legal document section headers
LEGAL_SECTION_PATTERNS = [
    # US Legal Sections
    (r'^(?:ARTICLE|Article)\s+([IVXLCDM]+|\d+)[.:\s—–-](.+)', 'article'),
    (r'^(?:SECTION|Section|§)\s*(\d+(?:\.\d+)*)[.:\s—–-](.+)', 'section'),
    (r'^(\d+(?:\.\d+)+)\s+([A-Z][^.]+)', 'numbered_section'),

    # Contract-specific
    (r'^(?:RECITALS|WHEREAS|DEFINITIONS|REPRESENTATIONS|WARRANTIES)', 'contract_section'),
    (r'^(?:EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+([A-Z]|\d+)', 'exhibit'),

    # Litigation-specific
    (r'^(?:COUNT|CLAIM)\s+([IVXLCDM]+|\d+)', 'count'),
    (r'^(?:PRAYER\s+FOR\s+RELIEF|CONCLUSION|ARGUMENT)', 'litigation_section'),

    # UK/EU specific
    (r'^(?:Regulation|Directive|Article)\s+(\d+)', 'eu_article'),
    (r'^(?:Part|Schedule)\s+(\d+|[A-Z])', 'uk_section'),
]

def detect_sections(text: str) -> List[Dict]:
    """Detect legal section boundaries in text."""
    lines = text.split('\n')
    sections = []
    current_section = {"header": "Preamble", "type": "preamble", "start": 0}

    for i, line in enumerate(lines):
        stripped = line.strip()
        for pattern, section_type in LEGAL_SECTION_PATTERNS:
            match = re.match(pattern, stripped)
            if match:
                # Close current section
                current_section["end"] = i
                sections.append(current_section)
                # Start new section
                current_section = {
                    "header": stripped,
                    "type": section_type,
                    "start": i
                }
                break

    current_section["end"] = len(lines)
    sections.append(current_section)
    return sections
```

#### Step 3: Implement Semantic Chunker with Section Awareness

```python
# backend/services/chunking.py — Enhanced chunking

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter

# Markdown header splitter for structured documents
HEADERS_TO_SPLIT_ON = [
    ("#", "heading_1"),
    ("##", "heading_2"),
    ("###", "heading_3"),
]

def chunk_document_semantic(
    extracted_sections: List[Dict],
    file_type: str,
    jurisdiction: str = None
) -> List[Dict]:
    """
    Semantic chunking with legal document awareness.

    Strategy:
    1. If markdown format → split on headers first, then semantic split within sections
    2. If plain text → detect sections via regex, then semantic split
    3. Respect section boundaries (never split across sections)
    4. Enrich with section metadata
    """
    # Combine all sections into full text with location markers
    full_text = ""
    location_map = []  # Maps character offset to location

    for section in extracted_sections:
        start = len(full_text)
        full_text += section["content"] + "\n\n"
        end = len(full_text)
        location_map.append({
            "start": start, "end": end,
            "location": section["location"],
            "location_type": section.get("location_type", "page")
        })

    # Phase 1: Split by headers (for markdown-formatted content)
    if any(s.get("format") == "markdown" for s in extracted_sections):
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON
        )
        header_splits = md_splitter.split_text(full_text)
    else:
        # Use legal section detection for non-markdown
        sections = detect_sections(full_text)
        header_splits = [...]  # Convert sections to splits

    # Phase 2: Semantic split within each section
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    semantic_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=75  # Split when similarity drops below 75th percentile
    )

    final_chunks = []
    chunk_idx = 0

    for split in header_splits:
        section_text = split.page_content if hasattr(split, 'page_content') else split
        section_header = split.metadata.get("heading_1", "")

        # Semantic split within section
        if len(section_text) > CHUNK_SIZE:
            sub_chunks = semantic_splitter.split_text(section_text)
        else:
            sub_chunks = [section_text]

        for sub_chunk in sub_chunks:
            if sub_chunk.strip():
                # Find location from map
                location = find_location(sub_chunk, location_map, full_text)

                final_chunks.append({
                    "content": sub_chunk,
                    "page_num": location,
                    "section_name": section_header or f"Chunk {chunk_idx + 1}",
                    "section_type": split.metadata.get("section_type", "general"),
                    "jurisdiction": jurisdiction,
                })
                chunk_idx += 1

    return final_chunks
```

#### Step 4: Enhanced Qdrant Payload with Section Metadata

```python
# backend/services/vector_store.py — Enhanced metadata

metadata = {
    "chunk_id": chunk_id,
    "chunk_sequence": idx,
    "page_num": str(chunk.get("page_num", "")),
    "section_name": chunk.get("section_name", ""),
    "section_type": chunk.get("section_type", "general"),
    "jurisdiction": chunk.get("jurisdiction", ""),
    "content_preview": chunk.get("content", "")[:500],  # Increased from 200
}
```

#### Step 5: Update `LEGAL_SYSTEM_PROMPT` with Jurisdiction Context

```python
# backend/services/rag_engine.py

def get_system_prompt(jurisdiction: str = None) -> str:
    base = LEGAL_SYSTEM_PROMPT
    if jurisdiction:
        base += f"\n\nJURISDICTION CONTEXT: This matter is governed by {jurisdiction} law. "
        base += "Ensure all analysis, citations, and legal reasoning are appropriate for this jurisdiction."
    return base
```

### 6.4 Dependencies to Add

```
# requirements.txt additions
pymupdf4llm>=0.0.10
langchain-experimental>=0.3.0   # For SemanticChunker
tenacity>=8.2.0                 # For retry logic
```

### 6.5 Migration Plan

1. **Phase 1 (Week 1-2):** Fix all CRITICAL issues (CRIT-1 through CRIT-4)
2. **Phase 2 (Week 2-3):** Add retry logic (RAG-1, RAG-2, ING-4), file size limits (ING-1), batch upserts (ING-5)
3. **Phase 3 (Week 3-4):** Implement semantic chunking (Steps 1-3 above)
4. **Phase 4 (Week 4-5):** Add jurisdiction support (Section 5.4), integrate caches (RAG-6, RAG-7)
5. **Phase 5 (Week 5-6):** Re-process existing documents with new pipeline, update documentation

---

## 7. Unused / Dead Code

| File | Item | Status |
|---|---|---|
| `backend/services/chunking.py:165-201` | `chunk_pdf_from_blob()` | Deprecated, never called |
| `backend/services/cache_manager.py` | Entire `QueryCache` class | Complete but not integrated |
| `backend/services/embedding_cache.py` | Entire `EmbeddingCache` class | Complete but not integrated |
| `backend/services/chunking.py:26-102` | `chunk_pdf()` (file-path based) | Only used by deprecated `chunk_pdf_from_blob` |
| `backend/models.py` | `ProcessingJob` model | Defined but never used in tasks.py or main.py |
| `backend/services/embeddings.py:160-182` | `estimate_embedding_cost()` | Never called; also has wrong pricing |

---

## 8. Summary Table

| ID | Issue | Severity | Category | File(s) |
|---|---|---|---|---|
| CRIT-1 | Chunk ID mismatch — full content fetch always fails | CRITICAL | Ingestion/RAG | `tasks.py:115`, `rag_engine.py:1001` |
| CRIT-2 | `recreate_collection()` destroys existing vectors | CRITICAL | Ingestion | `vector_store.py:122` |
| CRIT-3 | Race condition — case committed before blob upload | CRITICAL | Ingestion | `main.py:149-157` |
| CRIT-4 | Error response confidence shape inconsistency | CRITICAL | RAG | `rag_engine.py:844` |
| ING-1 | No file size limit on upload | HIGH | Ingestion | `main.py:122` |
| ING-2 | No streaming upload for large files | MEDIUM | Ingestion | `main.py:122` |
| ING-3 | Retry restarts entire pipeline from scratch | HIGH | Ingestion | `tasks.py:167` |
| ING-4 | No per-batch retry for embeddings | HIGH | Ingestion | `tasks.py:96-104` |
| ING-5 | Single upsert call — no vector batching | MEDIUM | Ingestion | `vector_store.py:183-210` |
| ING-6 | All chunks in single DB transaction | MEDIUM | Ingestion | `tasks.py:131-144` |
| ING-7 | Content preview too short (200 chars) | MEDIUM | Ingestion | `vector_store.py:199` |
| ING-8 | No HNSW index configuration | MEDIUM | Ingestion | `vector_store.py:122` |
| ING-9 | No payload indexing on Qdrant | LOW | Ingestion | `vector_store.py` |
| ING-10 | Sync download in async context | MEDIUM | Ingestion | `storage.py` |
| ING-11 | Deprecated `chunk_pdf_from_blob` not removed | LOW | Ingestion | `chunking.py:165` |
| ING-12 | No duplicate document detection | MEDIUM | Ingestion | `main.py:140` |
| ING-13 | Temp file cleanup not guaranteed on kill | LOW | Ingestion | `text_extraction.py` |
| ING-14 | `section_name` always generic "Chunk N" | MEDIUM | Ingestion | `chunking.py:92,154` |
| RAG-1 | No retry logic on LLM API calls | HIGH | RAG | `rag_engine.py:764-796` |
| RAG-2 | No retry logic on embedding API calls | HIGH | RAG | `embeddings.py:62-157` |
| RAG-3 | Missing `[Section "X"]` citation pattern | MEDIUM | RAG | `rag_engine.py:360` |
| RAG-4 | Confidence thresholds differ from docs | MEDIUM | RAG | `rag_engine.py:729` |
| RAG-5 | Citation coverage uses crude bracket heuristic | MEDIUM | RAG | `rag_engine.py:684` |
| RAG-6 | Query cache not integrated | MEDIUM | RAG | `cache_manager.py` |
| RAG-7 | Embedding cache not used | LOW | RAG | `embedding_cache.py` |
| RAG-8 | Sync embed in async context | LOW | RAG | `rag_engine.py:862` |
| RAG-9 | Cross-encoder reranking has no error handling | MEDIUM | RAG | `rag_engine.py` |
| RAG-10 | No concurrent retrieval optimization | LOW | RAG | `rag_engine.py` |
| DOC-1 | `call_with_retry()` doesn't exist | HIGH | Documentation | `RAG_PIPELINE.md` |
| DOC-2 | Confidence thresholds mismatch | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-3 | Error response shape mismatch | CRITICAL | Documentation | `RAG_PIPELINE.md` |
| DOC-4 | Document summary fields mismatch | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-5 | Missing Section citation pattern | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-6 | Dual citation coverage mismatch | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-7 | Cache integration claimed but absent | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-8 | Content preview size mismatch | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-9 | Async embedding claim incorrect | LOW | Documentation | `RAG_PIPELINE.md` |
| DOC-10 | Async parallelization not implemented | LOW | Documentation | `RAG_PIPELINE.md` |
| DOC-11 | HNSW config not applied | MEDIUM | Documentation | `RAG_PIPELINE.md` |
| DOC-12 | Wrong file name reference | LOW | Documentation | `RAG_PIPELINE.md` |
| DOC-13 | Embedding cost incorrect ($0.02 vs $0.13) | LOW | Documentation | `RAG_PIPELINE.md` |
| JURIS-1 | No jurisdiction field on Case model | HIGH | Multi-Jurisdiction | `models.py` |
| JURIS-2 | No jurisdiction in system prompt | MEDIUM | Multi-Jurisdiction | `rag_engine.py` |
| JURIS-3 | No jurisdiction-specific citations | MEDIUM | Multi-Jurisdiction | `rag_engine.py` |
| JURIS-4 | No multi-language support | LOW | Multi-Jurisdiction | `text_extraction.py` |

**Totals:** 4 CRITICAL, 8 HIGH, 21 MEDIUM, 10 LOW = **43 issues identified**
