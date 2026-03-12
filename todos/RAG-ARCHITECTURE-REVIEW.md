# RAG Architecture Review — Full Pipeline Analysis

**Date:** 2026-03-12
**Scope:** Complete end-to-end RAG pipeline review (text extraction → chunking → embeddings → vector store → retrieval → LLM generation → frontend)

---

## Executive Summary

Comprehensive review of the Lexintel legal RAG architecture identified **49 findings** across security, performance, data integrity, architecture, and code quality. The system has strong foundations (hybrid semantic chunking, SAC, cross-encoder reranking) but has critical gaps in security, data consistency, and accumulated dead code.

| Severity | Count | Description |
|----------|-------|-------------|
| P1 Critical | 14 | Blocks production readiness — security, data integrity |
| P2 Important | 18 | Should fix — performance, architecture, dead code |
| P3 Nice-to-have | 17 | Enhancements — logging, monitoring, polish |

---

## P1 — Critical (Must Fix)

### Security
| # | Issue | File | Risk |
|---|-------|------|------|
| 001 | No authentication on any endpoint | `main.py` | Data exfiltration, unauthorized access |
| 002 | No file size limit on upload | `main.py` | OOM crash, DoS |
| 008 | Content-Disposition header injection | `main.py:609,912` | Header injection, XSS |

### Data Integrity
| # | Issue | File | Risk |
|---|-------|------|------|
| 004 | PG-Qdrant consistency gap | `tasks.py:161→242` | Orphaned vectors if commit fails after upsert |
| 007 | No CASCADE on 5 foreign keys | `models.py` | Orphaned records on delete |
| 009 | Race condition: doc commit before task ID | `main.py:508-527` | Orphaned documents |
| 010 | Silent collection drop/recreate on dim mismatch | `vector_store.py:154-163` | Silent data loss |
| 011 | Soft delete doesn't propagate to children | `main.py:290` | Deleted matter's data still searchable |
| 012 | Matter.blob_storage_path broken for multi-doc | `main.py:241` | Blob reference overwritten |
| 013 | Chunk.embedding_hash never populated | `models.py:78` | Wasted column, misleading |
| 014 | MatterStatus enum defined but unenforced | `models.py:13-18` | Invalid status values |

### Performance
| # | Issue | File | Risk |
|---|-------|------|------|
| 003 | No pagination on list endpoints | `main.py:89,945-971` | OOM on growth |
| 005 | N+1 query in generate_document_summary() | `rag_engine.py:1145` | Latency scales with doc count |
| 006 | Cross-encoder loads ~100MB per worker | `rag_engine.py:191-218` | Memory pressure |

---

## P2 — Important (Should Fix)

### Dead Code (~530 LOC removable)
| # | Issue | File | LOC |
|---|-------|------|-----|
| 015 | job_processor.py — pre-Celery, zero imports | `services/job_processor.py` | 232 |
| 016 | schemas.py — Pydantic schemas never imported | `schemas.py` | 74 |
| 017 | cache_manager.py — Redis cache never integrated | `services/cache_manager.py` | 156 |
| 018 | ProcessingJob model — never queried | `models.py:116-132` | 16 |

### Architecture
| # | Issue | File |
|---|-------|------|
| 019 | rag_engine.py god object (1184 lines, 5+ concerns) | `services/rag_engine.py` |
| 021 | VECTOR_SIZE / EMBEDDING_DIMENSIONS duplication | `vector_store.py` + `embeddings.py` |
| 031 | Frontend Citation type differs from backend | `frontend/lib/types.ts` |

### Over-Engineering (~360 LOC removable)
| # | Issue | File |
|---|-------|------|
| 020 | 7 confidence scoring functions, frontend uses 1 float | `rag_engine.py:536-789` |
| 025 | Thin pass-through wrappers | `rag_engine.py` |
| 026 | source_document field computed but never consumed | `rag_engine.py:1170` |
| 027 | ground_citations redundant with extract_citations | `rag_engine.py:477-533` |

### Performance
| # | Issue | File |
|---|-------|------|
| 022 | SAC memory duplication — full parallel list | `tasks.py:167-171` |
| 023 | Per-process LRU cache — near-zero hit rate | `embedding_cache.py` |
| 024 | Frontend unconditional 10s polling | `hooks/use-matters.ts` |
| 028 | Citation regex recompiled per query | `rag_engine.py:354-474` |
| 029 | full_text concatenation duplicates entire document | `tasks.py:109` |
| 030 | BlobServiceClient recreated per operation | `services/storage.py` |
| 032 | No database connection pool configuration | `database.py` |

---

## P3 — Nice-to-Have (Enhancements)

| # | Issue | Category |
|---|-------|----------|
| 033 | Token estimation underestimates 20-30% for legal text | Quality |
| 034 | MD5-based Qdrant point ID collision risk | Quality |
| 035 | Qdrant timeout 10s too low for large upserts | Performance |
| 036 | content_preview fallback for nonexistent field | Quality |
| 037 | Progress SSE lacks document-level granularity | Frontend |
| 038 | bulk_insert_mappings skips ORM defaults | Data |
| 039 | No rate limiting on API endpoints | Security |
| 040 | Limited request validation | Security |
| 041 | No structured logging | Observability |
| 042 | No health check endpoint | Observability |
| 043 | Permissive CORS configuration | Security |
| 044 | No error tracking / APM integration | Observability |
| 045 | Celery task missing retry configuration | Reliability |
| 046 | No API versioning | Architecture |
| 047 | OpenAPI docs incomplete (no response_model) | Quality |
| 048 | No graceful shutdown for Celery workers | Reliability |
| 049 | No database backup strategy documented | Operations |

---

## Architecture Strengths (What's Working Well)

- Hybrid semantic chunking (markdown headers → semantic → fallback recursive)
- SAC (Summary-Augmented Chunking) — research-backed, reduces retrieval mismatch
- Cross-encoder reranking with weighted scoring (40% vector + 60% rerank)
- NUPunkt legal sentence boundary detection (4,000+ abbreviations)
- Structured PDF extraction with pymupdf4llm
- YAKE unsupervised keyword extraction at ingestion
- Section detector with multi-jurisdiction support (UK, EU, US, CA, SG, UN)
- Celery-based async processing with Redis pub/sub progress

---

## Recommended Priority

### Week 1 (Zero-risk, high impact)
1. Delete dead code files (#015-018) — 530 LOC removal, zero risk
2. Fix Content-Disposition injection (#008) — 5-line fix
3. Add file size limit (#002) — 10-line fix
4. Fix silent collection recreate (#010) — raise error instead
5. Remove unused source_document computation (#026)

### Week 2-3
6. Add pagination (#003)
7. Fix PG-Qdrant consistency (#004) — reorder operations
8. Add CASCADE to foreign keys (#007)
9. Fix race condition (#009)
10. Consolidate dimension constants (#021)

### Medium-term
11. Add authentication (#001)
12. Split rag_engine.py god object (#019)
13. Simplify confidence scoring (#020)
14. Fix N+1 query (#005)

---

## Files Referenced

```
backend/main.py                          — FastAPI endpoints (1073 lines)
backend/models.py                        — SQLAlchemy models
backend/tasks.py                         — Celery pipeline orchestration
backend/config.py                        — Pydantic settings
backend/database.py                      — DB session factory
backend/services/rag_engine.py           — RAG query logic (1184 lines)
backend/services/vector_store.py         — Qdrant wrapper
backend/services/embeddings.py           — Cohere embeddings
backend/services/chunking.py             — Hybrid semantic chunking
backend/services/document_summary.py     — Gemini enrichment
backend/services/text_extraction.py      — PDF/DOCX/TXT extraction
backend/services/storage.py              — Azure Blob storage
backend/services/progress.py             — Redis SSE progress
backend/services/embedding_cache.py      — Per-process LRU cache
backend/services/job_processor.py        — DEAD CODE
backend/services/cache_manager.py        — DEAD CODE
backend/schemas.py                       — DEAD CODE
frontend/hooks/use-matters.ts            — Matter polling
frontend/lib/types.ts                    — TypeScript types
frontend/lib/api-services.ts             — API client
```
