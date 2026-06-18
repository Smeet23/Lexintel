---
status: pending
priority: p1
issue_id: "006"
tags: [code-review, performance]
dependencies: []
---

# Cross-Encoder Loads ~100MB per Celery Worker

## Problem Statement
The cross-encoder model (`ms-marco-MiniLM-L-6-v2`) is loaded per Celery worker process at `rag_engine.py:191-218`. With multiple workers, this multiplies memory usage significantly (~100MB each).

## Findings
- **Location:** `backend/services/rag_engine.py:191-218`
- **Risk:** High — memory pressure in production with multiple workers
- **Evidence:** Model loaded at module/class level, no shared memory mechanism

## Proposed Solutions

### Option A: Lazy Loading + Singleton
- Load model only on first rerank call, cache as module-level singleton
- **Pros:** Doesn't load if unused (e.g. in ingestion workers)
- **Cons:** Still per-process
- **Effort:** Small
- **Risk:** Low

### Option B: Separate Reranking Service
- Move reranking to a dedicated worker/microservice
- **Pros:** Single model instance, independent scaling
- **Cons:** More infrastructure
- **Effort:** Large
- **Risk:** Medium

## Acceptance Criteria
- [ ] Cross-encoder not loaded in workers that don't query
- [ ] Memory usage documented for production worker sizing
