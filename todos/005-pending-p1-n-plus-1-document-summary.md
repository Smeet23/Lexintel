---
status: pending
priority: p1
issue_id: "005"
tags: [code-review, performance]
dependencies: []
---

# N+1 Query in generate_document_summary()

## Problem Statement
`generate_document_summary(matter)` at `rag_engine.py:1145` lazy-loads all documents and chunks on every query. This triggers N+1 ORM queries per RAG request.

## Findings
- **Location:** `backend/services/rag_engine.py:1145`
- **Risk:** High — query latency scales linearly with document count
- **Evidence:** No `joinedload()` or `selectinload()`, lazy relationship loading

## Proposed Solutions

### Option A: Eager Loading
- Use `selectinload(Matter.documents).selectinload(Document.chunks)` in the query
- **Pros:** Single query, minimal code change
- **Cons:** None significant
- **Effort:** Small
- **Risk:** Low

### Option B: Cache Summary per Matter
- Compute and store summary once at ingestion, not per query
- **Pros:** Zero query-time cost
- **Cons:** Needs migration for stored summary
- **Effort:** Medium
- **Risk:** Low

## Acceptance Criteria
- [ ] No N+1 queries during RAG query flow
- [ ] Query latency independent of document count
