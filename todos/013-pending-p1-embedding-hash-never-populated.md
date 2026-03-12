---
status: pending
priority: p1
issue_id: "013"
tags: [code-review, data-integrity]
dependencies: []
---

# Chunk.embedding_hash Column Never Populated

## Problem Statement
`Chunk.embedding_hash` defined at `backend/models.py:78` is never written to during ingestion. The column exists in the schema but contains NULL for all rows.

## Findings
- **Location:** `backend/models.py:78`, `backend/tasks.py` (no write)
- **Risk:** Medium — wasted schema, misleading for developers
- **Evidence:** Grep for `embedding_hash` shows only model definition, no assignment

## Proposed Solutions

### Option A: Remove the Column
- Drop `embedding_hash` column via migration if unused
- **Pros:** Clean schema
- **Effort:** Small
- **Risk:** Low

### Option B: Populate It
- Compute hash of embedding vector at ingestion, store for cache invalidation
- **Pros:** Enables embedding cache validation
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Column is either populated during ingestion or removed from schema
