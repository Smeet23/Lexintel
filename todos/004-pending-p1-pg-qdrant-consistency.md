---
status: pending
priority: p1
issue_id: "004"
tags: [code-review, data-integrity, architecture]
dependencies: []
---

# PG-Qdrant Consistency Gap

## Problem Statement
In `backend/tasks.py`, PostgreSQL flush happens at line 161 and commit at line 242. Qdrant upsert happens between them. If the final PG commit fails after Qdrant upsert, orphaned vectors exist in Qdrant with no corresponding PG rows.

## Findings
- **Location:** `backend/tasks.py:161` (flush), `backend/tasks.py:242` (commit)
- **Risk:** Critical — data inconsistency, ghost search results pointing to non-existent chunks
- **Evidence:** No compensating transaction or rollback on Qdrant side

## Proposed Solutions

### Option A: Qdrant After Commit
- Move Qdrant upsert to after PG commit succeeds
- **Pros:** Simple, guarantees PG exists before vectors
- **Cons:** Brief window where PG has chunks but no vectors (acceptable)
- **Effort:** Small
- **Risk:** Low

### Option B: Compensating Delete
- On PG commit failure, delete Qdrant points
- **Pros:** Preserves current ordering
- **Cons:** More complex, Qdrant delete can also fail
- **Effort:** Medium
- **Risk:** Medium

## Acceptance Criteria
- [ ] PG commit failure does not leave orphaned Qdrant vectors
- [ ] Search results always correspond to existing PG chunks
