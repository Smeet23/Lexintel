---
status: pending
priority: p1
issue_id: "010"
tags: [code-review, data-integrity]
dependencies: []
---

# Silent Collection Drop/Recreate on Dimension Mismatch

## Problem Statement
`backend/services/vector_store.py:154-163` silently drops and recreates the Qdrant collection if vector dimensions don't match. This destroys all existing embeddings without warning.

## Findings
- **Location:** `backend/services/vector_store.py:154-163`
- **Risk:** Critical — silent data loss of all embeddings
- **Evidence:** `recreate_collection()` called on mismatch with only a log warning

## Proposed Solutions

### Option A: Fail Loudly
- Raise an exception on dimension mismatch instead of silent recreate
- Log the mismatch, require explicit admin action to recreate
- **Pros:** Prevents accidental data loss
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Dimension mismatch raises an error, does not silently recreate
- [ ] Clear error message indicates current vs expected dimensions
