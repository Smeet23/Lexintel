---
status: pending
priority: p2
issue_id: "021"
tags: [code-review, quality]
dependencies: []
---

# VECTOR_SIZE / EMBEDDING_DIMENSIONS Duplication

## Problem Statement
`VECTOR_SIZE=1024` in `vector_store.py` and `EMBEDDING_DIMENSIONS=1024` in `embeddings.py` define the same value independently. If one changes without the other, dimension mismatch causes silent collection recreation (see #010).

## Findings
- **Location:** `backend/services/vector_store.py`, `backend/services/embeddings.py`
- **Risk:** Medium — divergence causes data loss

## Proposed Solutions

### Option A: Single Source in Config
- Move to `backend/config.py` as `EMBEDDING_DIMENSIONS`, import everywhere
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] Single constant for embedding dimensions
- [ ] Both services import from same source
