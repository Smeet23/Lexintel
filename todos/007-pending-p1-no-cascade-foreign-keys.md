---
status: pending
priority: p1
issue_id: "007"
tags: [code-review, data-integrity]
dependencies: []
---

# No CASCADE on Foreign Keys

## Problem Statement
All 5 foreign keys in `backend/models.py` lack `ondelete="CASCADE"`. Deleting a Matter leaves orphaned Documents, Chunks, and Queries with broken FK references.

## Findings
- **Location:** `backend/models.py` — all ForeignKey definitions
- **Risk:** Critical — orphaned records, integrity constraint violations
- **Evidence:** No `ondelete` parameter on any FK, soft delete at line 290 doesn't propagate

## Proposed Solutions

### Option A: Add CASCADE to All FKs
- Alembic migration to add `ondelete="CASCADE"` to all foreign keys
- **Pros:** Database-enforced cleanup, simple
- **Cons:** Irreversible deletes (mitigated by soft-delete pattern)
- **Effort:** Small
- **Risk:** Low (test with existing data first)

## Acceptance Criteria
- [ ] Deleting a Matter cascades to Documents, Chunks, Queries
- [ ] Deleting a Document cascades to Chunks
- [ ] Migration tested against existing data
