---
status: pending
priority: p1
issue_id: "011"
tags: [code-review, data-integrity]
dependencies: ["007"]
---

# Soft Delete Doesn't Propagate to Children

## Problem Statement
Soft-deleting a Matter at `backend/main.py:290` sets `is_deleted=True` on the Matter only. Child Documents, Chunks, and Queries remain active and queryable.

## Findings
- **Location:** `backend/main.py:290`
- **Risk:** High — deleted matter's data still returned in searches and queries
- **Evidence:** Only matter row updated, no cascade to related records

## Proposed Solutions

### Option A: Cascade Soft Delete
- When soft-deleting a Matter, also soft-delete all Documents, Chunks, Queries
- Also delete corresponding Qdrant vectors
- **Pros:** Complete cleanup
- **Effort:** Medium
- **Risk:** Low

## Acceptance Criteria
- [ ] Soft-deleting a Matter marks all children as deleted
- [ ] Qdrant vectors for deleted chunks are removed
- [ ] Deleted chunks excluded from search results
