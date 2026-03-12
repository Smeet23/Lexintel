---
status: pending
priority: p1
issue_id: "009"
tags: [code-review, data-integrity]
dependencies: []
---

# Race Condition: Document Commit Before Task ID Assignment

## Problem Statement
In `backend/main.py:508-527`, the document is committed to PostgreSQL before the Celery task ID is assigned back. If the commit succeeds but task dispatch fails, the document exists with no processing task. If the user queries status immediately after upload, they get stale state.

## Findings
- **Location:** `backend/main.py:508-527`
- **Risk:** High — orphaned documents with no processing task, incorrect status
- **Evidence:** Sequential commit → task dispatch with no transaction wrapping

## Proposed Solutions

### Option A: Assign Task ID After Dispatch
- Dispatch Celery task first, then update document with task_id and commit
- **Pros:** Atomic — document only exists with valid task reference
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Document always has valid celery_task_id after creation
- [ ] Failed task dispatch rolls back document creation
