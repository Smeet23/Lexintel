---
status: pending
priority: p2
issue_id: "016"
tags: [code-review, quality]
dependencies: []
---

# Dead Code: schemas.py (74 LOC)

## Problem Statement
`backend/schemas.py` defines Pydantic schemas that are never imported anywhere. 74 lines of dead code.

## Findings
- **Location:** `backend/schemas.py`
- **Evidence:** Zero imports across entire codebase

## Proposed Solutions

### Option A: Delete the File
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] File deleted, no import errors
