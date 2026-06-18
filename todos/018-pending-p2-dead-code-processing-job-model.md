---
status: pending
priority: p2
issue_id: "018"
tags: [code-review, quality]
dependencies: []
---

# Dead Code: ProcessingJob Model (16 LOC)

## Problem Statement
`ProcessingJob` model at `backend/models.py:116-132` is never queried or written to. Leftover from pre-Celery architecture.

## Findings
- **Location:** `backend/models.py:116-132`
- **Evidence:** No queries, no creates, no references outside model definition

## Proposed Solutions

### Option A: Remove Model + Drop Table
- Remove model class, create migration to drop table
- **Effort:** Small | **Risk:** Low

## Acceptance Criteria
- [ ] Model removed from models.py
- [ ] Migration drops the table
