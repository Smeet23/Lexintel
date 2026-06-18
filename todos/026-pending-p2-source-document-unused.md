---
status: pending
priority: p2
issue_id: "026"
tags: [code-review, quality]
dependencies: []
---

# source_document Response Field Never Consumed

## Problem Statement
`source_document` field at `rag_engine.py:1170` is computed and returned but the frontend never reads it. The computation includes `extract_key_concepts()` which iterates all chunks — wasted work.

## Findings
- **Location:** `backend/services/rag_engine.py:1170`, `backend/services/document_summary.py`
- **Risk:** Wasted CPU per query
- **Evidence:** Frontend `AskResponse` type doesn't include `source_document`

## Proposed Solutions

### Option A: Remove source_document Computation
- Stop computing and returning `source_document` until frontend needs it
- Also removes the `extract_key_concepts()` and `calculate_page_count()` calls
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] No wasted computation for unused response fields
