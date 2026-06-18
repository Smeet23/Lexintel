---
status: pending
priority: p3
issue_id: "036"
tags: [code-review, quality]
dependencies: []
---

# content_preview Fallback for Nonexistent Field

## Problem Statement
`backend/services/vector_store.py:369` falls back to `content_preview` payload field that doesn't exist in the Qdrant schema — always returns None.

## Findings
- **Location:** `backend/services/vector_store.py:369`

## Proposed Solutions
- Remove the fallback or add the field to schema
- **Effort:** Small | **Risk:** None
