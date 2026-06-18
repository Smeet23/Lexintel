---
status: pending
priority: p1
issue_id: "012"
tags: [code-review, data-integrity]
dependencies: []
---

# Matter.blob_storage_path Broken for Multi-Document Matters

## Problem Statement
`Matter.blob_storage_path` at `backend/main.py:241` stores a single path. When multiple documents are uploaded to the same matter, only the last document's path is stored, overwriting previous ones.

## Findings
- **Location:** `backend/main.py:241`
- **Risk:** High — data loss of blob references for all but last document
- **Evidence:** Single string field overwritten on each upload

## Proposed Solutions

### Option A: Use Document.blob_storage_path Instead
- Store blob path on Document model (already has the field), remove from Matter
- **Pros:** Correct 1:many relationship, no data loss
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Each document retains its own blob storage path
- [ ] Multi-document matters preserve all blob references
