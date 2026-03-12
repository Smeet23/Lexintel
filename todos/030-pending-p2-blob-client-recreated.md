---
status: pending
priority: p2
issue_id: "030"
tags: [code-review, performance]
dependencies: []
---

# BlobServiceClient Recreated Per Operation

## Problem Statement
`backend/services/storage.py` creates a new `BlobServiceClient` for every upload/download operation instead of reusing a singleton.

## Findings
- **Location:** `backend/services/storage.py`
- **Risk:** Low — connection overhead, unnecessary TLS handshakes

## Proposed Solutions

### Option A: Module-Level Singleton
- Create `BlobServiceClient` once, reuse across operations
- **Effort:** Small | **Risk:** Low

## Acceptance Criteria
- [ ] Single client instance reused across operations
