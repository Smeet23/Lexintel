---
status: pending
priority: p1
issue_id: "002"
tags: [code-review, security]
dependencies: []
---

# No File Size Limit on Upload

## Problem Statement
`backend/main.py` calls `await file.read()` with no size cap. A single large upload can exhaust server memory and crash the process.

## Findings
- **Location:** `backend/main.py` — upload endpoints
- **Risk:** Critical — OOM crash, denial of service
- **Evidence:** No `Content-Length` check, no streaming read, no middleware limit

## Proposed Solutions

### Option A: Streaming Read with Cap
- Read file in chunks, abort if exceeding limit (e.g. 100MB)
- **Pros:** Memory-safe, simple
- **Cons:** Slight code change
- **Effort:** Small
- **Risk:** Low

### Option B: Nginx/Reverse Proxy Limit
- Set `client_max_body_size` at proxy layer
- **Pros:** No code change
- **Cons:** Doesn't protect in dev/direct access
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Uploads exceeding size limit return 413
- [ ] Memory usage stays bounded during large uploads
