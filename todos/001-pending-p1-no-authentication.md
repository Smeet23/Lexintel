---
status: pending
priority: p1
issue_id: "001"
tags: [code-review, security]
dependencies: []
---

# No Authentication on Any Endpoint

## Problem Statement
Every endpoint in `backend/main.py` is publicly accessible. No authentication middleware, API keys, or session validation exists. Anyone with the URL can upload documents, query matters, delete data.

## Findings
- **Location:** `backend/main.py` — all routes
- **Risk:** Critical — data exfiltration, unauthorized uploads, deletion of all matters
- **Evidence:** No `Depends()` auth guards, no middleware, no API key checks

## Proposed Solutions

### Option A: API Key Authentication (Quick)
- Add `X-API-Key` header validation via FastAPI dependency
- Store keys in environment variables
- **Pros:** Fast to implement, minimal code
- **Cons:** Not user-scoped, can't revoke per-user
- **Effort:** Small
- **Risk:** Low

### Option B: JWT + OAuth2 (Production-grade)
- FastAPI OAuth2PasswordBearer + JWT tokens
- User registration/login endpoints
- Token refresh flow
- **Pros:** Industry standard, per-user access control
- **Cons:** More complex, needs user management
- **Effort:** Large
- **Risk:** Medium

## Acceptance Criteria
- [ ] All endpoints require valid credentials
- [ ] Unauthorized requests return 401
- [ ] Existing frontend sends credentials with requests
