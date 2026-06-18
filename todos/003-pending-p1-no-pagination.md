---
status: pending
priority: p1
issue_id: "003"
tags: [code-review, performance, security]
dependencies: []
---

# No Pagination on List Endpoints

## Problem Statement
`GET /matters` (line 89) and `GET /chunks` (lines 945-971) return all records unbounded. With growth, these become OOM risks and performance bottlenecks.

## Findings
- **Location:** `backend/main.py:89` (matters), `backend/main.py:945-971` (chunks)
- **Risk:** High — unbounded queries, memory exhaustion, slow responses
- **Evidence:** No `limit`/`offset` or cursor parameters

## Proposed Solutions

### Option A: Offset Pagination
- Add `?page=1&per_page=20` query params
- **Pros:** Simple, familiar
- **Cons:** Offset drift on concurrent writes
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] List endpoints accept pagination parameters
- [ ] Default page size is reasonable (20-50)
- [ ] Response includes total count and page metadata
- [ ] Frontend updated to paginate
