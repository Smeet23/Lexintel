---
status: pending
priority: p2
issue_id: "024"
tags: [code-review, performance, frontend]
dependencies: []
---

# Frontend Unconditional 10s Polling

## Problem Statement
`frontend/hooks/use-matters.ts` polls `/matters` every 10 seconds regardless of whether any matters are processing. Wasted bandwidth and server load when idle.

## Findings
- **Location:** `frontend/hooks/use-matters.ts`
- **Risk:** Medium — unnecessary load on backend

## Proposed Solutions

### Option A: Conditional Polling
- Only poll when `matters.some(m => m.status === 'processing')`
- Stop polling when all matters are complete
- **Effort:** Small | **Risk:** Low

### Option B: SSE/WebSocket
- Use existing SSE progress system for status updates
- **Pros:** Real-time, no polling
- **Cons:** More infrastructure
- **Effort:** Medium | **Risk:** Low

## Acceptance Criteria
- [ ] No polling when all matters are idle
