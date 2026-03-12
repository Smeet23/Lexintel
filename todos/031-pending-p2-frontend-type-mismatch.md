---
status: pending
priority: p2
issue_id: "031"
tags: [code-review, quality, frontend]
dependencies: []
---

# Frontend Citation Type Differs from Backend

## Problem Statement
`frontend/lib/types.ts` Citation type shape differs from what the backend actually returns. Missing fields like `confidence_explanation`, `source_document` in `AskResponse`.

## Findings
- **Location:** `frontend/lib/types.ts`, `frontend/lib/api-services.ts`
- **Risk:** Medium — runtime errors, missing data display

## Proposed Solutions

### Option A: Sync Types with Backend
- Update frontend types to match actual backend response shape
- **Effort:** Small | **Risk:** Low

## Acceptance Criteria
- [ ] Frontend types match backend response schema exactly
