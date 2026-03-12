---
status: pending
priority: p3
issue_id: "040"
tags: [code-review, security]
dependencies: []
---

# Limited Request Validation on Endpoints

## Problem Statement
Several endpoints accept raw strings without validation (e.g., query text length, file type verification beyond extension).

## Findings
- **Location:** `backend/main.py` — various endpoints

## Proposed Solutions
- Add Pydantic request models with field validators
- **Effort:** Medium | **Risk:** Low
