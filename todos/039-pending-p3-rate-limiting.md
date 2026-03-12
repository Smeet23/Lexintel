---
status: pending
priority: p3
issue_id: "039"
tags: [code-review, security]
dependencies: ["001"]
---

# No Rate Limiting on API Endpoints

## Problem Statement
No rate limiting exists on any endpoint. External-facing API calls (especially `/ask` which triggers Gemini API calls) can be abused to exhaust API quotas and compute.

## Findings
- **Location:** `backend/main.py` — all routes

## Proposed Solutions
- Add `slowapi` or similar rate limiter middleware
- **Effort:** Small | **Risk:** Low
