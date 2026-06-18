---
status: pending
priority: p3
issue_id: "043"
tags: [code-review, security]
dependencies: []
---

# Permissive CORS Configuration

## Problem Statement
CORS middleware likely allows all origins. Should restrict to frontend domain in production.

## Findings
- **Location:** `backend/main.py` — CORS middleware setup

## Proposed Solutions
- Configure `allow_origins` from environment variable
- **Effort:** Small | **Risk:** Low
