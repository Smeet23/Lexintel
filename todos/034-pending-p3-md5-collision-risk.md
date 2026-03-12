---
status: pending
priority: p3
issue_id: "034"
tags: [code-review, quality]
dependencies: []
---

# MD5-Based Qdrant Point ID Collision Risk

## Problem Statement
`backend/services/vector_store.py:86-104` uses MD5 hash truncated to UUID for Qdrant point IDs. Theoretical collision risk, and MD5 is deprecated for security contexts.

## Findings
- **Location:** `backend/services/vector_store.py:86-104`
- **Risk:** Very low — theoretical collision, not a practical concern at current scale

## Proposed Solutions
- Switch to SHA-256 truncated to UUID format
- **Effort:** Small | **Risk:** Low
