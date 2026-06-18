---
status: pending
priority: p3
issue_id: "046"
tags: [code-review, architecture]
dependencies: []
---

# No API Versioning

## Problem Statement
All endpoints are unversioned (e.g., `/matters` not `/v1/matters`). Breaking changes will affect all clients simultaneously.

## Proposed Solutions
- Add `/api/v1/` prefix to all routes
- **Effort:** Small | **Risk:** Low (coordinate with frontend)
