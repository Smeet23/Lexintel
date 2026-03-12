---
status: pending
priority: p3
issue_id: "047"
tags: [code-review, quality]
dependencies: []
---

# OpenAPI Docs Incomplete

## Problem Statement
FastAPI auto-generates OpenAPI docs but response models aren't specified on most endpoints. Swagger UI shows generic responses.

## Proposed Solutions
- Add `response_model` to all endpoints
- **Effort:** Medium | **Risk:** None
