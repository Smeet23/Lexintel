---
status: pending
priority: p3
issue_id: "042"
tags: [code-review, quality]
dependencies: []
---

# No Health Check Endpoint

## Problem Statement
No `/health` or `/ready` endpoint exists for load balancers, Kubernetes probes, or monitoring.

## Findings
- **Location:** `backend/main.py`

## Proposed Solutions
- Add `/health` checking DB + Qdrant + Redis connectivity
- **Effort:** Small | **Risk:** None
