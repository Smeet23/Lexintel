---
status: pending
priority: p3
issue_id: "035"
tags: [code-review, performance]
dependencies: []
---

# Qdrant Timeout 10s Too Low for Large Upserts

## Problem Statement
`backend/services/vector_store.py:68` sets Qdrant client timeout to 10 seconds. Large batch upserts (100+ vectors) may exceed this.

## Findings
- **Location:** `backend/services/vector_store.py:68`
- **Risk:** Low — occasional timeout on large documents

## Proposed Solutions
- Increase to 30s or make configurable
- **Effort:** Small | **Risk:** None
