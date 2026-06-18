---
status: pending
priority: p3
issue_id: "037"
tags: [code-review, quality, frontend]
dependencies: []
---

# Progress SSE Lacks Document-Level Granularity

## Problem Statement
`backend/services/progress.py` reports progress at the matter level only. For multi-document matters, users can't see which document is currently processing.

## Findings
- **Location:** `backend/services/progress.py`

## Proposed Solutions
- Add `document_id` and `document_name` to progress events
- **Effort:** Small | **Risk:** Low
