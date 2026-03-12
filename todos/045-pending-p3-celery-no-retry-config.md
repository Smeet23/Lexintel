---
status: pending
priority: p3
issue_id: "045"
tags: [code-review, quality]
dependencies: []
---

# Celery Task Missing Retry Configuration

## Problem Statement
`process_document_task` in `backend/tasks.py` has no `max_retries`, `retry_backoff`, or `autoretry_for` configuration. Transient failures (API timeouts, DB blips) cause permanent task failure.

## Proposed Solutions
- Add `autoretry_for=(Exception,)`, `max_retries=3`, `retry_backoff=True`
- **Effort:** Small | **Risk:** Low
