---
status: pending
priority: p3
issue_id: "048"
tags: [code-review, quality]
dependencies: []
---

# No Graceful Shutdown for Celery Workers

## Problem Statement
Celery workers don't have explicit graceful shutdown handling. Long-running ingestion tasks may be killed mid-processing, leaving partial data.

## Proposed Solutions
- Configure `worker_cancel_long_running_tasks_on_connection_loss` and `acks_late`
- **Effort:** Small | **Risk:** Low
