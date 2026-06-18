---
status: pending
priority: p2
issue_id: "023"
tags: [code-review, performance]
dependencies: []
---

# Per-Process LRU Cache Near-Zero Benefit

## Problem Statement
`backend/services/embedding_cache.py` uses a per-process LRU cache for embeddings. Celery workers are separate processes, so each has its own empty cache. Cache hit rate approaches zero in production.

## Findings
- **Location:** `backend/services/embedding_cache.py`
- **Risk:** Low — wastes memory, provides false sense of caching
- **Evidence:** Process-local dict, Celery forks new processes

## Proposed Solutions

### Option A: Redis-Based Cache
- Store embedding hashes → vectors in Redis
- Shared across all workers
- **Pros:** Actually effective
- **Effort:** Medium
- **Risk:** Low

### Option B: Remove Cache
- Remove the cache since it's ineffective
- **Pros:** Less code, less confusion
- **Effort:** Small
- **Risk:** None

## Acceptance Criteria
- [ ] Cache is either shared across workers or removed
