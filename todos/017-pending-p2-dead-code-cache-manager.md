---
status: pending
priority: p2
issue_id: "017"
tags: [code-review, quality]
dependencies: []
---

# Dead Code: cache_manager.py (156 LOC)

## Problem Statement
`backend/services/cache_manager.py` implements a Redis `QueryCache` that is never integrated into any endpoint or service. 156 lines of dead code. Config has `cache_enabled`/`cache_ttl_seconds` that reference nothing.

## Findings
- **Location:** `backend/services/cache_manager.py`, `backend/config.py` (unused cache settings)
- **Evidence:** Zero imports, config fields unreferenced

## Proposed Solutions

### Option A: Delete File + Remove Config Fields
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] File deleted
- [ ] Unused config fields removed
