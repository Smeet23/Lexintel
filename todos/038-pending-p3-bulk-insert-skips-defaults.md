---
status: pending
priority: p3
issue_id: "038"
tags: [code-review, data-integrity]
dependencies: []
---

# bulk_insert_mappings Skips ORM Defaults

## Problem Statement
`backend/tasks.py:160` uses `bulk_insert_mappings()` which bypasses ORM-level column defaults like `created_at`. Rows may have NULL timestamps.

## Findings
- **Location:** `backend/tasks.py:160`

## Proposed Solutions
- Explicitly set `created_at` in mapping dicts, or use `bulk_save_objects()`
- **Effort:** Small | **Risk:** Low
