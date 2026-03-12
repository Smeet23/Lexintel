---
status: pending
priority: p2
issue_id: "029"
tags: [code-review, performance]
dependencies: []
---

# full_text Concatenation Duplicates Entire Document

## Problem Statement
At `backend/tasks.py:109`, the full document text is concatenated into a single string for enrichment that only uses the first 30K chars. This duplicates the entire document text in memory unnecessarily.

## Findings
- **Location:** `backend/tasks.py:109`
- **Risk:** Medium — memory waste for large documents

## Proposed Solutions

### Option A: Truncate at Concatenation
- Only concatenate up to 30K chars since that's all enrichment uses
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] No full document duplication when only partial text needed
