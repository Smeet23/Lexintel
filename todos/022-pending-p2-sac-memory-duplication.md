---
status: pending
priority: p2
issue_id: "022"
tags: [code-review, performance]
dependencies: []
---

# SAC Memory Duplication in tasks.py

## Problem Statement
At `backend/tasks.py:167-171`, the SAC (Summary-Augmented Chunking) step creates a full parallel list of augmented strings — duplicating the entire document's chunk content in memory.

## Findings
- **Location:** `backend/tasks.py:167-171`
- **Risk:** Medium — doubles memory usage during ingestion for large documents
- **Evidence:** List comprehension creating `[summary + chunk for chunk in chunks]`

## Proposed Solutions

### Option A: Generator-Based Approach
- Use a generator instead of materializing the full list
- Prepend summary during embedding batch iteration
- **Effort:** Small | **Risk:** Low

## Acceptance Criteria
- [ ] No full duplication of chunk content in memory
- [ ] Embedding quality unchanged
