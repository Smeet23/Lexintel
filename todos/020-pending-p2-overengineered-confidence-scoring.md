---
status: pending
priority: p2
issue_id: "020"
tags: [code-review, quality]
dependencies: []
---

# Over-Engineered Confidence Scoring (~200 LOC)

## Problem Statement
7 functions spanning ~200 lines (lines 536-789) compute detailed confidence breakdowns. Frontend only reads the final `score` float — all sub-scores are wasted computation.

## Findings
- **Location:** `backend/services/rag_engine.py:536-789`
- **Risk:** Wasted CPU cycles, code complexity
- **Evidence:** Frontend `AskResponse` type only uses `confidence` float

## Proposed Solutions

### Option A: Simplify to Single Score Function
- Replace 7 functions with one that returns a float
- Keep the algorithm but remove intermediate data structures
- **Pros:** ~150 LOC removal, faster
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Confidence scoring reduced to single function
- [ ] Same score quality, less code
