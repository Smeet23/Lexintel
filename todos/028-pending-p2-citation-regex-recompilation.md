---
status: pending
priority: p2
issue_id: "028"
tags: [code-review, performance]
dependencies: []
---

# Citation Regex Patterns Recompiled Per Query

## Problem Statement
Citation extraction regex patterns at `rag_engine.py:354-474` are compiled on every query call instead of once at module level.

## Findings
- **Location:** `backend/services/rag_engine.py:354-474`
- **Risk:** Low — minor performance waste

## Proposed Solutions

### Option A: Module-Level Compilation
- Compile regex patterns once as module constants
- **Effort:** Small | **Risk:** None

## Acceptance Criteria
- [ ] Regex patterns compiled once at module load
