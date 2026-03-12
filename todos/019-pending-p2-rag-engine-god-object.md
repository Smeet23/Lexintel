---
status: pending
priority: p2
issue_id: "019"
tags: [code-review, architecture]
dependencies: []
---

# rag_engine.py God Object (1184 Lines, 5+ Responsibilities)

## Problem Statement
`backend/services/rag_engine.py` handles query orchestration, embedding, reranking, citation extraction, confidence scoring, and document summary generation. Single file with 1184 lines and 5+ distinct responsibilities.

## Findings
- **Location:** `backend/services/rag_engine.py`
- **Risk:** Hard to test, modify, and understand
- **Evidence:** 1184 lines, multiple unrelated concerns mixed together

## Proposed Solutions

### Option A: Extract Services
- Split into: `query_service.py`, `reranker.py`, `citation_extractor.py`, `confidence_scorer.py`
- Keep `rag_engine.py` as thin orchestrator
- **Pros:** Testable, maintainable
- **Cons:** Refactoring effort
- **Effort:** Large
- **Risk:** Medium (regressions if not well-tested)

### Option B: Incremental Extraction
- Extract one concern at a time starting with confidence scoring (~200 LOC)
- **Pros:** Lower risk per change
- **Effort:** Medium (per extraction)
- **Risk:** Low

## Acceptance Criteria
- [ ] No single file >400 LOC
- [ ] Each extracted service has focused tests
