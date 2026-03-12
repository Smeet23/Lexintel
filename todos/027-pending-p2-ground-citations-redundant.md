---
status: pending
priority: p2
issue_id: "027"
tags: [code-review, quality]
dependencies: []
---

# ground_citations_in_source() Redundant with extract_citations()

## Problem Statement
`ground_citations_in_source()` at `rag_engine.py:477-533` re-processes citations that were already extracted by `extract_citations()`. Both functions parse the LLM response for citation markers.

## Findings
- **Location:** `backend/services/rag_engine.py:477-533`
- **Risk:** Low — duplicated work, harder to maintain

## Proposed Solutions

### Option A: Merge into Single Function
- Combine extraction + grounding into one pass
- **Effort:** Medium | **Risk:** Low

## Acceptance Criteria
- [ ] Citation processing happens in a single pass
