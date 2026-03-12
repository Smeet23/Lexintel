---
status: pending
priority: p2
issue_id: "025"
tags: [code-review, quality]
dependencies: ["019"]
---

# Thin Wrappers in rag_engine.py

## Problem Statement
Several functions in `rag_engine.py` are trivial pass-throughs: `embed_query()` wraps embeddings service, `count_tokens_estimate()` is `len(text)//4`, `retrieve_chunks()` wraps `search_vectors()`. These add indirection without value.

## Findings
- **Location:** `backend/services/rag_engine.py`
- **Risk:** Low — code noise, misleading abstraction layers

## Proposed Solutions

### Option A: Inline the Calls
- Replace wrapper calls with direct service calls
- **Effort:** Small | **Risk:** Low

## Acceptance Criteria
- [ ] No single-line pass-through wrappers remain
