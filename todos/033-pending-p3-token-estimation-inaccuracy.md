---
status: pending
priority: p3
issue_id: "033"
tags: [code-review, quality]
dependencies: []
---

# Token Estimation Underestimates by 20-30%

## Problem Statement
`count_tokens_estimate()` uses `len(text)//4` which underestimates token count by 20-30% for legal text with specialized terminology.

## Findings
- **Location:** `backend/services/rag_engine.py`
- **Risk:** Low — may exceed context window occasionally

## Proposed Solutions
- Use `tiktoken` or a proper tokenizer for accurate counts
- **Effort:** Small | **Risk:** Low
