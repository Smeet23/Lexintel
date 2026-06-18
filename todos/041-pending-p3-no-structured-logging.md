---
status: pending
priority: p3
issue_id: "041"
tags: [code-review, quality]
dependencies: []
---

# No Structured Logging

## Problem Statement
All logging uses format strings without structured fields. Makes log aggregation and searching difficult in production.

## Findings
- **Location:** All backend services

## Proposed Solutions
- Use `structlog` or JSON logging format
- **Effort:** Medium | **Risk:** Low
