---
status: pending
priority: p1
issue_id: "008"
tags: [code-review, security]
dependencies: []
---

# Content-Disposition Header Injection

## Problem Statement
File download endpoints at `backend/main.py:609` and `main.py:912` inject user-supplied filenames directly into `Content-Disposition` headers without sanitization. Malicious filenames can inject headers.

## Findings
- **Location:** `backend/main.py:609`, `backend/main.py:912`
- **Risk:** High — header injection, potential XSS via filename
- **Evidence:** No filename sanitization before header construction

## Proposed Solutions

### Option A: Sanitize Filename
- Strip/escape special characters, use RFC 6266 compliant encoding
- **Pros:** Simple, covers the attack vector
- **Effort:** Small
- **Risk:** Low

## Acceptance Criteria
- [ ] Filenames sanitized before use in headers
- [ ] Special characters (quotes, newlines, semicolons) escaped or removed
