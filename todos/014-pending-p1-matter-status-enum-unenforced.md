---
status: pending
priority: p1
issue_id: "014"
tags: [code-review, data-integrity]
dependencies: []
---

# MatterStatus Enum Defined but Never Enforced

## Problem Statement
`MatterStatus` enum at `backend/models.py:13-18` is defined but the `Matter.status` column uses a plain string. Status transitions aren't validated — any string can be written.

## Findings
- **Location:** `backend/models.py:13-18`
- **Risk:** Medium — invalid status values, no state machine enforcement
- **Evidence:** Column type is String, not Enum; no validation on writes

## Proposed Solutions

### Option A: Enforce Enum at DB Level
- Change column type to use SQLAlchemy `Enum(MatterStatus)`
- Add migration to convert existing values
- **Pros:** Database-enforced validity
- **Effort:** Small
- **Risk:** Low (verify existing values match enum)

## Acceptance Criteria
- [ ] Status column uses Enum type
- [ ] Invalid status values rejected at DB level
