---
status: pending
priority: p2
issue_id: "002"
tags: [code-review, security, agent-native, api]
---

# Deferred: Rate limiting + standalone verification/lookup endpoints + /ask response_model

> **Note (2026-06-03):** The rate-limiting portion is superseded by the detailed, P1
> design in [[003-pending-p1-rate-limiting-and-cost-dos-protection]] (from the no-auth/rate-limiter
> review). This todo now tracks only the **standalone verification/lookup endpoints** and the typed
> `/ask` `response_model`.

## Problem Statement
1. **No rate limiting** on any public endpoint. `POST /matters/{id}/ask` and `/admin/*` drive paid
   external API spend (Cohere/Voyage/Gemini) and can saturate the DB pool.
2. **Agent-native gaps**: citation verification, claim verification, conflict detection, issue
   analysis, and raw citation lookup are only reachable as side effects of `/ask` — no standalone
   primitives. `/ask` declares `response_model=dict`, so its rich contract is not published via
   OpenAPI (the frontend hand-mirrors ~120 lines of types).

## Proposed Solution
- Add `slowapi` limiter: `/ask` 10/min/IP, graph endpoints 30/min/IP, `/admin/*` 5/min/IP (429 + Retry-After).
- Add `POST /matters/{id}/verify-citations`, `/verify-claims`, `/conflicts`, `POST /problem-formulation`,
  `GET /citations/lookup`, `POST /citations/lookup-batch` wrapping existing services (schemas already exist).
- Build a composed `QueryResponse` model (with optional verification/conflict/issue fields) and set it
  as `/ask`'s `response_model`; always echo effective `as_of_date`/`temporal_scope`.

## Acceptance Criteria
- [ ] Rate limiting active with 429 responses
- [ ] Every capability has a standalone, schema-typed endpoint
- [ ] `/ask` publishes a typed response_model in OpenAPI

## Work Log
- 2026-06-02: Identified by security + agent-native reviewers. Deferred (additive, lower risk).
- 2026-06-03: **DONE (non-rate-limit parts).** Added standalone endpoints — `POST /problem-formulation`,
  `GET /citations/lookup`, `POST /citations/lookup-batch`, `POST /matters/{id}/conflicts`,
  `POST /matters/{id}/verify-citations`, `POST /matters/{id}/verify-claims` — and a typed
  `AskResponse` `response_model` on `/ask` (permissive `extra='allow'`, full payload preserved,
  now in OpenAPI). Also Referrer-Policy `no-referrer` + `/precedents/search` fan-out cap (50).
  All endpoints live-verified (200 + real data; 422/400 guards); 364 tests green. Rate limiting
  intentionally NOT implemented (per user). Remaining: nothing — close after triage.
