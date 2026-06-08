---
status: pending
priority: p1
issue_id: "004"
tags: [code-review, security, auth, access-control, go-live]
dependencies: []
---

# No-auth posture — go-live hardening (must-fix before non-localhost exposure)

## Problem Statement
The app intentionally ships with **no authentication / no per-user authorization** for MVP
(`CLAUDE.md:6-26`). That is acceptable **only** for single-user loopback use — but
`uvicorn.run(host="0.0.0.0")` (`main.py:2297`) and the documented `--host 0.0.0.0` run command mean
the service already binds to all interfaces. On any shared network the no-auth posture becomes a set
of real P1/P2 exposures. This todo captures the **pre-public-exposure** hardening; full per-user auth
is tracked separately ([[005-pending-p2-tenancy-and-auth-seam]]).

## Findings (evidence)
**Load-bearing mitigation:** every PK is random **UUIDv4** (`models.py:24,77,110,238,260,...`) and
non-UUID input is 400-rejected — so there is **no mass IDOR by enumeration**. The residual risks are
the routes needing **no ID**, destructive/expensive ops gated only by knowing a UUID, and UUID leakage.

- **P1 `/precedents/*` is globally unscoped** (`main.py:1981 search`, `2084 list`, `2110 delete`):
  `GET /precedents` returns **every** saved precedent (verbatim `chunk_content`, query, matter_id,
  notes) to any caller with **no ID and no auth**; `/precedents/search` returns content snippets from
  **all** matters; `DELETE /precedents/{id}` deletes any precedent. This is the one place the UUID
  mitigation does **not** apply — the canary proving there is no tenancy boundary.
- **P1 destructive ops need only a known UUID** (`DELETE /matters/{id}` `:443`,
  `DELETE /documents/{id}` `:803` which **hard-deletes** chunks+Qdrant+blob irreversibly `:845-888`,
  `DELETE /queries` `:1373` wipes queries + soft-deletes conversations, `DELETE /precedents/{id}`).
- **P2 full per-matter exfil from one leaked UUID**: document blobs (`/document`, `/documents/{id}/download`),
  full chunk text (`/chunks`), Q&A history, drafts, contract reviews, and the **audit log**
  (`/audit-log` `:1940` — a compliance record readable by anyone with the matter UUID). Leak vectors:
  URL path (SPA), proxy logs, `Referrer-Policy: strict-origin-when-cross-origin` (`:108`), SSE channel.
- **P2 admin gate open under misconfig** (`_require_admin` `main.py:2140-2167`): well-designed
  (constant-time `hmac.compare_digest`, fail-closed 503 in prod) BUT a no-op when `ADMIN_API_KEY` unset
  + `debug=True` (a common copy-dev-`.env` mistake) → `/admin/reindex` open.
- **P2 CORS + no-auth** (`main.py:87-99`): `allow_origin_regex` for any localhost + `allow_credentials=True`.
  Loopback-only today, but CORS gives **zero** protection against scripted (curl) access — the dominant
  threat with no auth. Must lock to exact prod origins before exposure.
- **P3 upload surface** reachable without auth but well-constrained (MIME allow-list, magic-byte check,
  50MB cap pre+post, filename traversal block). Residual = unbounded jobs (see [[003-...]]).
- **P3 SSRF minimal**: only outbound is to fixed CourtListener host (user controls the query string, not
  the URL) — not classic SSRF. No SQL injection (ORM, parameterized; IDs coerced via `UUID()`).

## Go-live checklist (must-fix before ANY public/non-localhost exposure)
1. **Reverse proxy / firewall** in front enforcing per-IP limits + network allowlist (also covers [[003]]).
2. **Set `ADMIN_API_KEY`** and confirm **`debug=False`**; verify `/admin/reindex` → 401 without header.
   Consider also requiring loopback source IP for the dev no-op so a misconfigured public `debug=True` fails.
3. **Lock CORS**: drop the localhost regex (`main.py:95`) in prod; `allowed_origins` = exact prod origin(s).
4. **Gate or remove `/precedents` + `/precedents/search`** (cross-tenant reads, zero barrier).
5. **Add minimal auth + per-resource ownership** before real client data (see [[005-pending-p2-tenancy-and-auth-seam]]).
6. **Make document deletion recoverable or admin-gated** (currently hard-purges blob+vectors).
7. **Treat matter UUIDs as secrets**: `Referrer-Policy: no-referrer` for SPA pages; scrub UUIDs from proxy logs.
8. **Cap `/precedents/search` fan-out** to max-N matters regardless of auth.

## Proposed Solution
Two tracks: (a) the checklist above as a release gate; (b) wire the global auth seam from
[[005-pending-p2-tenancy-and-auth-seam]] so prod hardening is a config flip, not a 40-handler refactor.

## Acceptance Criteria
- [ ] Documented release gate: app is never bound to a non-loopback interface without items 1-4 done
- [ ] `ADMIN_API_KEY` enforced + `debug=False` verified in any deployed env
- [ ] CORS restricted to explicit origins in prod
- [ ] `/precedents/*` gated or removed for non-single-user deployments

## Work Log
- 2026-06-03: Surfaced by security-auditor review (theme: no auth & rate limiter). Random-UUID PKs
  noted as the key mitigating control; `/precedents/*` is the highest-priority pre-public fix.
