---
status: pending
priority: p2
issue_id: "005"
tags: [code-review, architecture, auth, multi-tenancy, seam]
dependencies: []
---

# Auth + rate-limit seam & tenancy column (make MVP→prod a config flip, not a refactor)

## Problem Statement
Auth and rate-limiting can be added cleanly later **only if the seams exist now**. Two are
well-positioned (global dependency / middleware); one — **tenancy/ownership** — is a latent invasive
retrofit because no model has an owner column. Doing the cheap forward-prep now converts a ~51-site
refactor + data backfill into a handful of edits.

## Findings (evidence)
- **No global auth/rate-limit seam in use** (`main.py`): 40 route decorators, each declaring its own
  `Depends(get_db)`; no `FastAPI(dependencies=[...])` and no per-router `dependencies=`. BUT the seam
  *exists* — there's already one `@app.middleware("http")` (`add_security_headers` `:102-109`) proving
  the pattern. App-level `dependencies=[...]` or a second middleware injects auth+limit globally
  **without** editing the 40 handlers. (P2: structure permits the clean path but doesn't signal it —
  invites copy-paste-into-every-signature.)
- **`_require_admin` is the right pattern, wired wrong** (`main.py:2140-2167`): constant-time compare +
  tri-state config gate (key→enforce, unset+debug→warn-open, unset+prod→503) is exactly the MVP→prod
  toggle to generalize. But it's called **imperatively** (`_require_admin(request)` at `:2179,:2249`),
  so it's not in OpenAPI, can't be dependency-overridden in tests, and a new admin route that forgets
  the call is silently open. (P3)
- **No tenancy on any model** (`models.py:21-45`): `Matter` has no `user_id`/`owner_id`/`tenant_id`;
  no `User` table among the 14 models; `AuditLog.user` is free-text `"System"`. Retrofitting per-tenant
  authZ later touches **51** `Matter.id ==`/`matter_id ==` filter sites + a FK migration + a data
  backfill of existing rows — risky once there's prod data. (P1 if deferred whole.)

## Proposed Solution (MVP-pragmatic, behavior-neutral now)
1. **`backend/auth.py`** with two dependencies — `verify_caller` and `rate_limit` — each a **no-op when
   its env var is unset, fail-closed in prod**, reusing `_require_admin`'s tri-state gate. Attach
   globally via `FastAPI(dependencies=[Depends(verify_caller), Depends(rate_limit)])` at `main.py:79`.
   MVP behavior unchanged (no-ops); prod hardening = set env vars.
2. **Promote `_require_admin` → `Depends`** and group `/admin/*` under
   `APIRouter(prefix="/admin", dependencies=[Depends(require_admin)])` so the imperative calls vanish
   and new admin routes are protected by construction. Hoist `import os` to module scope.
3. **Add nullable `tenant_id` to `Matter` NOW** (forward migration, default `"default"`) — zero behavior
   change, but removes the schema-migration+backfill from the future critical path (the painful part).
4. **Route reads through `get_matter_or_404(matter_id, ctx)`** so the 51 filter sites resolve scope from
   one injected context instead of inlining the predicate — collapses the future authZ change to ~1 place.
5. Have `verify_caller` set `request.state.tenant` (constant `"default"` today); per-tenant filtering
   later becomes "read `request.state.tenant`," already plumbed.

## Acceptance Criteria
- [ ] Global auth + rate-limit attachable via one app-level change (verified no-op in MVP)
- [ ] `/admin/*` protected via router dependency, not imperative calls; covered by a test using dependency-override
- [ ] `Matter` has a nullable `tenant_id` (migration applied; default tenant) with no behavior change
- [ ] Matter fetches funnel through a single `get_matter_or_404` helper

## Technical Details
Files: `main.py` (`:79` app deps, `:102` middleware, `:2140` admin), new `backend/auth.py`,
`backend/models.py` (Matter), new alembic migration, `backend/config.py`.

## Work Log
- 2026-06-03: Surfaced by architecture review (theme: no auth & rate limiter). Auth/limit seams = low
  future cost; tenancy column = highest-leverage pre-launch prep.
