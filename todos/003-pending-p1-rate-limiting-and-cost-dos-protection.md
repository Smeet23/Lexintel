---
status: pending
priority: p1
issue_id: "003"
tags: [code-review, security, performance, rate-limiting, dos, cost]
dependencies: []
---

# Rate limiting — cost-amplification & DoS protection (no limiter exists)

Supersedes the rate-limit half of [[002-pending-p2-rate-limiting-and-standalone-verification-endpoints]].

## Problem Statement
The API has **no rate limiting anywhere** (`requirements.txt` has no `slowapi`/limiter;
`main.py` has only `CORSMiddleware` + `add_security_headers`). Every expensive endpoint is
unauthenticated AND unthrottled, and `uvicorn.run(host="0.0.0.0")` (`main.py:2297`, and the
documented run command) means the service is reachable off-loopback — so the "localhost MVP"
assumption does not actually bound exposure. A trivial request loop can drain paid Gemini/Cohere
budget, saturate the Celery queue, and exhaust the DB connection pool.

## Findings (evidence)
- **P1 `POST /ask`** (`main.py:1153`): one request = Cohere embed + Qdrant(30) + cross-encoder
  rerank + conflict NLI + Gemini generate + per-citation (CourtListener HTTP + cosine + **2nd
  Gemini**) + 2 DeBERTa NLI claim-verify passes (+ optional CourtListener research). 2+ Gemini calls
  and N HTTP calls per request. Hammering burns quota in minutes and pins a DB connection for the
  multi-second duration.
- **P1 uploads → unbounded Celery jobs** (`POST /matters` `main.py:317`, `POST /matters/{id}/documents`
  `main.py:616`): no per-request **file-count** cap; each file `send_task`s to the single `default`
  queue (`worker_prefetch_multiplier=1`) → queue flood, indefinite "processing", embedding-quota burn.
- **P1 `/admin/reindex`** (`main.py:2238`): no-arg call dispatches `reindex_all_matters_task`
  (re-embeds entire corpus); gated only by `_require_admin`, a permissive no-op when `ADMIN_API_KEY`
  unset + `debug=True`.
- **P2 `/precedents/search`** (`main.py:1981`): one request fans out a Qdrant search across **every
  ready matter** (`asyncio.gather` over all collections) — cost scales with corpus size.
- **P2 `/contract-review` / `/drafts`** (`main.py:1710,1825`): synchronous full-document Gemini per call.
- **P2 DB pool** (`database.py:17`): SQLAlchemy defaults = 5+10 = **15** connections; every route holds
  one via `Depends(get_db)`. ~15 concurrent slow LLM requests exhaust the pool → 30s `pool_timeout`
  stalls **all** endpoints (cascading 500s). Rate limiting is the front-line mitigation; raising
  `pool_size`/`max_overflow` is complementary.
- **P3 `/graph/*`**, default reads, and SSE `/progress` (600s hold) — connection tie-up under loops.
- **P3 X-Forwarded-For**: a naive XFF key func is spoofable (limit bypass). A `429` must still pass
  back through `CORSMiddleware` (added first, so it wraps the response — verify the handler returns a
  real `Response`).

## Proposed Solution (recommended: slowapi + Redis)
`slowapi` (FastAPI-native, backed by `limits`); Redis is already in the stack (Celery broker /
`settings.redis_url`) → distributed counters for free, no new infra.

**Single seam** (new `backend/ratelimit.py`, keep main.py from growing):
- `limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url, default_limits=["60/minute"])`
  with in-memory fallback if Redis is down (dev never hard-fails).
- In `main.py` after `app = FastAPI(...)`: `app.state.limiter = limiter`;
  `app.add_exception_handler(RateLimitExceeded, handler)` returning `{"detail": "Rate limit exceeded"}`
  + `Retry-After`. Keep `CORSMiddleware` first.

**Per-endpoint limits (per IP):**
| `/ask` 10/min · `/precedents/search` 10/min · `/contract-review`,`/drafts` 5/min · uploads 5/min ·
`/admin/*` 2/min · `/graph/*` 30/min · SSE establishment 10/min · default 60/min |

- Key on `request.client.host`; trust `X-Forwarded-For` **only** behind a new `settings.trust_proxy`
  flag (off by default).
- Add a **per-request file-count cap** (≤10) in `POST /matters` before the enqueue loop (blunts queue
  flooding independent of the limiter).

**MVP increment (ship first, ~1 file + decorators on 6 routes):** in-memory slowapi, default 60/min,
explicit decorators on `/ask`, `/contract-review`, `/drafts`, uploads, `/admin/reindex`, + the
file-count cap. **Full:** Redis storage, `/precedents/search`+`/graph/*`+SSE+default, proxy-aware key
func, DB pool sizing.

## Acceptance Criteria
- [ ] Hammering `/ask` past the limit returns **429 + Retry-After** (with CORS headers intact)
- [ ] `POST /matters` rejects > N files with 4xx before enqueuing any Celery job
- [ ] `/admin/*` limited to a few/min
- [ ] Limiter backed by Redis in prod, in-memory fallback in dev
- [ ] DB `pool_size`/`max_overflow` set explicitly (follow-up)

## Technical Details
Files: `backend/main.py`, `backend/requirements.txt` (`slowapi>=0.1.9`), `backend/config.py`
(`trust_proxy`, pool settings), new `backend/ratelimit.py`, `backend/database.py`, `backend/celery_app.py`.

## Work Log
- 2026-06-03: Surfaced by security-engineer + security-auditor review (theme: no auth & rate limiter).
