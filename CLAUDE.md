# LexIntel — Project Notes for Claude

Legal RAG application: PDF/DOCX/TXT → chunk → embed (Cohere) → Qdrant → answer
(Gemini/Groq). Backend: FastAPI + Celery + PostgreSQL + Qdrant. Frontend: Next.js 14.

## Authentication — NONE (MVP)

**We are in MVP. There is intentionally NO authentication. Do not add auth gating.**

- The frontend has **no auth guard / middleware / redirect**. Any route is directly
  accessible. `frontend/app/login/page.tsx` is a non-enforced **demo stub** that only
  sets a `lexintel_token` value in `localStorage`; nothing checks it. It exists for
  future use and can be ignored.
- Backend API endpoints are **open** (no login, no per-user authorization). This is
  deliberate for MVP.
- The only gate is `/admin/*` (reindex), which requires `X-Admin-Api-Key` **only when
  `ADMIN_API_KEY` is set**. In local/dev (no key, `debug=True`) it is effectively open.
  This is a destructive-operation safety guard, not user login — leave it.
- Do **not** introduce login flows, session/cookie auth, JWT, or route guards during
  MVP unless explicitly asked.

## CORS (local dev)

`backend/main.py` allows any **localhost / 127.0.0.1 port** via `allow_origin_regex`
(in addition to the configured `allowed_origins`). So the Next.js dev server works on
any port (3000, 3001, 3100, …). Loopback-only — no production exposure.

## Running locally

- Infra: `docker compose up -d postgres redis qdrant azurite`
- Migrate: `python -m alembic -c backend/alembic.ini upgrade head`
- API: from `backend/` → `uvicorn main:app --host 0.0.0.0 --port 8000`
- Worker: `bash backend/run_worker.sh` (runs from project root; uses the `default`
  Celery queue — producer and worker must agree on the queue name).
- Frontend: from `frontend/` → `npm run dev` (defaults to :3000).

## Agentic loop — context management / summarisation

The LangGraph agentic pipeline (`backend/services/agentic_rag.py`) starts with a
`context_manager_node` (`START → context_manager → router`). Once a conversation
exceeds `agentic_context_summarize_after` turns, it summarises the OLDER turns
into a rolling brief (via the fast/free Groq LLM) and keeps the most-recent
`agentic_context_keep_recent` turns verbatim — bounding prompt context for long
multi-turn sessions. Best-effort: any failure leaves history untouched. The
managed history flows to every node and to the fast-path delegation. Config:
`agentic_context_summarize_after` (6), `agentic_context_keep_recent` (4),
`agentic_context_summary_max_chars` (1500).

## Citation resolution (correctness-critical)

Inline `[n]` markers map POSITIONALLY to score-desc-ordered chunks. `final_chunks`
is sorted into one canonical order before `format_legal_context`,
`extract_citations`, the verifiers, and the sources list, so `[n]` resolves to the
SAME source everywhere (otherwise a claim verifies against the wrong chunk).
Grounded `supporting_excerpt` is the query-relevant window of the chunk (not
`chunk[:500]`).

## LLM provider routing

- `settings.llm_answer_provider` (env `LLM_ANSWER_PROVIDER`) selects the RAG
  answer LLM: `"gemini"` (default) or `"groq"` (fast/free). The other provider is the
  automatic fallback. Set `LLM_ANSWER_PROVIDER=groq` to avoid Gemini token cost in dev.
- The **ingest/graph** LLM services now also honor `llm_answer_provider` and fall
  back to the other backend: `authority_detector`, `document_summary`,
  `temporal_extractor`, `citation_extractor` (LLM extraction),
  `citation_graph.classify_treatment`, and `extract_relationships_llm`
  (provider-ordered). Previously these were Gemini-first with no Groq fallback, so
  a Gemini 429 silently degraded the citation graph (regex-only) even on Groq.
- Unified LLM access lives in `backend/services/llm.py` (`agenerate`/`generate`).
- **Quota note:** Gemini free tier ≈ 20 req/day, Groq ≈ 100k tokens/day. When BOTH
  are exhausted, RAG answers + relationship extraction return empty (graph builds 0
  edges) — an external wall, not a bug. The graph-injection path is verifiable
  token-free by calling `rag_engine._build_graph_context` directly.

## Local models / Celery worker (macOS)

`SemanticChunker` (all-MiniLM) and the CrossEncoder reranker are pinned to CPU by
default (`LOCAL_MODEL_DEVICE`, default `cpu`). Apple-MPS initialised inside a
Celery **prefork** child crashes the worker (`MPSKernel ... MTLCompilerService
unavailable`). CPU is fine for these tiny models; the prefork pool (`-c 4`) works.
Override `LOCAL_MODEL_DEVICE=mps`/`cuda` only on a non-forking pool / GPU box.

## Testing

- Unit/integration: `cd backend && ./venv/bin/python -m pytest tests/ --ignore=tests/test_real_e2e_rag.py --ignore=tests/test_real_pdfs.py`
  (those two are stale real-API suites). Current: 364 passing.
- **Browser regression (agent-browser as QA):** `scripts/browser_regression.sh [FRONTEND_URL] [MATTER_ID]`
  — re-runnable; drives every route + matter-workspace tab + the Ask-AI flow, asserting each renders
  with no console errors. Exit 0 = all pass. Needs frontend + backend running and `agent-browser` on PATH.
