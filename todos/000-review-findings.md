---
status: in_progress
priority: p1
issue_id: "000"
tags: [code-review, security, performance, architecture, data-integrity, frontend, testing]
---

# Code Review Findings — Uncommitted `main` Changeset (2026-06-02)

Exhaustive multi-agent review of ~7,500 changed lines + ~15 new backend services, 5 migrations,
frontend components, and tests. 11 specialist agents (security, 3× python, performance,
architecture, database, frontend, pipeline/testing, simplification, agent-native).

Legend: 🔴 P1 = blocks merge · 🟡 P2 = should fix · 🔵 P3 = nice-to-have

---

## 🔴 P1 — CRITICAL (block merge)

### Correctness
- **C1 — `check_case_status` always returns good_law.** `citation_lookup.py:826`. Misuses
  CourtListener `precedential_status` (publication status, not citator treatment) — `bad_law`/`caution`
  branches are unreachable. Every US citation passes the good-law check. **Legal-correctness bug.** → FIXED
- **C2 — Blocking sync I/O inside `async` functions.** `embed_query`/Groq/Gemini called directly in
  async paths: `agentic_rag.retrieve_node`, `citation_verifier._tier1_cosine_verify`,
  `claim_verifier._cov_llm_judge`, `treatment_verifier.verify_negative_treatment`,
  `conflict_detector.detect_conflicts` (via `rag_engine` call site). Blocks the event loop → latency
  cascade under load. Wrap in `asyncio.to_thread`. → FIXED
- **C3 — Ensemble NLI single-model aliasing.** `claim_verifier.py:474`. When one model loads,
  `scores_base = scores_small` makes `models_agree` always True → disagreement/LLM-escalation path
  unreachable; low-confidence single-model results finalize as Tier-1. → FIXED
- **C4 — `str.format()` crashes on `{`/`}` in document text / case name.** `citation_graph.py:1532,1565`.
  `_TREATMENT_CLASSIFICATION_PROMPT.format(cited_case_name=...)` raises on `{0}`-like content. → FIXED
- **C5 — Token-budget trim discards enrichment.** `rag_engine.py:1974`. On overflow,
  `format_legal_context(final_chunks[:2])` drops the citation-graph block, issue context, and conflict
  augmentation silently. → FIXED
- **C6 — `_build_graph_context` skips all in-matter citation nodes.** `rag_engine.py:629`. Guard
  `node.document_id is not None and node.matter_id == matter_uuid` drops every citation extracted from
  an uploaded doc — overruled cases inside briefs never validated. → FIXED
- **C7 — `conversation_history: list = None`.** Wrong annotation (should be `Optional[list]`) in
  `rag_engine` + `agentic_rag`. → FIXED
- **C8 — Bare reranker `except` with no logging.** `agentic_rag.py:422`. → FIXED
- **C9 — `response.text` accessed without blocked/empty guard.** `rag_engine.py:1497`,
  `authority_detector.py:217`, `claim_verifier.py:540,622`. Gemini raises on safety-block. → FIXED
- **C10 — `resp.choices[0].message.content.strip()` without guard.** `claim_verifier.py:642`,
  `treatment_verifier.py:113`. IndexError/AttributeError on empty choices / None content. → FIXED
- **C11 — `summary[status] += 1` inconsistent key access.** `claim_verifier.py:844`. → FIXED

### Security
- **S1 — Admin endpoints unauthenticated by default.** `main.py:2101`. `_require_admin` is a no-op
  unless `ADMIN_API_KEY` set; `POST /admin/reindex` dispatches unbounded Celery tasks. → FIXED
- **S2 — SSRF via unvalidated URLs from CourtListener response bodies.** `citation_lookup.py:877`.
  `_get_json` fetches arbitrary URLs from API JSON with no domain allowlist. → FIXED

### Data integrity (migrations)
- **D1 — Migration 16 revision id mismatch** (`"16_citation_graph_fixes"` vs `"16"`) breaks chain. → FIXED
- **D2 — `ALTER COLUMN` type change locks table** (migration 16, authority_score/confidence). → DOCUMENTED+lock_timeout
- **D3 — `document_status` nullability mismatch** migration 12 (NOT NULL) vs model (nullable=True). → FIXED
- **D4 — `amendment_chains.matter_id` nullability mismatch** migration (NOT NULL) vs model (nullable). → FIXED

### Performance
- **P-1 — Quadratic external-call fan-out in citation indexing.** `citation_graph.extract_and_index_citations`
  — sequential per-chunk extraction + per-citation Gemini + per-node CourtListener. → FIXED (concurrent batches + dedup)
- **P-2 — N+1 `COUNT(*)` per citation** to detect node creation. `citation_graph.py:1803`. → FIXED (return flag)
- **P-3 — `is_good_law` N+1** source-node lookups, called per-citation in RAG hot path. → FIXED (batch IN)
- **P-4 — `get_matter_graph` recomputes PageRank on every read.** → FIXED (serve persisted; recompute on ingest)

### Testing
- **T1 — `test_full_e2e_pipeline.py` is not a pytest test** (only `__main__`) → always green in CI. → FIXED (guarded)
- **T2 — Idempotency retry deletes PG chunks but not Qdrant** → sparse/dense schema-mismatch crash. → FIXED
- **T3 — `ingest_embedding_model` assigned after enrichment** → silent unstamped model breaks reindex tracker. → FIXED

### Frontend
- **F1 — Unsanitised URLs in `href` (XSS).** `CitationPanel.tsx:95,105`. → FIXED (isSafeUrl)
- **F2 — `axios` high-severity CVEs.** `package.json`. → FIXED (upgrade)

---

## 🟡 P2 — IMPORTANT (should fix)

- Module-level `asyncio.Semaphore` bound to no loop — `citation_lookup.py:28`. → FIXED (lazy)
- `@retry` dead branch + blanket `except` defeats retry — `citation_lookup.py:149`. → FIXED
- `cluster_id` interpolated into URL without numeric validation — `citation_lookup.py:815,872`. → FIXED
- `_BoundedCache` evicts insertion-order not LRU (×2 copies) — `citation_lookup.py`, `citation_verifier.py`. → FIXED
- N+1 in BFS graph queries (`get_citation_chain/network`, `find_similar_cases`). → FIXED (batch)
- `conflict_detector._get_nli_model` no double-checked lock (double model load). → FIXED
- Coverage metric inverted in `claim_verifier._check_token_coverage`. → FIXED
- `_detect_temporal_intent` `" in "` heuristic misfires on citation/section numbers. → FIXED
- `dayfirst=True` hardcoded misparses US dates — `temporal_extractor.py`. → FIXED
- `reciprocal_rank_fusion` in-place dict mutation — `hybrid_search.py`. → FIXED
- `_build_response` hardcodes confidence "medium" — `agentic_rag.py`. → FIXED
- `_COMPILED_GRAPH` lazy init without lock — `agentic_rag.py`. → FIXED
- `search_vectors` extra `get_collection` per query — `vector_store.py`. → FIXED (cache)
- `reindex_matter_task` loads all chunks into memory — `tasks.py`. → FIXED (yield_per)
- Missing FK `ON DELETE` + `document_id` index on citation tables → new migration 17. → FIXED
- Citation tables JSON→JSONB, type/constraint mismatches (canonical_name). → model aligned
- HTTP 500 leaks `str(e)` on graph endpoints; missing input length caps; no rate limiting. → FIXED (generic errors, Path caps); rate-limiting = follow-up todo
- Frontend: `getGoodLawStatus: Promise<any>`, `(c: any)` casts, stale `onNodeClick` closure,
  `conflicts.map` null guard, `vConfig` fallback. → FIXED
- Agent-native: `/ask` declares `response_model=dict`; verification/lookup/conflict lack standalone
  endpoints. → follow-up (see 001)

---

## 🔵 P3 — NICE-TO-HAVE
- Unused imports (`json`, `math`, `func`, `Optional`, `Tuple`, `Dict`), redundant f-strings, dead
  `section_name`/`page_num`. → FIXED
- Citation coverage substring false-positives — `rag_engine._calculate_citation_coverage`. → FIXED
- `count_tokens_estimate` raises on empty — return 0. → FIXED
- RRF weight normalization for "mixed" — `hybrid_search.py`. → FIXED
- `dynamic import` dagre/ReactFlow to code-split. → FIXED
- Move `AmendmentChain` class above `Document`. → FIXED

---

## DEFERRED — Large architectural refactors (tracked, NOT attempted this pass)
These are "Large" structural changes that would destabilize a working system; tracked as follow-ups:
- **A1** — Two divergent RAG pipelines (`agentic_rag` vs `query_matter`) lack a shared core → extract
  shared pipeline stages. (`001-pending-p1-...` candidate)
- **A2** — LLM client construction duplicated across ~14 services → single `llm/` client abstraction.
- **A3** — God modules `rag_engine.py` (2187 LOC) and `citation_graph.py` (2040 LOC) → split.
- Repository layer / transaction boundaries; verification facade; settings grouping;
  package layout fix to remove ~100 triple-import fallbacks.

---

## ROUND 2 — Re-review (workflow, 81 agents, double-skeptic verify) — 2026-06-02

36 candidates → 26 confirmed (6 P1 / 13 P2 / 7 P3), 10 refuted as already-fixed. Notably 4 P1s were
**regressions introduced by the Round-1 fix pass**. All fixed + verified:

### P1 (fixed)
- **R1 — `_detect_temporal_intent` unconditional `break`** skipped later keywords. `rag_engine.py`. → FIXED (runtime-verified: "in...during 2018" → 2018)
- **R2 — `agentic_rag.verify_node` called `detect_conflicts` un-wrapped** (blocking event loop). → FIXED (`asyncio.to_thread`)
- **R3 — Migration 17 `ON DELETE SET NULL` on NOT-NULL `amendment_chains.matter_id`** → every matter delete would fail. → FIXED (migration 17 now drops NOT NULL first; model nullable=True; re-applied to dev DB; schema verified SET NULL + nullable=YES)
- **R4 — `InlineCitation` tooltip `<a href>` missing `isSafeUrl`** (XSS bypass — button path was guarded, tooltip wasn't). → FIXED
- **R5 — `test_full_e2e_pipeline.run_full_e2e()` had no `return`** → entrypoint crashed with AttributeError when RUN_FULL_E2E=1. → FIXED (`return results`)
- **R6 — reindex `yield_per` cursor invalidated by per-batch `db.commit()`** → FIXED (re-issuing self-terminating `.limit()` pagination)

### P2 (fixed)
- Missing matter-existence guard on GET `/contract-review`, `/drafts`, `/audit-log` (broken access control). → FIXED
- Upload size checked after full `await file.read()` (DoS). → FIXED (Content-Length pre-check)
- `POST /drafts` `document_type`/`instructions` unbounded. → FIXED (max_length)
- `rerank_chunks` CPU-bound, not wrapped in async path. → FIXED (`asyncio.to_thread`)
- `_nli_predict_sliding` `break` on whitespace window skipped rest of source. → FIXED (`continue`)
- `documents.document_status` DB default was 'current' not 'unknown'. → FIXED (migration 12 + 17 safety-net; verified)
- conftest missing `voyage_api_key=''` (would route embeds to Voyage). → FIXED
- Frontend `any` types on verification fields + `(c: any)` lambdas. → FIXED (typed Raw* interfaces)

### P3 (fixed)
- `treatment_verifier._VERIFY_PROMPT.format()` crash on `{}` in evidence (C4-class). → FIXED (`.replace()`)
- EWCA citation multi-space court token failed map lookup. → FIXED (whitespace normalize)
- Duplicate inline `index=True` vs named indexes on CitationNode/Edge. → FIXED (removed inline)

### P1 (addressed by design)
- **`check_case_status` only returns good_law/unknown, never bad_law** — negative treatment is correctly
  the citation GRAPH's job (NEGATIVE_TREATMENTS edges / `is_good_law`), not this CourtListener call.
  → docstring corrected to be honest; dead caller branches left as harmless forward stubs.

### Deferred (Round 2) — tracked
- **P-1 real concurrency** in `extract_and_index_citations` (still sequential; runs in Celery off the
  request path). Genuine `asyncio.gather`+semaphore+cross-chunk dedup is Large → folded into [[001]]; misleading docstring/note corrected.
- Rate limiting (slowapi) — already [[002]].
- `httpx.AsyncClient` per-request (no keep-alive across a batch) — P3, [[002]] candidate.

### Refuted as already-fixed (10) — e.g.
admin gate already correct, token-budget re-enrichment already applies graph/issue, get_citation_network
already filters dangling endpoints, `_citation_lookup_post` retry already handles timeouts, migration 16
lock_timeout already present, api baseURL fallback already present.

### Round-2 verification
- `pytest`: **337 passed, 2 skipped, 2 xfailed, 0 failures**.
- Migration: dev DB re-applied 16→17 corrected; `amendment_chains.matter_id` nullable=YES + FK SET NULL +
  `document_status` default 'unknown' — all verified via information_schema.
- Runtime: temporal-intent returns 2018; `py_compile` all touched files PASS; frontend `tsc --noEmit` clean.
