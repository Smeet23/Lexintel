---
status: pending
priority: p1
issue_id: "001"
tags: [code-review, architecture, refactor, deferred]
---

# Deferred: Shared RAG pipeline + LLM client abstraction + god-module split

## Problem Statement
The changeset introduced two parallel RAG orchestrators (`agentic_rag.agentic_query_matter` and
`rag_engine.query_matter`) that are **not feature-equivalent** — the agentic path omits temporal
filtering, citation-graph injection, issue-context injection, token-budget trimming, and DB source
hydration. `main.py` picks one at runtime and silently falls back on exception, so the same question
yields materially different answers depending on a flag. Separately, LLM clients are constructed in
~14 services with duplicated JSON-fence-stripping/fallback logic, and `rag_engine.py` (2187 LOC) /
`citation_graph.py` (2040 LOC) are god modules.

These are **Large** refactors deliberately NOT attempted in the review-fix pass to avoid
destabilizing a working system. Concrete correctness/security/perf bugs were fixed first.

## Proposed Solution (phased)
1. Extract shared pipeline stages into `rag_pipeline/`: `retrieve+hybrid+rerank`, `temporal_filter`,
   `build_context`, `verify`, `hydrate_sources`. Both orchestrators call the same functions so parity
   is structural. Pass `as_of_date` through the agentic path.
2. Add `services/llm/` exposing `generate(prompt, *, model, json) -> str|dict` for Gemini + Groq,
   owning configure/retry/timeout/fence-strip. Replace the ~14 duplicate client sites.
3. Split god modules: `rag_engine` → `retrieval.py`/`context_builder.py`/`confidence.py`/
   `query_filters.py` + thin orchestrator; `citation_graph` → `graph_repository.py`/
   `graph_algorithms.py`/`graph_queries.py`/`relationship_extractor.py`.
4. Introduce repositories + single unit-of-work commit boundary; fix package layout to remove the
   ~100 triple-import fallbacks; group settings into nested config models; verification facade.

## Acceptance Criteria
- [ ] Agentic and standard paths produce identical enrichment (temporal/graph/conflict/issue) for the same input
- [ ] One LLM client module; no `genai.GenerativeModel(...)` construction outside it
- [ ] No service module > 800 LOC
- [ ] Single import style; no triple `except ImportError` fallback blocks

## Work Log
- 2026-06-02: Identified by architecture-strategist + simplicity reviewers. Deferred as Large refactor.
- 2026-06-03: **Pipeline parity DONE.** Agentic `generate_node` now mirrors `query_matter`'s synthesis
  stage: canonical `[n]` ordering, doc-summaries, conflict detection + context augmentation,
  citation-graph good/bad-law injection, issue analysis, query-relevant grounding, and rich sources;
  `verify_node` reuses the precomputed conflict; `_build_response` surfaces grounded citations +
  issue_analysis. **Also fixed two reachability bugs**: `/ask` agentic gating and `_build_response`
  used non-fallback `from backend.*` imports that raised under the `backend/`-cwd uvicorn launch, so
  the agentic path silently fell back to single-pass and was never used via HTTP. Now verified live
  (HTTP → complexity=complex, rewrites=3). Remaining parity gap: agentic path doesn't accept
  `as_of_date` / DB temporal filter (tracked for a later pass). 364 tests green.
- 2026-06-03: **LLM abstraction extended** — added a `system=` parameter to `services/llm`
  (Gemini `system_instruction` / Groq system message; +2 unit tests). Migrated `problem_formulation`
  to `llm.generate(system=…, provider="groq", fallback=True, json=True)` (live-verified). Added a
  re-runnable **browser regression suite** `scripts/browser_regression.sh` (15 checks, all pass).
  366 tests green. **Remaining (deferred — own focused passes):** migrate the bespoke LLM sites
  (agentic `_fast_llm_json`, `treatment_verifier`, `citation_graph` window extractors,
  `rag_engine` answer-gen, `claim_verifier` CoV); Phase 4 import-fallback overhaul + run-from-root;
  Phase 5 god-module split. These are structural refactors with high regression surface and are
  Gemini-quota-limited to live-verify today, so not rushed at the tail of this batch.
- 2026-06-02: **DONE — the folded-in concurrency sub-item** (genuine concurrency in
  `citation_graph.extract_and_index_citations`). Implemented `_gather_bounded` (semaphore-bounded
  `asyncio.gather`, `return_exceptions`), parallelized the three pure-I/O fan-outs — `extract_all_citations`
  (per chunk), `classify_treatment` (per unique cite), `verify_negative_treatment` (per negative triple)
  — and added cross-chunk citation **dedup** (each authority created/enriched/classified once). Added
  `settings.citation_graph_max_concurrency` (default 8, coerced defensively).
  **Safety:** sync SQLAlchemy session → only pure-I/O is gathered; `find_or_create_node`, `create_edge`,
  `enrich_node_from_courtlistener` (two internal awaits around a `db.flush()`), and `_merge_node_by_cluster`
  stay **serial** (a python-reviewer pass caught a P1 where gathering enrich could flush half-built node
  state — reverted to serial; dedup keeps it once-per-authority). Removed a redundant double-enrich of
  `cited_node` in the triple loop.
  **Verified:** 104 citation_graph tests + 337 full suite green; empirical bench shows 5.9× speedup
  (1.40s→0.24s) with identical result dicts, semaphore bound respected (max in-flight = 8), dedup
  confirmed (classify once per unique cite).
  **Still deferred (this todo):** the shared-pipeline / god-module-split items above;
  and moving the triple-path enrich/merge/outbound to an async session for full parallelism.
- 2026-06-02: **LLM client abstraction (item #2) — substantially DONE.** Created `services/llm.py`
  (`agenerate`/`generate` + `parse_json`/`strip_json_fences`; owns Gemini/Groq configure, JSON mode,
  fence-strip, the empty/blocked-`candidates` guard, provider ordering + Gemini⇄Groq fallback). 15 unit
  tests (`tests/test_llm_client.py`). **Migrated 8 single-prompt sites** (authority_detector,
  citation_extractor, citation_graph.classify_treatment, citation_verifier, contract_review,
  document_summary, draft_service, temporal_extractor) — each verified behavior-identical (exact
  temperature/max_output_tokens/json/fallback preserved; degrade-on-failure unchanged). Full suite
  **352 passed, 0 failures**.
  **Intentionally NOT migrated (5, with reasons — would need a `system=` param or richer return, i.e.
  behavior risk):** `rag_engine` core answer-gen (custom `system_instruction`, usage-metadata token
  count, 3-try backoff, 45s-timeout→typed exception, tuple return, own Groq fallback); `problem_formulation`
  + `agentic_rag` fast-LLM (role-separated system+user messages + prebuilt Groq client); `claim_verifier`
  CoV judge (canary-gated before the real call); `citation_graph` window relationship-extractors
  (`_extract_relationships_window_gemini/_groq` already a deliberate per-window quota/fallback split);
  `treatment_verifier` (Groq verdict — migratable later). Future: add an optional `system=` parameter to
  `llm` to absorb the role-separated sites without behavior change.
