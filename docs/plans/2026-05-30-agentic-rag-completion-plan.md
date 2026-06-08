# Plan: Agentic RAG Completion (Phase 2 delta)

**Created:** 2026-05-30
**Status:** Ready for implementation
**Priority:** P1
**Estimated effort:** ~16.5 dev-days (P0 5.5 / P1 5 / P2 6)
**Builds on:** existing `backend/services/agentic_rag.py` (working CRAG variant) — extend, do not redesign.
**Migration:** needs a new Alembic rev (next free is **14**; coordinate with reindex plan which also wants 14 — first to merge takes 14, other takes 15).

## Ground truth

- Installed: `langgraph>=1.1.0`, `groq>=0.4.0`. Idioms confirmed: `StateGraph(TypedDict)`, `add_conditional_edges`, `Annotated[list, reducer]`, `ToolNode` (`langgraph.prebuilt`), `recursion_limit` via `ainvoke` config (raises `GraphRecursionError`). Current code is already idiomatic.
- Existing graph (`agentic_rag.py:463-511`): `START → router → {fast_path=END | clarify} → retrieve → evaluate → {generate | rewrite→retrieve} → generate → verify → END`. Single flat module (~641 lines). Keep flat unless it passes 800 lines.
- Gated on `settings.agentic_rag_enabled` (default False), `main.py:1217-1248`, with double fallback (main.py wrapper + internal self-fallback). `include_legal_research` passed but unused in agentic path.
- `Query` model has JSON cols `citations`/`citation_verification`/`claim_verification`/`conflict_analysis`; **no** `routing_decision`/`agent_iterations`/`agent_latency_ms`. `agentic_metadata` returned but never persisted.

## Gap table

| Deliverable | Current (agentic_rag.py:line) | Missing |
|---|---|---|
| Router | `router_node:200` | regex patterns/conv-depth — low priority |
| Clarification full schema | `clarify_node:226` (jurisdiction/entities/refined_query only) | `sub_questions`, `requires_external_research`, `temporal_scope` — **blocks planning + sufficiency** |
| Planning agent | absent (`clarify→retrieve:490`) | whole node + state + rewiring (**P0**) |
| Execution tools / CourtListener | `retrieve_node:250` hardcoded hybrid | gated external research step (**P0**) |
| Reflection / sufficiency | `evaluate_node:300` (top-1 grade) | `_compute_sufficiency_signals` + strategy selection (**P0**) |
| Verification | `verify_node:396` (sequential) | parallelize + `verification_passed` (**P1**) |
| Synthesis | absent (reuses `generate_node:348`, runs *before* verify) | post-verify synthesis injecting verification (**P1**) |
| Metrics | absent | `agentic_metrics.py` per-node timing (**P1**) |
| DB columns | absent | 4 cols + migration + persist (**P1**) |
| SSE steps | absent (`progress.py` ingestion-only) | `publish_agentic_step` + UI (**P2**) |
| AgenticQueryResponse schema | absent | Pydantic + frontend type (**P1/P2**) |
| Shadow mode / A-B | on/off only | shadow + traffic pct (**P2**) |
| `_build_response` | stub (`confidence="medium":616`, `citations:[]:611`, `has_hallucinations:False:621`) | wire real values (**P1**) |

## P0 — Core "agentic" quality (~5.5d, no DB/frontend, behind existing flag)

### 1. Extend `clarify_node` (0.5d)
Widen the `_fast_llm_json` schema to also return `sub_questions` (default `[query]`), `requires_external_research` (bool), `temporal_scope`. Prompt + return-dict change only; reuse the Groq→Gemini fallback (`_fast_llm_json:161`).

### 2. Planning node (1d)
- State additions: `sub_questions`, `requires_external_research`, `temporal_scope`, `research_plan: Optional[Dict]` (last-writer-wins, no reducer).
- `plan_node`: derive `ResearchPlan` from clarification (`estimated_complexity`, `max_iterations = min(len(sub_questions)+1, settings.agentic_max_iterations)`, `strategy_hint`). **Deterministic for ≤1 sub-question** (no LLM); `_fast_llm_json` only for multi-hop.
- Rewire: `graph.add_node("plan", plan_node)`, replace `clarify→retrieve` with `clarify→plan→retrieve`. Flag `agentic_planning_enabled=True`.

### 3. CourtListener tool — gated step, NOT full ToolNode (1d)
- Reuse `legal_research.search_cases` + `format_as_context` (`legal_research.py:76,123`) — already emit the chunk schema; non-agentic path uses them at `rag_engine.py:1360-1371`.
- New `external_research_node` gated on `requires_external_research or include_legal_research` (flag `agentic_external_research_enabled`). Placement: after `retrieve`, before `evaluate`. Merge into `chunks`, dedupe by `chunk_id`. Non-blocking try/except. Strategy `external` (below) can set `force_external` to trigger on later iterations.
- Optional: annotate external chunks with `is_good_law` status (`citation_graph.py:543`) for the binding-authority signal.

### 4. Sufficiency signals + strategy selection (2d)
Replace shallow grade in `evaluate_node`, keep rewrite loop + `MAX_REWRITE_ITERATIONS=3` cap.
- `_compute_sufficiency_signals(state)`: `avg_relevance` (mean chunk score), `sub_questions_covered`/`uncovered` (≥3-token overlap heuristic), `jurisdiction_match` (chunk `authority_metadata.jurisdiction` vs state; unknown→True), `has_binding_authority` (authority_metadata tier or good-law case_law), `chunk_count`.
- Strategy chooser → writes `next_strategy`, routes to a `strategy_node`:
  - `avg≥0.70 & coverage≥0.80` → sufficient → verify/generate
  - `rewrite_count≥cap` → generate (existing)
  - `avg<0.45` → `reformulate` (existing rewrite)
  - `coverage<0.50` → `targeted` (search uncovered sub-questions)
  - filtered `<3` chunks → `filter_relax` (drop filters, mirror `rag_engine.py:1303-1308`)
  - `not has_binding_authority` & external allowed → `external` (set `force_external`)
  - marginal → `expand`
- Constants: add `SUFFICIENCY_HIGH=0.70`, `COVERAGE_HIGH=0.80`, `RELEVANCE_LOW=0.45`, `COVERAGE_LOW=0.50` (tunable via config). Pass `config={"recursion_limit": N}` to `ainvoke` as hard backstop.

### 5. Unit tests (1d)
`test_agentic_planning.py`, `test_agentic_sufficiency.py`, `test_agentic_external.py` — mock `_fast_llm_json`/`search_cases`.

## P1 — Synthesis + observability (~5d)

### 6. Parallelize verify (0.5d)
`verify_node`: `asyncio.gather` the three checks; compute `verification_issues`/`verification_passed`.

### 7. Synthesis node + reorder (1.5d)
Move generation after verification: `generate(draft) → verify → synthesize → END`. `synthesize_node` regenerates **only when** `verification_passed == False` (bounds latency to ~30% of queries), injecting verification badges/conflict notices/authority/good-law. Reuse `format_legal_context`, `LEGAL_SYSTEM_PROMPT`, `generate_answer`. Fix `_build_response`: real confidence via `calculate_answer_confidence` (`rag_engine.py:957`), populate `citations` from `citation_verification`, derive `has_hallucinations` from `claim_verification.summary`.

### 8. Metrics (1d)
`agentic_metrics.py` with `node_timer(node_name)` context manager → per-node `latency_ms` into state. Aggregate `agent_latency_ms`, `agent_iterations = rewrite_count+1`, `routing_decision`. Flag-guarded, never blocks.

### 9. DB columns + migration + persist (1d)
`models.py:238` Query: add `routing_decision String(50)`, `agent_iterations Integer`, `agent_latency_ms Integer`, `agentic_metadata JSON` (all nullable). Alembic rev **14** (`down_revision="13"`). Persist in `main.py:1252-1263` via `.get()` (non-agentic stores None). Enrich `agentic_metadata` (`agentic_rag.py:631`) with sub_questions/strategy_log/sufficiency_signals/node_timings.

### 10. Integration tests (1d)
Full graph `ainvoke` (stubbed LLM), metadata persisted, migration up/down.

## P2 — UX + safe rollout (~6d)

### 11–12. SSE step indicators (3d)
Reuse Redis channel (`progress.py:49`): `publish_agentic_step(matter_id, step, total, agent, iteration, message)` with `stage:"agentic"`; each node fire-and-forget publishes. Frontend `AgenticStepIndicator.tsx` + `types.ts` `AgenticStepEvent`/`agenticMetadata`; `api-services.ts` parse `agentic_metadata`.

### 13. Shadow mode / A-B (1.5d)
`config.py`: `agentic_shadow_mode=False`, `agentic_traffic_pct=0`. Shadow: serve `query_matter`, fire-and-forget agentic run, log comparison without returning. A-B: deterministic hash(matter_id/query) bucketing.

### 14. Quality A/B tests (1.5d)
Assert agentic ≥ single-pass source coverage on multi-doc queries; identical fast-path results.

## Risks
- Latency (8–15s): deterministic plan for ≤1 sub-q, parallel verify, skip synthesis regen when passed, SSE streaming.
- Cost (+~67%): Groq for router/clarify/plan/strategy, Gemini only for generate/synthesis.
- Groq quota: existing Gemini fallback + add circuit-breaker.
- Infinite loop: `MAX_REWRITE_ITERATIONS` + `recursion_limit`.
- Migration: all cols nullable/additive.

## Correction to original plan
Original plan §901 assigned the agentic migration to rev 13, but **13 = citation graph**. Use **14** (coordinate with reindex plan).
