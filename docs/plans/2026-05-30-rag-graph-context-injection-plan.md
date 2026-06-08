# Plan: RAG Graph-Context Injection (citation_graph spec §9)

**Created:** 2026-05-30
**Status:** Ready for implementation
**Priority:** P1 (highest value-per-effort of the four; P0 ≈ 0.5–1 day)
**Builds on:** existing citation graph (models, `is_good_law`, endpoints) + a crude precursor at `rag_engine.py:1492-1526`.

## Problem

The citation knowledge graph (good-law status, overruling/treatment edges, PageRank authority) is never injected into the RAG prompt, so answers can't warn about bad law or weight precedent. A crude precursor exists (`rag_engine.py:1492-1526`: regex-matches only `Name v. Name` in 5 chunks, appends `[GRAPH] X: STATUS`). Replace it with a proper validator.

> **Spec drift warning:** spec §9 references `graph_queries.is_good_law` / `get_precedent_chain` / `extract_citations_regex` and `good_law`/`bad_law` status strings — these names DO NOT exist. Real code: `citation_graph.is_good_law` / `get_citation_chain` / `find_similar_cases`, `citation_extractor.extract_all_citations`, and status enum `good`/`bad`/`caution`/`unknown`. Implement against the real code; reviewers must not "fix" it back to spec.

## What to inject (per cited authority, all from `citation_graph.py`)

| Field | Source |
|---|---|
| good-law status (`good`/`bad`/`caution`/`unknown`) | `is_good_law(db, text)["status"]` (`:543`) |
| negative treatments (OVERRULES/REVERSES/...) + citing case | `is_good_law(...)["negative_treatments"]` (`:592`) — same call, free |
| positive/cautionary counts | `is_good_law(...)["positive_count"]`/`["cautionary_count"]` (`:621`) |
| authority score + jurisdiction + name + year | node `_node_to_dict` fields (`:208`) — direct fetch for ranking/cap |
| key cited authorities (P1) | `get_citation_chain(db, text, depth=1)["forward"]` (`:630`) |

`find_similar_cases` — skip for RAG (reserve for API/viz).

## Where citations come from

**Union of citations in retrieved chunks (primary) + query text (cheap add).** Validate what the answer will actually cite. Pull from `final_chunks` (post-rerank/temporal, `rag_engine.py:1450/1484`) + the raw query.

**Extraction, ~0 latency:** `extract_all_citations(text, use_llm=False)` (`citation_extractor.py:282`) — regex+eyecite only, no API. `_likely_has_citations` (`:271`) short-circuits citation-free chunks. Bound to query + top-5 chunk contents (mirror the existing `[:5]` cap). It's async but does no network with `use_llm=False`.

## Design — `_build_graph_context(db, matter_id, citations, max_citations=5) -> Optional[str]`

Sync core (DB work) wrapped in `asyncio.to_thread` at call site (`is_good_law` etc. are sync). Steps:
1. Dedup `raw_text` case-insensitively (union of query+chunks needs re-dedup).
2. Resolve node: `db.query(CitationNode).filter(CitationNode.citation_text == _normalise_text(raw)).first()` (mirrors `is_good_law:564`). Two-tier: matter-scoped first, global fallback (overrulers may be outside the matter). Skip if no node (→ unknown, no value).
3. Cap top-N by `float(authority_score)` desc (default 5).
4. `is_good_law(db, raw)` once per citation (returns status + treatments + counts together; note it `db.flush()`es the `is_good_law` flag — harmless in-session).
5. Format compact block, emit a line only when `status != "unknown"`; cap ~1.5–2k chars.

**Block shape:**
```
--- CITATION VALIDATION (Knowledge Graph) ---
Heed these validity signals. If an authority is BAD LAW, do not rely on it for a
conclusion; warn the reader and prefer the controlling authority.

• Roe v. Wade, 410 U.S. 113 — BAD LAW (overruled by Dobbs, 597 U.S. 215). Authority: high. US.
• Chevron v. NRDC, 467 U.S. 837 — CAUTION (questioned/limited by 2 authorities). Authority: high. US.
• Brown v. Board, 347 U.S. 483 — GOOD LAW (followed by 12 authorities). Authority: high. US.
```
Return `None` if no line emitted (caller skips cleanly).

## Integration in `query_matter`

Replace the crude block at `rag_engine.py:1492-1526` (correct location: after `format_legal_context:1484` + conflict augmentation, before issue injection + token budget, after rerank/temporal, before `generate_answer:1572`):
```python
if settings.citation_graph_enabled:
    try:
        from backend.services.citation_extractor import extract_all_citations  # triple try/except
        cite_text = query + "\n" + "\n".join(c.get("content","") for c in final_chunks[:5])
        found = await extract_all_citations(cite_text, use_llm=False)
        graph_block = await asyncio.to_thread(_build_graph_context, db, matter_id, found)
        if graph_block:
            formatted_context = graph_block + "\n\n" + formatted_context  # PREPEND (spec §9.2)
    except Exception as e:
        logger.warning(f"Graph validation context failed (non-blocking): {e}")
```

**System-prompt addendum** (both `LEGAL_SYSTEM_PROMPT:62` and `LEGAL_RESEARCH_SYSTEM_PROMPT:124`, near CONFLICTS/HALLUCINATION FLAGS):
```
CITATION VALIDATION:
- Context may include a "CITATION VALIDATION (Knowledge Graph)" block with good-law status.
- If an authority is BAD LAW (overruled/reversed/superseded), do NOT rely on it; warn the reader and mark [BAD LAW].
- If CAUTION, note uncertain validity before relying.
- Weight authorities by stated authority score and jurisdiction when sources conflict.
```
Add `[BAD LAW]` to the HALLUCINATION FLAGS inline-marker list (`:108`/`:184`). No `generate_answer` signature change.

## Config & graceful degradation
- Reuse `settings.citation_graph_enabled` (`config.py:72`); optionally add `citation_graph_rag_injection_enabled=True` to disable just injection.
- Disabled / import fail / no citations / node unknown / lookup raises → skip cleanly; `formatted_context` is already valid before the block runs, so the standard answer is never broken (non-blocking).

## Latency / budget
- ~15–30 indexed Postgres queries for 5 citations (single-digit ms each); run off-loop via `asyncio.to_thread`. Block ≈ 400–500 tokens vs 50k `CONTEXT_TOKEN_BUDGET` — negligible. Added before the budget trim (`:1543`); prepend means it survives the trim-to-2-chunks fallback.

## Edge cases
- Citation in query not in graph → unknown → skip (no "not found" noise).
- Conflicting treatments → `is_good_law` precedence negative>cautionary>positive → `bad`; surface positive_count for the split.
- **Self-citation:** ingestion creates a node for the document itself (`document_name` as citation_text, `:1078`); won't match `extract_all_citations` output, but add a defensive filter (skip nodes whose `document_id` == chunk's document).
- Short-form cites (`id.`, `supra`) → unknown → skipped.

## Test plan
- **P0 unit:** `_build_graph_context` over a seeded graph — `BAD LAW` for overruled, `GOOD LAW` for followed, omits unknown, caps at N, returns None when empty; dedup across union; `use_llm=False` returns immediately. Prompt: assert addendum present in both prompts.
- **P0 integration:** `query_matter` with a matter referencing a seeded overruled case → assembled context contains the block (mock `generate_answer` to capture context). Degradation: disabled / import fail / lookup raises → valid answer, no exception.
- **P1:** `get_citation_chain(depth=1)` forward authorities appended + capped; authority-bucket labeling.

## Phases
1. **P0 (~0.5–1d):** `_build_graph_context` + replace crude block + prompt addendum + tests. (Integration point + flag already exist.)
2. **P1 (~0.5d):** treatment chains + authority bucketing.

## Risks
- **Citation-text matching brittle:** graph keys on exact `_normalise_text` (whitespace-collapse only); `347 U. S. 483` vs `347 U.S. 483` misses → degrade to unknown/skip (never wrong warnings). Consider shared canonicalization between ingest + lookup.
- **Stale graph → false GOOD LAW:** `is_good_law` returns `good` when no negative edges exist; un-ingested overruling reads as good. Lean on `node.is_good_law` when `is_verified` (CourtListener override, `enrich_node_from_courtlistener:77`); frame signals as graph-derived.
- **Write in read path:** `is_good_law` flushes — acceptable, flag for awareness.
