# 006 — Litigator-Grade Citation & Validity Trust (P0→P2)

Status: **pending** · Source: advocate revalidation (Jun 3 2026) + 6-agent planning workflow
(`lawyer-grade-improvement-plan`). Constraints: **MVP — no auth, no rate limiter.** Plan only;
incremental edits to existing services, no rewrites.

---

## Enhancement Summary (deepened 2026-06-08)

Deepened via a 17-agent research+review workflow (6 research: CourtListener API, eyecite, citator
domain, RAG faithfulness, Alembic, FastAPI; 5 reviewers: architecture, security, performance,
data-integrity, simplicity). Sections enhanced: §0, Pre-Wave-1 blockers, CORR-1/2/4/5/8,
SAFE-1/3/4, QUOTE-1/2, RECALL-1/2, TAX-1/3/4/5/6, STAT-1, PARALLEL-1, BRIEF-1, PROV-1. Per-item
detail in **Research Insights by item** + **Reviewer-flagged adjustments** at the end.

**Highest-impact new findings:**
1. **CourtListener has NO editorial citator** — only the citation network (who-cites-whom + `depth`).
   "Corroboration" = "the citing opinion's cited list contains the cited case" (identity/network
   presence), **never** "an editor confirmed good law." Reframes the CORR badges + SAFE-1 copy (see C1).
2. **Async/sync hot-path hazard (PARALLEL-1):** `_build_graph_context`/`is_good_law` are sync inside
   `asyncio.to_thread` — you cannot `await lookup_citation()` there. Split into a **sync cache-peek
   resolver** (answer path) + an **async resolver** (BRIEF-1). This also breaks the "inviolable"
   PARALLEL-1→BRIEF-1 coupling.
3. **CORR-4 cost is badly understated** — `get_outbound_case_citations` is 2+2×N serial GETs (N≤25),
   holds the shared `Semaphore(5)` throughout, no cache → fan-out → CL throttle (5/min,50/hr,125/day)
   → silent `[]` → falsely `ai_inferred`. Needs cache + per-GET semaphore + membership short-circuit.
4. **`cove_confirmed` has no producer (YAGNI)** → ship a two-value enum (`ai_inferred`|`courtlistener`),
   reserve the third in a comment, keep `VARCHAR(20)`.
5. **`_merge_node_by_cluster` doesn't dedup re-pointed edges** → parallel-cite merges can create
   duplicate rows on the dedup key, splitting corroboration. CORR-5 upgrade-in-place needs post-merge
   reconciliation **first**.
6. **CitationLookup status ≠ HTTP status:** 404 = valid-but-absent (`unknown`, NOT fabricated);
   300 = ambiguous (surface candidates, never auto-pick); 400 = malformed. Conflating 404 with "fake"
   = false accusations. Shapes SAFE-1, SAFE-2, BRIEF-1, the TAX taxonomy.

**Conflicts flagged for human go/no-go:**
- **TAX-4/TAX-5 scope vs North Star:** reviewers split — ship TAX-1 (caution/superseded) now, **gate
  TAX-4 (partial) + TAX-5 (binding) on RECALL-2** proving a concrete miss they fix. If deferred, **drop
  the `context_excerpt`-text-encoding MVP option entirely** (it's a banned brittle-string pattern).
- **PARALLEL-1 uncached-bail** makes the warning inert on the first/highest-stakes query (answer path).
  Accept on the answer path; **override on BRIEF-1** (bounded live lookups). Confirm this path-split.

## New cross-cutting considerations

- **C1 — "Corroboration" semantics:** CL = identity + citation-network only (no Shepard's/KeyCite).
  The `courtlistener` badge MAY say "resolves to a real case (cited N times)"; it MUST NOT say "good
  law / not overruled / treatment verified." All validity stays in our `treatment_verifier`+graph. A CL
  **404** is `unknown`, never `bad`/quarantine (CL corpus is non-exhaustive).
- **C2 — SSRF / URL allowlist on every fetched-or-surfaced URL** (CORR-4, QUOTE-1/2, BRIEF-1, PROV-1):
  reuse the existing `_get_json` origin guard (`citation_lookup.py:899-915`), `cluster_id` `^\d+$`
  validation, redirect checks; opinion-fetch gets origin allowlist + `REQUEST_TIMEOUT` + body-size cap;
  any surfaced `source_url` validated `https://`+trusted-origin before serialization (omit otherwise);
  frontend external links `rel="noopener noreferrer"`.
- **C3 — Hot-path async/sync boundary:** the validity core is sync/DB-only inside `to_thread`. Never
  `await` network there — expose a **sync `_lookup_cache.peek()`**, bail to canonical-only on miss; all
  live resolution in async ingest/endpoints; keep `is_good_law`/`get_version_chain` pure-DB + bounded;
  use `asyncio.wait_for` (venv is 3.9, no `asyncio.timeout`).
- **C4 — `_gather_bounded` + `return_exceptions=True` is the universal fan-out** (CORR-4, BRIEF-1):
  per-coro semaphore + map exceptions → that item's `status='unknown'`; never plain `gather`
  (one failure cancels siblings → 500). Heavy outbound fan-out gets a **separate, smaller** semaphore so
  it can't starve cheap lookups on the shared `Semaphore(5)`.
- **C5 — Negation polarity has ONE owner** (`treatment_verifier`): SAFE-3 consumes adjudicated edges,
  never re-derives polarity; share one RECALL-1 negation-trap fixture across TAX-3 and SAFE-4.
- **C6 — Privilege-safe logging:** brief text is work-product — don't log full text/windows; citation
  strings at `debug`; treat uploaded text as DATA (prompt-injection: sanitize/length-bound any node
  display name before it enters the SAFE-3 banner); add an adversarial RECALL-1 fixture.
- **C7 — Config-flag interdependency:** **corroboration (CORR-4) requires enrichment (CORR-1)**;
  emit a sanity log when `citation_edge_corroboration_enabled=True` while `citation_graph_enrich_enabled=False`.

---

## 0. Cross-cutting engineering principle — NO BRITTLE STRING HEURISTICS

**Library/resolution over patterns.** For any decision about whether a string is a legal
authority, what authority it is, or its court/jurisdiction:

- **Recognition →** eyecite / reporters-db (`get_citations` → `FullCaseCitation`/`FullLawCitation`).
  Never a hand-rolled reporter regex or a maintained abbreviation list (lossy across
  jurisdictions; the old `_likely_has_citations` substring gate was already removed for this).
- **Identity / merge / bare-name admission →** CourtListener cluster + name resolution.
  Never a `" v. "` / `" vs "` caption substring (under-matches `In re`/`Ex parte`/foreign
  neutral cites — the dangerous under-warn — and over-matches prose).
- **Court hierarchy →** CourtListener's structured court id/field, not court-name substrings.
- **Answer-side matching (banners) →** match on the resolved **reporter citation string**, not
  the bare party name (avoids "Roe"/"Smith" collisions).
- **Where nothing resolves → OMIT, don't guess.** Better to withhold an unverifiable authority
  than to assert/deny good law on a pattern. Regex is acceptable only as a *non-lossy
  pre-filter* (e.g. digit-presence), never as the deciding authority.

Every item below that previously leaned on a substring (TAX-5, SAFE-3, PARALLEL-1, BRIEF-1)
has been written to follow this. New code must too.

---

## Already shipped this session (anchors current state — do NOT re-plan)

Fixed + tested (376 backend tests pass; validated against live data):
- Phantom-node guard: `_parse_relationship_items` drops triples whose endpoint is not a real
  authority — now **eyecite-only** (`_looks_like_case_citation`), no regex / no `" v. "` substring.
- Ingest/graph LLM services honor `LLM_ANSWER_PROVIDER` + cross-provider fallback
  (authority_detector, document_summary, temporal_extractor, citation_extractor,
  classify_treatment, relationship extraction).
- `_normalize_case_name` at the `find_or_create_node` choke point (no multi-line header leak).
- Good-law: controlling authority (source of OVERRULES, no incoming negative) → `good`;
  non-citation nodes never marked `bad` (`_is_real_authority`).
- `_build_graph_context` prompt hygiene (name de-dup + overruler newline collapse).
- macOS: local models pinned to CPU (`LOCAL_MODEL_DEVICE`) — Celery prefork no longer crashes.

---

## North Star

"Trustworthy for a litigator" = **every validity signal is honest about its source and its
limits**: a green badge means an authoritative source confirmed it; an absent warning never
silently means "clear"; and no answer relies on overruled law without a **code-enforced** warning.
**Single biggest lever: authoritative corroboration** — today every good-law verdict rests on one
LLM-extracted edge whose endpoints usually have `courtlistener_id=null`, so the CourtListener
path is dead. Fix enrichment (CORR-1) + edge provenance (CORR-2/4/5) to turn "AI guess" into
"checked." Everything else guards the failure mode where a confident-but-wrong signal is worse
than "unknown."

---

## Wave 0 — Quick Wins (~days, no deps, ship independently)

### QUOTE-1 — Turn on quote verification in the live answer path
- **Lawyer problem:** Citations read "verified" because the cite *string* resolves; the quoted
  holding is never checked, so a hallucinated quote on a real cite passes silently.
- **Change:** Add `settings.citation_quote_verification_enabled` (default `True`); pass
  `enable_quote_verification=...` into the existing `verify_response_citations()` calls. The
  Step-5 opinion-fetch + `verify_quote` block already exists, just unreached. Best-effort.
- **Files:** `rag_engine.py:~2209-2219`, `main.py:~2440`, `config.py`.
- **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** none.
- **Acceptance:** Real cite + fabricated quote → `partial`, quote_match<0.5; accurate quote stays
  `verified`; flag `False` → unchanged.

### CORR-3 — Node provenance accessor (read-time projection)
- **Lawyer problem:** No label distinguishes "exists in CourtListener" (identity) from
  "CourtListener confirms good law" (validity); UI can over-claim.
- **Change:** In `_node_to_dict` add derived `provenance` = `'courtlistener'` when
  `courtlistener_id` set OR `is_verified`, else `'ai_inferred'`. Pure projection, no column.
- **Files:** `citation_graph.py:_node_to_dict` · **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** none.
- **Acceptance:** id-bearing node → `'courtlistener'`; bare node → `'ai_inferred'`.

### SAFE-1 — Reframe `no_adverse` as abstention, not soft-positive
- **Lawyer problem:** A green "No adverse treatment" badge reads as a clearance; it only means no
  overruling edge in this matter's tiny graph (recall unmeasured).
- **Change:** Keep the enum value (frozen contract); change copy/colors only. Frontend
  `getStatusColors` → neutral/amber + Info icon, label "No adverse treatment found — not a
  clearance"; minimap off `#10b981`; detail caption "Based only on authorities in this matter;
  not a Shepard's/KeyCite clearance." `good`/`bad` colors unchanged.
- **Files:** `CitationGraph.tsx`, `CitationGraphTab.tsx`, `types.ts`, `citation_graph.py:_good_law_status` (docstring).
- **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** none.
- **Acceptance:** `no_adverse` badge says "not a clearance", non-green, non-CheckCircle icon.

---

## ⚠ Pre-Wave-1 blockers (decide before any Wave-1 code)
- **Migration numbering:** CORR-2 = **migration 18** (revises 17); TAX-4 = **migration 19**
  (revises 18). Never two migrations on `down_revision='17'` (forks Alembic, breaks `upgrade head`).
- **CORR-4 de-risk spike (½–1 day, no commit):** confirm that for ONE real overruled pair,
  `get_outbound_case_citations(citing_cluster_id)` contains the cited cluster id, and **pin the
  exact dict key** holding that id. Wrong key → every edge silently stamped `ai_inferred`.

## Wave 1 — P0 Trust & Safety (corroboration spine + deterministic enforcement)

### CORR-2 — Edge corroboration columns (migration 18)
- **Lawyer problem:** A Gemini-only "overruled" looks identical to one CourtListener confirmed.
- **Change:** `CitationEdge.corroboration VARCHAR(20) NOT NULL server_default 'ai_inferred'`
  (`ai_inferred`|`courtlistener`|`cove_confirmed`) + `verified_by JSON NULL`; index; surface in
  `_edge_to_dict`. Migration 18 backfills existing rows.
- **Files:** `models.py:CitationEdge`, `alembic/versions/18_add_edge_corroboration.py`, `citation_graph.py:_edge_to_dict`.
- **Migration:** **18** · **Effort:** M · **Risk:** med · **Deps:** none.
- **Acceptance:** up→down→up clean; existing rows `'ai_inferred'`; `_edge_to_dict` has both keys.

### CORR-1 — Backfill `courtlistener_id` + `is_verified` for ALL case nodes at ingest
- **Lawyer problem:** A node first seen as a bare cite then later treated is never re-enriched, so
  most relationship endpoints stay unverified — an overruled-but-unverified case can be cited.
- **Change:** Widen the enrichment predicate in `extract_and_index_citations` from
  `_was_created` to "any case node where `courtlistener_id is None and not is_verified`". Keep
  serial + `citation_graph_enrich_enabled` gate; `lookup_citation` cache makes it idempotent.
- **Files:** `citation_graph.py:extract_and_index_citations`, `:enrich_node_from_courtlistener`.
- **Migration:** none · **Effort:** S · **Risk:** med · **Deps:** none (do alongside CORR-2).
- **Acceptance:** pre-existing unverified cited node gets id+`is_verified` after ingest; id-bearing node triggers zero new lookups.

### CORR-5 — `create_edge` persists corroboration (non-destructive upgrade)
- **Change:** Add `corroboration='ai_inferred'`, `verified_by=None` params; validate; on dedup
  upgrade-in-place only (precedence `courtlistener > cove_confirmed > ai_inferred`, never downgrade).
- **Files:** `citation_graph.py:create_edge` · **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** **CORR-2**.
- **Acceptance:** persists; weaker re-call no downgrade; stronger upgrades; invalid rejected.

### CORR-4 — Cross-check negative `case_cites` edges against CourtListener outbound
- **Lawyer problem:** The most consequential signal (OVERRULES/REVERSES) is single-source; nothing
  consults CL's own record of what the citing case cites.
- **Change:** Pure-I/O `corroborate_edge_against_courtlistener(citing, cited)` — runs only when both
  carry `courtlistener_id`. `get_outbound_case_citations(citing.courtlistener_id)`; match cited
  cluster id on the **spike-pinned key**. Returns `('courtlistener',…)` on match, `('ai_inferred',…)`
  when CL data present but no match, `(None,None)` when CL unavailable (abstain). Pre-compute via
  `_gather_bounded`; pass into `create_edge`. **Annotates provenance only — never flips treatment.**
- **Files:** `citation_graph.py:extract_and_index_citations`, `:corroborate_edge_against_courtlistener`, `:create_edge`, `citation_lookup.py:get_outbound_case_citations`.
- **Migration:** none · **Effort:** L · **Risk:** high · **Deps:** **CORR-1, CORR-2, CORR-5**.
- **Acceptance:** outbound includes cited → `courtlistener`/matched; omits → `cove_confirmed`/`ai_inferred`; CL error/`[]` → never silently `courtlistener`; treatment unchanged.

### CORR-8 — Config flag + offline/no-key honest-degradation contract
- **Change:** `settings.citation_edge_corroboration_enabled` (default `True`) gates CORR-4. Disabled
  or CL `[]` (no token/offline) → corroboration MUST stay `ai_inferred`/`cove_confirmed`. Document in
  `extract_and_index_citations` docstring + CLAUDE.md.
- **Files:** `config.py`, `citation_graph.py`, `CLAUDE.md` · **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** **CORR-4**.
- **Acceptance:** flag False → no calls; flag True + no token + outbound `[]` → no edge `courtlistener`.

### CORR-6 — Expose edge + node provenance through API → frontend types → mappers
- **Change:** Add `corroboration?`/`verified_by?` to `GraphEdge`, `provenance?` to `GraphNode`;
  thread through `mapGraphEdge`/`mapGraphNode`. No route change (returns `get_matter_graph` verbatim).
- **Files:** `types.ts`, `api-services.ts` · **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** **CORR-2, CORR-3**.
- **Acceptance:** `tsc` passes; corroboration+provenance round-trip through `normalizeCitationNetwork`.

### CORR-7 — Render "Verified (CourtListener)" vs "AI-inferred" badges
- **Change:** Badge from `node.provenance` + `edge.corroboration` (check icon vs dashed/amber);
  legend. Keep `good_law_status` (validity) badge separate from provenance (identity) badge.
- **Files:** `CitationGraph.tsx`, `CitationGraphTab.tsx` · **Migration:** none · **Effort:** M · **Risk:** low · **Deps:** **CORR-6**; **release after CORR-4 runs** (else all `ai_inferred`).
- **Acceptance:** corroborated vs ai_inferred edges render distinct badges.

### SAFE-2 — Quarantine unresolved/hallucinated authorities out of the verified list
- **Lawyer problem:** A fabricated cite returns in `verified_citations` as `not_found`, rendered
  beside real ones — mistakable for checked.
- **Change:** In `verify_response_citations`, split off `quarantined_citations` (not_found/unverified
  where `lookup.found is False AND matched_source_idx is None`) + `summary.quarantined`; never delete
  silently. Frontend renders a distinct red "Unverifiable — excluded; do not rely" block. Answer untouched.
- **Files:** `citation_agent.py:verify_response_citations`, `VerificationBar.tsx`, `CitationPanel.tsx`, `api-services.ts`, `types.ts`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** none.
- **Acceptance:** fabricated cite in `quarantined_citations` not `verified_citations`; real source-matched only in verified.

### SAFE-3 — Deterministic bad-law banner appended by code (not LLM-dependent)
- **Lawyer problem:** The only bad-law safeguard today is a prompt instruction the LLM may ignore.
- **Change:** `_build_graph_context` also returns its structured `bad_law_findings` (or a sibling
  `collect_bad_law_findings` reusing the same resolution). In `query_matter` (and agentic
  `generate_node`), after the answer is finalized, if a bad-law authority is **matched in the answer**,
  append: "⚠ VALIDITY WARNING: {Name} appears to be BAD LAW (overruled by {X} per the citation graph).
  Do not rely on it without independent verification." Best-effort, gated on `citation_graph_enabled`.
- **NO-BRITTLE:** match on the resolved **reporter citation string / node identity**, NOT the bare
  party name (avoid "Roe"/"Smith" false-positive banners). Reuse the already-resolved
  `verified_citations`/node identity rather than substring-scanning the prose.
- **Files:** `rag_engine.py:_build_graph_context`, `:query_matter`, `agentic_rag.py:generate_node`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** needs existing `_build_graph_context`
  bad-law findings (the scoped dep on SAFE-2 was mislabeled).
- **Acceptance:** bad+matched authority → answer contains "VALIDITY WARNING" + name even when the LLM
  wrote none; no banner when not in the answer; no banner on a mere party-name collision; agentic identical.

### SAFE-4 — Eval: top-retrieved authority is bad law → answer must not silently rely on it
- **Change:** `backend/tests/test_bad_law_safety.py` (offline): graph where the top chunk's authority
  has an incoming negative edge; stub `generate_answer` to cite it; assert (1) banner naming it,
  (2) verification shows `bad` not `no_adverse`/`good`, (3) `no_adverse` not serialized as positive.
- **Files:** `backend/tests/test_bad_law_safety.py` · **Migration:** none · **Effort:** M · **Risk:** low · **Deps:** **SAFE-1, SAFE-3**.
- **Acceptance:** passes offline; fails if the banner is removed from either pipeline.

---

## Wave 2 — P1 Recall & Treatment Taxonomy
**Measure recall first.** All TAX `is_good_law`/`_good_law_status` items edit the same ~100 lines —
**serialize them (TAX-1 keystone first), do not branch in parallel.**

### RECALL-1 — Labeled treatment/overruling benchmark fixtures
- **Change:** `backend/tests/fixtures/treatment_benchmark/cases.jsonl` (~40-60 hand-labeled windows
  from public opinions): clean OVERRULES, negation traps ("we do not overrule X", "remains good law"),
  DISTINGUISHES-not-overrules, ABROGATES/SUPERSEDES, QUESTIONS/CRITICIZES/LIMITS, partial overrule,
  passive voice, pure CITES. Schema `{text_window, citing, cited, expected_treatment, polarity, notes}`
  + rubric README. **Honesty note:** 40-60 windows is a regression tripwire, not a statistical guarantee.
- **Files:** fixtures + README · **Migration:** none · **Effort:** M · **Risk:** low · **Deps:** none.
- **Acceptance:** loader asserts keys, polarity set, ≥1 record per hard category.

### RECALL-2 — Offline recall/precision harness over the gold set
- **Change:** `backend/tests/test_treatment_benchmark.py` runs `_parse_relationship_items` over
  **recorded** extractions (zero quota in CI); per-polarity P/R/F1; assert conservative thresholds
  (e.g. negative-recall ≥ 0.7, precision ≥ 0.8). `scripts/run_treatment_benchmark.py` (live, env-gated)
  refreshes recordings with a printed quota-cost note.
- **Files:** test + script + recorded json · **Migration:** none · **Effort:** M · **Risk:** low · **Deps:** **RECALL-1**.
- **Acceptance:** zero-network pytest prints P/R/F1, fails below threshold.

### TAX-1 — Cautionary & abrogation/supersession first-class in the verdict (keystone)
- **Change:** Split the axis in `is_good_law`/`_good_law_status`/`_live_good_law`: OVERRULES/REVERSES →
  `bad`; ABROGATES/SUPERSEDES → `superseded` (carry the superseding authority); QUESTIONS/CRITICIZES/
  LIMITS → `caution` (`is_good_law` None-with-reason); DISTINGUISHES stays strictly NEUTRAL (regression
  test). Return contributing tokens + authorities. **Preserve the GL-006 CourtListener-precedence guard.**
  Ensure new status strings flow through `_node_to_dict`/`get_matter_graph`.
- **Files:** `citation_graph.py:is_good_law`, `:_good_law_status/_live_good_law`, `:_node_to_dict`, `rag_engine.py:_build_graph_context`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** none, but **precedes TAX-2/3/4/5/6**.
- **Acceptance:** DISTINGUISHES→never bad; QUESTIONS→`caution`+None; SUPERSEDES→`superseded`; OVERRULES→`bad`; GL-006 tests pass.

### TAX-3 — Negation-discipline second-pass for ALL status-affecting treatments
- **Change:** Broaden `treatment_verifier._NEGATIVE` to the status-affecting set (add QUESTIONS/
  CRITICIZES/LIMITS) with verb phrases; tighten the extraction prompt's negation/scope block; wire so
  `rejected` downgrades cautionary edges to neutral CITES too. Fail-safe: missing key/timeout → keep edge.
- **Files:** `treatment_verifier.py`, `citation_graph.py:_RELATIONSHIP_EXTRACTION_PROMPT`, ingest loop.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** **TAX-1**.
- **Acceptance:** "did not criticize or limit X" → CITES; passive "X was overruled by Y" → confirmed; no key → kept. Use RECALL-1 negation fixtures.

### TAX-2 — Surface new statuses to API + frontend (caution/superseded)
- **Change:** Extend `NodeGoodLawStatus`/`GoodLawStatus` unions with `caution`, `superseded`; add
  color/badge mappings; older payloads fall through to existing buckets.
- **Files:** `types.ts`, `CitationGraph.tsx`, `CitationGraphTab.tsx`, `api-services.ts`.
- **Migration:** none · **Effort:** S · **Risk:** low · **Deps:** **TAX-1** (deployed + emitting the new statuses).
- **Acceptance:** `superseded`/`caution` render distinct badges; existing unchanged; `tsc` passes.

### TAX-4 — Partial-overruling qualifier (migration 19)
- **Change:** Parse `scope` from evidence ("in part", "on issue Y") in `_parse_relationship_items`; add
  nullable `CitationEdge.is_partial` + `overrule_scope` (**migration 19**, revises 18). In `is_good_law`,
  all-negative-edges-partial → `partial` (still-good-with-caveat) listing issues; flat `bad` only on an
  unqualified overruling. *(MVP option: ride `is_partial` on `context_excerpt` text to defer the migration.)*
- **Files:** `citation_graph.py:_parse_relationship_items`, `:is_good_law`, prompt, `models.py`, `alembic/versions/19_add_edge_overrule_scope.py`.
- **Migration:** **19** · **Effort:** M · **Risk:** med · **Deps:** **TAX-1**; migration **19 not 18**.
- **Acceptance:** partial-only → `partial` w/ issue; unqualified → `bad`; migration up/down clean.

### TAX-6 — Thread as-of date into the good-law answer
- **Change:** Reuse `rag_engine._detect_temporal_intent` + `node.year`; pass `as_of_date` into
  `is_good_law`/`_build_graph_context`; ignore negative/cautionary edges whose acting authority year is
  after `as_of_date`; render "as of <date>" / "(current)". Best-effort: missing dates → include edge.
- **Files:** `citation_graph.py:is_good_law`, `rag_engine.py:_build_graph_context` + injection, `agentic_rag.py:generate_node`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** **TAX-1** (merge after TAX-4/TAX-5).
- **Acceptance:** overruled-2022 case with `as_of_date=2020` → good; `None` → bad. Extends `test_temporal_as_of.py`.

### TAX-5 — Court-hierarchy binding vs persuasive (de-scoped; NO-BRITTLE)
- **Lawyer problem:** A lower/sibling court "questioning" a SCOTUS holding doesn't make it bad law.
- **Change (MVP slice only):** `_treatment_is_binding(treating, treated)` returns binding only on clear
  same-or-higher hierarchy derived from **CourtListener's structured court id/field** — NOT court-name
  substrings. Cross-jurisdiction or unknown court → downgrade negative to `caution`/persuasive, **never
  strengthen to bad**. **Defer** the full ranked per-jurisdiction table until a dedicated labeled
  court-hierarchy fixture set exists (court-name string matching is exactly the brittle, confident-but-
  wrong heuristic to avoid; under-warning a binding overrule is sanctionable).
- **Files:** `citation_graph.py:is_good_law`, `:_good_law_status/_live_good_law`, `:_treatment_is_binding`.
- **Migration:** none · **Effort:** M (slice) / L (full) · **Risk:** med · **Deps:** **TAX-1**.
- **Acceptance (slice):** District→SCOTUS OVERRULES yields at most `caution` for SCOTUS; unknown court never stronger than `caution`.

---

## Wave 3 — P2 Verification Depth & Workflow Fit

### QUOTE-2 — Verify the quoted passage (not the 200-char window) + pincite checking
- **Change:** Verify the quoted span adjacent to `[n]` (fall back to window only when none). Extract
  pincite page via **eyecite** (`FullCaseCitation.pin_cite`); when opinion text exposes star-paging,
  check the matched passage is near the claimed page; surface `pincite_ok: true|false|null`. Additive.
- **Files:** `citation_agent.py`, `citation_verifier.py`, `citation_extractor.py`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** **QUOTE-1**.
- **Acceptance:** fabricated quote amid accurate prose → `partial`; pincite far → `pincite_ok=false`; no anchors → `null`.

### STAT-1 — Surface statute/regulation supersession from amendment chains in the answer
- **Change:** Statute branch in `_build_graph_context`: for statute/reg nodes (or docs in an amendment
  chain) call `get_version_chain()`; emit "• {name} — SUPERSEDED as of {date}; current: {successor}." Inject
  in agentic `generate_node` too. **First verify** chain population runs in a fresh dev DB (non-null `superseded_date`).
- **Files:** `rag_engine.py`, `amendment_chain_manager.py`, `agentic_rag.py`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** data-availability check above.
- **Acceptance:** ingest v1 then v2; answer retrieving v1 emits "SUPERSEDED…current v2"; only-v2 emits none.

### PARALLEL-1 — Resolve parallel & short-form cites to the same node (NO-BRITTLE; hot-path discipline)
- **Lawyer problem:** `is_good_law` matches `citation_text == canonical` only, so an overruled-case
  warning on the official-reporter node is MISSED when the brief uses the parallel regional cite or a
  short form — the warning exists but never fires on the variant the lawyer used.
- **Change:** Read-time `resolve_node_for_lookup(db, citation_text, matter_id)`: (1) canonical match;
  (2) on miss, **CourtListener** `lookup_citation()` → cluster_id, match by `courtlistener_id` (merges
  parallels — resolution, not pattern); (3) short-forms via **eyecite** `ShortCaseCitation`/antecedent.
  **Hot-path discipline (critical):** must hit the existing lookup cache and **hard fast-path bail to
  canonical-only when uncached** to avoid latency/quota spikes on the answer path; degrade to today's
  behavior on any failure (no rate limiter — rely on cache + fail-safe).
- **Files:** `citation_graph.py`, `rag_engine.py`, `citation_lookup.py`, `citation_extractor.py`.
- **Migration:** none · **Effort:** L · **Risk:** high · **Deps:** none, but **precedes BRIEF-1**.
- **Acceptance:** overruled case ingested under U.S. cite → `is_good_law('<parallel S.Ct. cite>')` = `bad`
  via cluster_id; short-form resolves to same node; canonical cases unaffected; uncached lookups don't block answers.

### BRIEF-1 — Brief-check endpoint: paste/upload a draft, flag every bad/questioned authority
- **Lawyer problem:** The highest-value pre-filing moment ("is any case I cite overruled?") has no endpoint.
- **Change:** `POST /matters/{id}/brief-check` (mirror `/verify-citations`): brief text (cap ~20000) or
  multipart via `text_extraction`. `extract_all_citations` → `resolve_node_for_lookup` (matter+global) →
  best-effort enrich → `is_good_law` → report `[{raw_text, case_name, status, negative_treatments,
  controlling_case, source_url}]`, cap 25. **Degrade:** never 500 on lookup failure — that cite → `unknown`.
- **Files:** `main.py`, `citation_extractor.py`, `citation_graph.py`, `text_extraction.py`.
- **Migration:** none · **Effort:** M · **Risk:** med · **Deps:** **PARALLEL-1 (inviolable** — a brief-check
  that misses parallel/short-form cites is worse than none).
- **Acceptance:** brief citing an overruled (enrichable) case → `bad` + non-empty controlling; good cite →
  right status; unparseable omitted; >25 capped; never 500s.

### PROV-1 — One-click provenance: overruling excerpt + link behind every flag
- **Change:** Plumb existing `negative_treatments[].context` + overruling node `source_url` through
  BRIEF-1 report + `get_matter_graph` node dicts; frontend expandable "why" panel with excerpt + link.
  Read-only surfacing.
- **Files:** `citation_graph.py`, `rag_engine.py`, `CitationGraph.tsx`, `types.ts`, `api-services.ts`.
- **Migration:** none · **Effort:** M · **Risk:** low · **Deps:** **BRIEF-1** + **CORR-1** (for `source_url`).
- **Acceptance:** bad-law flag includes `negative_treatments[]` w/ non-empty excerpt + clickable `source_url`; missing excerpt degrades to name+link without erroring.

---

## Dependency / Ordering Rationale
- **Corroboration before any "verified" UI:** CORR-1 + CORR-2 foundation → CORR-5 → CORR-4 (run before
  CORR-7 release, else badges read the `ai_inferred` default and look broken). CORR-3 independent (Wave 0).
- **Migration order hard blocker:** CORR-2 = 18, TAX-4 = 19. Never two on `down_revision='17'`.
- **De-risk CORR-4 with the spike** (pin the cluster-id key) before building the fan-out.
- **Measure before claiming:** RECALL-1 → RECALL-2 land before/with the taxonomy work.
- **TAX-1 keystone; TAX-2/3/4/5/6 serialize** (same `is_good_law` region). TAX-6 merges last.
- **PARALLEL-1 → BRIEF-1 inviolable.** PROV-1 last (3-deep chain + CORR-1).
- **SAFE-1 + SAFE-3 → SAFE-4** (eval locks them in). SAFE-3's scoped dep on SAFE-2 was mislabeled.

## Explicitly NOT Doing (MVP)
- **No auth / login / JWT / route guards** — CORR-8's offline contract handles no-key without auth.
- **No rate limiter** — PARALLEL-1/BRIEF-1 degrade to `unknown` via cache + fail-safe.
- **No brittle string heuristics** (see §0) — no hand-rolled reporter regex, no `" v. "` substring, no
  court-name substring hierarchy, no party-name banner matching. eyecite + CourtListener resolution; omit when unresolved.
- **No full per-jurisdiction court-hierarchy table** (TAX-5 ranked table deferred to a labeled fixture set).
- **No rename of the `no_adverse` enum** (frozen contract; copy/colors only).
- **No treatment-flipping from corroboration** — CORR-4 annotates provenance only.
- **No rewrites / new service modules** (except test/script helpers) — *but see reviewer adjustment #8:
  relax for two genuinely-new cohesive concerns (`citation_resolution.py`, `citation_corroboration.py`)
  rather than growing the already-oversized `citation_graph.py`/`rag_engine.py`.*

---

## Research Insights by item (deepened 2026-06-08)

### QUOTE-1
- **Best practices:** keep best-effort, gated on `base_status=='verified' && cluster_id`
  (`citation_agent.py:208`). Verbatim quotes need a **lexical gate (rapidfuzz)**, not cosine alone —
  cosine rates a fabricated-but-plausible holding "similar" (the silent pass). **Parallelize** the
  per-citation verify loop (`citation_agent.py:164-232`) under `Semaphore(3-5)` + per-cite `wait_for`
  (serial today = up to 18-24 sequential calls).
- **Concrete:** thresholds `citation_agent.py:306-309` (≥0.80 verified, <0.50 partial); `verify_quote`
  `citation_verifier.py:290`; `_opinion_cache` maxsize=200, **no TTL**. Consider **default-False in dev**
  (Cohere cost).
- **Edge/pitfall:** opinion unavailable/`cluster_id` null → `unverifiable` (never `verified`); reuse SSRF
  allowlist + body cap (C2); cosine-alone verbatim check is the bug.
- **Refs:** rapidfuzz docs; docs.ragas.io faithfulness.

### QUOTE-2
- **Best practices:** strict lexical threshold **only inside quotation marks** (fall back to window/cosine
  when no quoted span); `rapidfuzz.fuzz.token_set_ratio`/`partial_ratio` + `default_process`, threshold
  85-90; read pincite from **eyecite `citation.metadata.pin_cite`** (uniform across full/id/supra) — never
  re-parse page ranges; embed only the quoted span neighborhood.
- **Surface:** `pincite_ok: true|false|null` (`null` = no star-paging anchors).
- **Edge/pitfall:** normalize OCR/smart-quote/whitespace both sides; paraphrase must NOT get the verbatim
  threshold (false `partial`).
- **Refs:** freelawproject.github.io/eyecite/, rapidfuzz docs.

### CORR-1
- **Best practices:** prefer **DB-state idempotency** (predicate `courtlistener_id is None and not
  is_verified`); write `is_verified=True` even on found-but-no-id so a node isn't re-queried forever (cold
  worker = empty cache). **Scope to THIS ingest's nodes**, not a global unverified scan (the "ALL case
  nodes" wording is ambiguous — a global scan extends the open txn + mutates cross-matter rows under this
  commit). Cap lookups/run; keep serial.
- **Concrete:** `enrich_node_from_courtlistener` early-returns when id set (`:137`); single `db.commit()`
  (`:2300-2310`); shared `Semaphore(5)`.
- **Pitfall:** global scan; relying on warm cache for the idempotency acceptance.

### CORR-2
- **Best practices:** single `op.add_column(nullable=False, server_default='ai_inferred')` (PG11+
  metadata-only). Set **both** server_default AND model-side `default=` (this repo already shipped the
  `documents.document_status` missing-server_default bug, fixed in migration 17). Use **identical types
  both sides** (`String(20)`; avoid the Float-vs-Text drift seen on `confidence`). `verified_by` nullable
  JSON (JSONB if future querying), NULL for backfilled rows (`_edge_to_dict` emits `null`, not `'null'`).
- **Concrete:** `revision='18'`, `down_revision='17'`; `create_index('idx_citation_edge_corroboration',…)`;
  downgrade drops index THEN columns; optional `SET lock_timeout='5s'` (migration-16 precedent).
- **Pitfall:** **two-value enum, drop `cove_confirmed`** (no producer); model/migration type drift; no
  CREATE INDEX CONCURRENTLY (breaks the single-transaction 18+19 batch, no benefit on tiny tables).
- **Refs:** alembic cookbook; blog.jerrycodes.com/multiple-heads-in-alembic-migrations/.

### CORR-4
- **Best practices:** match on **`clusters[].id` (CLUSTER id)** — never opinion id. Corroboration needs a
  **membership test**, so short-circuit on first match (lower `max_results`). **Before building:** add a
  bounded+TTL cache to `get_outbound_case_citations`; **acquire the semaphore per-GET** (don't hold it
  across the N-fetch); use a **separate smaller semaphore** for this heavy fan-out. Annotation-only; CL
  unavailable/`[]` → abstain `(None,None)`; CL-present-no-match → `ai_inferred`.
- **Concrete:** `get_outbound_case_citations` (`citation_lookup.py:880`) = 2+2×N serial GETs (`:918` holds
  `Semaphore(5)` whole time, no cache). Cited id is a URI-tail string (`:969-974`, can be `''`);
  `node.courtlistener_id` = `str(cluster['id'])` (`:165`). **Normalize both sides** (`str(x).strip()`,
  reject empty) before compare; empty = abstain. Add matches/attempts metric. Bulk alt:
  `POST /api/rest/v4/citation-lookup/` returns `clusters[].id`; header `Authorization: Token <key>`.
- **Edge/pitfall:** CL throttle mid-run → silent `[]` → wrong `ai_inferred` (the North-Star failure);
  opinion-id/cluster-id confusion = 100% silent miss; ~52 serial GETs/edge × many edges (rated L,
  understated).
- **Refs:** courtlistener.com/help/api/rest/citation-lookup/, wiki.free.law citations.

### CORR-5
- **Best practices:** dedup branch = **insert-or-upgrade** — on a strictly-stronger incoming corroboration
  set fields + `db.flush()` but **keep returning None** (so `edges_created` counters at `:2116,2248,2292`
  don't double-count). Precedence rank dict `{'ai_inferred':0,'courtlistener':1}` ("never downgrade" = one
  compare). **Precondition:** fix `_merge_node_by_cluster` (`:680-686`) — it re-points edges without
  dedup, so parallel-cite merges create DUPLICATE rows on the dedup key; collapse keeping strongest
  corroboration + max confidence, with a test (parallel cites → one `case_cites` edge per pair/treatment).

### CORR-6 / CORR-7 / CORR-3 / CORR-8
- **CORR-3:** pure read-time projection; `provenance` = identity, separate from `good_law_status`. Note the
  `is_verified` OR is redundant today but `find_or_create_node` can set `courtlistener_id` directly
  (`:2280`) without the good-law check → add the round-trip test (`provenance='courtlistener'` AND
  `good_law='unknown'` simultaneously).
- **CORR-6:** pure type plumbing, no route change; `tsc` round-trip through `normalizeCitationNetwork`.
- **CORR-7:** badge = identity/network presence, **not** good law (C1). **Design it to render gracefully
  in the all-`ai_inferred` state** (the honest pre-corroboration truth) → ships independent of CORR-4; gate
  the "verified rate" messaging on an observable corroborated/total metric.
- **CORR-8:** document the CORR-1 prerequisite (C7) + sanity log; offline/`[]`/no-token → stays
  `ai_inferred`; doubles as a perf kill-switch.

### SAFE-1 / SAFE-2 / SAFE-3 / SAFE-4
- **SAFE-1:** `no_adverse` = abstention. Canonical proof: superseded-by-statute is invisible to
  citation-based citators (statute doesn't cite the case) → absence ≠ clearance even in Shepard's. CL 404 →
  `unknown`, distinct from `no_adverse`; never serialize either as positive.
- **SAFE-2:** quarantine only true non-resolution; **CL 404 (valid-but-absent) is `unknown`, NOT
  quarantine** — don't accuse a real-but-uncommon case of being fabricated.
- **SAFE-3:** deterministic **output rail** (treat the answer as untrusted; append from **node-identity**,
  not model self-disclosure or prose substrings). Prefer a sibling `collect_bad_law_findings` + a single
  shared `append_bad_law_banner(answer, findings, verified_citations)` imported by both `query_matter` and
  `generate_node` (one matcher, no drift). Keep the prompt instruction too, but SAFE-4 must pass with it
  deleted. Sanitize/length-bound the node name before concatenation (prompt-injection via case name).
- **SAFE-4:** fully offline/deterministic; stub `generate_answer`; assert banner+name, status `bad` (not
  `no_adverse`/`good`), no banner when bad authority absent, DISTINGUISHES→neutral; must fail if the banner
  is removed from EITHER pipeline. Note: a faithful (1.0) grounding score on a BAD-LAW chunk is still bad —
  validity is a separate assertion.
- **Refs:** docs.nvidia.com/nemo/guardrails output-rails; OWASP LLM prompt-injection; dho.stanford.edu
  Legal_RAG_Hallucinations; docs.ragas.io noise_sensitivity.

### RECALL-1 / RECALL-2
- Include negation traps ("we do not overrule X", "remains good law", "declined to overrule"),
  DISTINGUISHES-not-overrules, ABROGATES/SUPERSEDES, QUESTIONS/CRITICIZES/LIMITS, partial, **passive
  voice**, pure CITES, **plus ≥1 adversarial prompt-injection window**. RECALL-2: **per-polarity P/R/F1**
  (asymmetric — negative-recall floor ≥0.7; a blended F1 hides missed overrules), QAG-style binary
  verdicts over **recorded** extractions (zero network in CI), live refresh script env-gated.

### TAX-1 / TAX-2 / TAX-3 / TAX-4 / TAX-5 / TAX-6
- **TAX-1 (keystone):** adopt the 4-tier axis both citators converge on. Map OVERRULES/REVERSES/**VACATES**
  →`bad`; ABROGATES/SUPERSEDES→`superseded`; QUESTIONS/CRITICIZES/LIMITS→`caution`;
  **DISTINGUISHES→neutral**; FOLLOWS→positive; CITES/EXPLAINS/HARMONIZES→neutral. Encode ONE ordered
  precedence (`bad>superseded>partial>caution>no_adverse>good`) in a single place with a **truth-table
  test**. Preserve GL-006 CL-precedence guard; emit new statuses through `_node_to_dict`.
- **TAX-3:** centralize polarity in `treatment_verifier` (C5); broaden `_NEGATIVE` w/ verb phrases; treat
  the relationship window as data (robust delimiters; `.replace('{document}',window)` at `:1739,1772`).
- **TAX-4 (gate on RECALL-2):** partial = first-class tier (KeyCite red-striped). Migration **19** (revises
  18), nullable `is_partial BOOLEAN` + `overrule_scope TEXT`. **Drop the `context_excerpt`-text-encoding
  option** (banned brittle-string + second source of truth). "reversed on other grounds" → cited holding
  may retain value (don't auto-bad).
- **TAX-5 (gate on RECALL-2):** binding via **CL structured court id**, never court-name substrings;
  cross-jurisdiction/unknown → cap at `caution`, never escalate to `bad`. Defer ranked table to a labeled
  court-hierarchy fixture.
- **TAX-6:** suppress a negative/cautionary edge only when treating-authority date ≤ `as_of_date`; missing
  date → include (fail toward warning); keep inside `to_thread`; extends `test_temporal_as_of.py`.

### STAT-1
- Statute supersession comes from **amendment chains, not the citation graph** (C1). Confirm
  `get_version_chain()` is pure-DB single/bounded (not N+1); cap statute nodes (mirror `max_citations=5`);
  verify `superseded_date` actually populates in a fresh dev DB first.

### PARALLEL-1
- **Split into two layers** (fixes the async/sync hazard): (a) **sync** resolver for the answer path =
  canonical + `_lookup_cache.peek()`, no await/network, bail to canonical-only on miss; (b) **async**
  resolver for BRIEF-1, free to `await lookup_citation()` + enrich. Drive short-form/parallel grouping off
  **eyecite `resolve_citations(get_citations(text))`** (read the Resource grouping, not surface strings);
  dedupe parallels on `clusters[].id`. Make cache-only-vs-network a **parameter** so one resolver serves
  both call sites. Run `clean_text(..., steps=['html','inline_whitespace','underscores'])` before
  `get_citations` on PDF/DOCX (extraction underscores break spans); wrap `get_citations` in a timeout for
  uploads. Do NOT reorder before `resolve_citations` (`Id.` binds to the preceding Resource — same family
  as the canonical-[n] bug). Bare/foreign/neutral names → OMIT.
- **Refs:** freelawproject.github.io/eyecite/find.html; courtlistener citation-lookup.

### BRIEF-1
- **All-Form signature** (`brief_text: str|None = Form(None)`, `file: UploadFile|None = File(None)`) — you
  cannot bind JSON body + UploadFile on one handler; require exactly one → 400. Reuse the **matter-upload
  guards** (not just `/verify-citations`): `_guard_matter`, `validate_filename`, `MAX_UPLOAD_SIZE`
  (Content-Length **and** post-`read()` len), `validate_file_type`, magic-byte `validate_file_format`, then
  cap extracted text ~20000. Fan-out under a dedicated `Semaphore(~10)` + `gather(return_exceptions=True)`
  + per-item `wait_for`; cap 25 before building tasks; `await asyncio.to_thread(is_good_law,…)` (sync DB).
  **Allow live (uncached) resolution here** (latency-tolerant) — this is what decouples the inviolable
  PARALLEL-1 hot-path bail. Always 200 on a well-formed request.
- **Status map:** TAX verdicts + `unknown`; CL 404→`unknown` (not fabricated); CL 300→surface candidates;
  non-resolving→**OMIT** (distinct from `unknown`). Envelope mirrors `/verify-citations`.
- **Refs:** fastapi.tiangolo.com/tutorial/request-files/; FastAPI discussion #5666.

### PROV-1
- Read-only surfacing of `negative_treatments[].context` + overruling `source_url`; **validate https +
  trusted origin before serialization** (C2), omit otherwise; `rel="noopener noreferrer"`. Phrase labels:
  "Overruled by / Overruled in part by / Abrogated by / Superseded by statute / Reversed by / Vacated by /
  Questioned by / Criticized by / Limited by / Distinguished by." Keep validity separate from provenance.

---

## Reviewer-flagged adjustments (deepened 2026-06-08)

**HIGH**
1. **[PARALLEL-1, BRIEF-1]** Split the resolver (sync cache-peek for answer path; async for BRIEF-1) —
   fixes the async/sync layering violation and breaks the "inviolable" coupling.
2. **[CORR-4]** Hard go/no-go on the spike (pin `clusters[].id` on a real overruled pair) before any code;
   else descope CORR-4 to **abstain-only** (CORR-1/2/5 ship regardless). Add TTL cache + per-GET semaphore
   + separate smaller fan-out semaphore + membership short-circuit + normalized compare + matches/attempts
   metric.
3. **[TAX-1/4/5/6]** Ship TAX-1 (caution/superseded) as the line; **defer TAX-4 + TAX-5 unless RECALL-2
   surfaces a concrete miss.** One ordered precedence + truth-table test.
4. **[CORR-5]** Add `_merge_node_by_cluster` edge dedup as a precondition (collapse duplicate rows keeping
   strongest corroboration) + a test.
5. **[BRIEF-1]** Full upload-safety guards (size pre+post, type allowlist, magic-byte, extracted-text cap);
   oversized/wrong-type/malformed → 4xx, never OOM/500.
6. **[QUOTE-1/2]** Parallelize verification (`Semaphore(3-5)` + `wait_for`) + add a rapidfuzz lexical gate
   for quoted spans; reuse SSRF allowlist + body cap; consider default-False in dev.

**MED**
7. **[CORR-2/4/5]** Drop `cove_confirmed` → two-value enum, `VARCHAR(20)`, reserve the third in a comment.
8. **[Architecture]** Relax "no new service modules" for `citation_resolution.py` + `citation_corroboration.py`
   rather than growing `citation_graph.py` (~2331 lines) / `rag_engine.py` (~2370) — both ~3× the 800-line
   max (split pending, task #24); sequence to land with/after the split.
9. **[SAFE-3]** Sibling `collect_bad_law_findings` + single shared `append_bad_law_banner` (one matcher).
10. **[TAX-3+SAFE-3]** Centralize negation polarity in `treatment_verifier`; share one negation fixture.
11. **[CORR-1]** Scope enrichment to this-ingest nodes + DB-state idempotency; cap lookups/run.
12. **[CORR-4+CORR-1]** Don't share the single `Semaphore(5)`; run CORR-1 first or give CORR-4 a separate budget.
13. **[BRIEF-1/QUOTE/PROV-1]** SSRF/https allowlist on every fetched/surfaced URL (C2); privilege-safe logging (C6).
14. **[TAX-4]** Drop the `context_excerpt`-text-encoding option (banned brittle pattern) — migration 19 properly or defer.
15. **[TAX-3/RECALL-1/SAFE-3]** Add an adversarial prompt-injection fixture; sanitize/length-bound the banner case name.

**LOW**
16. **[CORR-2/TAX-4]** Downgrade drops index before column; verify up→down→up on a scratch DB.
17. **[UI sequencing]** Ship SAFE-1 + CORR-7 (all-`ai_inferred`-safe) first; hold TAX-2 palette until statuses
    emit + are benchmark-backed; consider provenance as icon/tooltip, not a second full badge.
18. **[STAT-1]** Confirm `get_version_chain()` is single/bounded pure-DB; cap statute nodes; record query count.
19. **[CORR-3/6]** Round-trip test: outbound-seeded node with `provenance='courtlistener'` AND `good_law='unknown'`.
20. **[CORR-2/5]** `verified_by` stays NULL for backfilled rows; `_edge_to_dict` emits `null`; populated only on upgrade.
