# Plan: As-of-Date Temporal Queries

**Created:** 2026-05-30
**Status:** Ready for implementation
**Priority:** P0 — only fully-missing headline temporal feature
**Estimated effort:** ~3.5–5 dev-days (phased)
**Depends on:** `effective_date` Qdrant DATETIME index (DONE, `vector_store.py:127`)

## Problem

Users cannot ask "what was the law as of `<date>`". Temporal filtering today is a post-retrieval Python filter on `document_status` only (`rag_engine.py` §4.55), with no `as_of_date` param, no intent detection, and no Qdrant-level date filtering. The spec (`temporal_awareness_spec.md` §6–§8) describes the feature against an older FLOAT-timestamp design; the code has since moved to ISO-string `effective_date` indexed as `PayloadSchemaType.DATETIME`.

## Retrieval semantics (the contract)

Given `as_of`:
- **INCLUDE** if `effective_date <= as_of` AND (`superseded_date` is null OR `superseded_date > as_of`) — i.e. a doc that is *now* superseded but was current *then* must appear.
- **EXCLUDE** `effective_date > as_of` (future versions not yet in force).
- **EXCLUDE** `superseded_date <= as_of`.
- **INCLUDE undated docs by default** (no `effective_date`), flagged `unknown` in context (config `temporal_include_undated=True`).

Default (no date): `as_of=None`, `exclude_superseded=True` → current behavior preserved. Historical/"all versions" intent → `exclude_superseded=False`.

## Prerequisite fix — `superseded_date` not in Qdrant payload

Server-side as-of supersession needs `superseded_date` in the payload:
- `tasks.py:269` — write `chunk["superseded_date"] = temporal_result.superseded_date.isoformat() if ... else None`.
- `vector_store.py:327` (upsert metadata) — add `"superseded_date": chunk.get("superseded_date")`.
- `vector_store.py:127` (index registration) — add `"superseded_date": PayloadSchemaType.DATETIME`.
- `vector_store.py:480` and `:584` (result dicts) — surface `superseded_date`.
- Existing points lack the key until re-ingested; the null-safe `must_not` range (below) treats them as never-superseded (safe, slightly permissive). Pair with the reindex tooling plan for backfill.

## Design

### Intent detection — `_detect_temporal_intent(query) -> Tuple[Optional[datetime], bool]`
Add after `rag_engine.py:251`. Returns `(as_of_date, exclude_superseded)`. Reuse `temporal_extractor._parse_date_string`. Patterns (case-insensitive, year band 1900–2100 to avoid matching citations/section numbers):
- `as of <date>` → `(date, False)`
- `in <year>` / `under the <year> act` → `(datetime(year,7,1,utc), False)`
- `before|prior to|pre <year>` → `(datetime(year,1,1,utc), False)`
- `current law|currently|in force` → `(None, True)`
- `historical|history of|evolution of|all versions|over time` → `(None, False)`
- default → `(None, True)`

### Explicit param (recommended, wins over NL)
- `schemas.py:52` `QueryCreate`: add `as_of_date: Optional[datetime] = Field(None, ...)`.
- `main.py` ask handler: pass `as_of_date=body.as_of_date` at all three `query_matter` call sites (~1231 agentic, ~1238 fallback, ~1244 standard).
- `rag_engine.py:1174` `query_matter`: add `as_of_date: Optional[datetime] = None`.
- Precedence: `if as_of_date: (as_of_date, False) else _detect_temporal_intent(query)`.

### Qdrant filter — `build_temporal_filter` + shared translation
- New `build_temporal_filter(as_of_date, exclude_superseded, base_filter)` in `vector_store.py` (~line 106) returns a plain dict with sentinel keys `_temporal_as_of` (ISO str) and `_temporal_current_only` (bool) merged with `base_filter` (jurisdiction etc.).
- Factor a single `_build_qdrant_filter(query_filter)` helper used by BOTH `search_vectors` (~`vector_store.py:421`) and `search_sparse_vectors` (~`:532`) — otherwise the sparse path bypasses temporal filtering. It pops the sentinels and builds:
  ```python
  must = [FieldCondition(key=k, match=MatchValue(value=v)) for k,v in qf.items()]
  must_not = []
  if temporal_as_of:
      # null-safe effective_date: (effective_date <= as_of) OR (effective_date is empty)
      must.append(Filter(should=[
          FieldCondition(key="effective_date", range=DatetimeRange(lte=temporal_as_of)),
          IsEmptyCondition(is_empty=PayloadField(key="effective_date")),
      ]))  # gate the IsEmpty branch on temporal_include_undated
      must_not.append(FieldCondition(key="superseded_date", range=DatetimeRange(lte=temporal_as_of)))
  elif current_only:
      must.append(FieldCondition(key="document_status", match=MatchValue(value="current")))
  Filter(must=must or None, must_not=must_not or None)
  ```
- `DatetimeRange` accepts ISO 8601 strings against a DATETIME-indexed field. `must` range auto-excludes null fields → wrap effective_date in `should` + `IsEmptyCondition` for undated inclusion. `must_not` range leaves null `superseded_date` in (desired).

### Python net (keep, make as-of-aware)
The existing post-retrieval filter (`rag_engine.py` §4.55) re-resolves authoritative `document_status` from Postgres — keep it as defense-in-depth for the **default current-law path**. When `as_of_date is not None`, **bypass** the `_effective_status` exclusion (it would delete exactly the in-force-then docs the feature targets) — server-side date range is authoritative there.

### Progressive fallback
Replace the `<3 results` retry to step down: temporal+jurisdiction → jurisdiction-only → unfiltered. Set `temporal_filter_relaxed: true` in the response when relaxed so the UI can warn.

### Config flags (`config.py`, temporal block ~76)
- `temporal_as_of_enabled: bool = True`
- `temporal_include_undated: bool = True`

### Frontend (Phase 3)
- `types.ts`: ask request `asOfDate?: string`; expose `temporal_filter_relaxed`.
- `api-services.ts` `askQuestion`: add `asOfDate?` → `as_of_date` in POST body.
- `ChatPanel.tsx`: optional `<input type="date">` next to the legal-research toggle; thread `asOfDate` through `app/matters/[id]/page.tsx`; active-date chip above input.
- `TemporalBadge.tsx` (new) on citations when `documentStatus !== "unknown"`.

## Edge cases
- No date → default, zero regression. Future date → harmless (optionally clamp to now behind flag). Undated doc → included + flagged. All filtered out → progressive fallback + `temporal_filter_relaxed`. Stale Qdrant status → Postgres net covers default path.

## Test plan
- **Unit:** `_detect_temporal_intent` per pattern + negatives; `build_temporal_filter` sentinel shape + base-filter merge + immutability; filter translation produces correct `must`/`must_not`/nested-`should` (introspect `Filter`, no live Qdrant).
- **Integration (live Qdrant):** seed 3 versions of one statute (2018 superseded, 2024 current, 2026 future). Assert: default → 2024 only; `as_of=2020` → 2018 only (headline); `as_of=2030` → 2024+2026; undated doc present in as-of + flagged; all-filtered → relaxed flag. API: `as_of_date` in body overrides NL; NL "as of 2019" works.

## Phases
1. **MVP server-side as-of (~1.5d):** `superseded_date` payload+index+writers; `build_temporal_filter` + shared translation w/ null handling; `QueryCreate`/`query_matter`/`main.py` passthrough; Python-net bypass for as-of. Explicit param only.
2. **NL intent + recall (~1d):** `_detect_temporal_intent`, progressive fallback, flags, `temporal_filter_relaxed`.
3. **Frontend (~1d):** types, `askQuestion` param, date-picker chip, `TemporalBadge`.
4. **Tests + buffer (~0.5–1.5d).**

## Risks
- `superseded_date` payload gap (must fix; existing points need reindex — see reindex plan).
- Undated docs silently dropped without `IsEmptyCondition` branch.
- Python net deleting in-force-then docs (bypass when as-of set).
- NL false positives (year band + adjacent keyword requirement).
- Hybrid divergence (shared `_build_qdrant_filter` helper).
