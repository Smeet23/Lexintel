# CourtListener On-Demand Legal Research Integration

**Date:** 2026-03-14
**Status:** Approved
**Scope:** Phase 1 — Quickest Win (US Federal case law)

---

## Overview

Add on-demand legal research to LexIntel by integrating CourtListener's free API. When a lawyer toggles "Include Legal Research" in the chat UI, the system queries CourtListener for relevant US case law in real-time and merges results with the user's uploaded documents in the RAG pipeline.

No pre-loading, no new database tables, no background jobs. Just a real-time API call added to the existing query pipeline.

---

## User Flow

1. Lawyer opens a Matter workspace and sees the chat input
2. A toggle appears near the chat input: **"Include Legal Research"** (off by default)
3. When **OFF** — works exactly as today (queries user's uploaded docs only)
4. When **ON**:
   - System searches user docs (existing RAG pipeline) AND queries CourtListener API in parallel
   - Merges both result sets, reranks together, sends combined context to Gemini
   - Response includes citations from BOTH user docs and real case law
5. Case law citations appear in the CitationPanel with:
   - Case name, court, date decided
   - Relevant excerpt
   - Link to full opinion on CourtListener

---

## Backend Technical Design

### New File: `backend/services/legal_research.py`

```
CourtListenerService
├── search_cases(query, jurisdiction=None)
│   → Calls CourtListener Search API
│   → Returns top 5-10 relevant opinions
│
├── get_opinion(opinion_id)
│   → Fetches full opinion text for a specific case
│
└── format_as_context(results)
    → Converts API results into the same chunk format
      the existing RAG pipeline expects
      (content, page_num, section_name, metadata)
```

### Changes to Existing Files

1. **`backend/services/rag_engine.py`**
   - `query_matter()` gets new parameter: `include_legal_research: bool`
   - When true: calls `CourtListenerService.search_cases()` in parallel with Qdrant vector search
   - Merges external results with user doc results
   - Modified prompt distinguishes "Your Documents" vs "Case Law"

2. **`backend/main.py`**
   - `/matters/{matter_id}/ask` endpoint gets new optional field: `include_legal_research: bool = False`

3. **`backend/config.py`**
   - Add `courtlistener_api_token: str` to Settings

4. **`backend/schemas.py`**
   - Update request schema with `include_legal_research` field

### CourtListener API Details

**Search endpoint:**
```
GET https://www.courtlistener.com/api/rest/v4/search/
  ?q={user's query}
  &type=o              (opinions only)
  &order_by=score      (relevance-ranked)
  &page_size=10
```

**Response fields used:**
- `caseName` — e.g., "Smith v. Jones"
- `court` — e.g., "Supreme Court of the United States"
- `dateFiled` — decision date
- `snippet` — relevant text excerpt
- `absolute_url` — link to full opinion
- `citation` — proper legal citation (e.g., "531 U.S. 98")
- `status` — precedential value (Published, Unpublished, etc.)

### Merging Strategy

1. CourtListener results get normalized scores based on search rank
2. Interleaved with user doc chunks
3. Combined set reranked using existing cross-encoder
4. Total context capped at existing token budget (50K tokens)
5. Each chunk labeled with source ("Your Documents" vs "Case Law")

### Error Handling

- CourtListener timeout: 3 seconds — if down, query proceeds with user docs only
- Rate limit tracking: 5,000 requests/hour — graceful degradation when approaching limit
- Network errors: logged, non-blocking

---

## Frontend Changes

### ChatPanel.tsx
- Add toggle switch near chat input: "Include Legal Research"
- Default: OFF
- Sends `include_legal_research: true` in request body when ON

### CitationPanel.tsx
- Visual distinction for case law results:
  - "Case Law" badge (different color from "Your Document")
  - Shows: case name, court, date, citation string
  - Clickable link to full opinion on CourtListener

### api-services.ts
- Update `askQuestion()` to accept and pass `include_legal_research` parameter

### types.ts
- Update request/response types for the new field

---

## Scope Summary

| Area | Changes |
|------|---------|
| **New file** | `backend/services/legal_research.py` |
| **Backend edits** | `rag_engine.py`, `main.py`, `config.py`, `schemas.py` |
| **Frontend edits** | `ChatPanel.tsx`, `CitationPanel.tsx`, `api-services.ts`, `types.ts` |
| **New infra** | None — just a free CourtListener API token |
| **New DB tables** | None |

---

## Future Enhancements (not in scope)

- Add GovInfo API for US Code / statutes
- Add eCFR API for federal regulations
- Add jurisdiction filtering (CA, NY, DE state courts)
- Cache frequently accessed case law in local DB
- Pre-load landmark cases for faster access
- Citation graph traversal (find cases that cite/overrule a result)
