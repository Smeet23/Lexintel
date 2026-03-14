# Functional Tabs & Precedents Design

**Date:** 2026-03-13
**Scope:** Contract Review, Draft Assistant, Audit Log, Precedents (Cross-Matter Search + Save)

---

## 1. Contract Review

### Backend

**New endpoint:**
- `POST /matters/{matter_id}/contract-review` — trigger analysis (optionally pass `document_id`, defaults to first document)
- `GET /matters/{matter_id}/contract-review` — fetch cached result

**New table: `contract_reviews`**
```sql
id              UUID PRIMARY KEY
matter_id       UUID FK → matters(id)
document_id     UUID FK → documents(id)
risks           JSONB       -- [{clause, section, risk_level, explanation, remedy}]
summary         JSONB       -- {total_clauses, high_risk, medium_risk, low_risk}
missing_clauses JSONB       -- ["Force Majeure", "Non-Compete", ...]
overall_score   INTEGER     -- 0-100
created_at      TIMESTAMP
```

**Flow:**
1. Fetch all chunks for the document
2. Send to Gemini with contract risk analysis system prompt
3. Gemini returns structured JSON: risks array, summary stats, missing clauses, overall score
4. Store in `contract_reviews` table
5. Return to frontend

**Re-run:** User clicks "Re-analyze" → deletes existing review → triggers new POST

**Gemini prompt structure:**
- System: "You are a legal contract risk analyst. Analyze the following document chunks and return a JSON object with..."
- Returns: `{risks: [{clause, section, risk_level, explanation, remedy}], summary: {total_clauses, high_risk, medium_risk, low_risk}, missing_clauses: [...], overall_score: 0-100}`

### Frontend

- On tab load, `GET` cached review. If none exists, show "Run Analysis" button
- Display risk cards (High=red, Medium=yellow, Low=green) with clause name, section, explanation, suggested remedy
- Right sidebar: summary stats (total clauses, risk breakdown, score bar) + missing clauses list
- "Re-analyze" button triggers new POST

---

## 2. Draft Assistant

### Backend

**New endpoints:**
- `POST /matters/{matter_id}/drafts` — generate a draft (body: `{document_type, instructions}`)
- `GET /matters/{matter_id}/drafts` — list past drafts
- `GET /matters/{matter_id}/drafts/{draft_id}` — get single draft

**New table: `drafts`**
```sql
id              UUID PRIMARY KEY
matter_id       UUID FK → matters(id)
document_type   VARCHAR     -- "Legal Memo", "Motion", "Response Letter", etc.
instructions    TEXT        -- user-provided instructions
content         TEXT        -- generated draft content
sources         JSONB       -- [{document_name, page_num, section_name, excerpt}]
created_at      TIMESTAMP
```

**Flow:**
1. Receive document_type + instructions
2. Embed the instructions using Cohere
3. Vector search matter's chunks for relevant context
4. Send context + instructions + document_type to Gemini with legal drafting prompt
5. Parse response, extract inline source references
6. Store in `drafts` table
7. Return draft to frontend

**Document types:** Legal Memo, Motion, Response Letter, Summary Brief, Case Analysis, Client Advisory

**Gemini prompt structure:**
- System: "You are a legal drafting assistant. Generate a {document_type} based on the provided context and instructions. Include inline source references [Source: document, page X] where applicable."

### Frontend

- Left panel: form (document type dropdown, instructions textarea, "Generate Draft" button) + list of past drafts
- Right panel: generated content display with inline source highlights
- Copy button and download as .txt
- Past drafts clickable to view again

---

## 3. Audit Log

### Backend

**New endpoint:**
- `GET /matters/{matter_id}/audit-log` — fetch all activity for a matter (ordered by created_at DESC)

**New table: `audit_logs`**
```sql
id              UUID PRIMARY KEY
matter_id       UUID FK → matters(id)
action          VARCHAR     -- "document_uploaded", "query_asked", etc.
user            VARCHAR     -- "System" for now
details         TEXT        -- context (question text, document name, etc.)
sources         VARCHAR     -- page/section references (nullable)
created_at      TIMESTAMP
```

**Tracked actions (written internally by backend, no write endpoint):**
- `document_uploaded` — when a document is uploaded
- `document_deleted` — when a document is deleted
- `query_asked` — when a question is asked via Ask AI
- `contract_review_run` — when contract review is triggered
- `draft_generated` — when a draft is generated
- `matter_created` — when the matter is first created
- `matter_cancelled` — when processing is cancelled

**User field:** "System" for now; swap to real user identity when auth is implemented.

### Frontend

- Replace `mockAuditLog` with real `GET /matters/{id}/audit-log` call
- Same table UI: action badge, user, details, sources, relative timestamp
- "Export Log" button exports as CSV

---

## 4. Precedents (Cross-Matter Search + Save)

### Backend

**New endpoints:**
- `POST /precedents/search` — vector search across all matters (body: `{query}`)
- `POST /precedents/save` — bookmark a result (body: `{title, query, matter_id, document_name, chunk_content, page_num, section_name, relevance_score, tags, notes}`)
- `GET /precedents` — list all saved precedents
- `DELETE /precedents/{id}` — remove a saved precedent

**New table: `saved_precedents`**
```sql
id              UUID PRIMARY KEY
title           VARCHAR
query           TEXT        -- original search query
document_name   VARCHAR
matter_id       UUID FK → matters(id)
chunk_content   TEXT
page_num        INTEGER
section_name    VARCHAR
relevance_score FLOAT
tags            JSONB       -- ["patent", "infringement", ...]
notes           TEXT        -- user notes (nullable)
created_at      TIMESTAMP
```

**Search flow:**
1. Receive query string
2. Embed query using Cohere
3. Search Qdrant across ALL matter collections (not filtered by single matter)
4. Rerank results with cross-encoder
5. Return results grouped by matter/document with relevance scores

**Save flow:**
1. User clicks "Save" on a search result
2. Frontend sends chunk details + optional tags/notes
3. Stored in `saved_precedents` table

### Frontend

- **Search tab:** search bar at top, results below grouped by matter → document. Each result shows: document name, section, relevance score (color-coded), content excerpt, "Save" button
- **Saved tab:** list of bookmarked precedents with tags, notes, relevance score. Filter by tags. Delete button.
- Save dialog: title (auto-filled), tags input, notes textarea

---

## Database Migration Summary

One Alembic migration adding 4 tables:
- `contract_reviews`
- `drafts`
- `audit_logs`
- `saved_precedents`

## API Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/matters/{id}/contract-review` | Trigger contract analysis |
| GET | `/matters/{id}/contract-review` | Fetch cached analysis |
| POST | `/matters/{id}/drafts` | Generate a draft |
| GET | `/matters/{id}/drafts` | List drafts |
| GET | `/matters/{id}/drafts/{draft_id}` | Get single draft |
| GET | `/matters/{id}/audit-log` | Fetch audit log |
| POST | `/precedents/search` | Cross-matter search |
| POST | `/precedents/save` | Save a precedent |
| GET | `/precedents` | List saved precedents |
| DELETE | `/precedents/{id}` | Delete saved precedent |
