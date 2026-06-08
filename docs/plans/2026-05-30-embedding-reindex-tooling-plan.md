# Plan: Embedding Provider Re-Index / Migration Tooling

**Created:** 2026-05-30
**Status:** Ready for implementation
**Priority:** P1
**Estimated effort:** ~1 working week (P0 2–3d / P1 1–2d / P2 1–2d)
**Migration:** new Alembic rev (next free is **14**; coordinate with agentic plan which also wants 14 — first to merge takes 14, other takes 15).

## Problem — cross-provider stale vectors

`_detect_provider()` (`embeddings.py:51-71`) picks Voyage if `VOYAGE_API_KEY` set, else Cohere — once, at startup, for BOTH ingestion and querying. Vectors in Qdrant carry **no record of which model produced them**. When the active provider flips, old documents keep stale vectors while queries embed with the new provider → cosine similarity across two unrelated vector spaces. No error, no log — silent retrieval degradation (plan R2). This tooling fixes it.

## Strategy — in-place batched re-upsert, tracked by `embedding_model`

Both providers are **1024-dim** (`VECTOR_SIZE=1024`, `EMBEDDING_DIMENSIONS=1024`), so the existing collection/HNSW/payload-index/sparse config all stay valid — no recreation. Point IDs are **deterministic** (`_generate_point_id` = MD5 of `f"{matter_id}:{chunk_id}"`, `vector_store.py:87`; chunk_id = `uuid5(...)`, `tasks.py:280`), so re-upserting the same chunk_id overwrites the vector+payload in place (Qdrant upsert is keyed by point ID). **In-place beats blue-green** here purely because of dim parity; blue-green is only needed if dims change.

Caveat: a matter mid-reindex transiently holds both providers' vectors (some chunks overwritten, some not) — same R2 degradation, scoped to seconds/minutes, self-healing on completion.

## Schema

`embedding_model String(64)` (store model name e.g. `"voyage-law-2"`, not provider) on **both**:
- `Matter.embedding_model` (authoritative per-matter vector space; drives admin status + all-matters filter).
- `Chunk.embedding_model` (per-chunk provenance for partial-failure resume).

Alembic rev **14** (`down_revision="13"`; 13 = citation graph). Add both columns nullable; backfill existing rows to `'embed-english-v3.0'` (Voyage was only just wired in, so no Voyage vectors predate this — verify before assuming; else leave NULL = "needs reindex"). Add matching ORM columns (`models.py` after `:32` Matter, after `:89` Chunk).

## Reindex tasks (`backend/tasks.py`, per convention)

### `reindex_matter_task(self, matter_id, target_model="voyage-law-2")`
`@shared_task(bind=True, max_retries=3, acks_late=True, track_started=True)`:
1. `create_collection(matter_id)` (idempotent).
2. Query chunks where `embedding_model != target OR is null` (resume-safe), ordered by `chunk_sequence`, in 96-batches.
3. **SAC-correct text** (critical): ingestion embeds summary-augmented text `f"{doc.summary}\n{chunk.content}"` (`tasks.py:313-315`) — reindex MUST replicate this, not raw content, or vectors diverge from fresh-ingest.
4. `embed_chunks_with_provider(texts, provider="voyage")` (§ forcing provider).
5. Build `_chunk_to_payload_dict(chunk)` hydrating from `Chunk` + its `Document` (document_name/type/jurisdiction live on Document; authority/temporal inside `Chunk.authority_metadata`). Re-upsert via `upsert_vectors` (unchanged — validates dim, same point ID).
6. **Preserve sparse vectors** (hybrid): regenerate via `generate_sparse_vectors_batch` and pass through, else dense-only re-upsert silently disables hybrid for those chunks.
7. Set `chunk.embedding_model = target` per batch, `db.commit()` per batch (resumable checkpoint).
8. After all chunks: `matter.embedding_model = target` (only on full success → half-done matters never falsely report done). Publish progress via existing `publish_embedding`/`publish_ready`/`publish_error` (`progress.py:152/196/207`).

Idempotent (re-run re-embeds 0 on a clean matter); resumable (per-batch commit + filter).

### `reindex_all_matters_task(target_model="voyage-law-2")`
Query non-deleted, `status="ready"`, `embedding_model != target`, ordered by `updated_at.desc()` (recency first); dispatch one `reindex_matter_task` per matter via `apply_async(queue="celery")` (sequential, rate-limit-bounded).

## Forcing provider (don't break public API)

**Option 1 (recommended):** add `embed_chunks_with_provider(chunks, provider)` to `embeddings.py` routing directly to the Voyage/Cohere batch branches (`:294-303`), bypassing `_detect_provider()`. Refactor `_cache_key` (`:74`) to accept `provider=` override so the reindex caches Voyage vectors under a `voyage:` key (not a detected-`cohere:` key). `embed_chunks(chunks)` becomes a thin wrapper = `embed_chunks_with_provider(chunks, _detect_provider())` — zero change to existing callers. Both clients can coexist in one process.

## Admin endpoints (`main.py`)

- `GET /admin/embedding-status`: per-matter `{embedding_model, total_chunks, chunks_on_target, drift = total - on_target}` + aggregate `{active_provider, target_model, reindexed_matters, pending_matters, dimensions:1024}`. Drift = partial-reindex detector.
- `POST /admin/reindex` (body: optional `matter_id`, `target_model`): dispatch `reindex_matter_task` or `reindex_all_matters_task` via `send_task(..., queue="celery")`; return task id. Poll status via `/admin/embedding-status` (drift→0) and the existing SSE progress channel.
- **Auth (hard prereq, out of scope):** app has no auth layer; `/admin/*` must be gated (API key / IP allowlist) before prod.

## Config (`config.py`)
Add `embedding_model: str = "voyage-law-2"` so the reindex target + admin report aren't hardcoded literals.

## Edge cases
- Dim mismatch: `upsert_vectors` raises `ValueError` (`vector_store.py:299`) — treat as fatal/non-retryable (wrong model), not retry-3x.
- Partial failure: per-batch commit resumes; bad chunk → log+continue but don't mark matter done.
- Mid-reindex query: Qdrant reads never block; transient mixed space self-heals (accept + low-traffic timing).
- Cache: provider-prefixed key already prevents cross-provider serving (`embeddings.py:74`) — just ensure explicit-provider path uses the explicit provider in the key.
- Cost: ~10k chunks ≈ 5M tokens ≈ $0.60 (within Voyage free tier); sequential + 96-batches.
- **Voyage batch lacks retry**: `_voyage_embed_batch_with_retry` (`:118`) is mis-named — no tenacity. Add a tenacity wrapper before bulk reindex (429s bite during bulk).

## Test plan
- **Unit** (`test_reindex.py`): `embed_chunks_with_provider` routing; `_cache_key(provider=...)` distinctness; `_chunk_to_payload_dict` completeness (esp. document_name + authority/temporal); resume filter selects only non-target chunks.
- **Integration:** vectors at same IDs change after reindex; metadata preserved; idempotent (2nd run re-embeds 0); `embedding_model` updated on Matter+Chunk; resume after partial; sparse preserved.
- **API:** drift math; dispatch correct task.
- **E2E:** ingest Cohere → flip Voyage → reindex → drift=0, sensible citations.

## Phases
1. **P0 (2–3d):** column + rev 14 + ORM; `embed_chunks_with_provider` + `_cache_key` override; `reindex_matter_task` (resume + SAC + sparse); Voyage tenacity; tests. *Fixes the bug for any chosen matter.*
2. **P1 (1–2d):** admin status + reindex endpoints; auth stub; API tests.
3. **P2 (1–2d):** `reindex_all_matters_task` + recency; progress UI surfacing drift/pending; E2E.

## Risks
- **SAC mismatch (HIGH):** raw vs summary-augmented re-embedding diverges — replicate `tasks.py:313-315`; test parity.
- **Sparse/hybrid regression (MED):** preserve sparse vectors on re-upsert.
- **Voyage no-retry (MED):** add tenacity.
- **Mixed-space window (MED, transient):** self-heals; low-traffic timing.
- **Unauth admin (HIGH security):** gate before prod.
- **Cache poisoning (MED):** explicit-provider `_cache_key`.
- **Migration numbering (LOW):** rev 14 not 13.
- **Backfill assumption (LOW):** confirm no Voyage vectors predate migration.
