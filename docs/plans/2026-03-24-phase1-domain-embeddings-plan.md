---
title: "Phase 1: Domain-Specific Legal Embeddings — Switch to Voyage-law-2"
type: feat
status: planned
date: 2026-03-24
priority: P0
estimated_effort: 2–3 weeks
---

# Phase 1: Domain-Specific Legal Embeddings

## 1. Problem Statement

Lexintel currently uses Cohere `embed-english-v3.0` (1024-dim), a general-purpose embedding
model. This is the single largest retrieval accuracy bottleneck in the system.

### Why General Embeddings Hurt Legal Retrieval

**Legal language is structurally different from general English.** Terms like "consideration,"
"remedy," "action," "party," "interest," and "holding" have domain-specific meanings that
diverge sharply from everyday usage. A general-purpose encoder maps "the court held that..."
and "she held the cup" into neighboring regions of vector space. Legal synonyms like
"estoppel" and "preclusion," or "tortfeasor" and "wrongdoer," land far apart because they
rarely co-occur in general training corpora.

**Citation-dense text confuses general models.** Legal documents are riddled with strings
like "42 U.S.C. 1983" and "523 U.S. 83, 89 (1998)" that general models treat as noise
rather than semantically meaningful anchors.

**Long, nested reasoning structures are common.** A single sentence in a judicial opinion
can span 80+ words with multiple subordinate clauses. General models trained primarily on
web text lose coherence over these spans.

**Measurable impact from research:**

| Source | Finding |
|--------|---------|
| Voyage AI blog (Apr 2024) | voyage-law-2 outperforms Cohere embed-english-v3.0 by ~20% average NDCG on 8 legal retrieval benchmarks |
| Voyage AI blog (Apr 2024) | voyage-law-2 outperforms OpenAI text-embedding-3-large by 6% average, >10% on LeCaRDv2, LegalQuAD, GerDaLIR |
| MLEB benchmark (Oct 2025) | On the 10-dataset Massive Legal Embedding Benchmark, legal-adapted models (Kanon 2 Embedder: 86.03, Voyage 3 Large: 85.71) significantly outperform general-purpose models |
| Harvey AI + Voyage (Jul 2024) | Fine-tuned voyage-law-2 on 20B+ case law tokens reduces irrelevant top results by 25% vs best general-purpose models |
| Weaviate migration guide (2025) | For high-risk domains like legal, even 5-8% retrieval improvement justifies model migration |

**Bottom line:** Switching from Cohere embed-english-v3.0 to a legal-domain model is
conservatively a 6-15% NDCG improvement — the single highest-ROI change available to
Lexintel's retrieval pipeline.

### Current Architecture (What Changes)

```
Current:
  chunk text --> Cohere embed-english-v3.0 --> 1024-dim vector --> Qdrant

After:
  chunk text --> Voyage voyage-law-2 --> 1024-dim vector --> Qdrant
                                         (same dimensions, same Qdrant config)
```

The vector dimension stays at 1024, so Qdrant collection configuration, HNSW parameters,
and all downstream code (reranker, RAG engine, hybrid search) remain untouched.

---

## 2. Options Analysis

### Option A: Switch to Voyage voyage-law-2 (Recommended)

| Dimension | Assessment |
|-----------|------------|
| **Accuracy gain** | +6-15% NDCG on legal retrieval (benchmarked on 8 legal datasets). Best on 7/8 legal benchmarks. MLEB rank: top 3. |
| **Output dimensions** | 1024 (same as current Cohere) — no Qdrant schema change needed |
| **Context length** | 16K tokens (vs Cohere's 512 tokens) — handles long legal passages without truncation |
| **Cost** | $0.12/1M tokens (first 50M free per account). Cohere embed-v3 is $0.10/1M. Marginal increase. |
| **Batch API** | Up to 128 texts per call, 120K total tokens per batch. Our current batch size of 96 fits. |
| **SDK** | `voyageai` PyPI package. Drop-in replacement: `vo.embed(texts, model="voyage-law-2", input_type="document")` |
| **Asymmetric embeddings** | Supports `input_type="document"` and `input_type="query"` (matches our current Cohere asymmetric pattern) |
| **Migration effort** | ~3-5 days of code changes. Main change is `embeddings.py`. Config adds one env var. |
| **Migration risk** | Existing vectors must be re-embedded (different model = incompatible vector space). Handled by background re-indexing task. |
| **Vendor stability** | Voyage AI acquired by MongoDB (Feb 2025, $220M). API remains available. voyage-law-2 also on AWS/Azure Marketplace. Voyage 4 family (Jan 2026) has shared embedding space — future upgrade path is smooth. |

**Verdict: Best accuracy-to-effort ratio. Same dimensions. Proven legal domain performance.**

### Option B: Fine-tune Cohere on Legal Corpus

| Dimension | Assessment |
|-----------|------------|
| **Accuracy gain** | Potentially +10-30% for fine-tuned domain models (per BAAI BGE research). Harvey achieved 25% reduction in irrelevant results by fine-tuning voyage-law-2 on 20B tokens. |
| **Cost** | High. Requires curated legal training pairs (question-passage pairs with hard negatives). Harvey had a dedicated legal research team annotating data. |
| **Effort** | 4-8 weeks minimum. Need training data curation, GPU compute, evaluation infrastructure. Cohere fine-tuning API exists but is limited. |
| **Migration** | Same re-embedding problem as Option A, plus ongoing model management. |
| **Risk** | Overfitting to training distribution. Requires continuous evaluation. Legal corpus licensing concerns. |

**Verdict: Highest ceiling but 5-10x the effort. Do this in Phase 3 after establishing Voyage baseline.**

### Option C: Open-Source Legal Model (nomic-embed-text, LEGAL-BERT, BGE-M3 fine-tuned)

| Dimension | Assessment |
|-----------|------------|
| **Accuracy gain** | nomic-embed-text-v2-moe: competitive on general MTEB but no legal-specific benchmarks published. LEGAL-BERT: 768-dim, older architecture, lower ceiling. BGE-M3 fine-tuned: COLIEE 2025 F1=0.23, promising but requires fine-tuning effort. |
| **Cost** | $0 API cost (self-hosted). But requires GPU infrastructure for inference (A10G minimum for production latency). |
| **Effort** | 2-4 weeks for model serving infrastructure (ONNX/TorchServe/vLLM), plus fine-tuning time if needed. |
| **Migration** | Different dimensions likely (768 for LEGAL-BERT, 1024 for nomic-v2). May need Qdrant collection recreation. |
| **Risk** | Inference latency and throughput become our problem. No SLA. Model updates require manual deployment. |

**Verdict: Attractive for cost control at scale, but premature. Lexintel's current volume does not justify self-hosted infrastructure overhead.**

### Option D: Hybrid — Keep Cohere for General, Add Legal Model for Reranking

| Dimension | Assessment |
|-----------|------------|
| **Accuracy gain** | Reranking already uses cross-encoder/ms-marco-MiniLM-L-6-v2. Adding a legal reranker on top of weak initial retrieval has diminishing returns — the bottleneck is recall in the first stage, not precision in reranking. |
| **Cost** | Double the embedding API calls (Cohere + Voyage). |
| **Effort** | 1-2 weeks, but architectural complexity increases (two embedding pipelines, dual caching). |
| **Migration** | Partial — only reranking changes, existing vectors stay. But recall problem persists. |
| **Risk** | Complexity without addressing root cause. "Polishing a wrong answer." |

**Verdict: Does not solve the fundamental retrieval quality problem. The bottleneck is first-stage recall, not reranking precision.**

### Decision Matrix

| Criteria (weight) | A: Voyage-law-2 | B: Fine-tune | C: Open-source | D: Hybrid |
|-------------------|-----------------|--------------|-----------------|-----------|
| Accuracy gain (40%) | 9 | 10 | 6 | 5 |
| Implementation effort (25%) | 9 | 3 | 5 | 7 |
| Cost efficiency (15%) | 8 | 4 | 10 | 5 |
| Migration risk (10%) | 7 | 5 | 6 | 9 |
| Future-proofing (10%) | 9 | 7 | 8 | 4 |
| **Weighted score** | **8.7** | **5.8** | **6.5** | **5.9** |

---

## 3. Recommended Approach

**Option A: Switch to Voyage voyage-law-2.**

Justification:

1. **Proven legal accuracy**: Best-in-class on 7/8 legal retrieval benchmarks. Top 3 on MLEB
   (the most comprehensive legal embedding benchmark, 10 datasets, 6 jurisdictions).

2. **Drop-in replacement**: Same 1024 dimensions as our current Cohere setup. Same asymmetric
   embedding pattern (`document` / `query` input types). Qdrant HNSW config stays identical.

3. **Minimal code change surface**: Only `embeddings.py`, `config.py`, and `requirements.txt`
   change. All downstream consumers (vector_store, rag_engine, hybrid_search, tasks) are
   untouched.

4. **Clear upgrade path**: Voyage 4 family (Jan 2026) introduces shared embedding spaces and
   MoE architecture. When ready, we can upgrade to voyage-4-large with minimal friction.
   Future Option B (fine-tuning) can start from voyage-law-2 as base — exactly what Harvey did.

5. **16K context length**: Our current chunks average 500-2000 tokens, so this is not an
   immediate win, but it eliminates any silent truncation on long legal sections that Cohere's
   512-token limit may cause.

---

## 4. Implementation Steps

### 4.1 Configuration Changes

**File: `backend/config.py`**

Add Voyage API key setting alongside existing Cohere key (keep Cohere for backward
compatibility during transition):

```python
# Voyage AI — used for legal-domain embeddings
voyage_api_key: str = ""

# Embedding provider toggle: "voyage" or "cohere"
embedding_provider: str = "voyage"

# Embedding model name (configurable for future upgrades to voyage-4-large etc.)
embedding_model: str = "voyage-law-2"
```

The `cohere_api_key` field stays required during the transition period. After all matters
are re-embedded, it can be made optional.

**File: `.env`**

```
VOYAGE_API_KEY=pa-...
EMBEDDING_PROVIDER=voyage
EMBEDDING_MODEL=voyage-law-2
```

### 4.2 Embedding Service Refactor

**File: `backend/services/embeddings.py`**

Refactor to a provider-agnostic interface. The public API (`embed_text`, `embed_query`,
`embed_chunks`) stays identical — callers are unaffected.

```python
# New constants
EMBEDDING_DIMENSIONS = 1024  # Same for both Cohere v3 and Voyage-law-2

# Provider selection at module level
_provider = settings.embedding_provider  # "voyage" or "cohere"
_model = settings.embedding_model        # "voyage-law-2" or "embed-english-v3.0"


@lru_cache(maxsize=1)
def get_voyage_client():
    """Get Voyage AI client for legal embeddings."""
    if not settings.voyage_api_key:
        raise ValueError("VOYAGE_API_KEY environment variable not set")
    import voyageai
    return voyageai.Client(api_key=settings.voyage_api_key)


def embed_text(text: str) -> List[float]:
    """Embed a single document text. Provider-agnostic."""
    # ... cache check (unchanged) ...
    if _provider == "voyage":
        return _voyage_embed_document([text])[0]
    else:
        return _cohere_embed_document([text])[0]


def embed_query(text: str) -> List[float]:
    """Embed a search query. Provider-agnostic."""
    # ... cache check (unchanged) ...
    if _provider == "voyage":
        return _voyage_embed_query(text)
    else:
        return _cohere_embed_query(text)


def _voyage_embed_document(texts: List[str]) -> List[List[float]]:
    """Embed document texts via Voyage AI."""
    client = get_voyage_client()
    response = client.embed(
        texts,
        model=_model,
        input_type="document",
    )
    return response.embeddings


def _voyage_embed_query(text: str) -> List[float]:
    """Embed a query via Voyage AI."""
    client = get_voyage_client()
    response = client.embed(
        [text],
        model=_model,
        input_type="query",
    )
    return response.embeddings[0]
```

Key design decisions:
- **Same public API**: `embed_text()`, `embed_query()`, `embed_chunks()` signatures unchanged.
- **Provider toggle**: `EMBEDDING_PROVIDER` env var selects at startup. No runtime switching.
- **Same cache**: SHA-256 key includes provider prefix (`voyage:` or `cohere:`) to avoid
  cross-contamination.
- **Same retry logic**: Tenacity retry wraps Voyage calls identically to Cohere.
- **Batch size**: Voyage allows 128 texts / 120K tokens per call. Keep our 96-text batch
  size (already within limits).

### 4.3 Dependencies

**File: `backend/requirements.txt`**

Add:
```
voyageai>=0.3.0
```

The `cohere>=5.0.0` dependency stays until Cohere is fully removed after migration.

### 4.4 Embedding Cache Update

**File: `backend/services/embedding_cache.py`**

No structural changes. The cache key already uses SHA-256 of text content. We prefix cache
keys with the provider name to prevent stale Cohere embeddings from being served when the
provider is switched to Voyage:

```python
# In embed_text():
cache_key = f"{_provider}:" + hashlib.sha256(text.encode()).hexdigest()

# In embed_query():
cache_key = f"query:{_provider}:" + hashlib.sha256(text.encode()).hexdigest()
```

### 4.5 Vector Store — No Changes Needed

**File: `backend/services/vector_store.py`**

No changes. `VECTOR_SIZE = 1024` stays the same. The `create_collection()` function already
handles dimension mismatch detection and recreation (lines 156-164), which would only trigger
if we changed dimensions in the future.

### 4.6 Re-Embedding Migration Task

**New file: `backend/services/reindex.py`**

A Celery task that re-embeds all existing matters with the new model. This is the critical
migration piece.

```python
@shared_task(bind=True, max_retries=3)
def reindex_matter_task(self, matter_id: str):
    """
    Re-embed all chunks for a matter using the current embedding provider.

    Strategy:
    1. Load all chunks from PostgreSQL for this matter.
    2. Re-embed in batches using the new provider.
    3. Upsert new vectors to Qdrant (overwrites old vectors via deterministic IDs).
    4. Mark matter as re-indexed in metadata.

    The deterministic point ID generation (_generate_point_id) means new vectors
    overwrite old ones in place — no orphaned vectors.
    """
    ...


@shared_task
def reindex_all_matters_task():
    """
    Queue re-indexing for all existing matters.
    Processes matters sequentially to avoid API rate limits.
    """
    ...
```

Design decisions:
- **Deterministic IDs**: Our `_generate_point_id(chunk_id, matter_id)` function produces
  the same Qdrant point ID regardless of embedding model. Re-upserting overwrites in place.
  No orphan cleanup needed.
- **Batch processing**: 96 chunks per Voyage API call (within 128-text limit). Matters
  processed one at a time to stay within rate limits.
- **Idempotent**: Re-running the task on a matter that is already re-indexed is safe
  (just overwrites with same vectors).
- **Progress tracking**: Uses existing `publish_*` SSE events so the frontend can show
  re-indexing status.
- **Metadata flag**: Add `embedding_model` field to Chunk or Matter metadata in PostgreSQL
  so we can track which model was used and identify any chunks that failed re-embedding.

### 4.7 Database Migration

**New file: `backend/alembic/versions/13_add_embedding_model_tracking.py`**

Add a column to track which embedding model was used for each matter's vectors:

```python
# Matter table
op.add_column('matters', sa.Column('embedding_model', sa.String(64), nullable=True))

# Optional: per-chunk tracking for partial migration recovery
op.add_column('chunks', sa.Column('embedding_model', sa.String(64), nullable=True))
```

This enables:
- Querying which matters still need re-embedding.
- Detecting mixed-model states (some chunks old, some new).
- Audit trail for model changes.

### 4.8 Health Check / Admin Endpoint

**File: `backend/main.py` (or new admin router)**

Add an endpoint to check migration status:

```
GET /admin/embedding-status

Response:
{
  "current_provider": "voyage",
  "current_model": "voyage-law-2",
  "total_matters": 42,
  "reindexed_matters": 38,
  "pending_matters": 4,
  "embedding_dimensions": 1024
}
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**File: `backend/tests/test_embeddings_voyage.py`**

```python
def test_voyage_embed_text_returns_1024_dim():
    """Verify Voyage embeddings have correct dimensions."""

def test_voyage_embed_query_returns_1024_dim():
    """Verify query embeddings have correct dimensions."""

def test_voyage_embed_chunks_batch():
    """Verify batch embedding works within Voyage API limits."""

def test_provider_toggle_selects_correct_client():
    """Verify EMBEDDING_PROVIDER env var routes to correct provider."""

def test_cache_key_includes_provider_prefix():
    """Verify cache keys are provider-specific to prevent cross-contamination."""

def test_cohere_fallback_still_works():
    """Verify Cohere path still works when EMBEDDING_PROVIDER=cohere."""
```

### 5.2 Retrieval Accuracy Benchmark (Before/After)

This is the most important test. Create a benchmark suite that measures retrieval quality
before and after the model switch.

**File: `backend/tests/benchmark_embeddings.py`**

**Methodology:**

1. **Select 3-5 existing matters** with known good queries and expected relevant chunks.

2. **Create a ground truth set**: For each query, manually identify the top 5 most relevant
   chunks (human-annotated). Store as JSON fixtures.

   ```json
   {
     "query": "What is the limitation period for breach of contract in this jurisdiction?",
     "matter_id": "...",
     "relevant_chunk_ids": ["chunk-uuid-1", "chunk-uuid-2", "chunk-uuid-3"],
     "relevant_document_names": ["Limitation Act 1980.pdf"]
   }
   ```

3. **Metrics**: For each query, measure:
   - **Recall@5**: What fraction of ground-truth chunks appear in top 5 results?
   - **Recall@10**: What fraction appear in top 10?
   - **NDCG@10**: Normalized Discounted Cumulative Gain (position-weighted).
   - **MRR**: Mean Reciprocal Rank of first relevant result.

4. **Run with Cohere** (current model) — record baseline metrics.

5. **Run with Voyage** (new model) — record new metrics.

6. **Compare**: We expect +6-15% improvement on NDCG@10 based on published benchmarks.

**Benchmark fixture format:**

```
backend/tests/fixtures/
  embedding_benchmark/
    queries.json          # 20-30 test queries across different legal topics
    ground_truth.json     # Human-annotated relevant chunks per query
```

### 5.3 Integration Tests

**File: `backend/tests/test_reindex_task.py`**

```python
def test_reindex_matter_updates_vectors():
    """Verify re-indexing produces new vectors in Qdrant."""

def test_reindex_preserves_metadata():
    """Verify chunk metadata (page_num, section, document_name) survives re-indexing."""

def test_reindex_is_idempotent():
    """Verify re-running reindex on same matter is safe."""

def test_reindex_updates_embedding_model_column():
    """Verify Matter.embedding_model is updated after successful re-index."""
```

### 5.4 End-to-End Test

**File: `backend/tests/test_e2e_voyage.py`**

Upload a legal PDF, process it with Voyage embeddings, query it, and verify the response
includes relevant citations. Compare answer quality subjectively against Cohere baseline.

---

## 6. Rollout Plan

### Phase 6.1: Parallel Deployment (Day 1-2)

1. Deploy code changes with `EMBEDDING_PROVIDER=cohere` (no behavior change).
2. Verify all existing tests pass.
3. Add `VOYAGE_API_KEY` to environment but keep Cohere as active provider.
4. Run Voyage embedding tests against the API to verify connectivity and dimensions.

### Phase 6.2: New Matters Use Voyage (Day 3-5)

1. Flip `EMBEDDING_PROVIDER=voyage` in production.
2. All NEW document uploads are embedded with Voyage.
3. Existing matters still have Cohere vectors — queries against them still work
   (query embedding via Voyage will be compared against Cohere vectors, which is
   suboptimal but not broken — cosine similarity still returns results, just
   with lower quality).
4. Monitor error rates, latency, and API costs.

**Important**: This is the one phase where mixed-model vectors exist. It is temporary
and acceptable because:
- Existing matters still return results (just slightly degraded).
- New matters get full Voyage quality immediately.
- The re-indexing task (next phase) resolves all mixed states.

### Phase 6.3: Background Re-Indexing (Day 5-10)

1. Trigger `reindex_all_matters_task()` via admin endpoint or Celery CLI.
2. The task processes matters one at a time:
   - Loads chunks from PostgreSQL.
   - Re-embeds via Voyage API.
   - Upserts new vectors to Qdrant (overwrites old Cohere vectors in place).
   - Updates `Matter.embedding_model = "voyage-law-2"`.
3. Monitor progress via `/admin/embedding-status`.
4. **No downtime**: Matters remain queryable throughout. Each matter has a brief
   window (~seconds) during upsert where some vectors are new and some are old,
   but Qdrant handles concurrent reads/writes safely.

**Re-indexing cost estimate:**
- Assume 10,000 chunks across all matters (generous estimate for early-stage app).
- At ~500 tokens/chunk average: 5M tokens total.
- Voyage pricing: $0.12/1M tokens = $0.60 total.
- First 50M tokens free per account = effectively $0.

### Phase 6.4: Validation & Cleanup (Day 10-14)

1. Run benchmark suite (Section 5.2) on re-indexed matters.
2. Verify NDCG@10 improvement matches expectations (6-15%).
3. Run full E2E test suite.
4. If metrics are satisfactory:
   - Remove Cohere client code from `embeddings.py` (or keep behind feature flag).
   - Make `cohere_api_key` optional in `config.py`.
   - Remove `cohere` from `requirements.txt` (optional — low urgency).
5. Update MEMORY.md to reflect new embedding provider.

### Rollback Plan

If Voyage causes issues at any phase:

1. **Phase 6.2 rollback**: Set `EMBEDDING_PROVIDER=cohere`. Immediate. No re-indexing needed
   for existing matters (they still have Cohere vectors).
2. **Phase 6.3 rollback**: Stop the re-indexing task. Matters that were already re-indexed
   will have Voyage vectors; switch to `EMBEDDING_PROVIDER=cohere` and re-run the reindex
   task with Cohere to restore. Deterministic point IDs mean re-upserting overwrites cleanly.
3. **Nuclear rollback**: Delete Qdrant collections and re-process all documents from scratch
   using Cohere. This is the existing `process_document_task` flow — no new code needed.

---

## 7. Risks & Mitigations

### R1: Voyage API Availability / Rate Limits

**Risk**: Voyage API may have different rate limits or availability than Cohere.

**Mitigation**:
- Voyage allows 128 texts per batch, 120K tokens per call. Our 96-text batches fit.
- Tenacity retry decorator (3 attempts, exponential backoff) applied to Voyage calls
  identically to Cohere.
- Voyage API is now backed by MongoDB infrastructure (post-acquisition). SLA is comparable
  to major cloud providers.
- Fallback: if Voyage is down, temporarily flip `EMBEDDING_PROVIDER=cohere` (existing
  Cohere vectors still in Qdrant for matters that were not yet re-indexed).

### R2: Mixed-Model Vectors During Migration

**Risk**: During Phase 6.2-6.3, some matters have Cohere vectors while queries use Voyage
embeddings. Cross-model cosine similarity is degraded.

**Mitigation**:
- This is a temporary state (5-10 days).
- Cosine similarity between models from the same family (1024-dim, trained on similar
  objectives) still produces reasonable results — just not optimal.
- The re-indexing task prioritizes high-usage matters first.
- If quality degradation is unacceptable, stay on Phase 6.1 (Cohere active) until re-indexing
  completes, then do an atomic flip.

### R3: Re-Indexing Failures

**Risk**: Some matters fail to re-index (API errors, corrupt chunks, etc.).

**Mitigation**:
- `embedding_model` column on Matter tracks state. Any matter without `voyage-law-2` can
  be retried.
- Celery task has `max_retries=3` with exponential backoff.
- Admin endpoint shows which matters are pending.
- Individual chunk failures are logged but do not abort the entire matter — partial
  re-indexing is better than none.

### R4: Voyage Pricing Changes

**Risk**: MongoDB/Voyage changes pricing post-acquisition.

**Mitigation**:
- Current cost is $0.12/1M tokens — marginal vs Cohere's $0.10/1M.
- At Lexintel's current scale (<1M tokens/month for embeddings), cost is negligible.
- The provider-agnostic architecture makes switching back to Cohere or to open-source
  a config change, not a code change.

### R5: Context Length Mismatch

**Risk**: Voyage's 16K context could behave differently than Cohere's 512 on short chunks.

**Mitigation**:
- Our chunks are 500-2000 tokens — well within both models' limits.
- Voyage handles short text fine; the 16K limit is an upper bound, not a minimum.
- No truncation behavior changes for our typical chunk sizes.

### R6: Embedding Cache Pollution

**Risk**: Cached Cohere embeddings served when provider switches to Voyage.

**Mitigation**:
- Cache keys include provider prefix (`voyage:sha256...` vs `cohere:sha256...`).
- On provider switch, old cache entries are never matched — they naturally age out via LRU.
- The in-memory cache is process-local and clears on restart anyway.

---

## 8. Timeline

### Week 1: Core Implementation

| Day | Task | Owner |
|-----|------|-------|
| Mon | Add `voyageai` dependency. Create `VOYAGE_API_KEY` config. Write provider toggle in `embeddings.py`. | Backend |
| Tue | Implement Voyage client functions (`_voyage_embed_document`, `_voyage_embed_query`). Update cache key prefixes. | Backend |
| Wed | Write unit tests for Voyage provider. Run existing test suite with `EMBEDDING_PROVIDER=cohere` to verify no regressions. | Backend |
| Thu | Create Alembic migration for `embedding_model` column. Implement `reindex_matter_task` and `reindex_all_matters_task`. | Backend |
| Fri | Write re-indexing integration tests. Create admin `/embedding-status` endpoint. | Backend |

### Week 2: Benchmarking & Rollout

| Day | Task | Owner |
|-----|------|-------|
| Mon | Create benchmark fixtures: select 3-5 matters, annotate 20-30 ground-truth query-chunk pairs. | Legal + Backend |
| Tue | Run baseline benchmark with Cohere. Record Recall@5, Recall@10, NDCG@10, MRR. | Backend |
| Wed | Deploy with `EMBEDDING_PROVIDER=voyage` for new matters (Phase 6.2). Run benchmark on new matter. | Backend + DevOps |
| Thu | Start background re-indexing of existing matters (Phase 6.3). Monitor progress. | Backend |
| Fri | Continue re-indexing. Spot-check re-indexed matters with benchmark queries. | Backend |

### Week 3: Validation & Cleanup

| Day | Task | Owner |
|-----|------|-------|
| Mon | Complete re-indexing. Run full benchmark suite on re-indexed matters. Compare before/after. | Backend |
| Tue | Run full E2E test suite. Fix any edge cases. | Backend |
| Wed | Make `cohere_api_key` optional. Update MEMORY.md and project documentation. | Backend |
| Thu | Code review. Merge to main. | Team |
| Fri | Buffer / overflow. | — |

### Success Criteria

- [ ] All matters re-indexed with voyage-law-2 (`/admin/embedding-status` shows 0 pending)
- [ ] NDCG@10 improvement of >= 5% on benchmark queries (conservative target)
- [ ] No increase in P95 query latency (embedding call time is similar)
- [ ] All existing tests pass with `EMBEDDING_PROVIDER=voyage`
- [ ] Rollback tested: verified switching back to Cohere works

---

## Appendix A: File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `backend/config.py` | Modify | Add `voyage_api_key`, `embedding_provider`, `embedding_model` settings |
| `backend/services/embeddings.py` | Modify | Add Voyage client, provider toggle, update cache keys |
| `backend/requirements.txt` | Modify | Add `voyageai>=0.3.0` |
| `backend/services/reindex.py` | New | Re-indexing Celery tasks |
| `backend/alembic/versions/13_add_embedding_model_tracking.py` | New | Add `embedding_model` column |
| `backend/models.py` | Modify | Add `embedding_model` field to Matter and/or Chunk |
| `backend/main.py` | Modify | Add `/admin/embedding-status` endpoint |
| `backend/tests/test_embeddings_voyage.py` | New | Unit tests for Voyage provider |
| `backend/tests/benchmark_embeddings.py` | New | Before/after accuracy benchmark |
| `backend/tests/test_reindex_task.py` | New | Re-indexing integration tests |
| `.env` | Modify | Add `VOYAGE_API_KEY`, `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL` |

**Files NOT changed** (by design):
- `backend/services/vector_store.py` — VECTOR_SIZE stays 1024
- `backend/services/rag_engine.py` — calls `embed_query()` which is provider-agnostic
- `backend/services/hybrid_search.py` — uses same embedding interface
- `backend/tasks.py` — calls `embed_chunks()` which is provider-agnostic
- `frontend/*` — no frontend changes needed

## Appendix B: Future Roadmap (Post Phase 1)

| Phase | What | When |
|-------|------|------|
| Phase 2 | Upgrade to voyage-4-large (MoE, shared embedding space, 40% cheaper inference) | Q2 2026 |
| Phase 3 | Fine-tune voyage-law-2 on Lexintel's own query-chunk pairs (a la Harvey approach) | Q3 2026 |
| Phase 4 | Evaluate Kanon 2 Embedder (MLEB #1, NDCG 86.03) as potential replacement | Q3 2026 |
| Phase 5 | Self-hosted open-source model (BGE-M3 fine-tuned) for cost optimization at scale | Q4 2026 |

## Appendix C: Research Sources

- [Voyage AI: Domain-Specific Embeddings — Legal Edition (voyage-law-2)](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/)
- [Harvey Partners with Voyage to Build Custom Legal Embeddings](https://www.harvey.ai/blog/harvey-partners-with-voyage-to-build-custom-legal-embeddings)
- [Massive Legal Embedding Benchmark (MLEB) — Isaacus](https://isaacus.com/mleb)
- [MLEB Paper — arXiv:2510.19365](https://arxiv.org/abs/2510.19365)
- [MongoDB Acquires Voyage AI ($220M, Feb 2025)](https://investors.mongodb.com/news-releases/news-release-details/mongodb-announces-acquisition-voyage-ai-enable-organizations)
- [Voyage 4 Model Family: Shared Embedding Space with MoE](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [voyage-3-large: SOTA General-Purpose Embedding Model](https://blog.voyageai.com/2025/01/07/voyage-3-large/)
- [Embedding Models Comparison 2026 (Cohere vs Voyage vs OpenAI vs BGE)](https://reintech.io/blog/embedding-models-comparison-2026-openai-cohere-voyage-bge)
- [Weaviate: When Good Models Go Bad — Migration Guide](https://weaviate.io/blog/when-good-models-go-bad)
- [Voyage AI Python SDK](https://pypi.org/project/voyageai/)
- [BAAI BGE Fine-Tuning for Legal Corpus](https://www.kdjingpai.com/en/embedding-weidiaoa/)
- [Embedding Fine-Tuning: Principles and Legal Applications](https://aws.amazon.com/blogs/machine-learning/fine-tune-a-bge-embedding-model-using-synthetic-data-from-amazon-bedrock/)
