# Conflict Detection Between Legal Sources

## Implementation Specification

**Status:** Proposed
**Date:** 2026-03-23
**Integration point:** `query_matter()` step 4.5 (after reranking) to step 5 (before context formatting)
**New file:** `backend/services/conflict_detector.py`
**Frontend component:** `frontend/components/ConflictAlert.tsx`

---

## 1. Problem Statement

When multiple legal documents are uploaded to a matter, retrieved chunks may contain contradictory legal positions. The current pipeline silently passes all chunks to Gemini without flagging conflicts. This risks the LLM producing an answer that cherry-picks one position without informing the user that an opposing position exists in their own documents.

**Example:** Document A (2024 statute) says "limitation period is 3 years." Document B (2019 case law) says "limitation period is 6 years." Current behavior: Gemini picks one. Desired behavior: surface both positions, rank by authority, explain the conflict.

---

## 2. Architecture Overview

```
retrieve_chunks() → filter → rerank_chunks()
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │  detect_conflicts()  │  ← NEW (step 4.6)
                         │                      │
                         │  1. Group by doc_id   │
                         │  2. Cluster similar   │
                         │  3. Cross-doc NLI     │
                         │  4. Score credibility │
                         │  5. Resolve by auth   │
                         └─────────┬─────────────┘
                                   │
                                   ▼
                         format_legal_context()  → generate_answer()
                                                        │
                                                        ▼
                                                  (response includes
                                                   conflict_analysis)
```

---

## 3. Algorithm: Google DRAGged Approach (Adapted)

### 3.1 Claim Extraction

Extract the main legal claim from each chunk. Uses a lightweight prompt to Gemini (`gemini-2.5-flash-lite`) with constrained output.

```python
CLAIM_EXTRACTION_PROMPT = """Extract the single main legal claim from this text.
Return ONLY the claim as one sentence. No explanation.
If the text contains no clear legal claim, return "NO_CLAIM".

Text: {chunk_content}"""
```

**Optimization:** Batch all chunks into a single Gemini call with numbered inputs to reduce API round-trips. Fallback: use the first sentence of each chunk as a heuristic claim (zero API cost).

### 3.2 Pairwise NLI Scoring (Cross-Source Only)

Only compare chunks from DIFFERENT documents. This reduces comparisons by ~85% for a typical matter with 2-3 documents.

**Pair reduction math:**
- 8 final chunks, 2 documents (4 chunks each)
- Naive O(N^2): C(8,2) = 28 pairs
- Cross-doc only: 4 x 4 = 16 pairs (43% reduction)
- With clustering (2 representatives per doc): 2 x 2 = 4 pairs (86% reduction)

### 3.3 Contradiction Detection

NLI model outputs 3-class probabilities: `[contradiction, entailment, neutral]`.

```
if contradiction_score > 0.5  → CONFLICT DETECTED
if contradiction_score > 0.35 → POTENTIAL CONFLICT (flagged for review)
```

### 3.4 Source Credibility Scoring

When a conflict is detected, rank the conflicting sources:

```python
credibility = (
    authority_type_weight * 0.4 +    # statute=1.0, case=0.95, commentary=0.75
    recency_weight * 0.3 +           # <1yr=1.0, <5yr=0.9, <10yr=0.7
    specificity_weight * 0.2 +       # dates/amounts=higher
    citation_count_weight * 0.1      # frequently cited=higher
)
```

### 3.5 Conflict Resolution

The system does NOT silently discard the lower-authority source. Instead:
1. Both chunks remain in context
2. The LLM prompt is augmented with conflict metadata
3. The frontend shows a conflict alert with side-by-side comparison

---

## 4. Optimization: O(N log N) Instead of O(N^2)

### 4.1 Cross-Document Filtering

```python
# Group chunks by document_id
doc_groups: Dict[str, List[Dict]] = defaultdict(list)
for chunk in final_chunks:
    doc_id = chunk.get("document_id", "unknown")
    doc_groups[doc_id].append(chunk)

# Only generate cross-document pairs
pairs = []
doc_ids = list(doc_groups.keys())
for i in range(len(doc_ids)):
    for j in range(i + 1, len(doc_ids)):
        for chunk_a in doc_groups[doc_ids[i]]:
            for chunk_b in doc_groups[doc_ids[j]]:
                pairs.append((chunk_a, chunk_b))
```

### 4.2 Semantic Clustering

Group semantically similar chunks within each document, then only compare cluster representatives.

```python
from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer

def cluster_chunks(chunks: List[Dict], n_clusters: int = None) -> List[List[Dict]]:
    """Cluster chunks by semantic similarity, return representative per cluster."""
    if len(chunks) <= 2:
        return [[c] for c in chunks]

    # Use existing Cohere embeddings from retrieval (already computed)
    # Fall back to lightweight local model if embeddings unavailable
    embeddings = []
    for chunk in chunks:
        if "embedding" in chunk:
            embeddings.append(chunk["embedding"])
        else:
            # Embeddings already computed during retrieval; re-embed if missing
            embeddings.append(embed_query_fn(chunk.get("content", "")[:256]))

    if n_clusters is None:
        n_clusters = max(1, len(chunks) // 3)  # ~3 chunks per cluster

    clustering = AgglomerativeClustering(
        n_clusters=min(n_clusters, len(chunks)),
        metric="cosine",
        linkage="average"
    )
    labels = clustering.fit_predict(embeddings)

    clusters = defaultdict(list)
    for idx, label in enumerate(labels):
        clusters[label].append(chunks[idx])

    return list(clusters.values())


def get_cluster_representative(cluster: List[Dict]) -> Dict:
    """Pick the highest-scoring chunk as cluster representative."""
    return max(cluster, key=lambda c: c.get("score", 0))
```

### 4.3 Batch NLI Inference

Process NLI pairs in batches of 32 to maximize throughput on CPU.

```python
NLI_BATCH_SIZE = 32

def batch_nli_predict(model, pairs: List[Tuple[str, str]]) -> List[Dict]:
    """Run NLI inference in batches, return per-pair scores."""
    all_scores = []
    for batch_start in range(0, len(pairs), NLI_BATCH_SIZE):
        batch = pairs[batch_start:batch_start + NLI_BATCH_SIZE]
        # NLI models return [contradiction, entailment, neutral] logits
        raw_scores = model.predict(
            [(a, b) for a, b in batch],
            apply_softmax=True
        )
        for scores in raw_scores:
            all_scores.append({
                "contradiction": float(scores[0]),
                "entailment": float(scores[1]),
                "neutral": float(scores[2]),
            })
    return all_scores
```

---

## 5. Backend Implementation

### 5.1 New File: `backend/services/conflict_detector.py`

```python
"""Cross-source conflict detection for legal RAG pipeline.

Detects contradictions between retrieved chunks from different documents
using NLI cross-encoder inference. Scores source credibility and resolves
conflicts based on legal authority hierarchy.

Integration: called from query_matter() after reranking, before context formatting.

Algorithm (adapted from Google DRAGged):
  1. Group chunks by document_id
  2. Cluster similar chunks within each document
  3. Pairwise NLI on cross-document representative pairs (O(N log N))
  4. Score source credibility for conflicting pairs
  5. Return structured conflict analysis for frontend display
"""
import logging
import re
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONTRADICTION_THRESHOLD = 0.50      # Hard conflict
POTENTIAL_CONFLICT_THRESHOLD = 0.35  # Soft conflict (flagged for review)
NLI_BATCH_SIZE = 32
MAX_CHUNK_PAIRS = 100               # Safety cap to avoid runaway computation
CLAIM_MAX_CHARS = 512               # NLI input truncation limit

# Authority hierarchy weights (higher = more authoritative)
AUTHORITY_WEIGHTS: Dict[str, float] = {
    "constitution": 1.00,
    "statute": 1.00,
    "regulation": 0.95,
    "case_law": 0.90,
    "supreme_court": 0.95,
    "high_court": 0.88,
    "district_court": 0.80,
    "tribunal": 0.75,
    "commentary": 0.70,
    "practice_note": 0.65,
    "opinion": 0.60,
    "unknown": 0.50,
}

# Recency decay: years since publication → weight
RECENCY_BRACKETS: List[Tuple[int, float]] = [
    (1, 1.0),    # < 1 year old
    (3, 0.95),   # 1-3 years
    (5, 0.90),   # 3-5 years
    (10, 0.70),  # 5-10 years
    (20, 0.50),  # 10-20 years
    (999, 0.30), # 20+ years
]

# Singleton NLI model (shared with claim_verifier.py)
_NLI_MODEL = None


# ═══════════════════════════════════════════════════════════════
# NLI MODEL (reuse from claim_verifier)
# ═══════════════════════════════════════════════════════════════

def _get_nli_model():
    """Lazy-load nli-deberta-v3-base. Shares the singleton from claim_verifier
    if already loaded, otherwise loads independently."""
    global _NLI_MODEL
    if _NLI_MODEL is not None:
        return _NLI_MODEL

    # Try to reuse the model instance from claim_verifier (same process)
    try:
        try:
            from backend.services.claim_verifier import _get_nli_base
        except ImportError:
            try:
                from services.claim_verifier import _get_nli_base
            except ImportError:
                from .claim_verifier import _get_nli_base
        model = _get_nli_base()
        if model is not None:
            _NLI_MODEL = model
            return _NLI_MODEL
    except Exception:
        pass

    # Fallback: load our own instance
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading NLI model for conflict detection (cross-encoder/nli-deberta-v3-base)")
        _NLI_MODEL = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        return _NLI_MODEL
    except Exception as e:
        logger.warning(f"Failed to load NLI model for conflict detection: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# SOURCE CREDIBILITY SCORING
# ═══════════════════════════════════════════════════════════════

def _classify_authority_type(chunk: Dict) -> str:
    """Infer the authority type of a chunk from its metadata and content.

    Uses section_name, document_name, and content heuristics. Returns one of
    the keys from AUTHORITY_WEIGHTS.
    """
    doc_name = (chunk.get("document_name") or "").lower()
    section = (chunk.get("section_name") or "").lower()
    content = (chunk.get("content") or "")[:300].lower()
    source_type = chunk.get("source_type", "document")

    # External case law from CourtListener
    if source_type == "case_law":
        if any(kw in content for kw in ["supreme court", "scotus"]):
            return "supreme_court"
        if any(kw in content for kw in ["high court", "court of appeals", "circuit"]):
            return "high_court"
        return "case_law"

    # Statute indicators
    if any(kw in doc_name for kw in ["act", "statute", "code", "law no", "ordinance"]):
        return "statute"
    if any(kw in section for kw in ["section", "article", "clause"]):
        if any(kw in doc_name for kw in ["act", "code"]):
            return "statute"

    # Regulation indicators
    if any(kw in doc_name for kw in ["regulation", "rule", "directive", "order"]):
        return "regulation"

    # Constitution
    if "constitution" in doc_name:
        return "constitution"

    # Commentary/opinion
    if any(kw in doc_name for kw in ["commentary", "guide", "handbook", "manual"]):
        return "commentary"
    if any(kw in doc_name for kw in ["opinion", "advisory", "memo"]):
        return "opinion"

    # Case law in uploaded documents
    if any(kw in content for kw in [" v. ", " vs. ", " versus ", "plaintiff", "defendant", "court held"]):
        return "case_law"

    return "unknown"


def _estimate_recency(chunk: Dict) -> float:
    """Estimate recency weight from chunk metadata.

    Looks for year patterns in document_name, section_name, and content.
    Returns a weight from RECENCY_BRACKETS.
    """
    text = f"{chunk.get('document_name', '')} {chunk.get('section_name', '')} {chunk.get('content', '')[:200]}"

    # Find 4-digit years
    years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
    if not years:
        return 0.50  # Unknown recency → middle weight

    # Use the most recent year found
    most_recent = max(int(y) for y in years)
    current_year = datetime.now().year
    age = current_year - most_recent

    for bracket_years, weight in RECENCY_BRACKETS:
        if age <= bracket_years:
            return weight

    return 0.30  # Very old


def _estimate_specificity(chunk: Dict) -> float:
    """Estimate specificity weight from chunk content.

    More specific chunks (dates, amounts, section refs) score higher.
    """
    content = chunk.get("content", "")
    score = 0.5  # Baseline

    # Dates
    if re.search(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', content):
        score += 0.15
    # Monetary amounts
    if re.search(r'[$\u00a3\u20ac\u20b9]\s*[\d,]+(?:\.\d{2})?', content):
        score += 0.15
    # Section/article references
    if re.search(r'\b(?:section|article|clause|rule)\s+\d+', content, re.IGNORECASE):
        score += 0.10
    # Specific case citations
    if re.search(r'\b\d+\s+[A-Z][a-z]+\.?\s+\d+', content):
        score += 0.10

    return min(1.0, score)


def _estimate_citation_count(chunk: Dict) -> float:
    """Estimate citation count weight.

    Uses the relevance_score as a proxy — chunks that appear in many queries
    tend to have higher retrieval scores. Future: use actual citation graph.
    """
    score = chunk.get("score", 0.5)
    # Normalize retrieval score to 0-1 range (already is, but clamp)
    return min(1.0, max(0.0, score))


def calculate_credibility(chunk: Dict) -> Dict:
    """Calculate composite credibility score for a chunk.

    Returns dict with overall score and factor breakdown.
    """
    authority_type = _classify_authority_type(chunk)
    authority_weight = AUTHORITY_WEIGHTS.get(authority_type, 0.50)
    recency_weight = _estimate_recency(chunk)
    specificity_weight = _estimate_specificity(chunk)
    citation_weight = _estimate_citation_count(chunk)

    credibility = (
        authority_weight * 0.4 +
        recency_weight * 0.3 +
        specificity_weight * 0.2 +
        citation_weight * 0.1
    )

    return {
        "score": round(credibility, 3),
        "authority_type": authority_type,
        "authority_weight": round(authority_weight, 2),
        "recency_weight": round(recency_weight, 2),
        "specificity_weight": round(specificity_weight, 2),
        "citation_weight": round(citation_weight, 2),
    }


# ═══════════════════════════════════════════════════════════════
# SEMANTIC CLUSTERING
# ═══════════════════════════════════════════════════════════════

def _cluster_chunks(chunks: List[Dict], max_representatives: int = 3) -> List[Dict]:
    """Cluster chunks by semantic similarity and return one representative per cluster.

    Uses agglomerative clustering on chunk content. Falls back to returning
    all chunks if clustering dependencies are unavailable.

    Args:
        chunks: List of chunk dicts from the same document.
        max_representatives: Maximum number of cluster representatives to return.

    Returns:
        List of representative chunk dicts (one per cluster, up to max_representatives).
    """
    if len(chunks) <= max_representatives:
        return chunks

    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics.pairwise import cosine_distances
    except ImportError:
        logger.debug("sklearn not available, skipping clustering optimization")
        # Fallback: return top-scoring chunks as representatives
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
        return sorted_chunks[:max_representatives]

    # Build embeddings from chunk content using lightweight local model
    try:
        try:
            from backend.services.embeddings import embed_query as _embed
        except ImportError:
            try:
                from services.embeddings import embed_query as _embed
            except ImportError:
                from .embeddings import embed_query as _embed

        embeddings = []
        for chunk in chunks:
            text = (chunk.get("content") or "")[:256]
            embeddings.append(_embed(text))

        X = np.array(embeddings)
        distance_matrix = cosine_distances(X)

        n_clusters = min(max_representatives, len(chunks))
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
        labels = clustering.fit_predict(distance_matrix)

        # Pick highest-scoring chunk from each cluster
        cluster_map: Dict[int, List[Dict]] = defaultdict(list)
        for idx, label in enumerate(labels):
            cluster_map[int(label)].append(chunks[idx])

        representatives = []
        for cluster_chunks in cluster_map.values():
            best = max(cluster_chunks, key=lambda c: c.get("score", 0))
            representatives.append(best)

        return representatives

    except Exception as e:
        logger.debug(f"Clustering failed, using score-based fallback: {e}")
        sorted_chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)
        return sorted_chunks[:max_representatives]


# ═══════════════════════════════════════════════════════════════
# BATCH NLI INFERENCE
# ═══════════════════════════════════════════════════════════════

def _batch_nli_predict(
    model,
    pairs: List[Tuple[str, str]],
) -> List[Dict[str, float]]:
    """Run NLI inference in batches.

    The nli-deberta-v3-base model returns logits for 3 classes in the order:
    [contradiction, entailment, neutral].

    Args:
        model: CrossEncoder model instance.
        pairs: List of (premise, hypothesis) string tuples.

    Returns:
        List of dicts with keys "contradiction", "entailment", "neutral",
        each containing a float probability (sums to ~1.0).
    """
    if not pairs:
        return []

    all_scores: List[Dict[str, float]] = []

    for batch_start in range(0, len(pairs), NLI_BATCH_SIZE):
        batch = pairs[batch_start:batch_start + NLI_BATCH_SIZE]
        # predict() returns numpy array of shape (batch_size, 3)
        # Class order for nli-deberta-v3-base: [contradiction, entailment, neutral]
        try:
            raw_scores = model.predict(
                [(a, b) for a, b in batch],
                apply_softmax=True,
            )
        except TypeError:
            # Older sentence-transformers versions may not support apply_softmax
            import numpy as np
            raw_logits = model.predict([(a, b) for a, b in batch])
            # Manual softmax
            def _softmax(x):
                e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return e_x / e_x.sum(axis=-1, keepdims=True)
            raw_scores = _softmax(np.array(raw_logits))

        for scores in raw_scores:
            all_scores.append({
                "contradiction": float(scores[0]),
                "entailment": float(scores[1]),
                "neutral": float(scores[2]),
            })

    return all_scores


# ═══════════════════════════════════════════════════════════════
# MAIN DETECTION PIPELINE
# ═══════════════════════════════════════════════════════════════

def detect_conflicts(
    chunks: List[Dict],
    query: str = "",
) -> Optional[Dict]:
    """Detect contradictions between chunks from different documents.

    This is the main entry point, called from query_matter() after reranking.

    Pipeline:
      1. Group chunks by document_id
      2. If only one document, return None (no cross-doc conflicts possible)
      3. Cluster chunks within each document → select representatives
      4. Generate cross-document pairs
      5. Run batch NLI on pairs
      6. For each contradiction, score source credibility
      7. Return structured conflict analysis

    Args:
        chunks: List of reranked chunk dicts. Each must have at minimum:
            - "content": str
            - "document_id": str
            - "document_name": str
            - "page_num": str
            - "section_name": str
            - "score": float
        query: The user's query string (used for context in conflict explanation).

    Returns:
        None if no conflicts detected, otherwise a dict:
        {
            "has_conflicts": True,
            "conflicts": [
                {
                    "id": "conflict_0",
                    "severity": "high" | "medium",
                    "contradiction_score": 0.82,
                    "chunk_a": {
                        "document_id": "...",
                        "document_name": "...",
                        "page_num": "...",
                        "section_name": "...",
                        "content_snippet": "...",    # first 300 chars
                        "credibility": { ... },
                    },
                    "chunk_b": { ... },
                    "recommended_source": "a" | "b",
                    "explanation": "Source A (statute, 2024) outranks Source B (case law, 2019) ..."
                }
            ],
            "summary": {
                "total_conflicts": 2,
                "high_severity": 1,
                "medium_severity": 1,
                "recommended_action": "Review conflicting sources before relying on answer"
            }
        }
    """
    if not chunks or len(chunks) < 2:
        return None

    # 1. Group by document_id
    doc_groups: Dict[str, List[Dict]] = defaultdict(list)
    for chunk in chunks:
        doc_id = chunk.get("document_id", "unknown")
        doc_groups[doc_id].append(chunk)

    # 2. Need at least 2 documents for cross-doc conflicts
    if len(doc_groups) < 2:
        logger.debug("Single document in context — skipping conflict detection")
        return None

    # 3. Cluster within each document and select representatives
    doc_representatives: Dict[str, List[Dict]] = {}
    for doc_id, doc_chunks in doc_groups.items():
        representatives = _cluster_chunks(doc_chunks, max_representatives=3)
        doc_representatives[doc_id] = representatives
        logger.debug(
            f"Document {doc_id}: {len(doc_chunks)} chunks → "
            f"{len(representatives)} representatives"
        )

    # 4. Generate cross-document pairs (only between different documents)
    nli_pairs: List[Tuple[str, str]] = []
    pair_metadata: List[Tuple[Dict, Dict]] = []  # Track which chunks each pair came from

    doc_ids = list(doc_representatives.keys())
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            for chunk_a in doc_representatives[doc_ids[i]]:
                for chunk_b in doc_representatives[doc_ids[j]]:
                    content_a = (chunk_a.get("content") or "")[:CLAIM_MAX_CHARS]
                    content_b = (chunk_b.get("content") or "")[:CLAIM_MAX_CHARS]
                    if content_a.strip() and content_b.strip():
                        nli_pairs.append((content_a, content_b))
                        pair_metadata.append((chunk_a, chunk_b))

    if not nli_pairs:
        return None

    # Safety cap
    if len(nli_pairs) > MAX_CHUNK_PAIRS:
        logger.warning(
            f"Too many NLI pairs ({len(nli_pairs)}), capping at {MAX_CHUNK_PAIRS}"
        )
        nli_pairs = nli_pairs[:MAX_CHUNK_PAIRS]
        pair_metadata = pair_metadata[:MAX_CHUNK_PAIRS]

    total_pairs = len(nli_pairs)
    logger.info(
        f"Conflict detection: {len(doc_groups)} documents, "
        f"{sum(len(r) for r in doc_representatives.values())} representatives, "
        f"{total_pairs} cross-doc pairs"
    )

    # 5. Run batch NLI
    nli_model = _get_nli_model()
    if nli_model is None:
        logger.warning("NLI model unavailable — skipping conflict detection")
        return None

    try:
        nli_results = _batch_nli_predict(nli_model, nli_pairs)
    except Exception as e:
        logger.warning(f"NLI inference failed: {e}")
        return None

    # 6. Identify conflicts and score credibility
    conflicts: List[Dict] = []
    for idx, (scores, (chunk_a, chunk_b)) in enumerate(zip(nli_results, pair_metadata)):
        contradiction_score = scores["contradiction"]

        if contradiction_score < POTENTIAL_CONFLICT_THRESHOLD:
            continue

        severity = "high" if contradiction_score >= CONTRADICTION_THRESHOLD else "medium"

        # Calculate credibility for both sources
        cred_a = calculate_credibility(chunk_a)
        cred_b = calculate_credibility(chunk_b)

        # Determine recommended source
        if cred_a["score"] > cred_b["score"]:
            recommended = "a"
        elif cred_b["score"] > cred_a["score"]:
            recommended = "b"
        else:
            recommended = "a"  # Tie-break: first source wins

        # Generate explanation
        explanation = _generate_conflict_explanation(
            chunk_a, chunk_b, cred_a, cred_b, recommended, contradiction_score
        )

        conflict = {
            "id": f"conflict_{len(conflicts)}",
            "severity": severity,
            "contradiction_score": round(contradiction_score, 3),
            "chunk_a": {
                "document_id": chunk_a.get("document_id", ""),
                "document_name": chunk_a.get("document_name", ""),
                "page_num": chunk_a.get("page_num", ""),
                "section_name": chunk_a.get("section_name", ""),
                "content_snippet": (chunk_a.get("content") or "")[:300],
                "credibility": cred_a,
            },
            "chunk_b": {
                "document_id": chunk_b.get("document_id", ""),
                "document_name": chunk_b.get("document_name", ""),
                "page_num": chunk_b.get("page_num", ""),
                "section_name": chunk_b.get("section_name", ""),
                "content_snippet": (chunk_b.get("content") or "")[:300],
                "credibility": cred_b,
            },
            "recommended_source": recommended,
            "explanation": explanation,
        }
        conflicts.append(conflict)

    if not conflicts:
        logger.debug("No conflicts detected between documents")
        return None

    # 7. Build summary
    high_count = sum(1 for c in conflicts if c["severity"] == "high")
    medium_count = sum(1 for c in conflicts if c["severity"] == "medium")

    if high_count > 0:
        action = "Conflicting legal positions detected — review sources before relying on answer"
    else:
        action = "Potential inconsistencies detected — consider reviewing flagged sources"

    result = {
        "has_conflicts": True,
        "conflicts": conflicts,
        "summary": {
            "total_conflicts": len(conflicts),
            "high_severity": high_count,
            "medium_severity": medium_count,
            "recommended_action": action,
        },
    }

    logger.info(
        f"Conflict detection complete: {len(conflicts)} conflicts "
        f"({high_count} high, {medium_count} medium)"
    )

    return result


def _generate_conflict_explanation(
    chunk_a: Dict,
    chunk_b: Dict,
    cred_a: Dict,
    cred_b: Dict,
    recommended: str,
    contradiction_score: float,
) -> str:
    """Generate a human-readable explanation for a detected conflict.

    Args:
        chunk_a: First conflicting chunk.
        chunk_b: Second conflicting chunk.
        cred_a: Credibility assessment for chunk_a.
        cred_b: Credibility assessment for chunk_b.
        recommended: "a" or "b" indicating recommended source.
        contradiction_score: NLI contradiction probability.

    Returns:
        A plain-English explanation string.
    """
    name_a = chunk_a.get("document_name", "Source A")
    name_b = chunk_b.get("document_name", "Source B")
    type_a = cred_a["authority_type"].replace("_", " ").title()
    type_b = cred_b["authority_type"].replace("_", " ").title()
    score_a = cred_a["score"]
    score_b = cred_b["score"]

    strength = "Strong" if contradiction_score >= 0.7 else "Moderate"

    winner_name = name_a if recommended == "a" else name_b
    winner_type = type_a if recommended == "a" else type_b
    loser_name = name_b if recommended == "a" else name_a
    loser_type = type_b if recommended == "a" else type_a

    parts = [
        f"{strength} contradiction detected (score: {contradiction_score:.0%}).",
    ]

    if score_a != score_b:
        parts.append(
            f'"{winner_name}" ({winner_type}, credibility {max(score_a, score_b):.0%}) '
            f'outranks "{loser_name}" ({loser_type}, credibility {min(score_a, score_b):.0%}).'
        )
    else:
        parts.append(
            f'Both sources have equal credibility ({score_a:.0%}). '
            f'Manual review recommended.'
        )

    # Add recency note if relevant
    if cred_a["recency_weight"] != cred_b["recency_weight"]:
        more_recent = "A" if cred_a["recency_weight"] > cred_b["recency_weight"] else "B"
        parts.append(f"Source {more_recent} is more recent.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# CONTEXT AUGMENTATION
# ═══════════════════════════════════════════════════════════════

def augment_context_with_conflicts(
    formatted_context: str,
    conflict_analysis: Dict,
) -> str:
    """Append conflict metadata to the LLM context so Gemini acknowledges conflicts.

    This does NOT replace any context — it adds a CONFLICT NOTICE section at the end
    so the LLM can reference both positions in its answer.

    Args:
        formatted_context: The existing formatted context string.
        conflict_analysis: Output from detect_conflicts().

    Returns:
        Augmented context string with conflict notices appended.
    """
    if not conflict_analysis or not conflict_analysis.get("conflicts"):
        return formatted_context

    notice_parts = [
        "\n" + "=" * 60,
        "CONFLICT NOTICE: The following sources contain contradictory positions.",
        "You MUST acknowledge these conflicts in your answer.",
        "Present BOTH positions and indicate which source is more authoritative.",
        "=" * 60 + "\n",
    ]

    for conflict in conflict_analysis["conflicts"]:
        chunk_a = conflict["chunk_a"]
        chunk_b = conflict["chunk_b"]
        notice_parts.append(
            f"CONFLICT ({conflict['severity'].upper()}): "
            f'"{chunk_a["document_name"]}" (p. {chunk_a["page_num"]}) '
            f'vs "{chunk_b["document_name"]}" (p. {chunk_b["page_num"]})'
        )
        notice_parts.append(f"  Contradiction score: {conflict['contradiction_score']:.0%}")
        notice_parts.append(f"  Recommended: {conflict['explanation']}")
        notice_parts.append("")

    return formatted_context + "\n".join(notice_parts)
```

### 5.2 Changes to `backend/schemas.py`

Add the following schemas after the `ClaimVerificationResponse` class (line 148):

```python
# ============================================
# CONFLICT DETECTION SCHEMAS
# ============================================

class ConflictChunkInfo(BaseModel):
    """One side of a detected conflict"""
    document_id: str
    document_name: str
    page_num: str
    section_name: Optional[str] = None
    content_snippet: str
    credibility: dict  # {score, authority_type, authority_weight, recency_weight, ...}


class ConflictItem(BaseModel):
    """Single detected conflict between two sources"""
    id: str
    severity: str  # "high" or "medium"
    contradiction_score: float = Field(ge=0.0, le=1.0)
    chunk_a: ConflictChunkInfo
    chunk_b: ConflictChunkInfo
    recommended_source: str  # "a" or "b"
    explanation: str


class ConflictSummary(BaseModel):
    """Summary of all conflicts detected"""
    total_conflicts: int = 0
    high_severity: int = 0
    medium_severity: int = 0
    recommended_action: str = ""


class ConflictAnalysisResponse(BaseModel):
    """Full conflict analysis response"""
    has_conflicts: bool = False
    conflicts: list[ConflictItem] = []
    summary: ConflictSummary = ConflictSummary()
```

### 5.3 Changes to `backend/config.py`

Add a new setting after the `claim_verification_enabled` line (line 51):

```python
    # Conflict Detection
    conflict_detection_enabled: bool = True
```

### 5.4 Changes to `backend/services/rag_engine.py`

#### 5.4.1 Add import (after line 44, inside the first `try` block)

```python
    from backend.services.conflict_detector import detect_conflicts, augment_context_with_conflicts
```

With the same fallback pattern for the second and third `try` blocks:

```python
    from services.conflict_detector import detect_conflicts, augment_context_with_conflicts
```

```python
    from .conflict_detector import detect_conflicts, augment_context_with_conflicts
```

#### 5.4.2 Add conflict detection step (new step 4.6)

Insert after the reranking merge on line 1112 (`final_chunks = reranked_docs + case_law_chunks`) and before the context formatting on line 1114 (`# 5. Format context with token budgeting`):

```python
        # 4.6. Cross-source conflict detection
        conflict_analysis = None
        if settings.conflict_detection_enabled:
            try:
                conflict_analysis = detect_conflicts(final_chunks, query=query)
                if conflict_analysis:
                    logger.info(
                        f"Conflict detection: {conflict_analysis['summary']['total_conflicts']} "
                        f"conflicts found ({conflict_analysis['summary']['high_severity']} high)"
                    )
            except Exception as e:
                logger.warning(f"Conflict detection failed (non-blocking): {e}")
```

#### 5.4.3 Augment context with conflict notices

Insert after the `formatted_context = format_legal_context(...)` call on line 1132, before the token count on line 1135:

```python
            # Augment context with conflict notices (if any)
            if conflict_analysis:
                formatted_context = augment_context_with_conflicts(
                    formatted_context, conflict_analysis
                )
```

#### 5.4.4 Add conflict_analysis to response dict

Add to the return dict on line 1309 (inside the successful response block), after `"claim_verification": claim_verification,` on line 1333:

```python
            "conflict_analysis": conflict_analysis,
```

The final return dict becomes:

```python
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,
            "matter_id": matter_id,
            "query": query,
            "model": GEMINI_MODEL,
            "tokens_used": tokens_used,
            "confidence": {
                "level": confidence,
                "score": answer_confidence_score,
                "factors": {
                    "has_hallucinations": has_hallucinations,
                    "unsupported_claims": len(unsupported_claims),
                    "grounded_citations": len(grounded_citations),
                    "avg_citation_relevance": (
                        sum(c.get("relevance_score", 0) for c in grounded_citations) / len(grounded_citations)
                        if grounded_citations else 0.0
                    )
                }
            },
            "confidence_explanation": confidence_explanation,
            "source_document": doc_summary,
            "citation_verification": citation_verification,
            "claim_verification": claim_verification,
            "conflict_analysis": conflict_analysis,
            "error": None
        }
```

---

## 6. Frontend Implementation

### 6.1 Changes to `frontend/lib/types.ts`

Add after the `ClaimVerificationSummary` interface (line 83):

```typescript
// ============================================
// Conflict Detection Types
// ============================================

export interface ConflictCredibility {
  score: number
  authority_type: string
  authority_weight: number
  recency_weight: number
  specificity_weight: number
  citation_weight: number
}

export interface ConflictChunkInfo {
  document_id: string
  document_name: string
  page_num: string
  section_name?: string
  content_snippet: string
  credibility: ConflictCredibility
}

export interface ConflictItem {
  id: string
  severity: "high" | "medium"
  contradiction_score: number
  chunk_a: ConflictChunkInfo
  chunk_b: ConflictChunkInfo
  recommended_source: "a" | "b"
  explanation: string
}

export interface ConflictSummary {
  total_conflicts: number
  high_severity: number
  medium_severity: number
  recommended_action: string
}

export interface ConflictAnalysis {
  has_conflicts: boolean
  conflicts: ConflictItem[]
  summary: ConflictSummary
}
```

Add `conflictAnalysis` to the `QueryMessage` interface (after `verifiedClaims` on line 102):

```typescript
  /** Conflict detection results */
  conflictAnalysis?: ConflictAnalysis
```

### 6.2 Changes to `frontend/lib/api-services.ts`

Add `conflict_analysis` to the `AskResponse` interface (after `claim_verification` on line 121):

```typescript
  conflict_analysis?: {
    has_conflicts: boolean
    conflicts: {
      id: string
      severity: "high" | "medium"
      contradiction_score: number
      chunk_a: {
        document_id: string
        document_name: string
        page_num: string
        section_name?: string
        content_snippet: string
        credibility: {
          score: number
          authority_type: string
          authority_weight: number
          recency_weight: number
          specificity_weight: number
          citation_weight: number
        }
      }
      chunk_b: {
        document_id: string
        document_name: string
        page_num: string
        section_name?: string
        content_snippet: string
        credibility: {
          score: number
          authority_type: string
          authority_weight: number
          recency_weight: number
          specificity_weight: number
          citation_weight: number
        }
      }
      recommended_source: "a" | "b"
      explanation: string
    }[]
    summary: {
      total_conflicts: number
      high_severity: number
      medium_severity: number
      recommended_action: string
    }
  } | null
```

### 6.3 New Component: `frontend/components/ConflictAlert.tsx`

```tsx
"use client"

import React, { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  Scale,
  FileText,
  Shield,
  Clock,
  Award,
} from "lucide-react"
import { cn } from "@/lib/utils"
import type { ConflictAnalysis, ConflictItem, ConflictCredibility } from "@/lib/types"

// ─── Props ───────────────────────────────────────────────────

interface ConflictAlertProps {
  analysis: ConflictAnalysis
  className?: string
}

// ─── Severity colors ─────────────────────────────────────────

const SEVERITY_STYLES = {
  high: {
    bg: "bg-red-50 border-red-200",
    badge: "bg-red-100 text-red-800 border-red-300",
    icon: "text-red-600",
    bar: "bg-red-500",
  },
  medium: {
    bg: "bg-amber-50 border-amber-200",
    badge: "bg-amber-100 text-amber-800 border-amber-300",
    icon: "text-amber-600",
    bar: "bg-amber-500",
  },
}

const AUTHORITY_LABELS: Record<string, string> = {
  constitution: "Constitution",
  statute: "Statute",
  regulation: "Regulation",
  case_law: "Case Law",
  supreme_court: "Supreme Court",
  high_court: "High Court",
  district_court: "District Court",
  tribunal: "Tribunal",
  commentary: "Commentary",
  practice_note: "Practice Note",
  opinion: "Opinion",
  unknown: "Unknown",
}

// ─── Credibility breakdown ───────────────────────────────────

function CredibilityBreakdown({ cred }: { cred: ConflictCredibility }) {
  return (
    <div className="mt-2 space-y-1 text-[11px]">
      <div className="flex items-center gap-1.5">
        <Shield className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Authority:</span>
        <span className="font-medium text-slate-700">
          {AUTHORITY_LABELS[cred.authority_type] || cred.authority_type}
        </span>
        <span className="text-slate-400">({(cred.authority_weight * 100).toFixed(0)}%)</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Clock className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Recency:</span>
        <span className="font-medium text-slate-700">{(cred.recency_weight * 100).toFixed(0)}%</span>
      </div>
      <div className="flex items-center gap-1.5">
        <Award className="h-3 w-3 text-slate-400" />
        <span className="text-slate-500">Overall:</span>
        <span className="font-semibold text-slate-800">{(cred.score * 100).toFixed(0)}%</span>
      </div>
    </div>
  )
}

// ─── Single conflict card ────────────────────────────────────

function ConflictCard({ conflict }: { conflict: ConflictItem }) {
  const [expanded, setExpanded] = useState(false)
  const styles = SEVERITY_STYLES[conflict.severity]
  const recommended = conflict.recommended_source === "a" ? conflict.chunk_a : conflict.chunk_b
  const other = conflict.recommended_source === "a" ? conflict.chunk_b : conflict.chunk_a

  return (
    <div className={cn("rounded-lg border p-3", styles.bg)}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-start justify-between text-left"
      >
        <div className="flex items-start gap-2">
          <Scale className={cn("mt-0.5 h-4 w-4 flex-shrink-0", styles.icon)} />
          <div>
            <div className="flex items-center gap-2">
              <span className={cn("inline-flex items-center rounded border px-1.5 py-0.5 text-[10px] font-semibold uppercase", styles.badge)}>
                {conflict.severity}
              </span>
              <span className="text-xs text-slate-600">
                {(conflict.contradiction_score * 100).toFixed(0)}% contradiction
              </span>
            </div>
            <p className="mt-1 text-xs text-slate-700 leading-relaxed">
              {conflict.explanation}
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronUp className="h-4 w-4 text-slate-400 flex-shrink-0" />
        ) : (
          <ChevronDown className="h-4 w-4 text-slate-400 flex-shrink-0" />
        )}
      </button>

      {/* Side-by-side comparison */}
      <AnimatePresence>
        {expanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 grid grid-cols-2 gap-3">
              {/* Recommended source */}
              <div className="rounded-md border border-emerald-200 bg-emerald-50/50 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <FileText className="h-3.5 w-3.5 text-emerald-600" />
                  <span className="text-[11px] font-semibold text-emerald-800 truncate">
                    {recommended.document_name}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-1">
                  p. {recommended.page_num}
                  {recommended.section_name ? `, ${recommended.section_name}` : ""}
                </p>
                <p className="text-[11px] text-slate-700 leading-relaxed line-clamp-4">
                  {recommended.content_snippet}
                </p>
                <CredibilityBreakdown cred={recommended.credibility} />
                <div className="mt-1.5">
                  <span className="inline-flex items-center rounded bg-emerald-100 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 border border-emerald-200">
                    Recommended
                  </span>
                </div>
              </div>

              {/* Other source */}
              <div className="rounded-md border border-slate-200 bg-white/50 p-2.5">
                <div className="flex items-center gap-1.5 mb-1.5">
                  <FileText className="h-3.5 w-3.5 text-slate-500" />
                  <span className="text-[11px] font-semibold text-slate-700 truncate">
                    {other.document_name}
                  </span>
                </div>
                <p className="text-[10px] text-slate-500 mb-1">
                  p. {other.page_num}
                  {other.section_name ? `, ${other.section_name}` : ""}
                </p>
                <p className="text-[11px] text-slate-700 leading-relaxed line-clamp-4">
                  {other.content_snippet}
                </p>
                <CredibilityBreakdown cred={other.credibility} />
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Main component ──────────────────────────────────────────

export default function ConflictAlert({ analysis, className }: ConflictAlertProps) {
  const [collapsed, setCollapsed] = useState(false)
  const { summary, conflicts } = analysis

  if (!conflicts.length) return null

  return (
    <div className={cn("rounded-xl border border-amber-200 bg-amber-50/30 p-3", className)}>
      {/* Summary header */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between text-left"
      >
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-4 w-4 text-amber-600" />
          <span className="text-sm font-semibold text-amber-900">
            {summary.total_conflicts} Source Conflict{summary.total_conflicts !== 1 ? "s" : ""} Detected
          </span>
          {summary.high_severity > 0 && (
            <span className="inline-flex items-center rounded border border-red-300 bg-red-100 px-1.5 py-0.5 text-[10px] font-semibold text-red-800">
              {summary.high_severity} HIGH
            </span>
          )}
        </div>
        {collapsed ? (
          <ChevronDown className="h-4 w-4 text-amber-500" />
        ) : (
          <ChevronUp className="h-4 w-4 text-amber-500" />
        )}
      </button>

      {/* Action text */}
      <p className="mt-1 text-xs text-amber-700 pl-6">
        {summary.recommended_action}
      </p>

      {/* Conflict cards */}
      <AnimatePresence>
        {!collapsed && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="mt-3 space-y-2">
              {conflicts.map((conflict) => (
                <ConflictCard key={conflict.id} conflict={conflict} />
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
```

### 6.4 Changes to `frontend/components/ChatPanel.tsx`

Add the import at the top (after the existing component imports around line 22):

```typescript
import ConflictAlert from "@/components/ConflictAlert"
```

Add the `ConflictAnalysis` type to the import from `@/lib/types` (line 20):

```typescript
import type { QueryMessage, Citation, CitationVerification, ConflictAnalysis } from "@/lib/types"
```

In the message rendering section, add the `ConflictAlert` component above the answer text for assistant messages. The exact insertion point depends on the message rendering JSX, but it should be placed after the `VerificationBar` and before the answer content:

```tsx
{msg.conflictAnalysis?.has_conflicts && (
  <ConflictAlert
    analysis={msg.conflictAnalysis}
    className="mb-3"
  />
)}
```

### 6.5 Wire `conflict_analysis` into `QueryMessage`

In the `ChatPanel.tsx` handler that processes `AskResponse` into `QueryMessage` objects, map the field:

```typescript
conflictAnalysis: response.conflict_analysis
  ? {
      has_conflicts: response.conflict_analysis.has_conflicts,
      conflicts: response.conflict_analysis.conflicts.map((c) => ({
        ...c,
        chunk_a: {
          ...c.chunk_a,
          credibility: c.chunk_a.credibility,
        },
        chunk_b: {
          ...c.chunk_b,
          credibility: c.chunk_b.credibility,
        },
      })),
      summary: response.conflict_analysis.summary,
    }
  : undefined,
```

---

## 7. Performance Budget

| Operation | Cost | Notes |
|-----------|------|-------|
| Clustering (per doc) | ~5ms | sklearn AgglomerativeClustering, <10 chunks |
| NLI batch (4-16 pairs) | ~50-200ms | CPU, nli-deberta-v3-base, batch of 32 |
| Credibility scoring | ~1ms per chunk | Regex + string matching, no model |
| Context augmentation | ~0ms | String concatenation |
| **Total overhead** | **~60-210ms** | Acceptable within existing 2-5s query latency |

### Worst-case analysis

- 8 chunks from 4 documents (2 each)
- 4 clusters (1 representative each)
- Cross-doc pairs: C(4,2) = 6 pairs
- Single NLI batch: ~50ms
- Total: ~60ms

---

## 8. Testing Plan

### 8.1 Unit Tests: `backend/tests/test_conflict_detector.py`

```python
"""Unit tests for conflict detection service."""
import pytest
from unittest.mock import patch, MagicMock

from backend.services.conflict_detector import (
    detect_conflicts,
    calculate_credibility,
    _classify_authority_type,
    _estimate_recency,
    _estimate_specificity,
    _cluster_chunks,
    _batch_nli_predict,
    augment_context_with_conflicts,
)


# ─── Fixtures ─────────────────────────────────────────────────

def _make_chunk(doc_id="doc_1", doc_name="Test Act 2024", content="Test content",
                page_num="1", section_name="Section 1", score=0.85, source_type="document"):
    return {
        "document_id": doc_id,
        "document_name": doc_name,
        "content": content,
        "page_num": page_num,
        "section_name": section_name,
        "score": score,
        "source_type": source_type,
        "chunk_id": f"chunk_{doc_id}_{page_num}",
    }


# ─── Authority classification ─────────────────────────────────

class TestClassifyAuthority:
    def test_statute_from_name(self):
        chunk = _make_chunk(doc_name="Companies Act 2006")
        assert _classify_authority_type(chunk) == "statute"

    def test_case_law_from_content(self):
        chunk = _make_chunk(doc_name="judgment.pdf", content="The court held that the defendant v. plaintiff...")
        assert _classify_authority_type(chunk) == "case_law"

    def test_case_law_source_type(self):
        chunk = _make_chunk(source_type="case_law")
        assert _classify_authority_type(chunk) in ("case_law", "supreme_court", "high_court")

    def test_commentary(self):
        chunk = _make_chunk(doc_name="Halsbury's Commentary on Torts")
        assert _classify_authority_type(chunk) == "commentary"

    def test_unknown_fallback(self):
        chunk = _make_chunk(doc_name="notes.pdf", content="Some general text")
        assert _classify_authority_type(chunk) == "unknown"


# ─── Recency estimation ───────────────────────────────────────

class TestEstimateRecency:
    def test_recent_year(self):
        chunk = _make_chunk(content="The 2025 amendment provides...")
        weight = _estimate_recency(chunk)
        assert weight >= 0.90

    def test_old_year(self):
        chunk = _make_chunk(content="Under the 1990 Act...")
        weight = _estimate_recency(chunk)
        assert weight <= 0.50

    def test_no_year(self):
        chunk = _make_chunk(content="No date here")
        weight = _estimate_recency(chunk)
        assert weight == 0.50


# ─── Specificity estimation ───────────────────────────────────

class TestEstimateSpecificity:
    def test_monetary_amounts(self):
        chunk = _make_chunk(content="Damages of $500,000 were awarded")
        weight = _estimate_specificity(chunk)
        assert weight > 0.5

    def test_section_references(self):
        chunk = _make_chunk(content="Pursuant to Section 42 of the Act")
        weight = _estimate_specificity(chunk)
        assert weight > 0.5

    def test_no_specifics(self):
        chunk = _make_chunk(content="General legal principles apply")
        weight = _estimate_specificity(chunk)
        assert weight == 0.5


# ─── Credibility scoring ──────────────────────────────────────

class TestCredibility:
    def test_statute_scores_higher_than_commentary(self):
        statute = _make_chunk(doc_name="Finance Act 2024", content="The 2024 provisions state...")
        commentary = _make_chunk(doc_name="Tax Commentary", content="Scholars suggest...")
        cred_statute = calculate_credibility(statute)
        cred_commentary = calculate_credibility(commentary)
        assert cred_statute["score"] > cred_commentary["score"]

    def test_credibility_has_all_fields(self):
        chunk = _make_chunk()
        cred = calculate_credibility(chunk)
        assert "score" in cred
        assert "authority_type" in cred
        assert "authority_weight" in cred
        assert "recency_weight" in cred
        assert "specificity_weight" in cred
        assert "citation_weight" in cred


# ─── Clustering ────────────────────────────────────────────────

class TestClustering:
    def test_fewer_chunks_than_max(self):
        chunks = [_make_chunk(page_num=str(i)) for i in range(2)]
        result = _cluster_chunks(chunks, max_representatives=3)
        assert len(result) == 2

    @patch("backend.services.conflict_detector.embed_query_fn")
    def test_clustering_reduces_count(self, mock_embed):
        import numpy as np
        mock_embed.side_effect = lambda x: np.random.rand(1024).tolist()
        chunks = [_make_chunk(page_num=str(i), content=f"Content {i}") for i in range(6)]
        result = _cluster_chunks(chunks, max_representatives=2)
        assert len(result) <= 2


# ─── Conflict detection (integration) ─────────────────────────

class TestDetectConflicts:
    def test_single_document_returns_none(self):
        chunks = [_make_chunk(doc_id="doc_1", page_num=str(i)) for i in range(4)]
        result = detect_conflicts(chunks)
        assert result is None

    def test_empty_chunks_returns_none(self):
        assert detect_conflicts([]) is None
        assert detect_conflicts(None) is None

    @patch("backend.services.conflict_detector._get_nli_model")
    @patch("backend.services.conflict_detector._cluster_chunks")
    def test_detects_contradiction(self, mock_cluster, mock_nli):
        # Mock NLI model to return high contradiction score
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            [0.85, 0.05, 0.10],  # contradiction, entailment, neutral
        ]
        mock_nli.return_value = mock_model

        # Mock clustering to pass through
        mock_cluster.side_effect = lambda chunks, **kw: chunks

        chunk_a = _make_chunk(doc_id="doc_1", doc_name="Act 2024",
                              content="The limitation period is 3 years.")
        chunk_b = _make_chunk(doc_id="doc_2", doc_name="Case 2019",
                              content="The limitation period is 6 years.")

        result = detect_conflicts([chunk_a, chunk_b])
        assert result is not None
        assert result["has_conflicts"] is True
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["severity"] == "high"
        assert result["conflicts"][0]["contradiction_score"] == 0.85

    @patch("backend.services.conflict_detector._get_nli_model")
    @patch("backend.services.conflict_detector._cluster_chunks")
    def test_no_conflict_when_entailment(self, mock_cluster, mock_nli):
        mock_model = MagicMock()
        mock_model.predict.return_value = [
            [0.05, 0.85, 0.10],  # low contradiction
        ]
        mock_nli.return_value = mock_model
        mock_cluster.side_effect = lambda chunks, **kw: chunks

        chunk_a = _make_chunk(doc_id="doc_1", content="The period is 3 years.")
        chunk_b = _make_chunk(doc_id="doc_2", content="The period is 3 years.")

        result = detect_conflicts([chunk_a, chunk_b])
        assert result is None

    @patch("backend.services.conflict_detector._get_nli_model")
    def test_nli_unavailable_returns_none(self, mock_nli):
        mock_nli.return_value = None
        chunks = [
            _make_chunk(doc_id="doc_1"),
            _make_chunk(doc_id="doc_2"),
        ]
        result = detect_conflicts(chunks)
        assert result is None


# ─── Context augmentation ─────────────────────────────────────

class TestAugmentContext:
    def test_no_analysis_returns_unchanged(self):
        ctx = "Original context"
        assert augment_context_with_conflicts(ctx, None) == ctx
        assert augment_context_with_conflicts(ctx, {}) == ctx

    def test_adds_conflict_notice(self):
        ctx = "Original context"
        analysis = {
            "conflicts": [{
                "severity": "high",
                "contradiction_score": 0.82,
                "chunk_a": {"document_name": "Act", "page_num": "5"},
                "chunk_b": {"document_name": "Case", "page_num": "12"},
                "explanation": "Act outranks Case",
            }],
        }
        result = augment_context_with_conflicts(ctx, analysis)
        assert "CONFLICT NOTICE" in result
        assert "Act" in result
        assert "Case" in result
```

### 8.2 E2E Test Scenario

Add to the existing `backend/tests/test_all_phases_e2e.py`:

```python
class TestConflictDetectionE2E:
    """E2E: upload 2 contradictory documents, ask question, verify conflict surfaced."""

    async def test_conflict_detected_in_response(self, client, matter_with_two_docs):
        """Query a matter with contradictory docs and verify conflict_analysis is populated."""
        response = await client.post(
            f"/matters/{matter_with_two_docs}/ask",
            json={"question": "What is the limitation period?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("conflict_analysis") is not None
        assert data["conflict_analysis"]["has_conflicts"] is True
        assert len(data["conflict_analysis"]["conflicts"]) > 0
```

---

## 9. Configuration & Feature Flag

The feature is gated by `conflict_detection_enabled` in `backend/config.py`. Set to `True` by default but can be disabled via environment variable:

```bash
CONFLICT_DETECTION_ENABLED=false
```

When disabled, `query_matter()` skips step 4.6 entirely, adding zero overhead.

---

## 10. Error Handling

All conflict detection is non-blocking. Failures are caught and logged as warnings without affecting the main RAG pipeline:

```python
try:
    conflict_analysis = detect_conflicts(final_chunks, query=query)
except Exception as e:
    logger.warning(f"Conflict detection failed (non-blocking): {e}")
    conflict_analysis = None
```

This follows the same pattern used by `citation_verification` (step 7.6) and `claim_verification` (step 7.65) in the existing `query_matter()` function.

---

## 11. File Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/services/conflict_detector.py` | **CREATE** | Core detection service (NLI, credibility, clustering) |
| `backend/services/rag_engine.py` | **MODIFY** | Add step 4.6, import, context augmentation, response field |
| `backend/schemas.py` | **MODIFY** | Add `ConflictChunkInfo`, `ConflictItem`, `ConflictSummary`, `ConflictAnalysisResponse` |
| `backend/config.py` | **MODIFY** | Add `conflict_detection_enabled: bool = True` |
| `frontend/components/ConflictAlert.tsx` | **CREATE** | Inline alert + side-by-side comparison UI |
| `frontend/lib/types.ts` | **MODIFY** | Add `ConflictCredibility`, `ConflictChunkInfo`, `ConflictItem`, `ConflictSummary`, `ConflictAnalysis` types; add `conflictAnalysis` to `QueryMessage` |
| `frontend/lib/api-services.ts` | **MODIFY** | Add `conflict_analysis` to `AskResponse` |
| `frontend/components/ChatPanel.tsx` | **MODIFY** | Import `ConflictAlert`, render it in assistant messages |
| `backend/tests/test_conflict_detector.py` | **CREATE** | Unit + integration tests |

---

## 12. Dependencies

No new dependencies required. All models and libraries are already in the project:

- `sentence-transformers` (CrossEncoder, already used by claim_verifier.py and rag_engine.py reranker)
- `cross-encoder/nli-deberta-v3-base` (already loaded by claim_verifier.py)
- `sklearn` (AgglomerativeClustering, already a transitive dependency of sentence-transformers)
- `numpy` (already a transitive dependency)

---

## 13. Migration Path

1. Create `backend/services/conflict_detector.py` with the complete implementation
2. Add schemas to `backend/schemas.py`
3. Add config flag to `backend/config.py`
4. Modify `backend/services/rag_engine.py` (3 insertion points)
5. Add frontend types to `frontend/lib/types.ts`
6. Add response type to `frontend/lib/api-services.ts`
7. Create `frontend/components/ConflictAlert.tsx`
8. Wire `ConflictAlert` into `frontend/components/ChatPanel.tsx`
9. Add tests in `backend/tests/test_conflict_detector.py`
10. Test with 2+ document matter to verify end-to-end flow
