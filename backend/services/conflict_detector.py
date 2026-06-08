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
import threading
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

# Recency decay: years since publication -> weight
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
_NLI_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════
# NLI MODEL (reuse from claim_verifier)
# ═══════════════════════════════════════════════════════════════

def _get_nli_model():
    """Lazy-load nli-deberta-v3-base. Shares the singleton from claim_verifier
    if already loaded, otherwise loads independently. Thread-safe via double-checked locking."""
    global _NLI_MODEL
    if _NLI_MODEL is not None:
        return _NLI_MODEL

    with _NLI_LOCK:
        if _NLI_MODEL is not None:  # double-check after acquiring lock
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
        except Exception as exc:
            logger.debug(f"Could not reuse claim_verifier NLI model: {exc}")

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
        return 0.50  # Unknown recency -> middle weight

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

    Uses the relevance_score as a proxy -- chunks that appear in many queries
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
        for cluster_chunks_list in cluster_map.values():
            best = max(cluster_chunks_list, key=lambda c: c.get("score", 0))
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

    import numpy as np

    def _softmax(x: "np.ndarray") -> "np.ndarray":
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

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
            raw_logits = model.predict([(a, b) for a, b in batch])
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
      3. Cluster chunks within each document -> select representatives
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
                    "explanation": "Source A (statute, 2024) outranks Source B ..."
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
        logger.debug("Single document in context -- skipping conflict detection")
        return None

    # 3. Cluster within each document and select representatives
    doc_representatives: Dict[str, List[Dict]] = {}
    for doc_id, doc_chunks in doc_groups.items():
        representatives = _cluster_chunks(doc_chunks, max_representatives=3)
        doc_representatives[doc_id] = representatives
        logger.debug(
            f"Document {doc_id}: {len(doc_chunks)} chunks -> "
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
        logger.warning("NLI model unavailable -- skipping conflict detection")
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
        action = "Conflicting legal positions detected -- review sources before relying on answer"
    else:
        action = "Potential inconsistencies detected -- consider reviewing flagged sources"

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

    This does NOT replace any context -- it adds a CONFLICT NOTICE section at the end
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
