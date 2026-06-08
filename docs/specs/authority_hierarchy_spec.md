# Legal Authority Hierarchy Scoring — Implementation Specification

**Status:** Draft
**Date:** 2026-03-23
**Scope:** Backend ingestion pipeline, RAG query engine, Qdrant payload, frontend badges

---

## 1. Overview

Legal documents carry different weights depending on the court that issued them,
the jurisdiction they belong to, and whether they constitute binding authority.
A US Supreme Court opinion outweighs a district court ruling; a statute from the
queried jurisdiction outweighs a sister-state regulation.

This specification adds **authority hierarchy scoring** to every chunk in the
Lexintel RAG pipeline so that:

1. Authority metadata is extracted at **ingestion time** (one Gemini call per
   document, not per chunk).
2. Authority fields are stored in **PostgreSQL** (on the Chunk model) and
   **Qdrant** (as payload fields with indexes).
3. At **query time**, retrieved chunks are reranked using a combined
   semantic + authority score.
4. The **frontend** displays authority badges on citations.

---

## 2. Scoring Formula

### 2.1 Authority Score (per chunk)

```
authority_score = (court_level_weight * 0.5)
               + (jurisdiction_weight * 0.3)
               + (binding_bonus * 0.2)
```

### 2.2 Court Level Weights

| Court Level    | Weight |
|----------------|--------|
| `supreme`      | 1.00   |
| `appellate`    | 0.85   |
| `trial`        | 0.70   |
| `administrative` | 0.55 |
| `unknown`      | 0.50   |

### 2.3 Jurisdiction Weights (resolved at query time)

| Relationship       | Weight |
|--------------------|--------|
| `exact_match`      | 1.00   |
| `federal`          | 0.85   |
| `sister_state`     | 0.50   |
| `foreign`          | 0.30   |
| `unknown`          | 0.40   |

The jurisdiction relationship is determined by comparing the chunk's
`jurisdiction_code` against the query's target jurisdiction (either
auto-detected from the query text or explicitly selected by the user).

### 2.4 Binding Bonus

| Binding Authority | Value |
|-------------------|-------|
| `true`            | 1.00  |
| `false`           | 0.00  |
| `unknown`         | 0.30  |

### 2.5 Combined Reranking Score (query time)

```
final_score = (semantic_score * 0.6) + (authority_score * 0.4)
```

This replaces the current combined score formula in `rerank_chunks()` which
uses `(original_score * 0.4) + (rerank_score * 0.6)`. The new three-way
formula becomes:

```
final_score = (cross_encoder_score * 0.45) + (vector_score * 0.15) + (authority_score * 0.4)
```

This preserves the existing cross-encoder > vector-similarity ordering while
introducing authority as a major signal.

---

## 3. Database Changes

### 3.1 Chunk Model (`backend/models.py`)

Add a new JSON column to the `Chunk` model:

```python
# In class Chunk(Base):
# After the existing `concepts` column (line 80)

authority_metadata = Column(JSON, nullable=True, default=dict)
# Stores: {
#     "court_level": "supreme"|"appellate"|"trial"|"administrative"|"unknown",
#     "court_name": "Supreme Court of the United States",
#     "jurisdiction_code": "US.federal"|"US.state.CA"|"UK"|"IN"|...|"unknown",
#     "authority_score": 0.95,           # pre-computed float 0.0-1.0
#     "binding_authority": true|false,    # null treated as unknown
#     "source_type": "case_law"|"statute"|"regulation"|"contract"|"commentary"|"other",
#     "confidence": 0.9                   # Gemini's self-reported confidence
# }
```

**Exact diff for `backend/models.py`:**

```python
# --- a/backend/models.py
# +++ b/backend/models.py
# @@ line 80 (after concepts column)

     concepts = Column(JSON, nullable=True, default=list)  # YAKE-extracted keywords
+    authority_metadata = Column(JSON, nullable=True, default=dict)  # Court/jurisdiction hierarchy
     chunk_sequence = Column(Integer, nullable=True)  # Order within document
```

### 3.2 Document Model — no changes

The `Document` model already has `document_type` and `jurisdiction` columns.
The authority detector will additionally populate finer-grained fields
(`court_level`, `court_name`, `jurisdiction_code`, `binding_authority`) which
are stored per-chunk via `authority_metadata` JSON rather than adding
individual columns (avoids schema fragmentation for metadata that is always
read/written together).

---

## 4. Alembic Migration

**File:** `backend/alembic/versions/12_add_authority_metadata_to_chunks.py`

```python
"""Add authority_metadata JSON column to chunks table

Revision ID: 12
Revises: 11
Create Date: 2026-03-23
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "12"
down_revision: Union[str, None] = "11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("authority_metadata", sa.JSON, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chunks", "authority_metadata")
```

---

## 5. New Service: Authority Detector

**File:** `backend/services/authority_detector.py`

This service wraps a single Gemini call per document that classifies
court level, jurisdiction, and binding authority. Results are propagated
to every chunk from that document.

```python
"""Authority hierarchy detection for legal documents via Gemini.

Called once per document at ingestion time (alongside summary generation
and document classification). Returns structured authority metadata that
is stored on every chunk from that document.
"""
import json
import logging
from typing import Dict, Optional

import google.generativeai as genai

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)

# Court level weight lookup (used to pre-compute authority_score)
COURT_LEVEL_WEIGHTS: Dict[str, float] = {
    "supreme": 1.00,
    "appellate": 0.85,
    "trial": 0.70,
    "administrative": 0.55,
    "unknown": 0.50,
}

# Source type defaults — non-court documents get baseline scores
SOURCE_TYPE_DEFAULTS: Dict[str, Dict] = {
    "statute": {
        "court_level": "supreme",       # Statutes carry legislative authority
        "binding_authority": True,
        "default_score": 0.90,
    },
    "regulation": {
        "court_level": "administrative",
        "binding_authority": True,
        "default_score": 0.70,
    },
    "contract": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.40,
    },
    "commentary": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.30,
    },
    "other": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.35,
    },
}

AUTHORITY_EXTRACTION_PROMPT = """\
Analyze this legal document excerpt and classify its authority.
Respond with ONLY valid JSON (no markdown, no explanation).

{
  "source_type": "<statute|case_law|regulation|contract|commentary|other>",
  "court_level": "<supreme|appellate|trial|administrative|unknown>",
  "court_name": "<exact court name or 'unknown'>",
  "jurisdiction_code": "<ISO-style code: US.federal, US.state.CA, US.state.NY, UK, UK.england, IN, AU, EU, SG, CA.federal, CA.province.ON, etc. Use 'unknown' if uncertain>",
  "binding_authority": <true|false>,
  "confidence": <0.0 to 1.0>
}

Rules:
- For statutes/regulations: court_level reflects the legislative body level.
- For case law: court_level is the issuing court's tier.
- For contracts/commentary: court_level is "unknown", binding_authority is false.
- jurisdiction_code format: COUNTRY[.subdivision[.specific]]
  Examples: "US.federal", "US.state.CA", "UK.england", "IN", "AU.state.NSW"
- confidence reflects how certain you are about the classification.

Document excerpt:
"""


def _compute_authority_score(
    court_level: str,
    binding_authority: Optional[bool],
) -> float:
    """Compute the static portion of the authority score.

    The jurisdiction component (0.3 weight) is resolved at query time
    because it depends on the user's target jurisdiction.  At ingestion
    we compute:

        partial_score = (court_level_weight * 0.5) + (binding_bonus * 0.2)

    and store it.  At query time the full score is:

        authority_score = partial_score + (jurisdiction_weight * 0.3)

    However, for Qdrant payload filtering and rough pre-ranking we store
    a *default* authority_score that assumes ``jurisdiction_weight = 0.5``
    (sister-state baseline).  The query-time reranker overrides this.
    """
    cl_weight = COURT_LEVEL_WEIGHTS.get(court_level, 0.50)

    if binding_authority is True:
        binding_val = 1.0
    elif binding_authority is False:
        binding_val = 0.0
    else:
        binding_val = 0.30  # unknown

    # Default jurisdiction_weight for storage (overridden at query time)
    default_jurisdiction = 0.50

    score = (cl_weight * 0.5) + (default_jurisdiction * 0.3) + (binding_val * 0.2)
    return round(score, 4)


async def detect_authority(extracted_text: str) -> Dict:
    """Classify a document's legal authority using Gemini.

    Args:
        extracted_text: Full extracted document text (will be truncated
            to first 15 000 chars to stay within fast-response budget).

    Returns:
        Dict with keys: source_type, court_level, court_name,
        jurisdiction_code, binding_authority, authority_score, confidence.
        Falls back to safe defaults on any failure.
    """
    default_result = {
        "source_type": "other",
        "court_level": "unknown",
        "court_name": "unknown",
        "jurisdiction_code": "unknown",
        "binding_authority": None,
        "authority_score": _compute_authority_score("unknown", None),
        "confidence": 0.0,
    }

    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("No Google API key — skipping authority detection")
        return default_result

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)

    # Use first 15k chars — enough to identify court, jurisdiction, parties
    truncated = extracted_text[:15_000]
    prompt = AUTHORITY_EXTRACTION_PROMPT + truncated

    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=200,
                response_mime_type="application/json",
            ),
        )
        raw = response.text.strip()

        # Parse JSON (handle markdown code fences if model wraps output)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(raw)

        # Normalize values
        source_type = data.get("source_type", "other").lower()
        court_level = data.get("court_level", "unknown").lower()
        court_name = data.get("court_name", "unknown")
        jurisdiction_code = data.get("jurisdiction_code", "unknown")
        confidence = float(data.get("confidence", 0.5))

        # Normalize binding_authority to bool or None
        ba_raw = data.get("binding_authority")
        if isinstance(ba_raw, bool):
            binding_authority = ba_raw
        elif isinstance(ba_raw, str):
            binding_authority = ba_raw.lower() == "true" if ba_raw.lower() in ("true", "false") else None
        else:
            binding_authority = None

        # Normalize court_level aliases
        court_level_map = {
            "circuit": "appellate",
            "appeals": "appellate",
            "high_court": "appellate",
            "district": "trial",
            "magistrate": "trial",
        }
        court_level = court_level_map.get(court_level, court_level)

        # For non-case-law types, apply source-type defaults if Gemini
        # returned "unknown" for court_level
        if source_type != "case_law" and court_level == "unknown":
            defaults = SOURCE_TYPE_DEFAULTS.get(source_type, SOURCE_TYPE_DEFAULTS["other"])
            court_level = defaults["court_level"]
            if binding_authority is None:
                binding_authority = defaults["binding_authority"]

        authority_score = _compute_authority_score(court_level, binding_authority)

        result = {
            "source_type": source_type,
            "court_level": court_level,
            "court_name": court_name,
            "jurisdiction_code": jurisdiction_code,
            "binding_authority": binding_authority,
            "authority_score": authority_score,
            "confidence": round(confidence, 2),
        }

        logger.info(
            f"Authority detected: type={source_type}, court={court_level}, "
            f"jurisdiction={jurisdiction_code}, score={authority_score}, "
            f"confidence={confidence:.2f}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Authority detection JSON parse failed: {e}")
        return default_result
    except Exception as e:
        logger.warning(f"Authority detection failed (graceful degradation): {e}")
        return default_result


def compute_query_time_authority_score(
    chunk_authority: Dict,
    target_jurisdiction: Optional[str] = None,
) -> float:
    """Recompute authority score with query-specific jurisdiction weight.

    Called during reranking to replace the default jurisdiction_weight=0.5
    with an accurate weight based on the relationship between the chunk's
    jurisdiction and the user's target jurisdiction.

    Args:
        chunk_authority: The authority_metadata dict from the chunk.
        target_jurisdiction: The jurisdiction the user is asking about
            (e.g. "US.state.CA").  If None, uses the default 0.5 weight.

    Returns:
        Float authority score in [0.0, 1.0].
    """
    court_level = chunk_authority.get("court_level", "unknown")
    binding_authority = chunk_authority.get("binding_authority")
    chunk_jurisdiction = chunk_authority.get("jurisdiction_code", "unknown")

    cl_weight = COURT_LEVEL_WEIGHTS.get(court_level, 0.50)

    # Binding bonus
    if binding_authority is True:
        binding_val = 1.0
    elif binding_authority is False:
        binding_val = 0.0
    else:
        binding_val = 0.30

    # Jurisdiction weight
    jw = _resolve_jurisdiction_weight(chunk_jurisdiction, target_jurisdiction)

    score = (cl_weight * 0.5) + (jw * 0.3) + (binding_val * 0.2)
    return round(score, 4)


def _resolve_jurisdiction_weight(
    chunk_jurisdiction: str,
    target_jurisdiction: Optional[str],
) -> float:
    """Determine the jurisdiction relationship weight.

    Hierarchy:
      exact_match  -> 1.0   (chunk="US.state.CA", target="US.state.CA")
      federal      -> 0.85  (chunk="US.federal",   target="US.state.*")
      sister_state -> 0.50  (chunk="US.state.NY",  target="US.state.CA")
      foreign      -> 0.30  (chunk="UK",           target="US.*")
      unknown      -> 0.40  (either side unknown)
    """
    if not target_jurisdiction or target_jurisdiction == "unknown":
        return 0.40
    if not chunk_jurisdiction or chunk_jurisdiction == "unknown":
        return 0.40

    c = chunk_jurisdiction.lower()
    t = target_jurisdiction.lower()

    # Exact match
    if c == t:
        return 1.00

    c_parts = c.split(".")
    t_parts = t.split(".")

    # Same country?
    if c_parts[0] == t_parts[0]:
        # Federal covers all states in same country
        if len(c_parts) >= 2 and c_parts[1] == "federal":
            return 0.85
        # Both are sub-national but different → sister state
        if len(c_parts) >= 2 and len(t_parts) >= 2:
            return 0.50
        # One is country-level, other is sub-national
        return 0.60

    # Different countries → foreign
    return 0.30
```

---

## 6. Ingestion Pipeline Changes

### 6.1 `backend/tasks.py` — `process_document_task()`

**Goal:** Call `detect_authority()` alongside the existing `generate_doc_summary()`
and `classify_document()` calls, then propagate authority metadata to every
chunk dict before PostgreSQL insert and Qdrant upsert.

#### 6.1.1 Add import

```python
# --- a/backend/tasks.py  (imports section, after line 17)
# +++ b/backend/tasks.py

 from backend.services.document_summary import generate_doc_summary, classify_document
+from backend.services.authority_detector import detect_authority
```

Add the same to the `except ImportError` block:

```python
 from services.document_summary import generate_doc_summary, classify_document
+from services.authority_detector import detect_authority
```

#### 6.1.2 Call `detect_authority` in the enrichment step

Replace the existing enrichment block (lines 111-133) with:

```python
        # 2c. Enrich document: summary + classification + authority via Gemini (parallel)
        publish_enriching(matter_id, detail="Generating summary, classification, and authority...")
        full_text = "\n".join(chunk.get("content", "") for chunk in chunks)

        async def _enrich():
            return await asyncio.gather(
                generate_doc_summary(full_text),
                classify_document(full_text),
                detect_authority(full_text),
            )

        doc_summary, classification, authority = asyncio.run(_enrich())

        # Store enrichment results on Document record
        document.summary = doc_summary
        document.document_type = classification["document_type"]
        document.jurisdiction = classification["jurisdiction"]
        db.commit()

        logger.info(
            f"[Task {self.request.id}] Enrichment complete: "
            f"summary={'yes' if doc_summary else 'no'}, "
            f"type={classification['document_type']}, "
            f"jurisdiction={classification['jurisdiction']}, "
            f"authority_score={authority.get('authority_score', 'N/A')}"
        )

        # Propagate document-level metadata to chunk dicts for Qdrant payload
        for chunk in chunks:
            chunk["document_type"] = classification["document_type"]
            chunk["jurisdiction"] = classification["jurisdiction"]
            # Authority metadata (stored as JSON on Chunk model + Qdrant payload)
            chunk["authority_metadata"] = authority
```

#### 6.1.3 Store `authority_metadata` in PostgreSQL chunk mappings

In the chunk_mappings loop (around line 151), add the new field:

```python
        chunk_mappings.append({
            "id": chunk_id,
            "matter_id": UUID(matter_id),
            "document_id": doc_uuid,
            "page_num": chunk.get("page_num"),
            "section_name": chunk.get("section_name"),
            "section_type": chunk.get("section_type"),
            "content": chunk.get("content"),
            "concepts": chunk.get("concepts"),
            "chunk_sequence": idx,
            "authority_metadata": chunk.get("authority_metadata"),   # NEW
        })
```

---

## 7. Qdrant Payload Changes

### 7.1 `backend/services/vector_store.py` — `upsert_vectors()`

**Goal:** Store authority fields as top-level payload keys so Qdrant can
filter and sort on them.

#### 7.1.1 Update metadata dict in `upsert_vectors()` (around line 252)

```python
            # Create metadata from chunk — store full content for RAG quality
            authority = chunk.get("authority_metadata", {})
            metadata = {
                "chunk_id": chunk_id,
                "chunk_sequence": chunk.get("chunk_sequence", 0),
                "page_num": str(chunk.get("page_num", "")),
                "section_name": str(chunk.get("section_name", "")),
                "content": chunk.get("content", ""),
                "document_id": str(chunk.get("document_id", "")),
                "document_name": str(chunk.get("document_name", "")),
                "concepts": chunk.get("concepts", []),
                "document_type": str(chunk.get("document_type", "")),
                "jurisdiction": str(chunk.get("jurisdiction", "")),
                # Authority hierarchy fields (new)
                "court_level": authority.get("court_level", "unknown"),
                "jurisdiction_code": authority.get("jurisdiction_code", "unknown"),
                "authority_score": authority.get("authority_score", 0.0),
                "binding_authority": authority.get("binding_authority", False),
                "source_type": authority.get("source_type", "other"),
            }
```

#### 7.1.2 Add payload indexes in `_ensure_payload_indexes()` (around line 113)

```python
    index_fields = {
        "page_num": PayloadSchemaType.KEYWORD,
        "section_name": PayloadSchemaType.KEYWORD,
        "document_type": PayloadSchemaType.KEYWORD,
        "jurisdiction": PayloadSchemaType.KEYWORD,
        # Authority hierarchy indexes (new)
        "court_level": PayloadSchemaType.KEYWORD,
        "jurisdiction_code": PayloadSchemaType.KEYWORD,
        "source_type": PayloadSchemaType.KEYWORD,
        "binding_authority": PayloadSchemaType.KEYWORD,
        "authority_score": PayloadSchemaType.FLOAT,
    }
```

#### 7.1.3 Update `search_vectors()` return dict (around line 370)

Add authority fields to the result dict so the RAG engine can access them:

```python
            result_dict = {
                "score": hit.score,
                "chunk_id": payload.get("chunk_id", ""),
                "chunk_sequence": payload.get("chunk_sequence", 0),
                "page_num": payload.get("page_num", ""),
                "section_name": payload.get("section_name", ""),
                "content": full_content,
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "concepts": payload.get("concepts", []),
                "document_type": payload.get("document_type", ""),
                "jurisdiction": payload.get("jurisdiction", ""),
                # Authority fields (new)
                "court_level": payload.get("court_level", "unknown"),
                "jurisdiction_code": payload.get("jurisdiction_code", "unknown"),
                "authority_score": payload.get("authority_score", 0.0),
                "binding_authority": payload.get("binding_authority", False),
                "source_type": payload.get("source_type", "other"),
            }
```

---

## 8. RAG Engine Changes

### 8.1 `backend/services/rag_engine.py` — Authority-Weighted Reranking

#### 8.1.1 Add import

```python
# After existing imports (around line 26)
from backend.services.authority_detector import compute_query_time_authority_score
```

(And matching fallback imports in the `except ImportError` blocks.)

#### 8.1.2 Detect target jurisdiction from query

Update `_detect_query_filters()` to also return a structured jurisdiction code
(not just the short code used for Qdrant filtering):

```python
# New helper function (add after _detect_query_filters around line 143)

def _detect_target_jurisdiction(query: str) -> Optional[str]:
    """Detect the target jurisdiction code from query text.

    Returns a hierarchical jurisdiction code compatible with
    authority_detector's _resolve_jurisdiction_weight().

    Returns None if no jurisdiction is detected.
    """
    query_lower = query.lower()

    jurisdiction_map = {
        "US.federal": ["federal court", "federal law", "scotus", "supreme court of the united states"],
        "US.state.CA": ["california", "cal.", "ca law"],
        "US.state.NY": ["new york", "n.y."],
        "US.state.TX": ["texas", "tex."],
        "US.state.FL": ["florida", "fla."],
        "US.state.IL": ["illinois", "ill."],
        "UK": ["united kingdom", "english law", "uk law", "england and wales"],
        "UK.england": ["english court", "queen's bench", "king's bench"],
        "EU": ["european union", "eu law", "cjeu", "ecj"],
        "IN": ["india", "indian law", "supreme court of india"],
        "AU": ["australia", "australian law", "high court of australia"],
        "CA.federal": ["canada", "canadian law", "supreme court of canada"],
        "SG": ["singapore", "singaporean law"],
    }

    for code, hints in jurisdiction_map.items():
        if any(h in query_lower for h in hints):
            return code

    # Broader fallback: "us " → US.federal
    broad_map = {
        "US.federal": ["us ", "united states", "american"],
        "UK": ["uk ", "british"],
        "AU": ["australian"],
        "IN": ["indian"],
        "CA.federal": ["canadian"],
    }
    for code, hints in broad_map.items():
        if any(h in query_lower for h in hints):
            return code

    return None
```

#### 8.1.3 Update `rerank_chunks()` to incorporate authority score

Replace the existing `rerank_chunks()` function (lines 255-324) with:

```python
def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = FINAL_CHUNK_COUNT,
    target_jurisdiction: Optional[str] = None,
) -> List[Dict]:
    """
    Rerank retrieved chunks using cross-encoder relevance + authority score.

    Scoring formula:
      final_score = (cross_encoder_score * 0.45)
                  + (vector_score * 0.15)
                  + (authority_score * 0.4)

    When the cross-encoder is unavailable, falls back to:
      final_score = (vector_score * 0.6) + (authority_score * 0.4)

    Args:
        query: User query string
        chunks: List of chunks from vector search
        top_k: Number of top chunks to return after reranking
        target_jurisdiction: Optional jurisdiction code for authority scoring

    Returns:
        List of reranked chunks sorted by final_score, descending
    """
    if not chunks:
        logger.debug("No chunks to rerank")
        return []

    # Compute authority scores for all chunks
    for chunk in chunks:
        authority_meta = {
            "court_level": chunk.get("court_level", "unknown"),
            "jurisdiction_code": chunk.get("jurisdiction_code", "unknown"),
            "binding_authority": chunk.get("binding_authority"),
        }
        chunk["authority_score_computed"] = compute_query_time_authority_score(
            authority_meta, target_jurisdiction
        )

    reranker = _get_reranker()

    if reranker is None:
        # No cross-encoder: vector similarity (0.6) + authority (0.4)
        logger.debug("Reranker not available, using vector + authority scores")
        for chunk in chunks:
            original_score = chunk.get("score", 0)
            auth_score = chunk.get("authority_score_computed", 0)
            chunk["combined_score"] = (original_score * 0.6) + (auth_score * 0.4)

        ranked = sorted(chunks, key=lambda x: x.get("combined_score", 0), reverse=True)
        return ranked[:top_k]

    try:
        # Prepare pairs for cross-encoder
        pairs = []
        for chunk in chunks:
            content = chunk.get("content", "")[:512]
            pairs.append([query, content])

        logger.debug(f"Reranking {len(chunks)} chunks (with authority weighting)")

        scores = reranker.predict(pairs)

        for i, chunk in enumerate(chunks):
            original_score = chunk.get("score", 0)
            rerank_score = float(scores[i])
            auth_score = chunk.get("authority_score_computed", 0)

            chunk["rerank_score"] = rerank_score
            chunk["combined_score"] = (
                (rerank_score * 0.45)
                + (original_score * 0.15)
                + (auth_score * 0.40)
            )

        reranked = sorted(chunks, key=lambda x: x.get("combined_score", 0), reverse=True)

        if reranked:
            top = reranked[0]
            logger.debug(
                f"Top reranked chunk: "
                f"vector={top.get('score', 0):.3f}, "
                f"rerank={top.get('rerank_score', 0):.3f}, "
                f"authority={top.get('authority_score_computed', 0):.3f}, "
                f"combined={top.get('combined_score', 0):.3f}"
            )

        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed, using vector + authority fallback: {str(e)}")
        for chunk in chunks:
            original_score = chunk.get("score", 0)
            auth_score = chunk.get("authority_score_computed", 0)
            chunk["combined_score"] = (original_score * 0.6) + (auth_score * 0.4)
        ranked = sorted(chunks, key=lambda x: x.get("combined_score", 0), reverse=True)
        return ranked[:top_k]
```

#### 8.1.4 Update `query_matter()` to pass `target_jurisdiction`

In `query_matter()`, add target jurisdiction detection and pass it to
`rerank_chunks()`.

**After the query filter detection (around line 1034):**

```python
        query_filters = _detect_query_filters(query)
        target_jurisdiction = _detect_target_jurisdiction(query)  # NEW
```

**Update the `rerank_chunks()` call (around line 1105):**

```python
            reranked_docs = rerank_chunks(
                query, doc_chunks, top_k=top_k,
                target_jurisdiction=target_jurisdiction,  # NEW
            )
```

#### 8.1.5 Include authority metadata in source output

In `query_matter()`, where sources are built for the response (the section
that creates the `sources` list from `final_chunks`), add authority fields:

```python
            source = {
                "chunk_id": chunk.get("chunk_id", ""),
                "page_num": chunk.get("page_num", ""),
                "section_name": chunk.get("section_name", ""),
                "relevance_score": chunk.get("combined_score", chunk.get("score", 0)),
                "content": chunk.get("content", ""),
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("document_name", ""),
                "source_type": chunk.get("source_type", "document"),
                # Authority fields (new)
                "court_level": chunk.get("court_level", "unknown"),
                "jurisdiction_code": chunk.get("jurisdiction_code", "unknown"),
                "authority_score": chunk.get("authority_score_computed",
                                             chunk.get("authority_score", 0.0)),
                "binding_authority": chunk.get("binding_authority", False),
            }
```

### 8.2 `backend/services/rag_engine.py` — Context Formatting

#### 8.2.1 Update `format_legal_context()` to show authority labels

In the header line for each excerpt (around line 197), append authority info:

```python
            header = f"--- EXCERPT {i} ({location_label}"
            if section:
                header += f", Section: {section}"
            doc_name = chunk.get("document_name", "")
            if doc_name:
                header += f", Document: {doc_name}"

            # Authority label (new)
            court_level = chunk.get("court_level", "")
            if court_level and court_level != "unknown":
                header += f", Authority: {court_level.title()}"
            binding = chunk.get("binding_authority")
            if binding is True:
                header += " [BINDING]"

            header += f", Score: {score:.2f}) ---\n"
```

This gives Gemini visibility into authority signals when generating answers,
allowing it to naturally prioritize binding authority excerpts.

---

## 9. Document Summary Changes

### 9.1 `backend/services/document_summary.py`

No changes required. The existing `classify_document()` function already
extracts `document_type` and `jurisdiction`. The new `detect_authority()`
in `authority_detector.py` runs in parallel with it via `asyncio.gather()`
and handles the finer-grained authority classification.

---

## 10. Config Changes

### 10.1 `backend/config.py`

No new config settings are required. The authority detector uses the
existing `google_api_key` and `gemini_model` settings.

**Optional future enhancement:** Add an `authority_scoring_enabled` boolean
flag (default `True`) to allow disabling the feature without code changes:

```python
    # Authority Hierarchy Scoring
    authority_scoring_enabled: bool = True
```

---

## 11. Frontend Changes

### 11.1 Type Definitions — `frontend/lib/types.ts`

Add authority fields to `Citation` and `AskSourceItem`:

```typescript
// In interface Citation (after the existing fields):
export interface Citation {
  documentName: string
  pageNumber: number
  section?: string
  excerpt: string
  relevanceScore: number
  content?: string
  sourceType?: "document" | "case_law"
  url?: string
  verification?: CitationVerification
  // Authority hierarchy (new)
  courtLevel?: "supreme" | "appellate" | "trial" | "administrative" | "unknown"
  jurisdictionCode?: string
  authorityScore?: number
  bindingAuthority?: boolean
}

// In interface AskSourceItem (after the existing fields):
export interface AskSourceItem {
  chunk_id: string
  page_num: string
  section_name: string
  relevance_score: number
  content: string
  document_id: string
  document_name: string
  source_type?: "document" | "case_law"
  url?: string
  // Authority hierarchy (new)
  court_level?: string
  jurisdiction_code?: string
  authority_score?: number
  binding_authority?: boolean
}
```

### 11.2 Authority Badge Component — `frontend/components/AuthorityBadge.tsx`

New component for rendering authority tier badges:

```tsx
"use client"

import React from "react"
import { cn } from "@/lib/utils"
import { Scale, Shield, Landmark, Building2, HelpCircle } from "lucide-react"

interface AuthorityBadgeProps {
  courtLevel?: string
  bindingAuthority?: boolean
  authorityScore?: number
  className?: string
  /** Compact mode shows only the icon + short label */
  compact?: boolean
}

const COURT_CONFIG: Record<string, {
  label: string
  shortLabel: string
  bg: string
  text: string
  border: string
  Icon: React.ElementType
}> = {
  supreme: {
    label: "Supreme Court",
    shortLabel: "Supreme",
    bg: "bg-amber-50",
    text: "text-amber-800",
    border: "border-amber-200/60",
    Icon: Landmark,
  },
  appellate: {
    label: "Appellate Court",
    shortLabel: "Appellate",
    bg: "bg-blue-50",
    text: "text-blue-700",
    border: "border-blue-200/60",
    Icon: Scale,
  },
  trial: {
    label: "Trial Court",
    shortLabel: "Trial",
    bg: "bg-slate-50",
    text: "text-slate-600",
    border: "border-slate-200/60",
    Icon: Building2,
  },
  administrative: {
    label: "Administrative",
    shortLabel: "Admin",
    bg: "bg-gray-50",
    text: "text-gray-500",
    border: "border-gray-200/60",
    Icon: Building2,
  },
  unknown: {
    label: "Unclassified",
    shortLabel: "N/A",
    bg: "bg-gray-50",
    text: "text-gray-400",
    border: "border-gray-200/60",
    Icon: HelpCircle,
  },
}

export default function AuthorityBadge({
  courtLevel = "unknown",
  bindingAuthority,
  authorityScore,
  className,
  compact = false,
}: AuthorityBadgeProps) {
  const config = COURT_CONFIG[courtLevel] ?? COURT_CONFIG.unknown
  const { Icon } = config

  return (
    <div className={cn("inline-flex items-center gap-1", className)}>
      <span
        className={cn(
          "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] font-medium leading-none",
          config.bg, config.text, config.border
        )}
        title={`${config.label}${bindingAuthority ? " (Binding)" : ""}${
          authorityScore != null ? ` — Score: ${(authorityScore * 100).toFixed(0)}%` : ""
        }`}
      >
        <Icon className="h-3 w-3" />
        {compact ? config.shortLabel : config.label}
      </span>
      {bindingAuthority && (
        <span
          className="inline-flex items-center gap-0.5 rounded-md border border-emerald-200/60 bg-emerald-50 px-1.5 py-0.5 text-[10px] font-medium text-emerald-700 leading-none"
          title="Binding Authority"
        >
          <Shield className="h-3 w-3" />
          {!compact && "Binding"}
        </span>
      )}
    </div>
  )
}
```

### 11.3 Citation Panel Integration — `frontend/components/CitationPanel.tsx`

Import and render `AuthorityBadge` inside each citation card.

**Add import:**

```typescript
import AuthorityBadge from "@/components/AuthorityBadge"
```

**Add badge inside citation card** (after the document name / page info line,
before the excerpt):

```tsx
              {/* Authority badge */}
              {(citation.courtLevel && citation.courtLevel !== "unknown") && (
                <div className="mt-1.5">
                  <AuthorityBadge
                    courtLevel={citation.courtLevel}
                    bindingAuthority={citation.bindingAuthority}
                    authorityScore={citation.authorityScore}
                    compact
                  />
                </div>
              )}
```

### 11.4 API Response Mapping — `frontend/lib/api-services.ts`

Where sources from the `/ask` endpoint are mapped to `Citation` objects,
add the new fields:

```typescript
    // Map source to Citation (existing mapping location)
    const citation: Citation = {
      documentName: source.document_name,
      pageNumber: parseInt(source.page_num) || 0,
      section: source.section_name,
      excerpt: (source.content || "").slice(0, 200),
      relevanceScore: source.relevance_score,
      content: source.content,
      sourceType: source.source_type || "document",
      url: source.url,
      // Authority fields (new)
      courtLevel: source.court_level,
      jurisdictionCode: source.jurisdiction_code,
      authorityScore: source.authority_score,
      bindingAuthority: source.binding_authority,
    }
```

---

## 12. Qdrant Filtering at Query Time (Optional Enhancement)

Users or the system can optionally filter results by authority at search time.
Example: "Show only binding authority from appellate courts or higher."

**In `_detect_query_filters()` in `rag_engine.py`**, add authority-based
filter detection:

```python
    # Authority-based filters
    if "binding" in query_lower and ("authority" in query_lower or "precedent" in query_lower):
        filters["binding_authority"] = True
    if "supreme court" in query_lower:
        filters["court_level"] = "supreme"
    elif "appellate" in query_lower or "appeals" in query_lower:
        filters["court_level"] = "appellate"
```

These are passed directly to Qdrant's `FieldCondition(key=..., match=MatchValue(...))`
via the existing `query_filter` mechanism in `search_vectors()`.

---

## 13. Backfill Strategy for Existing Data

Existing chunks in PostgreSQL and Qdrant will have `authority_metadata = NULL`
and no authority payload fields. The system handles this gracefully:

1. **Query time:** `compute_query_time_authority_score()` returns a default
   score of ~0.45 when `court_level` is `"unknown"` and `binding_authority`
   is `None`. This means legacy chunks are not penalized severely but
   rank below properly classified chunks.

2. **Optional backfill script:** `backend/scripts/backfill_authority.py`
   can be run to re-process existing documents:

```python
"""Backfill authority metadata for existing documents.

Usage: python -m backend.scripts.backfill_authority

Iterates over all documents, calls detect_authority() on their text,
and updates chunk records in PostgreSQL + Qdrant payloads.
"""
import asyncio
import logging
from sqlalchemy import update
from backend.database import get_session_factory
from backend.models import Document, Chunk
from backend.services.storage import download_document_from_blob
from backend.services.text_extraction import extract_text
from backend.services.authority_detector import detect_authority
from backend.services.vector_store import get_qdrant_client, _get_collection_name, _generate_point_id

logger = logging.getLogger(__name__)


async def backfill_all():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        documents = db.query(Document).filter(Document.status == "ready").all()
        logger.info(f"Backfilling authority for {len(documents)} documents")

        for doc in documents:
            try:
                # Download and extract text
                blob = download_document_from_blob(doc.blob_storage_path)
                sections = extract_text(blob, doc.file_type or "pdf")
                full_text = "\n".join(s.get("content", "") for s in sections)

                # Detect authority
                authority = await detect_authority(full_text)

                # Update all chunks for this document
                db.execute(
                    update(Chunk)
                    .where(Chunk.document_id == doc.id)
                    .values(authority_metadata=authority)
                )
                db.commit()

                # Update Qdrant payloads
                chunks = db.query(Chunk).filter(Chunk.document_id == doc.id).all()
                client = get_qdrant_client()
                collection_name = _get_collection_name(str(doc.matter_id))

                if client.collection_exists(collection_name):
                    for chunk in chunks:
                        point_id = _generate_point_id(str(chunk.id), str(doc.matter_id))
                        client.set_payload(
                            collection_name=collection_name,
                            payload={
                                "court_level": authority.get("court_level", "unknown"),
                                "jurisdiction_code": authority.get("jurisdiction_code", "unknown"),
                                "authority_score": authority.get("authority_score", 0.0),
                                "binding_authority": authority.get("binding_authority", False),
                                "source_type": authority.get("source_type", "other"),
                            },
                            points=[point_id],
                        )

                logger.info(f"Backfilled {doc.name}: {authority.get('court_level')}, score={authority.get('authority_score')}")

            except Exception as e:
                logger.error(f"Failed to backfill {doc.name}: {e}")
                db.rollback()
                continue

    finally:
        db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(backfill_all())
```

---

## 14. Complete File Change Summary

| File | Action | Description |
|------|--------|-------------|
| `backend/models.py` | **MODIFY** | Add `authority_metadata` JSON column to Chunk |
| `backend/alembic/versions/12_add_authority_metadata_to_chunks.py` | **CREATE** | Migration for new column |
| `backend/services/authority_detector.py` | **CREATE** | Gemini-based authority classification + scoring |
| `backend/tasks.py` | **MODIFY** | Add `detect_authority()` to enrichment, propagate to chunks |
| `backend/services/vector_store.py` | **MODIFY** | Add authority fields to payload + indexes + search results |
| `backend/services/rag_engine.py` | **MODIFY** | Authority-weighted reranking, target jurisdiction detection, context labels |
| `backend/services/document_summary.py` | No change | (Already handles classification; authority is separate) |
| `backend/config.py` | Optional | Add `authority_scoring_enabled` flag |
| `frontend/lib/types.ts` | **MODIFY** | Add authority fields to Citation + AskSourceItem |
| `frontend/components/AuthorityBadge.tsx` | **CREATE** | Badge component for court level + binding indicator |
| `frontend/components/CitationPanel.tsx` | **MODIFY** | Render AuthorityBadge in citation cards |
| `frontend/lib/api-services.ts` | **MODIFY** | Map authority fields from API response |
| `backend/scripts/backfill_authority.py` | **CREATE** | One-time backfill for existing documents |

---

## 15. Testing Strategy

### 15.1 Unit Tests

**File:** `backend/tests/test_authority_detector.py`

```python
"""Unit tests for authority_detector.py"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from backend.services.authority_detector import (
    detect_authority,
    compute_query_time_authority_score,
    _compute_authority_score,
    _resolve_jurisdiction_weight,
)


class TestComputeAuthorityScore:
    def test_supreme_binding(self):
        score = _compute_authority_score("supreme", True)
        # (1.0 * 0.5) + (0.5 * 0.3) + (1.0 * 0.2) = 0.85
        assert score == 0.85

    def test_trial_not_binding(self):
        score = _compute_authority_score("trial", False)
        # (0.7 * 0.5) + (0.5 * 0.3) + (0.0 * 0.2) = 0.50
        assert score == 0.50

    def test_unknown_unknown(self):
        score = _compute_authority_score("unknown", None)
        # (0.5 * 0.5) + (0.5 * 0.3) + (0.3 * 0.2) = 0.46
        assert score == 0.46


class TestResolveJurisdictionWeight:
    def test_exact_match(self):
        assert _resolve_jurisdiction_weight("US.state.CA", "US.state.CA") == 1.0

    def test_federal_covers_state(self):
        assert _resolve_jurisdiction_weight("US.federal", "US.state.CA") == 0.85

    def test_sister_state(self):
        assert _resolve_jurisdiction_weight("US.state.NY", "US.state.CA") == 0.50

    def test_foreign(self):
        assert _resolve_jurisdiction_weight("UK", "US.federal") == 0.30

    def test_unknown_target(self):
        assert _resolve_jurisdiction_weight("US.federal", None) == 0.40

    def test_unknown_chunk(self):
        assert _resolve_jurisdiction_weight("unknown", "US.federal") == 0.40


class TestQueryTimeScore:
    def test_supreme_exact_match_binding(self):
        meta = {
            "court_level": "supreme",
            "jurisdiction_code": "US.federal",
            "binding_authority": True,
        }
        score = compute_query_time_authority_score(meta, "US.federal")
        # (1.0 * 0.5) + (1.0 * 0.3) + (1.0 * 0.2) = 1.0
        assert score == 1.0

    def test_trial_foreign_not_binding(self):
        meta = {
            "court_level": "trial",
            "jurisdiction_code": "UK",
            "binding_authority": False,
        }
        score = compute_query_time_authority_score(meta, "US.state.CA")
        # (0.7 * 0.5) + (0.3 * 0.3) + (0.0 * 0.2) = 0.44
        assert score == 0.44


@pytest.mark.asyncio
class TestDetectAuthority:
    @patch("backend.services.authority_detector.get_settings")
    async def test_no_api_key(self, mock_settings):
        mock_settings.return_value.google_api_key = ""
        result = await detect_authority("some text")
        assert result["court_level"] == "unknown"
        assert result["confidence"] == 0.0

    @patch("backend.services.authority_detector.genai")
    @patch("backend.services.authority_detector.get_settings")
    async def test_successful_classification(self, mock_settings, mock_genai):
        mock_settings.return_value.google_api_key = "test-key"
        mock_settings.return_value.gemini_model = "gemini-2.5-flash-lite"

        mock_response = MagicMock()
        mock_response.text = '{"source_type":"case_law","court_level":"supreme","court_name":"Supreme Court of the United States","jurisdiction_code":"US.federal","binding_authority":true,"confidence":0.95}'

        mock_model = MagicMock()
        mock_model.generate_content_async = AsyncMock(return_value=mock_response)
        mock_genai.GenerativeModel.return_value = mock_model

        result = await detect_authority("Marbury v. Madison, 5 U.S. 137 (1803)")

        assert result["source_type"] == "case_law"
        assert result["court_level"] == "supreme"
        assert result["jurisdiction_code"] == "US.federal"
        assert result["binding_authority"] is True
        assert result["authority_score"] == 0.85
        assert result["confidence"] == 0.95
```

### 15.2 Integration Tests

**File:** `backend/tests/test_authority_reranking.py`

Test that the full reranking pipeline correctly reorders chunks when
authority signals differ:

```python
"""Integration tests for authority-weighted reranking."""
from backend.services.rag_engine import rerank_chunks


def test_binding_supreme_outranks_trial():
    """Supreme Court binding opinion should outrank a trial court chunk
    even if the trial court chunk has higher vector similarity."""
    chunks = [
        {
            "content": "The trial court held that the contract was valid.",
            "score": 0.92,  # Higher vector similarity
            "court_level": "trial",
            "jurisdiction_code": "US.state.CA",
            "binding_authority": False,
        },
        {
            "content": "This Court holds that contract formation requires consideration.",
            "score": 0.78,  # Lower vector similarity
            "court_level": "supreme",
            "jurisdiction_code": "US.federal",
            "binding_authority": True,
        },
    ]

    # Rerank with cross-encoder disabled (mock scenario)
    # In practice the cross-encoder would also run, but this tests
    # the authority component in isolation.
    result = rerank_chunks(
        query="Is a contract valid without consideration?",
        chunks=chunks,
        top_k=2,
        target_jurisdiction="US.state.CA",
    )

    # The supreme court chunk should rank first due to authority weight
    assert result[0]["court_level"] == "supreme"


def test_legacy_chunks_get_default_score():
    """Chunks without authority metadata should get a mid-range default."""
    chunks = [
        {
            "content": "Some legacy chunk without authority fields.",
            "score": 0.80,
            # No court_level, jurisdiction_code, binding_authority
        },
    ]

    result = rerank_chunks(
        query="test query",
        chunks=chunks,
        top_k=1,
    )

    assert len(result) == 1
    assert "authority_score_computed" in result[0]
    # Default score should be moderate (around 0.40-0.50)
    assert 0.30 <= result[0]["authority_score_computed"] <= 0.60
```

### 15.3 E2E Test

Add a test to `backend/tests/test_all_phases_e2e.py` that ingests a known
court opinion PDF and verifies that authority metadata is correctly stored
in both PostgreSQL and Qdrant.

---

## 16. Performance Impact

| Operation | Current | With Authority | Delta |
|-----------|---------|---------------|-------|
| Ingestion (per doc) | 2 Gemini calls | 3 Gemini calls (+1 for authority) | +~200ms |
| Qdrant payload size | ~6 fields | ~11 fields | +5 small fields |
| Query-time reranking | Cross-encoder only | Cross-encoder + authority lookup | +<1ms (dict lookups) |
| Qdrant index memory | 4 keyword indexes | 8 keyword + 1 float indexes | Marginal |

The additional Gemini call runs in parallel with the existing two calls via
`asyncio.gather()`, so the wall-clock time increase is minimal (bounded by
the slowest of the three calls, not additive).

---

## 17. Rollout Plan

1. **Phase 1 — Backend only (no UI):**
   - Deploy migration, create `authority_detector.py`, update tasks.py,
     vector_store.py, rag_engine.py.
   - New documents get authority metadata. Old documents work with defaults.

2. **Phase 2 — Backfill:**
   - Run `backfill_authority.py` during a maintenance window.
   - Monitor Gemini API usage (1 call per existing document).

3. **Phase 3 — Frontend:**
   - Deploy AuthorityBadge component, update types and CitationPanel.
   - Authority badges appear only when metadata is present.

4. **Phase 4 — Feature flag:**
   - Add `authority_scoring_enabled` to config.
   - Allow disabling via environment variable if issues arise.

---

## 18. Open Questions

1. **Statute vs. case law scoring:** Should statutes always outrank case law
   from the same jurisdiction? Currently, a statute gets `court_level=supreme`
   by default, but this may not be correct for subordinate legislation
   (e.g., municipal ordinances).

2. **Multi-jurisdiction queries:** When a query mentions multiple
   jurisdictions (e.g., "Compare US and UK contract law"), how should the
   jurisdiction weight be computed? Current spec uses the first detected
   jurisdiction.

3. **User override:** Should users be able to set target jurisdiction
   explicitly via the UI (e.g., a dropdown), rather than relying on
   auto-detection from query text?

4. **Authority in system prompt:** Should the Gemini system prompt be
   updated to explicitly instruct the model to prioritize binding authority
   excerpts in its answer, beyond the `[BINDING]` label in context?
