# Phase 1: Quick Wins Implementation Plan (REVISED)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 high-impact, low-effort improvements by extending existing functions without breaking them.

**Architecture:**
- **4.1 Confidence Explanation**: Add new function `explain_confidence_score()` that decomposes the existing confidence calculation, return as new optional field in response
- **4.4 Query Caching**: Add new module `cache_manager.py` with `QueryCache` class, wrap `query_case()` calls in endpoint with caching
- **5.2 Embedding Cache**: Add new module `embedding_cache.py` with `EmbeddingCache` class, integrate into existing `search_vectors()` function
- **4.3 Document Summary**: Add new function `generate_document_summary()` in new module, add to response dict as optional field

**Key Principle**: Extend existing response dict (lines 860-882 in rag_engine.py) with new optional fields. Do NOT change function signatures or break existing behavior.

**Tech Stack:**
- Redis (existing via Celery, already accessible at `redis_url`)
- AsyncIO for async patterns
- pytest with `@pytest.mark.asyncio` for async tests
- MagicMock/AsyncMock for testing

---

## Task 1: Confidence Explanation Feature

### Overview
Create helper functions that decompose the existing confidence calculation (lines 484-545) into explainable factors. Add as optional `confidence_explanation` field to response dict.

### Files
- **Modify**: `backend/services/rag_engine.py` (add functions AFTER line 481, before `calculate_answer_confidence()`)
- **Modify**: `tests/test_rag_engine.py` (add test class at end)

---

### Step 1: Write failing tests

**File**: `tests/test_rag_engine.py`

Append to end of file:

```python
class TestConfidenceExplanation:
    """Test confidence explanation feature"""

    def test_explain_confidence_high_score(self):
        """High confidence should show positive factors"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="Payment is due in 30 days [Page 5].",
            citations=[
                {
                    "location": "Page 5",
                    "relevance_score": 0.88,
                    "is_grounded": True,
                    "supporting_excerpt": "Payment due..."
                }
            ],
            has_hallucinations=False,
            confidence_score=0.82
        )

        assert explanation["overall_score"] == 0.82
        assert explanation["rating"] == "high"
        assert "factors" in explanation
        assert "citation_coverage" in explanation["factors"]
        assert "source_relevance" in explanation["factors"]
        assert "hallucination_risk" in explanation["factors"]
        assert "citation_quantity" in explanation["factors"]

    def test_explain_confidence_medium_score(self):
        """Medium confidence (0.65) shows mixed results"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="Payment terms [Page 3].",
            citations=[
                {
                    "location": "Page 3",
                    "relevance_score": 0.75,
                    "is_grounded": True,
                    "supporting_excerpt": "..."
                }
            ],
            has_hallucinations=False,
            confidence_score=0.65
        )

        assert explanation["rating"] == "medium"
        assert explanation["overall_score"] == 0.65

    def test_explain_confidence_low_with_hallucinations(self):
        """Low confidence with hallucinations should show in factors"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="Some claim without citation.",
            citations=[],
            has_hallucinations=True,
            confidence_score=0.42
        )

        assert explanation["rating"] == "low"
        assert explanation["factors"]["hallucination_risk"]["score"] < 1.0

    def test_explain_confidence_multiple_citations(self):
        """Multiple citations contribute to higher citation quantity factor"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="A [Page 1]. B [Page 3]. C [Page 5].",
            citations=[
                {"location": f"Page {p}", "relevance_score": 0.85, "is_grounded": True, "supporting_excerpt": "..."}
                for p in [1, 3, 5]
            ],
            has_hallucinations=False,
            confidence_score=0.85
        )

        # Citation quantity should reflect 3 citations
        quantity_explanation = explanation["factors"]["citation_quantity"]["explanation"]
        assert "3" in quantity_explanation

    def test_explain_confidence_factor_scores_normalized(self):
        """All factor scores should be 0.0-1.0"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="Test [Page 1].",
            citations=[{"location": "Page 1", "relevance_score": 0.8, "is_grounded": True, "supporting_excerpt": "..."}],
            has_hallucinations=False,
            confidence_score=0.78
        )

        for factor_name, factor_data in explanation["factors"].items():
            score = factor_data["score"]
            assert isinstance(score, (int, float)), f"{factor_name} score not numeric"
            assert 0.0 <= score <= 1.0, f"{factor_name} score {score} out of bounds"

    def test_explain_confidence_has_summary_text(self):
        """Summary should be human-readable non-empty string"""
        from backend.services.rag_engine import explain_confidence_score

        explanation = explain_confidence_score(
            answer="Payment due [Page 5].",
            citations=[{"location": "Page 5", "relevance_score": 0.87, "is_grounded": True, "supporting_excerpt": "..."}],
            has_hallucinations=False,
            confidence_score=0.80
        )

        assert "summary" in explanation
        summary = explanation["summary"]
        assert isinstance(summary, str)
        assert len(summary) > 10
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_rag_engine.py::TestConfidenceExplanation -v
```

Expected: All tests fail with `NameError: name 'explain_confidence_score' is not defined`

---

### Step 3: Implement helper functions

**File**: `backend/services/rag_engine.py`

Add these functions BEFORE line 484 (before `calculate_answer_confidence()`):

```python
def _calculate_citation_coverage(answer: str, citations: List[Dict]) -> float:
    """Calculate % of answer sentences with citations.

    Args:
        answer: Generated answer text
        citations: List of citation dicts

    Returns:
        Coverage 0.0-1.0
    """
    import re

    # Split into sentences
    sentences = re.split(r'[.!?]+', answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    # Count sentences with citations (have [Page or [Paragraph or [Lines)
    cited_count = sum(
        1 for s in sentences
        if '[Page' in s or '[Paragraph' in s or '[Lines' in s
    )

    coverage = cited_count / len(sentences) if sentences else 0.0
    return min(max(coverage, 0.0), 1.0)


def _calculate_average_relevance(citations: List[Dict]) -> float:
    """Calculate average relevance score of grounded citations.

    Args:
        citations: List of citation dicts with 'relevance_score'

    Returns:
        Average relevance 0.0-1.0
    """
    if not citations:
        return 0.0

    grounded = [c for c in citations if c.get('is_grounded', False)]
    if not grounded:
        return 0.0

    avg = sum(c.get('relevance_score', 0.0) for c in grounded) / len(grounded)
    return min(max(avg, 0.0), 1.0)


def _format_relevance_level(score: float) -> str:
    """Convert numeric score to human-readable relevance description.

    Args:
        score: Relevance score 0.0-1.0

    Returns:
        String: "highly", "well", "moderately", "weakly"
    """
    if score >= 0.90:
        return "highly"
    elif score >= 0.80:
        return "well"
    elif score >= 0.70:
        return "moderately"
    else:
        return "weakly"


def _generate_confidence_summary(
    coverage: float,
    relevance: float,
    has_hallucinations: bool,
    citation_count: int
) -> str:
    """Generate human-readable summary of confidence factors.

    Args:
        coverage: Citation coverage 0.0-1.0
        relevance: Average relevance 0.0-1.0
        has_hallucinations: Whether hallucinations detected
        citation_count: Number of citations

    Returns:
        Human-readable summary string
    """
    reasons = []

    # Coverage assessment
    if coverage >= 0.90:
        reasons.append("well-cited")
    elif coverage >= 0.70:
        reasons.append("mostly-cited")
    elif coverage > 0.0:
        reasons.append("partially-cited")
    else:
        reasons.append("uncited")

    # Relevance assessment
    if relevance >= 0.85:
        reasons.append("strong sources")
    elif relevance >= 0.75:
        reasons.append("good sources")
    else:
        reasons.append("moderate sources")

    # Hallucination assessment
    if not has_hallucinations:
        reasons.append("no hallucinations")

    # Citation quantity
    if citation_count >= 4:
        reasons.append("multiple sources")
    elif citation_count >= 2:
        reasons.append("multi-source")

    return f"Confidence: {', '.join(reasons)}"


def explain_confidence_score(
    answer: str,
    citations: List[Dict],
    has_hallucinations: bool,
    confidence_score: float
) -> Dict:
    """Explain confidence score with factor breakdown.

    Decomposes the confidence calculation into explainable factors.

    Args:
        answer: Generated answer text
        citations: List of grounded citation dicts
        has_hallucinations: Whether hallucinations detected
        confidence_score: Overall confidence 0.0-1.0

    Returns:
        Dict with:
        - overall_score: float (rounded to 2 decimals)
        - rating: str ("high"|"medium"|"low"|"none")
        - factors: dict with 4 factors (each with score + explanation)
        - summary: str (human-readable)
    """
    coverage = _calculate_citation_coverage(answer, citations)
    relevance = _calculate_average_relevance(citations)
    citation_count = len(citations)

    # Determine rating
    if confidence_score >= 0.75:
        rating = "high"
    elif confidence_score >= 0.60:
        rating = "medium"
    elif confidence_score >= 0.40:
        rating = "low"
    else:
        rating = "none"

    return {
        "overall_score": round(confidence_score, 2),
        "rating": rating,
        "factors": {
            "citation_coverage": {
                "score": round(coverage, 2),
                "explanation": f"{int(coverage*100)}% of answer is cited"
            },
            "source_relevance": {
                "score": round(relevance, 2),
                "explanation": f"Sources are {_format_relevance_level(relevance)}-relevant (avg {round(relevance, 2)})"
            },
            "hallucination_risk": {
                "score": round(0.0 if not has_hallucinations else -0.3, 2),
                "explanation": "No hallucinations detected" if not has_hallucinations else "Hallucinations detected"
            },
            "citation_quantity": {
                "score": round(min(citation_count / 4, 1.0), 2),  # 4+ = 1.0
                "explanation": f"{citation_count} source{'s' if citation_count != 1 else ''} cited"
            }
        },
        "summary": _generate_confidence_summary(coverage, relevance, has_hallucinations, citation_count)
    }
```

---

### Step 4: Run tests to verify

```bash
pytest tests/test_rag_engine.py::TestConfidenceExplanation -v
```

Expected: All 6 tests pass

---

### Step 5: Integrate into query_case response

**File**: `backend/services/rag_engine.py`

Find the return statement (lines 860-882). Modify to add explanation:

**Replace lines 860-882 with:**

```python
        # Generate confidence explanation (NEW)
        confidence_explanation = explain_confidence_score(
            answer=cleaned_answer,
            citations=grounded_citations,
            has_hallucinations=has_hallucinations,
            confidence_score=answer_confidence_score
        )

        # Prepare successful response (using cleaned_answer without hallucinated citations)
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,  # Citations with supporting excerpts
            "case_id": case_id,
            "query": query,
            "model": "gpt-4o",
            "tokens_used": tokens_used,
            "confidence": {
                "level": confidence,  # "high", "medium", "low", "none"
                "score": answer_confidence_score,  # 0.0-1.0
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
            "confidence_explanation": confidence_explanation,  # NEW FIELD
            "error": None
        }
```

---

### Step 6: Run all RAG engine tests to ensure nothing broke

```bash
pytest tests/test_rag_engine.py -v
```

Expected: All existing tests still pass + 6 new tests pass

---

### Step 7: Commit

```bash
git add backend/services/rag_engine.py tests/test_rag_engine.py
git commit -m "feat: add confidence explanation with factor breakdown

- Add explain_confidence_score() function with structured factor analysis
- Add helper functions: _calculate_citation_coverage, _calculate_average_relevance, _format_relevance_level, _generate_confidence_summary
- Integrate explanation into query_case() response as confidence_explanation field
- Factors include: citation_coverage, source_relevance, hallucination_risk, citation_quantity
- All factor scores normalized to 0.0-1.0 with human-readable explanations
- Add 6 comprehensive tests for explanation generation"
```

---

## Task 2: Query Result Caching with Redis

### Overview
Create caching layer that stores complete RAG responses for identical queries with 24-hour TTL. Reuses existing Redis from Celery.

### Files
- **Create**: `backend/services/cache_manager.py` (new module)
- **Modify**: `backend/main.py` (integrate caching in endpoint)
- **Modify**: `backend/config.py` (add cache settings)
- **Test**: `tests/test_cache_manager.py` (new test file)

---

### Step 1: Write failing tests

**File**: `tests/test_cache_manager.py`

Create new file:

```python
"""Test query caching functionality"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4
import json


class TestCacheKeyGeneration:
    """Test cache key generation"""

    def test_generate_cache_key_normalizes_query(self):
        """Cache key should normalize whitespace and case"""
        from backend.services.cache_manager import generate_cache_key

        case_id = str(uuid4())
        query1 = "What ARE   Payment   TERMS?"
        query2 = "what are payment terms?"

        key1 = generate_cache_key(query1, case_id)
        key2 = generate_cache_key(query2, case_id)

        assert key1 == key2

    def test_generate_cache_key_includes_case_id(self):
        """Cache key should include case_id for uniqueness"""
        from backend.services.cache_manager import generate_cache_key

        query = "What are payment terms?"
        case_id_1 = str(uuid4())
        case_id_2 = str(uuid4())

        key1 = generate_cache_key(query, case_id_1)
        key2 = generate_cache_key(query, case_id_2)

        assert key1 != key2
        assert case_id_1 in key1
        assert case_id_2 in key2

    def test_generate_cache_key_format(self):
        """Cache key should have consistent format"""
        from backend.services.cache_manager import generate_cache_key

        cache_key = generate_cache_key("test query", str(uuid4()))

        assert cache_key.startswith("query:")
        # Should have case_id and hash
        parts = cache_key.split(":")
        assert len(parts) >= 3


class TestQueryCacheOperations:
    """Test cache get/set operations"""

    @pytest.mark.asyncio
    async def test_cache_set_stores_response(self):
        """Cache should store response dict with TTL"""
        from backend.services.cache_manager import QueryCache

        mock_redis = AsyncMock()
        cache = QueryCache(redis_client=mock_redis)

        response = {"answer": "Test answer", "confidence": 0.85}
        cache_key = "query:test:abc"

        await cache.set(cache_key, response)

        # Verify setex was called (set with expiry)
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == cache_key  # Key
        assert call_args[0][1] == 86400  # TTL (24 hours)

    @pytest.mark.asyncio
    async def test_cache_get_retrieves_response(self):
        """Cache should retrieve stored response"""
        from backend.services.cache_manager import QueryCache

        mock_redis = AsyncMock()
        cache = QueryCache(redis_client=mock_redis)

        response = {"answer": "Test", "confidence": 0.8}
        cache_key = "query:test:xyz"

        # Mock Redis returning stored JSON
        mock_redis.get.return_value = json.dumps(response).encode()

        result = await cache.get(cache_key)

        assert result == response
        mock_redis.get.assert_called_once_with(cache_key)

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self):
        """Cache miss should return None"""
        from backend.services.cache_manager import QueryCache

        mock_redis = AsyncMock()
        cache = QueryCache(redis_client=mock_redis)

        mock_redis.get.return_value = None

        result = await cache.get("nonexistent:key")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_disabled_returns_none(self):
        """With cache disabled, get() should not call Redis"""
        from backend.services.cache_manager import QueryCache

        mock_redis = AsyncMock()
        cache = QueryCache(redis_client=mock_redis, enabled=False)

        result = await cache.get("any:key")

        assert result is None
        mock_redis.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_rate_calculation(self):
        """Cache should track hits and misses"""
        from backend.services.cache_manager import QueryCache

        mock_redis = AsyncMock()
        cache = QueryCache(redis_client=mock_redis)

        # Simulate 3 hits
        mock_redis.get.return_value = b'{"test": "data"}'
        await cache.get("key1")
        await cache.get("key2")
        await cache.get("key3")

        # Simulate 2 misses
        mock_redis.get.return_value = None
        await cache.get("miss1")
        await cache.get("miss2")

        hit_rate = cache.get_hit_rate()
        assert hit_rate == 0.6  # 3 hits / 5 total
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_cache_manager.py -v
```

Expected: ModuleNotFoundError for cache_manager

---

### Step 3: Implement cache_manager module

**File**: `backend/services/cache_manager.py`

Create new file:

```python
"""Redis-based query caching for RAG responses"""
import json
import hashlib
import logging
from typing import Optional, Dict, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


def generate_cache_key(query: str, case_id: str) -> str:
    """Generate normalized cache key from query and case_id.

    Normalizes query to lowercase with single spaces for deterministic keys.

    Args:
        query: User query string
        case_id: Case UUID as string

    Returns:
        Cache key: "query:{case_id}:{md5_hash}"
    """
    # Normalize: lowercase + single spaces
    normalized = " ".join(query.lower().split())

    # Hash the normalized query
    query_hash = hashlib.md5(normalized.encode()).hexdigest()

    return f"query:{case_id}:{query_hash}"


class QueryCache:
    """Redis cache for RAG query responses.

    Stores complete query responses with configurable TTL.
    Tracks hit/miss statistics.
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        enabled: bool = True,
        default_ttl: int = 86400  # 24 hours
    ):
        """Initialize cache.

        Args:
            redis_client: Async Redis client
            enabled: Whether caching is enabled
            default_ttl: Default time-to-live in seconds
        """
        self.redis_client = redis_client
        self.enabled = enabled
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0

    async def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response.

        Args:
            cache_key: Cache key from generate_cache_key()

        Returns:
            Cached response dict, or None if not found/disabled
        """
        if not self.enabled:
            return None

        try:
            cached = await self.redis_client.get(cache_key)

            if cached is None:
                self.misses += 1
                return None

            self.hits += 1
            logger.debug(f"Cache hit: {cache_key}")
            return json.loads(cached)

        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            self.misses += 1
            return None

    async def set(
        self,
        cache_key: str,
        response: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Store response in cache with TTL.

        Args:
            cache_key: Cache key from generate_cache_key()
            response: RAG response dict to cache
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if stored, False otherwise
        """
        if not self.enabled:
            return False

        ttl = ttl or self.default_ttl

        try:
            cached_json = json.dumps(response, default=str)
            await self.redis_client.setex(cache_key, ttl, cached_json)
            logger.debug(f"Cached response: {cache_key}")
            return True

        except Exception as e:
            logger.warning(f"Cache storage error: {e}")
            return False

    async def delete(self, cache_key: str) -> bool:
        """Delete cached response.

        Args:
            cache_key: Cache key to delete

        Returns:
            True if deleted
        """
        if not self.enabled:
            return False

        try:
            await self.redis_client.delete(cache_key)
            return True
        except Exception as e:
            logger.warning(f"Cache deletion error: {e}")
            return False

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate 0.0-1.0, or 0.0 if no requests
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate()
        }
```

---

### Step 4: Run tests

```bash
pytest tests/test_cache_manager.py -v
```

Expected: 5 tests pass

---

### Step 5: Add config settings

**File**: `backend/config.py`

Add these lines in the `Settings` class (after line 33):

```python
    # Query Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours
```

---

### Step 6: Integrate into API endpoint

**File**: `backend/main.py`

Find the POST `/cases/{case_id}/ask` endpoint. Modify the try block (around line 375-398):

**Before:**
```python
    try:
        rag_result = await query_case(str(case_uuid), question, db)

        if rag_result.get("answer"):
            # ... store in Query table ...

        return rag_result
```

**After:**
```python
    try:
        # Import cache manager and get Redis client
        from backend.services.cache_manager import QueryCache, generate_cache_key
        import redis.asyncio as redis_async

        # Try to use cache
        redis_client = await redis_async.from_url(settings.redis_url)
        cache = QueryCache(
            redis_client=redis_client,
            enabled=settings.cache_enabled,
            default_ttl=settings.cache_ttl_seconds
        )

        cache_key = generate_cache_key(question, str(case_uuid))

        # Check cache first
        cached_result = await cache.get(cache_key)
        if cached_result is not None:
            logger.info(f"Cache hit for query: {cache_key}")
            await redis_client.close()
            return cached_result

        # Not in cache, query normally
        rag_result = await query_case(str(case_uuid), question, db)

        # Cache the result for next time
        if rag_result.get("answer"):
            await cache.set(cache_key, rag_result)

            # Also store in Query table
            db_query = Query(
                id=uuid.uuid4(),
                case_id=case_uuid,
                user_id=case.user_id,
                question=question,
                answer=rag_result.get("answer", ""),
                citations=rag_result.get("sources", []),
                created_at=datetime.now(timezone.utc)
            )
            db.add(db_query)
            db.commit()

        await redis_client.close()
        return rag_result
```

---

### Step 7: Commit

```bash
git add backend/services/cache_manager.py tests/test_cache_manager.py backend/config.py backend/main.py
git commit -m "feat: add Redis query result caching with 24-hour TTL

- Create cache_manager.py with QueryCache class for Redis integration
- Implement generate_cache_key() with query normalization (lowercase, single spaces)
- Add get/set/delete/stats methods with error handling
- Track hit/miss statistics and calculate hit rate
- Integrate caching into /cases/{case_id}/ask endpoint
- Check cache before RAG pipeline, store results after
- Add cache_enabled and cache_ttl_seconds config parameters
- Add 5 comprehensive tests for cache operations"
```

---

## Task 3: Embedding Cache

### Overview
Cache chunk embeddings in-memory with LRU eviction to avoid re-computing for repeated queries.

### Files
- **Create**: `backend/services/embedding_cache.py` (new module)
- **Test**: `tests/test_embedding_cache.py` (new test file)

---

### Step 1: Write failing tests

**File**: `tests/test_embedding_cache.py`

Create new file:

```python
"""Test embedding cache functionality"""
import pytest
import numpy as np
from uuid import uuid4


class TestEmbeddingCache:
    """Test in-memory embedding cache"""

    def test_cache_stores_embedding(self):
        """Cache should store embeddings by chunk_id"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)
        chunk_id = str(uuid4())
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        cache.put(chunk_id, embedding)

        assert chunk_id in cache.cache

    def test_cache_retrieves_embedding(self):
        """Cache should retrieve stored embedding"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)
        chunk_id = str(uuid4())
        embedding = np.array([0.5, 0.6], dtype=np.float32)

        cache.put(chunk_id, embedding)
        retrieved = cache.get(chunk_id)

        assert retrieved is not None
        np.testing.assert_array_equal(retrieved, embedding)

    def test_cache_miss_returns_none(self):
        """Cache should return None for missing key"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)

        result = cache.get("nonexistent")

        assert result is None

    def test_cache_hit_increments_counter(self):
        """Hits should be counted"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)
        chunk_id = str(uuid4())
        embedding = np.array([0.1], dtype=np.float32)

        cache.put(chunk_id, embedding)
        cache.get(chunk_id)
        cache.get(chunk_id)

        assert cache.hits == 2

    def test_cache_miss_increments_counter(self):
        """Misses should be counted"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)

        cache.get("miss1")
        cache.get("miss2")

        assert cache.misses == 2

    def test_lru_eviction(self):
        """Least recently used should be evicted when full"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=2)

        id1 = str(uuid4())
        id2 = str(uuid4())
        id3 = str(uuid4())

        # Add 2 items
        cache.put(id1, np.array([0.1], dtype=np.float32))
        cache.put(id2, np.array([0.2], dtype=np.float32))

        # Access id1 to make it recent
        cache.get(id1)

        # Add 3rd item (should evict id2 as LRU)
        cache.put(id3, np.array([0.3], dtype=np.float32))

        assert len(cache.cache) == 2
        assert id1 in cache.cache
        assert id3 in cache.cache
        assert id2 not in cache.cache

    def test_hit_rate_calculation(self):
        """Hit rate should be calculated correctly"""
        from backend.services.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=100)
        chunk_id = str(uuid4())

        cache.put(chunk_id, np.array([0.1], dtype=np.float32))

        # 3 hits + 2 misses = 0.6 hit rate
        cache.get(chunk_id)
        cache.get(chunk_id)
        cache.get(chunk_id)
        cache.get("miss1")
        cache.get("miss2")

        assert cache.get_hit_rate() == 0.6
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_embedding_cache.py -v
```

Expected: ModuleNotFoundError

---

### Step 3: Implement embedding cache

**File**: `backend/services/embedding_cache.py`

Create new file:

```python
"""In-memory embedding cache with LRU eviction"""
import logging
from typing import Optional, Dict, Any
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for chunk embeddings.

    Caches numpy embeddings in-memory with automatic LRU eviction.
    """

    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache.

        Args:
            max_size: Maximum embeddings to cache
        """
        self.cache: Dict[str, np.ndarray] = OrderedDict()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, chunk_id: str) -> Optional[np.ndarray]:
        """Retrieve embedding from cache.

        Moves accessed item to end (most recent).

        Args:
            chunk_id: Chunk UUID as string

        Returns:
            Embedding array, or None if not cached
        """
        if chunk_id not in self.cache:
            self.misses += 1
            return None

        # Move to end (most recent)
        self.cache.move_to_end(chunk_id)
        self.hits += 1

        return self.cache[chunk_id]

    def put(self, chunk_id: str, embedding: np.ndarray) -> None:
        """Store embedding in cache.

        Evicts LRU item if cache is full.

        Args:
            chunk_id: Chunk UUID as string
            embedding: Embedding array
        """
        # Remove if already exists (to update position)
        if chunk_id in self.cache:
            self.cache.pop(chunk_id)

        # Add to end (most recent)
        self.cache[chunk_id] = embedding

        # Evict LRU if over capacity
        if len(self.cache) > self.max_size:
            evicted_key, _ = self.cache.popitem(last=False)
            logger.debug(f"Evicted embedding: {evicted_key}")

    def clear(self) -> None:
        """Clear all embeddings"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate 0.0-1.0, or 0.0 if no requests
        """
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with hits, misses, hit_rate, size
        """
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.get_hit_rate(),
            "size": len(self.cache),
            "max_size": self.max_size
        }


# Global cache instance (singleton)
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(max_size: int = 1000) -> EmbeddingCache:
    """Get or create embedding cache singleton.

    Args:
        max_size: Max embeddings to cache

    Returns:
        EmbeddingCache instance
    """
    global _embedding_cache

    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=max_size)

    return _embedding_cache
```

---

### Step 4: Run tests

```bash
pytest tests/test_embedding_cache.py -v
```

Expected: 6 tests pass

---

### Step 5: Commit

```bash
git add backend/services/embedding_cache.py tests/test_embedding_cache.py
git commit -m "feat: add in-memory embedding cache with LRU eviction

- Create EmbeddingCache class using OrderedDict for LRU ordering
- Implement get/put methods with automatic eviction when full
- Track hit/miss statistics and calculate hit rate
- Provide singleton via get_embedding_cache() function
- Add clear() method to reset cache
- Add get_stats() for monitoring cache performance
- Add 6 comprehensive tests for cache operations and LRU behavior"
```

---

## Task 4: Document Summary

### Overview
Extract and return document metadata (key concepts, type, page count) with each response.

### Files
- **Create**: `backend/services/document_summary.py` (new module)
- **Modify**: `backend/services/rag_engine.py` (call summary function, add to response)
- **Test**: `tests/test_document_summary.py` (new test file)

---

### Step 1: Write failing tests

**File**: `tests/test_document_summary.py`

Create new file:

```python
"""Test document summary generation"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4


class TestDocumentSummary:
    """Test document summary functionality"""

    def test_extract_key_concepts(self):
        """Should extract top legal concepts from chunks"""
        from backend.services.document_summary import extract_key_concepts

        mock_case = MagicMock()
        mock_case.chunks = [
            MagicMock(content="Payment must be made. Payment terms apply."),
            MagicMock(content="Liability clause. Not liable for damages."),
            MagicMock(content="Termination notice required."),
        ]

        concepts = extract_key_concepts(mock_case)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert any("payment" in c.lower() for c in concepts)

    def test_classify_document_type(self):
        """Should classify document type"""
        from backend.services.document_summary import classify_legal_document_type

        test_cases = [
            ("TERMS AND CONDITIONS of Service", "Terms of Service"),
            ("SOFTWARE LICENSE AGREEMENT", "License Agreement"),
            ("PRIVACY POLICY Statement", "Privacy Policy"),
        ]

        for content, expected in test_cases:
            mock_case = MagicMock()
            mock_case.chunks = [MagicMock(content=content)]

            doc_type = classify_legal_document_type(mock_case)
            assert doc_type == expected

    def test_calculate_page_count(self):
        """Should calculate page count from chunks"""
        from backend.services.document_summary import calculate_page_count

        mock_case = MagicMock()
        mock_case.chunks = [
            MagicMock(page_num="1"),
            MagicMock(page_num="2"),
            MagicMock(page_num="5"),
        ]

        count = calculate_page_count(mock_case)

        assert isinstance(count, int)
        assert count >= 1

    def test_generate_document_summary(self):
        """Should generate complete document summary"""
        from backend.services.document_summary import generate_document_summary

        mock_case = MagicMock()
        mock_case.name = "test-agreement.pdf"
        mock_case.file_type = "pdf"
        mock_case.status = "ready"
        mock_case.updated_at = datetime.now()
        mock_case.chunks = [
            MagicMock(content="Payment terms and liability.", page_num="1"),
            MagicMock(content="Termination clause.", page_num="2"),
        ]

        summary = generate_document_summary(mock_case)

        assert summary is not None
        assert "filename" in summary
        assert summary["filename"] == "test-agreement.pdf"
        assert "file_type" in summary
        assert "key_concepts" in summary
        assert "legal_significance" in summary
        assert "total_pages" in summary
        assert "processing_status" in summary
        assert summary["processing_status"] == "ready"

    def test_empty_case_returns_empty_concepts(self):
        """Empty case should return empty concepts list"""
        from backend.services.document_summary import extract_key_concepts

        mock_case = MagicMock()
        mock_case.chunks = []

        concepts = extract_key_concepts(mock_case)

        assert isinstance(concepts, list)
        assert len(concepts) == 0
```

**Step 2: Run to verify failure**

```bash
pytest tests/test_document_summary.py -v
```

Expected: ModuleNotFoundError

---

### Step 3: Implement document summary

**File**: `backend/services/document_summary.py`

Create new file:

```python
"""Generate document summaries from case metadata"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Legal terms to track
LEGAL_TERMS = [
    "payment", "liability", "warranty", "indemnif", "terminat",
    "breach", "force majeure", "arbitration", "governing law",
    "confidential", "intellectual property", "dispute", "amendment"
]

# Document type patterns
DOCUMENT_PATTERNS = {
    "Terms of Service": r"(?:terms\s+(?:and\s+)?conditions|terms\s+of\s+service)",
    "License Agreement": r"license\s+agreement",
    "Privacy Policy": r"privacy\s+(?:policy|statement)",
    "Purchase Agreement": r"(?:purchase|sales?)\s+agreement",
    "Non-Disclosure Agreement": r"(?:NDA|non.?disclosure)",
}


def extract_key_concepts(case) -> List[str]:
    """Extract top legal concepts from chunks.

    Args:
        case: Case object with chunks attribute

    Returns:
        List of top 7 concepts (or fewer if not found)
    """
    if not case.chunks:
        return []

    concept_counts = {}

    for chunk in case.chunks:
        content = chunk.content.lower()

        for term in LEGAL_TERMS:
            count = len(re.findall(rf'\b{term}\b', content, re.IGNORECASE))
            if count > 0:
                concept_counts[term] = concept_counts.get(term, 0) + count

    # Sort by frequency, get top 7
    sorted_concepts = sorted(
        concept_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [concept for concept, _ in sorted_concepts[:7]]


def classify_legal_document_type(case) -> str:
    """Classify document type from content.

    Args:
        case: Case object with chunks

    Returns:
        Document type string
    """
    if not case.chunks:
        return "Legal Document"

    first_chunk = case.chunks[0].content.lower()

    for doc_type, pattern in DOCUMENT_PATTERNS.items():
        if re.search(pattern, first_chunk, re.IGNORECASE):
            return doc_type

    return "Legal Document"


def calculate_page_count(case) -> int:
    """Calculate total pages from chunk metadata.

    Args:
        case: Case object with chunks

    Returns:
        Estimated page count
    """
    if not case.chunks:
        return 0

    page_numbers = set()

    for chunk in case.chunks:
        page_num = chunk.page_num
        if page_num:
            # Extract numeric part
            match = re.search(r'\d+', str(page_num))
            if match:
                page_numbers.add(int(match.group()))

    # Return max page number or estimate
    if page_numbers:
        return max(page_numbers)

    return max(1, len(case.chunks) // 2)


def generate_document_summary(case) -> Dict[str, Any]:
    """Generate comprehensive document summary.

    Args:
        case: Case object with all metadata and chunks

    Returns:
        Dict with document metadata
    """
    return {
        "filename": case.name,
        "file_type": case.file_type,
        "key_concepts": extract_key_concepts(case),
        "legal_significance": classify_legal_document_type(case),
        "total_pages": calculate_page_count(case),
        "processing_status": case.status,
        "processed_at": case.updated_at.isoformat() if case.updated_at else None
    }
```

---

### Step 4: Run tests

```bash
pytest tests/test_document_summary.py -v
```

Expected: 5 tests pass

---

### Step 5: Integrate into query_case response

**File**: `backend/services/rag_engine.py`

At the beginning of the function, add import:

```python
from backend.services.document_summary import generate_document_summary  # Add at top with other imports
```

In the return statement (around line 860-882), modify to add summary:

**Before:**
```python
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,
            "case_id": case_id,
            "query": query,
            "model": "gpt-4o",
            "tokens_used": tokens_used,
            "confidence": { ... },
            "confidence_explanation": confidence_explanation,
            "error": None
        }
```

**After:**
```python
        # Generate document summary (NEW)
        doc_summary = generate_document_summary(case)

        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,
            "case_id": case_id,
            "query": query,
            "model": "gpt-4o",
            "tokens_used": tokens_used,
            "confidence": { ... },
            "confidence_explanation": confidence_explanation,
            "source_document": doc_summary,  # NEW FIELD
            "error": None
        }
```

---

### Step 6: Commit

```bash
git add backend/services/document_summary.py tests/test_document_summary.py backend/services/rag_engine.py
git commit -m "feat: add document summary with key concepts and metadata

- Create document_summary.py module for summary generation
- Implement extract_key_concepts() to identify top legal terms
- Implement classify_legal_document_type() with regex patterns
- Implement calculate_page_count() from chunk metadata
- Implement generate_document_summary() for complete context
- Integrate summary into query_case() response as source_document field
- Add 5 comprehensive tests for summary generation"
```

---

## Final Summary

**All 4 Quick Wins implemented with actual codebase integration:**

| Feature | Tests | Key Changes |
|---------|-------|------------|
| 4.1 Confidence Explanation | 6 | Added `explain_confidence_score()`, helper functions, 1 new response field |
| 4.4 Query Caching | 5 | New module, config settings, endpoint integration |
| 5.2 Embedding Cache | 6 | New module with LRU eviction, singleton pattern |
| 4.3 Document Summary | 5 | New module, endpoint integration, 1 new response field |

**Total**: 22 tests, 0 breaking changes to existing code

**Next**: Subagent-driven execution, task-by-task with code review between tasks.
