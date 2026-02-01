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
