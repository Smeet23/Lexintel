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
