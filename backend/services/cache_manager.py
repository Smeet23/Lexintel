"""Redis-based query caching for RAG responses"""
import json
import hashlib
import logging
from typing import Optional, Dict, Any
import redis.asyncio as redis

logger = logging.getLogger(__name__)


def generate_cache_key(query: str, matter_id: str) -> str:
    """Generate normalized cache key from query and matter_id.

    Normalizes query to lowercase with single spaces for deterministic keys.

    Args:
        query: User query string
        matter_id: Matter UUID as string

    Returns:
        Cache key: "query:{matter_id}:{md5_hash}"
    """
    # Normalize: lowercase + single spaces
    normalized = " ".join(query.lower().split())

    # Hash the normalized query
    query_hash = hashlib.md5(normalized.encode()).hexdigest()

    return f"query:{matter_id}:{query_hash}"


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
