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
