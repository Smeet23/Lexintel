"""Redis client management."""

from typing import Optional
import redis.asyncio as redis

_redis_client: Optional[redis.Redis] = None


async def get_redis_client(url: Optional[str] = None) -> redis.Redis:
    """Get or create Redis client (singleton pattern).

    Provides a single persistent Redis connection throughout the application.

    Args:
        url: Redis URL (only used for initial creation)

    Returns:
        Redis async client
    """
    global _redis_client

    if _redis_client is None:
        from ..config import settings
        url = url or settings.REDIS_URL
        _redis_client = await redis.from_url(url, decode_responses=True)

    return _redis_client


async def close_redis():
    """Close Redis connection gracefully."""
    global _redis_client

    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None


__all__ = ["get_redis_client", "close_redis"]
