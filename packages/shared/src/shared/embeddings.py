"""OpenAI embedding generation utilities."""

from typing import List
import logging
from openai import AsyncOpenAI, APIError

logger = logging.getLogger(__name__)


async def generate_embeddings_batch(
    texts: List[str],
    model: str = "text-embedding-3-small",
    api_key: str = None,
) -> List[List[float]]:
    """Generate embeddings for a batch of texts using OpenAI API.

    Args:
        texts: List of text chunks (max 2048 per API call, we use 20-25)
        model: OpenAI embedding model name
        api_key: OpenAI API key (uses env var if not provided)

    Returns:
        List of embedding vectors (512-dimensional for text-embedding-3-small)

    Raises:
        ValueError: If texts is empty or contains empty strings
        APIError: If OpenAI API call fails
    """
    if not texts:
        raise ValueError("No texts provided for embedding")

    # Validate inputs
    for i, text in enumerate(texts):
        if not text or len(text) == 0:
            raise ValueError(f"Empty text at index {i} cannot be embedded")

    from .config import settings

    api_key = api_key or settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")

    client = AsyncOpenAI(api_key=api_key)

    try:
        response = await client.embeddings.create(
            model=model,
            input=texts,
        )
    except APIError as e:
        logger.error(f"OpenAI API error: {e}")
        raise

    # Extract embeddings in order (matches input order)
    return [item.embedding for item in response.data]


__all__ = ["generate_embeddings_batch"]
