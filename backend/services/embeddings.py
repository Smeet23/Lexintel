"""Cohere embeddings service with caching"""
import hashlib
import logging
from functools import lru_cache
from typing import List

from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

try:
    from backend.exceptions import EmbeddingException
except ImportError:
    try:
        from exceptions import EmbeddingException
    except ImportError:
        from ..exceptions import EmbeddingException

try:
    from backend.services.embedding_cache import get_embedding_cache
except ImportError:
    try:
        from services.embedding_cache import get_embedding_cache
    except ImportError:
        from .embedding_cache import get_embedding_cache

logger = logging.getLogger(__name__)
settings = get_settings()

# Cohere embed-english-v3.0 output dimensions
EMBEDDING_MODEL = "embed-english-v3.0"
EMBEDDING_DIMENSIONS = 1024


@lru_cache(maxsize=1)
def get_cohere_client():
    """
    Get Cohere client for embeddings.

    Returns:
        cohere.Client instance

    Raises:
        ValueError: If Cohere API key is not configured
    """
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    import cohere
    # Support both Client(api_key=...) and Client(token=...) across SDK versions
    try:
        return cohere.Client(api_key=settings.cohere_api_key)
    except TypeError:
        return cohere.Client(token=settings.cohere_api_key)



def embed_text(text: str) -> List[float]:
    """
    Embed a single piece of text into a vector.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding (1024 dimensions)

    Raises:
        ValueError: If text is empty
        EmbeddingException: If API call fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache = get_embedding_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Embedding cache hit")
        return cached.tolist() if hasattr(cached, "tolist") else list(cached)

    try:
        logger.debug(f"Embedding text of length {len(text)}")
        client = get_cohere_client()
        response = client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_document",
        )
        if not response.embeddings or len(response.embeddings) != 1:
            raise ValueError("Cohere returned empty or invalid embedding")
        embedding = response.embeddings[0]

        import numpy as np
        cache.put(cache_key, np.array(embedding, dtype=np.float32))
        logger.debug(f"Successfully embedded text, dimension: {len(embedding)}")
        return embedding

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Cohere API error during text embedding: {str(e)}")
        raise EmbeddingException(
            "Failed to embed text with Cohere",
            detail=f"Cohere error: {str(e)}",
        ) from e


def embed_query(text: str) -> List[float]:
    """
    Embed a search query into a vector (use search_query input type for better retrieval).

    Args:
        text: Query text to embed

    Returns:
        List of floats representing the embedding (1024 dimensions)
    """
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty")

    cache_key = "query:" + hashlib.sha256(text.encode()).hexdigest()
    cache = get_embedding_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Query embedding cache hit")
        return cached.tolist() if hasattr(cached, "tolist") else list(cached)

    try:
        client = get_cohere_client()
        response = client.embed(
            texts=[text],
            model=EMBEDDING_MODEL,
            input_type="search_query",
        )
        if not response.embeddings or len(response.embeddings) != 1:
            raise ValueError("Cohere returned empty or invalid embedding")
        embedding = response.embeddings[0]
        import numpy as np
        cache.put(cache_key, np.array(embedding, dtype=np.float32))
        return embedding
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Cohere API error during query embedding: {str(e)}")
        raise EmbeddingException(
            "Failed to embed query with Cohere",
            detail=f"Cohere error: {str(e)}",
        ) from e


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _embed_batch_with_retry(client, texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts with retry on transient failures."""
    response = client.embed(
        texts=texts,
        model=EMBEDDING_MODEL,
        input_type="search_document",
    )
    if not response.embeddings:
        raise ValueError("Cohere returned empty embeddings")
    return response.embeddings


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed multiple chunks of text into vectors.

    Args:
        chunks: List of text chunks to embed

    Returns:
        List of embeddings (each 1024 dimensions)

    Raises:
        ValueError: If chunks list is empty or contains empty strings
        EmbeddingException: If API call fails
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            raise ValueError(f"Chunk at index {i} is empty")

    try:
        import numpy as np
        cache = get_embedding_cache()
        results = [None] * len(chunks)
        uncached_indices = []
        uncached_texts = []

        for i, chunk in enumerate(chunks):
            cache_key = hashlib.sha256(chunk.encode()).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                results[i] = cached.tolist() if hasattr(cached, "tolist") else list(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(chunk)

        cache_hits = len(chunks) - len(uncached_texts)
        if cache_hits > 0:
            logger.info(f"Embedding cache: {cache_hits}/{len(chunks)} hits, {len(uncached_texts)} to embed")

        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} chunks via Cohere API")
            client = get_cohere_client()
            # Cohere allows up to 96 texts per call; batch if needed
            batch_size = 96
            all_embeddings = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                batch_embeddings = _embed_batch_with_retry(client, batch)
                all_embeddings.extend(batch_embeddings)

            if len(all_embeddings) != len(uncached_texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(uncached_texts)}, got {len(all_embeddings)}"
                )

            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                cache_key = hashlib.sha256(chunks[idx].encode()).hexdigest()
                cache.put(cache_key, np.array(embedding, dtype=np.float32))

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return results

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Cohere API error during chunk embedding: {str(e)}")
        raise EmbeddingException(
            "Failed to embed chunks with Cohere",
            detail=f"Cohere error: {str(e)}",
        ) from e
