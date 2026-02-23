"""Google Generative AI embeddings service with retry logic and caching"""
import hashlib
import logging
from functools import lru_cache
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, retry_if_exception_type, wait_random_exponential, stop_after_attempt

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

try:
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
except ImportError:
    class GoogleAPIError(Exception):
        pass
    class ResourceExhausted(GoogleAPIError):
        pass

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

# Embeddings configuration
EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMENSIONS = 768  # Reduced from default 3072 to match Qdrant collection


@lru_cache(maxsize=1)
def get_embeddings_client() -> GoogleGenerativeAIEmbeddings:
    """
    Get Google Generative AI embeddings client with configured model.

    Returns:
        GoogleGenerativeAIEmbeddings instance for gemini-embedding-001

    Raises:
        ValueError: If Google API key is not configured
    """
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    logger.debug(f"Creating embeddings client for model {EMBEDDING_MODEL}")

    return GoogleGenerativeAIEmbeddings(
        google_api_key=settings.google_api_key,
        model=EMBEDDING_MODEL,
    )


@retry(
    retry=retry_if_exception_type((ResourceExhausted, GoogleAPIError)),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True
)
def embed_text(text: str) -> List[float]:
    """
    Embed a single piece of text into a vector.
    Uses SHA-256 cache key to avoid redundant API calls for repeated queries.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding (768 dimensions)

    Raises:
        ValueError: If text is empty
        EmbeddingException: If API call fails after retries
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    # Check cache first
    cache_key = hashlib.sha256(text.encode()).hexdigest()
    cache = get_embedding_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Embedding cache hit")
        return cached.tolist() if hasattr(cached, 'tolist') else list(cached)

    try:
        logger.debug(f"Embedding text of length {len(text)}")
        embeddings = get_embeddings_client()
        embedding = embeddings.embed_query(text, output_dimensionality=EMBEDDING_DIMENSIONS)

        if not embedding:
            raise ValueError("Google returned empty embedding")

        # Store in cache (float32 halves memory: ~3KB vs ~6KB per entry)
        import numpy as np
        cache.put(cache_key, np.array(embedding, dtype=np.float32))

        logger.debug(f"Successfully embedded text, dimension: {len(embedding)}")
        return embedding

    except ValueError:
        raise
    except (ResourceExhausted, GoogleAPIError):
        # Let tenacity handle these
        raise
    except Exception as e:
        logger.error(f"Google AI API error during text embedding: {str(e)}")
        raise EmbeddingException(
            "Failed to embed text with Google AI",
            detail=f"Google AI error: {str(e)}"
        ) from e


@retry(
    retry=retry_if_exception_type((ResourceExhausted, GoogleAPIError)),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
    reraise=True
)
def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed multiple chunks of text into vectors.

    Args:
        chunks: List of text chunks to embed

    Returns:
        List of embeddings (each 768 dimensions)

    Raises:
        ValueError: If chunks list is empty or contains empty strings
        EmbeddingException: If API call fails after retries
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Validate all chunks
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            raise ValueError(f"Chunk at index {i} is empty")

    try:
        import numpy as np
        cache = get_embedding_cache()

        # Check cache for each chunk — only send uncached ones to the API
        results = [None] * len(chunks)
        uncached_indices = []
        uncached_texts = []

        for i, chunk in enumerate(chunks):
            cache_key = hashlib.sha256(chunk.encode()).hexdigest()
            cached = cache.get(cache_key)
            if cached is not None:
                results[i] = cached.tolist() if hasattr(cached, 'tolist') else list(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(chunk)

        cache_hits = len(chunks) - len(uncached_texts)
        if cache_hits > 0:
            logger.info(f"Embedding cache: {cache_hits}/{len(chunks)} hits, {len(uncached_texts)} to embed")

        # Call API only for uncached chunks
        if uncached_texts:
            logger.info(f"Embedding {len(uncached_texts)} chunks via API")
            embeddings_client = get_embeddings_client()
            api_embeddings = embeddings_client.embed_documents(uncached_texts, output_dimensionality=EMBEDDING_DIMENSIONS)

            if not api_embeddings or len(api_embeddings) != len(uncached_texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(uncached_texts)}, "
                    f"got {len(api_embeddings) if api_embeddings else 0}"
                )

            # Store in cache and assemble results
            for idx, embedding in zip(uncached_indices, api_embeddings):
                results[idx] = embedding
                cache_key = hashlib.sha256(chunks[idx].encode()).hexdigest()
                cache.put(cache_key, np.array(embedding, dtype=np.float32))

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return results

    except ValueError:
        raise
    except (ResourceExhausted, GoogleAPIError):
        # Let tenacity handle these
        raise
    except Exception as e:
        logger.error(f"Google AI API error during chunk embedding: {str(e)}")
        raise EmbeddingException(
            "Failed to embed chunks with Google AI",
            detail=f"Google AI error: {str(e)}"
        ) from e
