"""Embedding service with dual-provider support: Voyage-law-2 (legal-specific) + Cohere (fallback).

When VOYAGE_API_KEY is set: Uses voyage-law-2 (1024-dim, +20% NDCG on legal text).
When VOYAGE_API_KEY is not set: Falls back to Cohere embed-english-v3.0 (1024-dim).

Both produce 1024-dimensional vectors — zero Qdrant schema changes needed.
"""
import hashlib
import logging
import threading
from functools import lru_cache
from typing import List, Optional

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

# Model configuration
EMBEDDING_DIMENSIONS = 1024

# Provider detection
_PROVIDER = None  # "voyage" or "cohere"
_PROVIDER_LOCK = threading.Lock()


def _reset_provider_for_testing() -> None:
    """Reset cached provider detection so tests that mutate VOYAGE_API_KEY see fresh results.

    TEST-ONLY — do not call in production code.
    """
    global _PROVIDER
    with _PROVIDER_LOCK:
        _PROVIDER = None


def _detect_provider() -> str:
    """Detect which embedding provider to use based on available API keys."""
    global _PROVIDER
    if _PROVIDER is not None:
        return _PROVIDER
    with _PROVIDER_LOCK:
        if _PROVIDER is not None:  # double-check after acquiring lock
            return _PROVIDER
        voyage_key = getattr(settings, 'voyage_api_key', '')
        if voyage_key:
            try:
                import voyageai  # noqa: F401
                _PROVIDER = "voyage"
                logger.info("Embedding provider: Voyage voyage-law-2 (legal-specific)")
                return _PROVIDER
            except ImportError:
                logger.warning("VOYAGE_API_KEY set but voyageai not installed. Falling back to Cohere.")

        _PROVIDER = "cohere"
        logger.info("Embedding provider: Cohere embed-english-v3.0 (general-purpose)")
        return _PROVIDER


def model_name_for_provider(provider: str) -> str:
    """Map a provider id ("voyage"/"cohere") to its concrete model name."""
    return "voyage-law-2" if provider == "voyage" else "embed-english-v3.0"


def active_embedding_model() -> str:
    """Return the model name of the currently-active embedding provider.

    Used to stamp chunks/matters with provenance at ingestion so the re-index
    drift report (GET /admin/embedding-status) is accurate.
    """
    return model_name_for_provider(_detect_provider())


def _cache_key(text: str, *, kind: str = "document", provider: Optional[str] = None) -> str:
    """Build a provider- and role-scoped embedding cache key.

    Prefixing with the provider prevents a warm cache from serving a vector
    produced by a different provider (e.g. a Cohere vector when the system has
    switched to Voyage), which would silently corrupt similarity. The `kind`
    ("document"/"query") keeps asymmetric embeddings separate.

    When ``provider`` is given it overrides auto-detection — this lets the
    re-index path cache forced-Voyage vectors under a ``voyage:`` key rather
    than a detected-``cohere:`` key (avoids cache poisoning across providers).
    """
    resolved = provider or _detect_provider()
    return f"{resolved}:{kind}:" + hashlib.sha256(text.encode()).hexdigest()


# ═══════════════════════════════════════════════════════════════
# VOYAGE CLIENT (legal-specific embeddings)
# ═══════════════════════════════════════════════════════════════

_VOYAGE_CLIENT = None
_VOYAGE_CLIENT_LOCK = threading.Lock()


def _get_voyage_client():
    """Get Voyage AI client singleton."""
    global _VOYAGE_CLIENT
    if _VOYAGE_CLIENT is not None:
        return _VOYAGE_CLIENT
    with _VOYAGE_CLIENT_LOCK:
        if _VOYAGE_CLIENT is not None:  # double-check after acquiring lock
            return _VOYAGE_CLIENT
        import voyageai
        voyage_key = getattr(settings, 'voyage_api_key', '')
        _VOYAGE_CLIENT = voyageai.Client(api_key=voyage_key)
        return _VOYAGE_CLIENT


def _voyage_embed_texts(texts: List[str], input_type: str = "document") -> List[List[float]]:
    """Embed texts using Voyage voyage-law-2."""
    client = _get_voyage_client()
    result = client.embed(
        texts,
        model="voyage-law-2",
        input_type=input_type,  # "document" or "query"
    )
    return result.embeddings


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _voyage_embed_single_batch(client, batch: List[str], input_type: str) -> List[List[float]]:
    """Embed one Voyage batch (<=128 texts) with retry.

    Bulk re-index hammers the Voyage API — 429s are common, so each
    sub-batch call is wrapped in exponential-backoff retry (3 attempts).
    """
    result = client.embed(batch, model="voyage-law-2", input_type=input_type)
    if not result.embeddings:
        raise ValueError("Voyage returned empty embeddings")
    return result.embeddings


def _voyage_embed_batch_with_retry(texts: List[str], input_type: str = "document") -> List[List[float]]:
    """Batch embed with retry for Voyage API."""
    client = _get_voyage_client()
    # Voyage allows up to 128 texts per call
    batch_size = 128
    all_embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        all_embeddings.extend(_voyage_embed_single_batch(client, batch, input_type))
    return all_embeddings


# ═══════════════════════════════════════════════════════════════
# COHERE CLIENT (general-purpose fallback)
# ═══════════════════════════════════════════════════════════════

COHERE_MODEL = "embed-english-v3.0"


@lru_cache(maxsize=1)
def get_cohere_client():
    """Get Cohere client singleton."""
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY environment variable not set")
    import cohere
    try:
        return cohere.Client(api_key=settings.cohere_api_key)
    except TypeError:
        return cohere.Client(token=settings.cohere_api_key)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _cohere_embed_batch_with_retry(client, texts: List[str], input_type: str = "search_document") -> List[List[float]]:
    """Embed a batch with Cohere with retry."""
    response = client.embed(texts=texts, model=COHERE_MODEL, input_type=input_type)
    if not response.embeddings:
        raise ValueError("Cohere returned empty embeddings")
    return response.embeddings


# ═══════════════════════════════════════════════════════════════
# PUBLIC API (provider-agnostic)
# ═══════════════════════════════════════════════════════════════

def embed_text(text: str) -> List[float]:
    """
    Embed a single document text into a 1024-dim vector.

    Uses Voyage voyage-law-2 when VOYAGE_API_KEY is set (legal-specific, +20% NDCG).
    Falls back to Cohere embed-english-v3.0 otherwise.
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    cache_key = _cache_key(text, kind="document")
    cache = get_embedding_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Embedding cache hit")
        return cached.tolist() if hasattr(cached, "tolist") else list(cached)

    try:
        provider = _detect_provider()

        if provider == "voyage":
            embeddings = _voyage_embed_texts([text], input_type="document")
            embedding = embeddings[0]
        else:
            client = get_cohere_client()
            response = client.embed(texts=[text], model=COHERE_MODEL, input_type="search_document")
            if not response.embeddings or len(response.embeddings) != 1:
                raise ValueError("Cohere returned empty or invalid embedding")
            embedding = response.embeddings[0]

        import numpy as np
        cache.put(cache_key, np.array(embedding, dtype=np.float32))
        logger.debug(f"Embedded text ({provider}), dimension: {len(embedding)}")
        return embedding

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Embedding API error: {type(e).__name__}: {str(e)[:100]}")
        raise EmbeddingException(
            f"Failed to embed text with {_detect_provider()}",
            detail=str(e),
        ) from e


def embed_query(text: str) -> List[float]:
    """
    Embed a search query into a 1024-dim vector.

    Uses asymmetric embedding: "query" input type for better retrieval.
    """
    if not text or not text.strip():
        raise ValueError("Query text cannot be empty")

    cache_key = _cache_key(text, kind="query")
    cache = get_embedding_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug("Query embedding cache hit")
        return cached.tolist() if hasattr(cached, "tolist") else list(cached)

    try:
        provider = _detect_provider()

        if provider == "voyage":
            embeddings = _voyage_embed_texts([text], input_type="query")
            embedding = embeddings[0]
        else:
            client = get_cohere_client()
            response = client.embed(texts=[text], model=COHERE_MODEL, input_type="search_query")
            if not response.embeddings or len(response.embeddings) != 1:
                raise ValueError("Cohere returned empty or invalid embedding")
            embedding = response.embeddings[0]

        import numpy as np
        cache.put(cache_key, np.array(embedding, dtype=np.float32))
        return embedding

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Query embedding API error: {type(e).__name__}: {str(e)[:100]}")
        raise EmbeddingException(
            f"Failed to embed query with {_detect_provider()}",
            detail=str(e),
        ) from e


def embed_chunks_with_provider(chunks: List[str], provider: str) -> List[List[float]]:
    """
    Embed multiple chunks into 1024-dim vectors with caching, forcing a
    specific provider ("voyage" or "cohere") regardless of auto-detection.

    This bypasses ``_detect_provider()`` so the re-index path can deliberately
    embed with the target provider (e.g. Voyage) even when the process started
    up detecting Cohere — both clients can coexist in one process. Cache keys
    are scoped to the explicit provider so forced-Voyage vectors never collide
    with detected-Cohere vectors.

    Uses batch API for efficiency.
    Voyage: up to 128 texts per call.
    Cohere: up to 96 texts per call.
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    if provider not in ("voyage", "cohere"):
        raise ValueError(f"Unknown embedding provider: {provider!r} (expected 'voyage' or 'cohere')")

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
            cache_key = _cache_key(chunk, kind="document", provider=provider)
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
            logger.info(f"Embedding {len(uncached_texts)} chunks via {provider}")

            if provider == "voyage":
                all_embeddings = _voyage_embed_batch_with_retry(uncached_texts, input_type="document")
            else:
                client = get_cohere_client()
                batch_size = 96
                all_embeddings = []
                for start in range(0, len(uncached_texts), batch_size):
                    batch = uncached_texts[start:start + batch_size]
                    batch_embeddings = _cohere_embed_batch_with_retry(client, batch)
                    all_embeddings.extend(batch_embeddings)

            if len(all_embeddings) != len(uncached_texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(uncached_texts)}, got {len(all_embeddings)}"
                )

            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                cache_key = _cache_key(chunks[idx], kind="document", provider=provider)
                cache.put(cache_key, np.array(embedding, dtype=np.float32))

        logger.info(f"Successfully embedded {len(chunks)} chunks via {provider}")
        return results

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Chunk embedding API error: {type(e).__name__}: {str(e)[:100]}")
        raise EmbeddingException(
            f"Failed to embed chunks with {provider}",
            detail=str(e),
        ) from e


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed multiple chunks into 1024-dim vectors with caching.

    Thin wrapper over :func:`embed_chunks_with_provider` using the
    auto-detected provider — zero behavioural change for existing callers.

    Uses batch API for efficiency.
    Voyage: up to 128 texts per call.
    Cohere: up to 96 texts per call.
    """
    return embed_chunks_with_provider(chunks, _detect_provider())
