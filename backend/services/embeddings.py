"""OpenAI embeddings service for text and chunk embedding"""
import logging
from functools import lru_cache
from typing import List
from langchain_community.embeddings import OpenAIEmbeddings
from backend.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Embeddings configuration
EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072  # Output dimension for text-embedding-3-large


@lru_cache(maxsize=1)
def get_embeddings_client() -> OpenAIEmbeddings:
    """
    Get OpenAI embeddings client with configured model.

    Returns:
        OpenAIEmbeddings instance for text-embedding-3-large

    Raises:
        ValueError: If OpenAI API key is not configured
    """
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    logger.debug(f"Creating embeddings client for model {EMBEDDING_MODEL}")

    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=EMBEDDING_MODEL
    )


def embed_text(text: str) -> List[float]:
    """
    Embed a single piece of text into a vector.

    Args:
        text: Text to embed

    Returns:
        List of floats representing the embedding (3072 dimensions)

    Raises:
        ValueError: If text is empty
        Exception: If API call fails
    """
    if not text or not text.strip():
        raise ValueError("Text cannot be empty")

    try:
        logger.debug(f"Embedding text of length {len(text)}")
        embeddings = get_embeddings_client()
        embedding = embeddings.embed_query(text)

        if not embedding:
            raise ValueError("OpenAI returned empty embedding")

        logger.debug(f"Successfully embedded text, dimension: {len(embedding)}")
        return embedding

    except Exception as e:
        logger.error(f"Failed to embed text: {str(e)}")
        raise


def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed multiple chunks of text into vectors.

    Args:
        chunks: List of text chunks to embed

    Returns:
        List of embeddings (each 3072 dimensions)

    Raises:
        ValueError: If chunks list is empty or contains empty strings
        Exception: If API call fails
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Validate all chunks
    for i, chunk in enumerate(chunks):
        if not chunk or not chunk.strip():
            raise ValueError(f"Chunk at index {i} is empty")

    try:
        logger.info(f"Embedding {len(chunks)} chunks")
        embeddings = get_embeddings_client()
        embeddings_list = embeddings.embed_documents(chunks)

        if not embeddings_list:
            raise ValueError("OpenAI returned empty embeddings list")

        if len(embeddings_list) != len(chunks):
            raise ValueError(
                f"Embedding count mismatch: expected {len(chunks)}, got {len(embeddings_list)}"
            )

        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return embeddings_list

    except Exception as e:
        logger.error(f"Failed to embed chunks: {str(e)}")
        raise


def estimate_embedding_cost(text_length: int) -> float:
    """
    Estimate the cost of embedding text with text-embedding-3-large.

    OpenAI charges $0.02 per 1M tokens for text-embedding-3-large.
    Rough estimate: 1 token ≈ 4 characters.

    Args:
        text_length: Total character length of text to embed

    Returns:
        Estimated cost in USD
    """
    # Estimate tokens (1 token per 4 characters)
    tokens = text_length // 4

    # Cost per 1M tokens
    cost_per_million = 0.02

    # Calculate cost
    cost = (tokens / 1_000_000) * cost_per_million

    return cost
