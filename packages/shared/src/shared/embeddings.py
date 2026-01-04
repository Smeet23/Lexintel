"""OpenAI embedding generation utilities."""

from typing import List
import logging
from openai import AsyncOpenAI, APIError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

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
        List of embedding vectors (1536-dimensional for text-embedding-3-small)

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


async def create_chunk_embeddings(
    session: AsyncSession,
    document_id: str,
    chunk_texts: List[str],
    batch_size: int = 20,
    model: str = "text-embedding-3-small",
) -> List[str]:
    """Generate embeddings for chunks and store in database.

    Args:
        session: AsyncSession for database operations
        document_id: ID of document these chunks belong to
        chunk_texts: List of text chunks to embed
        batch_size: How many chunks to embed per API call (20-25 recommended)
        model: OpenAI embedding model to use

    Returns:
        List of chunk IDs that had embeddings generated

    Raises:
        ValueError: If chunk_texts is empty
        APIError: If OpenAI API fails
    """
    if not chunk_texts:
        raise ValueError("No chunk texts provided for embedding")

    # Import DocumentChunk here to avoid circular imports
    from .models import DocumentChunk

    chunk_ids = []
    total_chunks = len(chunk_texts)

    logger.info(
        f"Starting embedding generation for {total_chunks} chunks",
        extra={"document_id": document_id, "batch_size": batch_size}
    )

    # Process in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_texts = chunk_texts[batch_start:batch_end]

        logger.debug(
            f"Processing batch {batch_start//batch_size + 1} "
            f"({batch_start}-{batch_end}/{total_chunks})"
        )

        # Generate embeddings for this batch
        embeddings = await generate_embeddings_batch(batch_texts, model=model)

        # Fetch chunks for this batch from database
        stmt = (
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .offset(batch_start)
            .limit(batch_size)
        )
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunk count mismatch: got {len(chunks)} chunks "
                f"but {len(embeddings)} embeddings"
            )

        # Update chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            chunk_ids.append(chunk.id)

        # Flush to database after each batch
        await session.flush()

    # Commit all changes
    await session.commit()

    logger.info(
        f"Successfully generated embeddings for {len(chunk_ids)} chunks",
        extra={"document_id": document_id}
    )

    return chunk_ids


__all__ = ["generate_embeddings_batch", "create_chunk_embeddings"]
