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
        raise ValueError(
            "OPENAI_API_KEY not configured. "
            "Please set OPENAI_API_KEY environment variable"
        )

    # Validate inputs before API call
    for i, text in enumerate(texts):
        if len(text) > 8000:
            logger.warning(
                f"Text at index {i} exceeds 8000 chars ({len(text)}), "
                f"OpenAI may truncate it"
            )

    client = AsyncOpenAI(api_key=api_key)

    try:
        logger.debug(
            f"Calling OpenAI embeddings API",
            extra={
                "model": model,
                "num_texts": len(texts),
                "total_chars": sum(len(t) for t in texts)
            }
        )
        response = await client.embeddings.create(
            model=model,
            input=texts,
        )
        logger.debug(
            f"OpenAI API success",
            extra={
                "embeddings_returned": len(response.data),
                "model_used": response.model
            }
        )
    except APIError as e:
        error_msg = f"OpenAI API error: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise
    except Exception as e:
        error_msg = f"Unexpected error calling OpenAI API: {type(e).__name__}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise

    # Extract embeddings in order (matches input order)
    if len(response.data) != len(texts):
        raise ValueError(
            f"OpenAI API returned {len(response.data)} embeddings "
            f"but {len(texts)} texts were provided"
        )

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

    # Fetch ALL chunks for this document once to ensure consistency
    # This prevents offset/limit mismatches and ensures correct ordering
    stmt = (
        select(DocumentChunk)
        .where(DocumentChunk.document_id == document_id)
        .order_by(DocumentChunk.chunk_index)
    )
    result = await session.execute(stmt)
    all_chunks = result.scalars().all()

    if len(all_chunks) != total_chunks:
        raise ValueError(
            f"Chunk count mismatch in database: expected {total_chunks} chunks "
            f"but found {len(all_chunks)} chunks for document {document_id}"
        )

    logger.debug(
        f"Fetched {len(all_chunks)} chunks from database for embedding",
        extra={"document_id": document_id}
    )

    # Process in batches
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        batch_texts = chunk_texts[batch_start:batch_end]
        batch_chunks = all_chunks[batch_start:batch_end]

        logger.debug(
            f"Processing batch {batch_start//batch_size + 1} "
            f"({batch_start}-{batch_end}/{total_chunks})"
        )

        try:
            # Generate embeddings for this batch
            embeddings = await generate_embeddings_batch(batch_texts, model=model)
            logger.debug(
                f"Generated {len(embeddings)} embeddings for batch",
                extra={"batch_start": batch_start, "batch_size": len(embeddings)}
            )

            # Verify count matches
            if len(embeddings) != len(batch_chunks):
                raise ValueError(
                    f"Embedding count mismatch in batch: got {len(batch_chunks)} chunks "
                    f"but {len(embeddings)} embeddings"
                )

            # Update chunks with embeddings using explicit vector casting
            for chunk, embedding in zip(batch_chunks, embeddings):
                if not chunk.embedding:
                    # Store embedding directly - pgvector SQLAlchemy handles type conversion
                    chunk.embedding = embedding
                    chunk_ids.append(chunk.id)
                else:
                    logger.warning(
                        f"Chunk {chunk.id} already has embedding, skipping",
                        extra={"document_id": document_id, "chunk_index": chunk.chunk_index}
                    )

            # Flush to database after each batch
            await session.flush()
            logger.info(
                f"Flushed batch with {len([c for c in batch_chunks if c.embedding is not None])} embeddings",
                extra={"batch_num": batch_start//batch_size + 1}
            )

        except Exception as e:
            logger.error(
                f"Failed to process batch {batch_start//batch_size + 1}: {e}",
                exc_info=True,
                extra={"document_id": document_id, "batch_start": batch_start}
            )
            raise

    # Commit all changes
    await session.commit()
    logger.info(
        f"Successfully committed embeddings for {len(chunk_ids)} chunks",
        extra={"document_id": document_id, "total_chunks": total_chunks}
    )

    return chunk_ids


__all__ = ["generate_embeddings_batch", "create_chunk_embeddings"]
