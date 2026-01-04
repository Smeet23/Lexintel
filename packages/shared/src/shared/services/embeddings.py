"""Service for managing document chunk embeddings."""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from shared import generate_embeddings_batch, DocumentChunk, setup_logging

logger = setup_logging(__name__)


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


__all__ = ["create_chunk_embeddings"]
