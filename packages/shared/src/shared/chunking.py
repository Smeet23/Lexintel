"""Document chunking utilities for creating and managing text chunks in database."""

import logging
from typing import List
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from .models import DocumentChunk

logger = logging.getLogger(__name__)


async def create_document_chunks(
    session: AsyncSession,
    document_id: str,
    chunk_texts: List[str],
) -> List[str]:
    """
    Create DocumentChunk records from text chunks.

    Centralized function to ensure consistent chunk creation across services.

    Args:
        session: SQLAlchemy async session
        document_id: ID of parent document
        chunk_texts: List of text chunks to save

    Returns:
        List of created chunk IDs

    Raises:
        Exception: If database insert fails
    """
    if not chunk_texts:
        logger.warning(f"[chunks] No chunks to create for document {document_id}")
        return []

    chunk_ids = []

    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk = DocumentChunk(
            id=str(uuid4()),
            document_id=document_id,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
        )
        session.add(chunk)
        chunk_ids.append(chunk.id)

        logger.debug(
            f"[chunks] Created chunk {chunk_index}: {len(chunk_text)} chars for doc {document_id}"
        )

    try:
        await session.flush()
        logger.info(f"[chunks] Created {len(chunk_ids)} chunks for document {document_id}")
        return chunk_ids
    except Exception as e:
        logger.error(f"[chunks] Failed to create chunks for {document_id}: {e}")
        raise


__all__ = ["create_document_chunks"]
