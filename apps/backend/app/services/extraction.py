"""
Text extraction and chunking service

Re-exports shared extraction utilities and provides database integration.

Shared utilities:
- extract_file() - Universal file extractor (auto-dispatches to PDF/DOCX/TXT)
- extract_pdf() - PDF text extraction
- extract_docx() - DOCX text extraction
- extract_txt() - TXT file extraction
- chunk_text() - Text chunking with overlap
- clean_text() - Text cleaning and normalization

Backend-specific:
- create_text_chunks() - Creates DocumentChunk DB records from text
"""

import logging
from typing import List
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from shared import (
    chunk_text,
    clean_text,
    extract_file,
    extract_pdf,
    extract_docx,
    extract_txt,
)
from shared.models import DocumentChunk

logger = logging.getLogger(__name__)


async def create_text_chunks(
    document_id: str,
    text: str,
    session: AsyncSession,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[str]:
    """
    Split text into chunks and create DocumentChunk records in database

    Args:
        document_id: ID of parent document
        text: Text to split
        session: SQLAlchemy async session
        chunk_size: Max chars per chunk
        overlap: Char overlap between chunks

    Returns:
        List of created chunk IDs

    Raises:
        Exception: If database insert fails
    """
    text = clean_text(text)

    if not text:
        logger.info(f"[chunks] No text to chunk for document {document_id}")
        return []

    chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunk_texts:
        logger.warning(f"[chunks] Text split resulted in no chunks for {document_id}")
        return []

    chunk_ids = []

    for chunk_index, chunk_text_item in enumerate(chunk_texts):
        chunk = DocumentChunk(
            id=str(uuid4()),
            document_id=document_id,
            chunk_text=chunk_text_item,
            chunk_index=chunk_index,
        )
        session.add(chunk)
        chunk_ids.append(chunk.id)

        logger.debug(
            f"[chunks] Created chunk {chunk_index}: {len(chunk_text_item)} chars for doc {document_id}"
        )

    try:
        await session.flush()
        logger.info(
            f"[chunks] Created {len(chunk_ids)} chunks for document {document_id}"
        )
        return chunk_ids
    except Exception as e:
        logger.error(f"[chunks] Failed to create chunks for {document_id}: {e}")
        raise


# Re-export shared extraction functions for convenience
__all__ = [
    "extract_file",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "chunk_text",
    "clean_text",
    "create_text_chunks",
]
