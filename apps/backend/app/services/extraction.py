"""
Text extraction and chunking service

Re-exports shared extraction utilities and provides a convenience wrapper
for the database-specific chunking operation.

Shared utilities (re-exported):
- extract_file() - Universal file extractor (auto-dispatches to PDF/DOCX/TXT)
- extract_pdf() - PDF text extraction
- extract_docx() - DOCX text extraction
- extract_txt() - TXT file extraction
- chunk_text() - Text chunking with overlap
- clean_text() - Text cleaning and normalization
- create_document_chunks() - Creates DocumentChunk DB records from text chunks

Backend convenience wrapper:
- create_text_chunks() - Combines extraction, cleaning, chunking, and DB insertion
"""

import logging
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession

from shared import (
    chunk_text,
    clean_text,
    extract_file,
    extract_pdf,
    extract_docx,
    extract_txt,
    create_document_chunks,
)

logger = logging.getLogger(__name__)


async def create_text_chunks(
    document_id: str,
    text: str,
    session: AsyncSession,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[str]:
    """
    Split text into chunks and create DocumentChunk records in database.

    Convenience wrapper that combines cleaning, chunking, and database insertion.

    Args:
        document_id: ID of parent document
        text: Text to split and clean
        session: SQLAlchemy async session
        chunk_size: Max chars per chunk (default 4000)
        overlap: Char overlap between chunks (default 400)

    Returns:
        List of created chunk IDs

    Raises:
        Exception: If database insert fails
    """
    # Clean the text first
    text = clean_text(text)

    if not text:
        logger.info(f"[chunks] No text to chunk for document {document_id}")
        return []

    # Split into chunks
    chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunk_texts:
        logger.warning(f"[chunks] Text split resulted in no chunks for {document_id}")
        return []

    # Create database records
    return await create_document_chunks(session, document_id, chunk_texts)


# Re-export shared extraction functions for convenience
__all__ = [
    "extract_file",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "chunk_text",
    "clean_text",
    "create_document_chunks",
    "create_text_chunks",
]
