"""
Text extraction and chunking service

Handles:
- Extracting text from various file formats (TXT, PDF, DOCX)
- Cleaning extracted text
- Splitting text into overlapping chunks
- Creating DocumentChunk database records
"""

import logging
import re
from typing import List
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DocumentChunk

logger = logging.getLogger(__name__)


# ===== FUNCTION 1: extract_file =====
async def extract_file(file_path: str) -> str:
    """
    Extract text from file based on file type

    Args:
        file_path: Path to file on filesystem

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file doesn't exist
    """

    # Phase 3: TXT files only
    if file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            logger.info(f"[extraction] Extracted {len(text)} chars from {file_path}")
            return text
        except FileNotFoundError:
            logger.error(f"[extraction] File not found: {file_path}")
            raise
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


# ===== FUNCTION 2: clean_text =====
def clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts and normalizing whitespace

    Removes:
    - NULL bytes (PostgreSQL UTF-8 incompatibility)
    - Control characters (except \n, \r, \t)
    - Extra newlines
    - Page numbers (OCR artifacts)
    - Form feeds

    Normalizes:
    - Line endings (CRLF → LF)
    - Spaces (multiple → single)

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """
    if not text:
        return text

    # Remove NULL bytes
    text = text.replace('\x00', '')

    # Normalize line endings (CRLF to LF) - do this early
    text = text.replace('\r\n', '\n')

    # Convert form feeds to newlines - before control char removal
    text = text.replace('\f', '\n')

    # Remove control characters (except \n=0x0A, \r=0x0D, \t=0x09)
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    # Remove page numbers (line with only digits/spaces) - remove the entire line including newlines
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

    # Remove page headers (line starting with "Page") - remove the entire line including newlines
    text = re.sub(r'\nPage\s+\d+.*\n', '\n', text)

    # Normalize multiple newlines to double newlines (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize multiple spaces to single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Final trim
    text = text.strip()

    logger.debug(f"[extraction] Cleaned text: {len(text)} chars")
    return text


# ===== FUNCTION 3: chunk_text =====
def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
    min_size: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks

    Algorithm:
    1. Start at position 0
    2. Extract chunk_size characters
    3. Trim whitespace from end
    4. If chunk >= min_size: keep it (or if it's the first chunk and we haven't collected any yet)
    5. Move start back by overlap
    6. Repeat until end of text

    Args:
        text: Text to chunk
        chunk_size: Max chars per chunk (default 4000)
        overlap: Char overlap between chunks (default 400)
        min_size: Minimum chunk size to keep (default 200)

    Returns:
        List of text chunks
    """
    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        slice_text = text[start:end].rstrip()

        # Keep chunk if it meets min_size OR if it's the only remaining content
        if len(slice_text) >= min_size or end >= text_len:
            chunks.append(slice_text)
            logger.debug(
                f"[chunking] Created chunk {len(chunks)}: "
                f"{len(slice_text)} chars at position {start}"
            )

        next_start = end - overlap
        if next_start <= start:
            break

        start = next_start

    logger.info(
        f"[chunking] Split {text_len} chars into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )

    return chunks


# ===== FUNCTION 4: create_text_chunks =====
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


# Stub implementations for other file types (to be implemented in Phase 3)
async def extract_pdf(file_path: str) -> str:
    """Extract text from PDF file (not yet implemented)"""
    raise NotImplementedError("PDF extraction coming in Phase 3")


async def extract_docx(file_path: str) -> str:
    """Extract text from DOCX file (not yet implemented)"""
    raise NotImplementedError("DOCX extraction coming in Phase 3")


async def extract_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    return await extract_file(file_path)
