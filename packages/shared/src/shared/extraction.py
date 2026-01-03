"""Text extraction and chunking utilities.

Handles extraction from PDF, DOCX, and TXT formats with text cleaning
and chunking capabilities. Uses optional dependencies PyPDF2 and python-docx.
"""

import logging
import re
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


# ===== CHUNKING AND CLEANING =====
def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
    min_size: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Algorithm:
    1. Start at position 0
    2. Extract chunk_size characters
    3. Trim whitespace from end
    4. If chunk >= min_size: keep it (or if it's the last chunk)
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


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts and normalizing whitespace.

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


# ===== FILE FORMAT EXTRACTORS =====
async def extract_pdf(file_path: str) -> str:
    """
    Extract text from PDF file using PyPDF2.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If PyPDF2 is not installed
        ValueError: If PDF is corrupted or unreadable
    """
    try:
        from PyPDF2 import PdfReader
    except ImportError as e:
        raise ImportError("PyPDF2 not installed. Install with: pip install lex-intel-shared[extraction]") from e

    path = Path(file_path)
    if not path.exists():
        logger.error(f"[extraction] PDF file not found: {file_path}")
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            text_parts = []

            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                else:
                    logger.warning(f"[extraction] No text found on PDF page {page_num + 1}: {file_path}")

            text = "\n".join(text_parts)
            logger.info(f"[extraction] Extracted {len(text)} chars from PDF: {file_path}")
            return text

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract PDF {file_path}: {e}")
        raise ValueError(f"Failed to extract PDF {file_path}: {e}") from e


async def extract_docx(file_path: str) -> str:
    """
    Extract text from DOCX file using python-docx.

    Args:
        file_path: Path to DOCX file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If python-docx is not installed
        ValueError: If DOCX is corrupted or unreadable
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx not installed. Install with: pip install lex-intel-shared[extraction]") from e

    path = Path(file_path)
    if not path.exists():
        logger.error(f"[extraction] DOCX file not found: {file_path}")
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        doc = Document(file_path)
        text_parts = []

        for para in doc.paragraphs:
            if para.text.strip():
                text_parts.append(para.text)

        text = "\n".join(text_parts)
        logger.info(f"[extraction] Extracted {len(text)} chars from DOCX: {file_path}")
        return text

    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract DOCX {file_path}: {e}")
        raise ValueError(f"Failed to extract DOCX {file_path}: {e}") from e


async def extract_txt(file_path: str) -> str:
    """
    Extract text from TXT file.

    Supports UTF-8 with fallback to latin-1 for different encodings.

    Args:
        file_path: Path to TXT file

    Returns:
        Extracted text content

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"[extraction] TXT file not found: {file_path}")
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    try:
        # Try UTF-8 first
        text = path.read_text(encoding='utf-8')
        logger.info(f"[extraction] Extracted {len(text)} chars from TXT: {file_path}")
        return text
    except UnicodeDecodeError:
        # Fallback to latin-1 for files with different encoding
        text = path.read_text(encoding='latin-1')
        logger.warning(f"[extraction] Used latin-1 encoding for TXT: {file_path}")
        return text
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"[extraction] Failed to extract TXT {file_path}: {e}")
        raise


async def extract_file(file_path: str) -> str:
    """
    Extract text from any supported file format.

    Auto-detects file type by extension and dispatches to appropriate extractor.

    Args:
        file_path: Path to file

    Returns:
        Extracted text

    Raises:
        ValueError: If file type not supported
        FileNotFoundError: If file doesn't exist
    """
    file_lower = file_path.lower()

    if file_lower.endswith('.pdf'):
        return await extract_pdf(file_path)
    elif file_lower.endswith('.docx'):
        return await extract_docx(file_path)
    elif file_lower.endswith('.txt'):
        return await extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


__all__ = [
    "chunk_text",
    "clean_text",
    "extract_file",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
]
