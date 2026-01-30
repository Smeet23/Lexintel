"""Multi-format text extraction service for PDFs, DOCX, and TXT files using Docling"""
import logging
import tempfile
import os
from typing import List, Dict
from docling.document_converter import DocumentConverter
from pathlib import Path

logger = logging.getLogger(__name__)

# Initialize Docling converter once (expensive operation)
_CONVERTER = None


def _get_converter() -> DocumentConverter:
    """Get cached Docling converter instance"""
    global _CONVERTER
    if _CONVERTER is None:
        logger.info("Initializing Docling DocumentConverter")
        _CONVERTER = DocumentConverter()
    return _CONVERTER


def extract_pdf_text(file_bytes: bytes) -> List[Dict[str, str]]:
    """
    Extract text from PDF file using Docling.

    Args:
        file_bytes: Raw PDF bytes

    Returns:
        List of dicts with keys: content, location (page number), location_type

    Raises:
        ValueError: If PDF is invalid or empty
        Exception: If PDF parsing fails
    """
    if not file_bytes:
        raise ValueError("PDF content is empty")

    temp_file = None
    try:
        # Write to temporary file for Docling processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            temp_file = tmp.name

        logger.info(f"Extracting text from PDF using Docling ({len(file_bytes)} bytes)")

        converter = _get_converter()
        result = converter.convert_single(Path(temp_file))
        doc = result.document

        if not doc:
            raise ValueError("PDF failed to convert or contains no content")

        sections = []
        page_number = 1
        current_page_text = []

        # Extract text from document tree, grouping by page
        for node in doc.iter_pages():
            if current_page_text:
                page_content = "\n".join(current_page_text).strip()
                if page_content:
                    sections.append({
                        "content": page_content,
                        "location": str(page_number),
                        "location_type": "page"
                    })
                current_page_text = []
                page_number += 1

            # Extract text from page
            page_text = node.export_to_markdown()
            if page_text.strip():
                current_page_text.append(page_text)

        # Don't forget the last page
        if current_page_text:
            page_content = "\n".join(current_page_text).strip()
            if page_content:
                sections.append({
                    "content": page_content,
                    "location": str(page_number),
                    "location_type": "page"
                })

        logger.info(f"Extracted {len(sections)} pages from PDF")
        return sections

    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")


def extract_docx_text(file_bytes: bytes) -> List[Dict[str, str]]:
    """
    Extract text from DOCX (Word) file using Docling.

    Args:
        file_bytes: Raw DOCX bytes

    Returns:
        List of dicts with keys: content, location (section number), location_type

    Raises:
        ValueError: If DOCX is invalid or empty
        Exception: If DOCX parsing fails
    """
    if not file_bytes:
        raise ValueError("DOCX content is empty")

    temp_file = None
    try:
        # Write to temporary file for Docling processing
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_bytes)
            temp_file = tmp.name

        logger.info(f"Extracting text from DOCX using Docling ({len(file_bytes)} bytes)")

        converter = _get_converter()
        result = converter.convert_single(Path(temp_file))
        doc = result.document

        if not doc:
            raise ValueError("DOCX failed to convert or contains no content")

        sections = []
        section_counter = 0

        # Extract structured content from document
        for node in doc.iter_text_paragraphs():
            para_text = node.export_to_markdown().strip()

            if para_text:
                section_counter += 1
                sections.append({
                    "content": para_text,
                    "location": f"para {section_counter}",
                    "location_type": "paragraph"
                })

        logger.info(f"Extracted {len(sections)} sections from DOCX")
        return sections

    except Exception as e:
        logger.error(f"Error extracting DOCX text: {str(e)}")
        raise
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")


def extract_txt_text(file_bytes: bytes, lines_per_section: int = 50) -> List[Dict[str, str]]:
    """
    Extract text from TXT file.

    Note: Docling doesn't handle plain text files, so we use basic extraction.

    Args:
        file_bytes: Raw TXT bytes
        lines_per_section: Number of lines per section (default 50)

    Returns:
        List of dicts with keys: content, location (line range), location_type

    Raises:
        ValueError: If TXT is invalid or empty
        UnicodeDecodeError: If file is not valid UTF-8 text
    """
    if not file_bytes:
        raise ValueError("TXT content is empty")

    try:
        # Decode bytes to string
        text_content = file_bytes.decode('utf-8')
        logger.info(f"Extracting text from TXT ({len(file_bytes)} bytes)")

        lines = text_content.split('\n')
        logger.info(f"Loaded {len(lines)} lines from TXT")

        sections = []
        for i in range(0, len(lines), lines_per_section):
            section_lines = lines[i:i + lines_per_section]
            section_text = '\n'.join(section_lines).strip()

            if section_text:
                start_line = i + 1  # 1-indexed
                end_line = min(i + lines_per_section, len(lines))
                location = f"line {start_line}-{end_line}"

                sections.append({
                    "content": section_text,
                    "location": location,
                    "location_type": "line_range"
                })

        logger.info(f"Extracted {len(sections)} sections from TXT")
        return sections

    except UnicodeDecodeError as e:
        logger.error(f"TXT file is not valid UTF-8: {str(e)}")
        raise ValueError(f"TXT file must be valid UTF-8 encoded text: {str(e)}") from e
    except Exception as e:
        logger.error(f"Error extracting TXT text: {str(e)}")
        raise


def extract_text(file_bytes: bytes, file_type: str) -> List[Dict[str, str]]:
    """
    Route text extraction to appropriate handler based on file type.

    Uses Docling for PDFs and DOCX files for better structure understanding.
    Uses basic extraction for TXT files (Docling doesn't support plain text).

    Args:
        file_bytes: Raw file bytes
        file_type: File type string ('pdf', 'docx', or 'txt')

    Returns:
        List of dicts with content and location information

    Raises:
        ValueError: If file_type is unsupported or content is invalid
        Exception: If extraction fails
    """
    if not file_bytes:
        raise ValueError("File content is empty")

    if file_type == "pdf":
        return extract_pdf_text(file_bytes)
    elif file_type == "docx":
        return extract_docx_text(file_bytes)
    elif file_type == "txt":
        return extract_txt_text(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
