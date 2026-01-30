"""Document chunking service for PDFs, DOCX, and TXT files"""
import logging
import tempfile
import os
from typing import List, Dict
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

try:
    from backend.services.text_extraction import extract_text
except ImportError:
    try:
        from services.text_extraction import extract_text
    except ImportError:
        from .text_extraction import extract_text

logger = logging.getLogger(__name__)

# Chunking configuration
CHUNK_SIZE = 1500  # Characters per chunk (~200-250 words, captures full paragraphs)
CHUNK_OVERLAP = 300  # Characters to overlap between chunks (maintains context)
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Ordered split preferences


def chunk_pdf(pdf_path: str) -> List[Dict[str, str]]:
    """
    Chunk a PDF file into semantic pieces with metadata.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of chunk dicts with keys: content, page_num, section_name

    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF parsing fails
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    try:
        # Load PDF using PyMuPDF (fitz)
        logger.info(f"Loading PDF from {pdf_path}")
        pdf_document = fitz.open(pdf_path)

        if pdf_document.page_count == 0:
            raise ValueError("PDF contains no pages or failed to load")

        logger.info(f"Loaded {pdf_document.page_count} pages from PDF")

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )

        chunks = []
        chunk_counter = 0

        # Extract text from each page
        for page_idx in range(pdf_document.page_count):
            logger.debug(f"Processing page {page_idx + 1}/{pdf_document.page_count}")

            # Extract text from page
            page = pdf_document[page_idx]
            page_text = page.get_text()

            if not page_text.strip():
                continue

            # Create a Document object for the page
            page_doc = Document(
                page_content=page_text,
                metadata={"page": page_idx, "source": pdf_path}
            )

            # Split single page into chunks
            split_docs = splitter.split_documents([page_doc])

            for doc in split_docs:
                chunk_counter += 1

                # Extract page number from metadata (0-indexed, convert to 1-indexed)
                page_num = doc.metadata.get("page", page_idx)

                chunk_dict = {
                    "content": doc.page_content,
                    "page_num": str(page_num + 1),  # Convert to 1-indexed
                    "section_name": f"Chunk {chunk_counter}"
                }
                chunks.append(chunk_dict)

        pdf_document.close()
        logger.info(f"Created {chunk_counter} chunks from PDF")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking PDF: {str(e)}")
        raise


def chunk_document_from_blob(blob_content: bytes, file_type: str = "pdf") -> List[Dict[str, str]]:
    """
    Chunk document (PDF, DOCX, or TXT) from blob storage content (bytes).

    Args:
        blob_content: Raw document bytes
        file_type: Document type ('pdf', 'docx', or 'txt'). Default: 'pdf'

    Returns:
        List of chunk dicts with keys: content, page_num (or para/line info), section_name

    Raises:
        ValueError: If blob_content is empty or invalid
        Exception: If chunking fails
    """
    if not blob_content:
        raise ValueError("Blob content is empty")

    try:
        logger.info(f"Chunking {file_type.upper()} document from blob ({len(blob_content)} bytes)")

        # Extract text using appropriate extractor
        extracted_sections = extract_text(blob_content, file_type)

        if not extracted_sections:
            logger.warning(f"No content extracted from {file_type.upper()} document")
            return []

        # Initialize text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=SEPARATORS
        )

        all_chunks = []
        chunk_counter = 0

        # Process each section (page/paragraph/line range)
        for section in extracted_sections:
            # Split section content into chunks
            split_docs = splitter.split_text(section["content"])

            for chunk_text in split_docs:
                if chunk_text.strip():  # Skip empty chunks
                    chunk_counter += 1
                    all_chunks.append({
                        "content": chunk_text,
                        "page_num": section["location"],  # Now stores page/para/line range
                        "section_name": f"Chunk {chunk_counter}"
                    })

        logger.info(f"Created {chunk_counter} chunks from {file_type.upper()}")
        return all_chunks

    except Exception as e:
        logger.error(f"Error chunking {file_type.upper()}: {str(e)}")
        raise


def chunk_pdf_from_blob(blob_content: bytes) -> List[Dict[str, str]]:
    """
    Deprecated: Use chunk_document_from_blob() instead.
    Chunk PDF from blob storage content (bytes).

    Args:
        blob_content: Raw PDF bytes

    Returns:
        List of chunk dicts

    Raises:
        ValueError: If blob_content is empty or invalid
        Exception: If chunking fails
    """
    if not blob_content:
        raise ValueError("Blob content is empty")

    # Write to temporary file for processing
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(blob_content)
            temp_file = tmp.name

        logger.info(f"Chunking PDF from blob ({len(blob_content)} bytes)")
        chunks = chunk_pdf(temp_file)
        return chunks

    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
                logger.debug(f"Cleaned up temporary file: {temp_file}")
            except OSError as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {str(e)}")


def estimate_tokens(chunk_content: str) -> int:
    """
    Estimate token count for a chunk (rough approximation).

    OpenAI uses roughly 1 token per 4 characters for English text.

    Args:
        chunk_content: Text content to estimate

    Returns:
        Approximate token count
    """
    return len(chunk_content) // 4
