"""Hybrid semantic chunking service for legal documents.

Strategy (structure-first, then semantics):
1. If markdown content (from pymupdf4llm/DOCX): split on headers first
2. If plain text: use section_detector to find legal boundaries
3. Within each section: SemanticChunker splits by semantic similarity
4. Fallback: RecursiveCharacterTextSplitter if SemanticChunker fails
"""
import bisect
import os
import re
import logging
import threading
from functools import lru_cache
from typing import List, Dict, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

try:
    from backend.services.text_extraction import extract_text
except ImportError:
    try:
        from services.text_extraction import extract_text
    except ImportError:
        from .text_extraction import extract_text

try:
    from backend.services.section_detector import detect_sections
except ImportError:
    try:
        from services.section_detector import detect_sections
    except ImportError:
        from .section_detector import detect_sections

logger = logging.getLogger(__name__)

# Chunking configuration
MAX_CHUNK_SIZE = 1000  # ~256-512 tokens, research-recommended for RAG
MIN_CHUNK_SIZE = 50    # Minimum chars for a standalone chunk (merge if smaller)
CHUNK_OVERLAP = 150
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
SEMANTIC_THRESHOLD_TYPE = "gradient"  # Best for legal text — anomaly detection on distance gradients
SEMANTIC_THRESHOLD_AMOUNT = 95  # Default for gradient; splits at anomalous similarity drops

# Markdown headers to split on (ordered by level)
MARKDOWN_HEADERS = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
    ("####", "h4"),
]


def _clean_markdown_artifacts(text: str) -> str:
    """Clean common PDF-to-markdown extraction artifacts.

    Fixes bold-separated words like '**P** **ART** **1**' → 'PART 1'
    and removes excessive whitespace.
    """
    if not text:
        return text

    # Fix bold-separated characters: **P** **ART** **1** → P ART 1 → PART 1
    # Pattern: **X** **Y** where X and Y are short fragments
    cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)

    # Fix single-letter-space artifacts from PDF: "P ART" → "PART", "C HAPTER" → "CHAPTER"
    # Only for ALL-CAPS sequences where a single letter is separated
    cleaned = re.sub(r'\b([A-Z])\s([A-Z]{2,})\b', r'\1\2', cleaned)

    # Collapse multiple spaces into one
    cleaned = re.sub(r'  +', ' ', cleaned)

    # Remove orphan single lowercase letters on their own line (OCR artifacts).
    # Preserve uppercase letters — they may be Roman numerals (I, V, X) or exhibit labels (A, B).
    cleaned = re.sub(r'\n\s*([a-z])\s*\n', '\n', cleaned)

    return cleaned


def _merge_small_chunks(chunks: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Merge chunks smaller than MIN_CHUNK_SIZE with their neighbors.

    Small chunks (e.g., standalone headers, page numbers, prices)
    are merged into the next chunk to improve RAG retrieval quality.
    """
    if not chunks:
        return chunks

    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        content = chunk["content"]

        # If chunk is too small and there's a next chunk
        if len(content.strip()) < MIN_CHUNK_SIZE and i + 1 < len(chunks):
            next_chunk = chunks[i + 1].copy()
            # Prepend small chunk's content to the next chunk
            next_chunk["content"] = content.strip() + "\n\n" + next_chunk["content"]
            # Keep the small chunk's section_name if the next one is generic
            if chunk.get("section_name") and not next_chunk.get("section_name"):
                next_chunk["section_name"] = chunk["section_name"]
            merged.append(next_chunk)
            i += 2  # Skip both
        elif len(content.strip()) < MIN_CHUNK_SIZE and merged:
            # Last chunk is small — merge into previous
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + content.strip()
            # Preserve small chunk's section_name if previous one is generic
            if chunk.get("section_name") and not merged[-1].get("section_name"):
                merged[-1]["section_name"] = chunk["section_name"]
            i += 1
        else:
            merged.append(chunk)
            i += 1

    return merged


def _extract_section_label(text: str, page_num: str = "") -> str:
    """Extract a meaningful label from chunk text when no header is available.

    Looks for the first significant line that could serve as a section name:
    bold markdown, numbered clauses, capitalized phrases, or first sentence.
    """
    if not text:
        return f"Page {page_num}" if page_num else ""

    # Strip markdown formatting for analysis
    clean = re.sub(r'[#*_`]', '', text).strip()
    lines = [ln.strip() for ln in clean.split('\n') if ln.strip()]
    if not lines:
        return f"Page {page_num}" if page_num else ""

    first_line = lines[0]

    # Try to find a meaningful first line (not just noise)
    # Look for numbered clauses: "1 Short title", "23 Right to reject"
    m = re.match(r'^(\d+(?:\.\d+)?)\s+([A-Z].*)', first_line)
    if m:
        label = first_line[:80]
        return label

    # Look for capitalized labels: "PART 1", "Consumer Rights Act", etc.
    if len(first_line) < 100 and first_line[0].isupper():
        return first_line[:80]

    # Fallback: use first meaningful phrase
    if len(first_line) > 80:
        # Truncate to first sentence or 80 chars
        period = first_line.find('. ')
        if 10 < period < 80:
            return first_line[:period]
        return first_line[:80]

    return first_line if len(first_line) > 5 else f"Page {page_num}" if page_num else ""


# Singleton — initialized once per process, reused across calls
# PID check ensures fork-safety for Celery prefork workers
_semantic_chunker_instance = None
_semantic_chunker_pid = None
_semantic_chunker_lock = threading.Lock()

# Local embedding model for semantic chunking (no API key needed)
SEMANTIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_semantic_chunker():
    """Initialize SemanticChunker with local HuggingFace embeddings.

    Uses sentence-transformers running locally — no OpenAI API key needed.
    The chunker is initialized once per process and reused (singleton).
    PID check ensures a fresh instance after Celery worker fork.

    Raises:
        RuntimeError: If SemanticChunker cannot be initialized
    """
    global _semantic_chunker_instance, _semantic_chunker_pid

    current_pid = os.getpid()
    if _semantic_chunker_instance is not None and _semantic_chunker_pid == current_pid:
        return _semantic_chunker_instance

    with _semantic_chunker_lock:
        # Double-checked locking
        if _semantic_chunker_instance is not None and _semantic_chunker_pid == current_pid:
            return _semantic_chunker_instance

        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info(f"Initializing SemanticChunker with local model: {SEMANTIC_EMBEDDING_MODEL} (pid={current_pid})")
        embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_EMBEDDING_MODEL)

        _semantic_chunker_instance = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_THRESHOLD_TYPE,
            breakpoint_threshold_amount=SEMANTIC_THRESHOLD_AMOUNT,
        )
        _semantic_chunker_pid = current_pid
        logger.info("SemanticChunker initialized successfully (local embeddings)")
        return _semantic_chunker_instance


@lru_cache(maxsize=1)
def _get_fallback_splitter() -> RecursiveCharacterTextSplitter:
    """Get fallback text splitter (cached singleton)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS
    )


def _split_text_semantic(text: str, semantic_chunker) -> List[str]:
    """Split text using semantic chunker.

    Uses SemanticChunker for similarity-based splitting. Oversized chunks
    are re-split with RecursiveCharacterTextSplitter to enforce max size.

    Args:
        text: Text to split
        semantic_chunker: SemanticChunker instance

    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []

    # SemanticChunker (gradient mode) needs at least 3 sentences for np.gradient.
    # For very short text or too few sentences, return as-is.
    stripped = text.strip()
    sentences = re.split(r'(?<=[.?!])\s+', stripped)
    if len(stripped) < 100 or len(sentences) < 3:
        return [stripped]

    try:
        docs = semantic_chunker.create_documents([text])
        chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
    except Exception as e:
        logger.warning(f"SemanticChunker failed, using fallback splitter: {e}")
        return _get_fallback_splitter().split_text(text)

    if not chunks:
        return [text.strip()]

    # Enforce max chunk size — re-split any oversized chunks
    final_chunks = []
    size_splitter = _get_fallback_splitter()
    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_SIZE * 1.5:
            final_chunks.extend(size_splitter.split_text(chunk))
        else:
            final_chunks.append(chunk)
    return final_chunks


def _chunk_markdown_content(sections: List[Dict]) -> List[Dict[str, str]]:
    """Chunk markdown-formatted sections using header splitting + semantics.

    Args:
        sections: Extracted sections with 'format': 'markdown'

    Returns:
        List of chunk dicts with content, page_num, section_name, section_type
    """
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=False
    )
    semantic_chunker = _get_semantic_chunker()

    all_chunks = []
    chunk_counter = 0

    for section in sections:
        content = section.get("content", "")
        location = section.get("location", "")

        if not content.strip():
            continue

        # Split by markdown headers first
        try:
            header_splits = md_splitter.split_text(content)
        except Exception:
            header_splits = []

        # Check if MarkdownHeaderTextSplitter found actual header-based splits
        has_real_headers = False
        if header_splits:
            for doc in header_splits:
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                if any(metadata.get(h) for h in ("h1", "h2", "h3", "h4")):
                    has_real_headers = True
                    break

        if header_splits and has_real_headers:
            for doc in header_splits:
                text = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                if not text.strip():
                    continue

                # Extract section name from metadata
                metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                section_name = (
                    metadata.get("h4") or metadata.get("h3") or
                    metadata.get("h2") or metadata.get("h1") or ""
                )

                # Semantic split within each header section
                sub_chunks = _split_text_semantic(text, semantic_chunker)
                for sub_chunk in sub_chunks:
                    chunk_counter += 1
                    name = section_name or _extract_section_label(sub_chunk, location)
                    all_chunks.append({
                        "content": sub_chunk,
                        "page_num": location,
                        "section_name": name or f"Page {location} chunk {chunk_counter}",
                        "section_type": "markdown_header",
                    })
        else:
            # No markdown headers found — use section detector on the text content
            # This catches legal headers (ARTICLE, SECTION, etc.) even in markdown format
            detected = detect_sections(content)
            for det_section in detected:
                sec_content = det_section.get("content", "")
                sec_header = det_section.get("header", "")
                sec_type = det_section.get("section_type", "body")

                if not sec_content.strip():
                    continue

                sub_chunks = _split_text_semantic(sec_content, semantic_chunker)
                for sub_chunk in sub_chunks:
                    chunk_counter += 1
                    name = sec_header or _extract_section_label(sub_chunk, location)
                    all_chunks.append({
                        "content": sub_chunk,
                        "page_num": location,
                        "section_name": name or f"Page {location} chunk {chunk_counter}",
                        "section_type": sec_type,
                    })

    return all_chunks


def _chunk_plain_text(sections: List[Dict]) -> List[Dict[str, str]]:
    """Chunk plain-text sections using section detection + semantics.

    Args:
        sections: Extracted sections without markdown formatting

    Returns:
        List of chunk dicts with content, page_num, section_name, section_type
    """
    semantic_chunker = _get_semantic_chunker()

    # Combine all sections into one text for section detection
    full_text = "\n\n".join(s.get("content", "") for s in sections if s.get("content", "").strip())

    if not full_text.strip():
        return []

    # Build a sorted location map for O(log n) bisect lookup
    location_starts = []
    location_values = []
    pos = 0
    for s in sections:
        content = s.get("content", "")
        if content.strip():
            location_starts.append(pos)
            location_values.append(s.get("location", ""))
            pos += len(content) + 2  # +2 for "\n\n" separator

    def _find_location(char_pos: int) -> str:
        """Find location string for a character position using bisect (O(log n))."""
        if not location_starts:
            return ""
        idx = bisect.bisect_right(location_starts, char_pos) - 1
        if 0 <= idx < len(location_values):
            return location_values[idx]
        return location_values[-1] if location_values else ""

    # Detect legal sections
    detected = detect_sections(full_text)

    all_chunks = []
    chunk_counter = 0

    for det_section in detected:
        section_content = det_section.get("content", "")
        section_header = det_section.get("header", "")
        section_type = det_section.get("section_type", "body")
        start_pos = det_section.get("start_pos", 0)

        if not section_content.strip():
            continue

        location = _find_location(start_pos)

        # Semantic split within each detected section
        sub_chunks = _split_text_semantic(section_content, semantic_chunker)
        for sub_chunk in sub_chunks:
            chunk_counter += 1
            name = section_header or _extract_section_label(sub_chunk, location)
            all_chunks.append({
                "content": sub_chunk,
                "page_num": location,
                "section_name": name or f"Page {location} chunk {chunk_counter}",
                "section_type": section_type,
            })

    return all_chunks


def chunk_document_from_blob(blob_content: bytes, file_type: str = "pdf") -> List[Dict[str, str]]:
    """
    Chunk document from blob storage using hybrid semantic strategy.

    Pipeline:
    1. Extract text (structured where possible)
    2. If markdown format: header split → semantic split
    3. If plain text: section detect → semantic split
    4. Fallback: RecursiveCharacterTextSplitter

    Args:
        blob_content: Raw document bytes
        file_type: Document type ('pdf', 'docx', or 'txt'). Default: 'pdf'

    Returns:
        List of chunk dicts with keys: content, page_num, section_name, section_type

    Raises:
        ValueError: If blob_content is empty or invalid
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

        # Check if content is markdown-formatted
        is_markdown = any(s.get("format") == "markdown" for s in extracted_sections)

        if is_markdown:
            logger.info("Using markdown header + semantic chunking strategy")
            chunks = _chunk_markdown_content(extracted_sections)
        else:
            logger.info("Using section detection + semantic chunking strategy")
            chunks = _chunk_plain_text(extracted_sections)

        # Final fallback if both strategies produced nothing
        if not chunks:
            logger.warning("Structured chunking produced no results, using fallback splitter")
            fallback = _get_fallback_splitter()
            chunk_counter = 0
            for section in extracted_sections:
                split_texts = fallback.split_text(section.get("content", ""))
                for text in split_texts:
                    if text.strip():
                        chunk_counter += 1
                        chunks.append({
                            "content": text,
                            "page_num": section.get("location", ""),
                            "section_name": f"Chunk {chunk_counter}",
                            "section_type": "body",
                        })

        # Post-processing: clean artifacts and merge tiny chunks
        for chunk in chunks:
            chunk["content"] = _clean_markdown_artifacts(chunk["content"])

        before_merge = len(chunks)
        chunks = _merge_small_chunks(chunks)
        if before_merge != len(chunks):
            logger.info(f"Merged {before_merge - len(chunks)} small chunks ({before_merge} → {len(chunks)})")

        logger.info(f"Created {len(chunks)} chunks from {file_type.upper()}")
        return chunks

    except Exception as e:
        logger.error(f"Error chunking {file_type.upper()}: {str(e)}")
        raise
