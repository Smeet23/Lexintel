"""Generate document summaries from matter metadata"""
import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Legal terms to track
LEGAL_TERMS = [
    "payment", "liability", "warranty", "indemnif", "terminat",
    "breach", "force majeure", "arbitration", "governing law",
    "confidential", "intellectual property", "dispute", "amendment"
]

# Document type patterns
DOCUMENT_PATTERNS = {
    "Terms of Service": r"(?:terms\s+(?:and\s+)?conditions|terms\s+of\s+service)",
    "License Agreement": r"license\s+agreement",
    "Privacy Policy": r"privacy\s+(?:policy|statement)",
    "Purchase Agreement": r"(?:purchase|sales?)\s+agreement",
    "Non-Disclosure Agreement": r"(?:NDA|non.?disclosure)",
}


def extract_key_concepts(matter) -> List[str]:
    """Extract top legal concepts from chunks.

    Args:
        matter: Matter object with chunks attribute

    Returns:
        List of top 7 concepts (or fewer if not found)
    """
    try:
        chunks = getattr(matter, 'chunks', None)
        if not chunks:
            return []
    except (AttributeError, TypeError):
        return []

    concept_counts = {}

    for chunk in chunks:
        try:
            content = getattr(chunk, 'content', '')
            if not isinstance(content, str):
                content = str(content)
            content = content.lower()
        except (AttributeError, TypeError):
            continue

        for term in LEGAL_TERMS:
            try:
                count = len(re.findall(rf'\b{term}\b', content, re.IGNORECASE))
                if count > 0:
                    concept_counts[term] = concept_counts.get(term, 0) + count
            except (TypeError, re.error):
                continue

    # Sort by frequency, get top 7
    sorted_concepts = sorted(
        concept_counts.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [concept for concept, _ in sorted_concepts[:7]]


def classify_legal_document_type(matter) -> str:
    """Classify document type from content.

    Args:
        matter: Matter object with chunks

    Returns:
        Document type string
    """
    try:
        chunks = getattr(matter, 'chunks', None)
        if not chunks:
            return "Legal Document"

        first_chunk = chunks[0]
        content = getattr(first_chunk, 'content', '')
        if not isinstance(content, str):
            content = str(content)
        first_chunk = content.lower()
    except (AttributeError, TypeError, IndexError):
        return "Legal Document"

    try:
        for doc_type, pattern in DOCUMENT_PATTERNS.items():
            if re.search(pattern, first_chunk, re.IGNORECASE):
                return doc_type
    except (TypeError, re.error):
        pass

    return "Legal Document"


def calculate_page_count(matter) -> int:
    """Calculate total pages from chunk metadata.

    Args:
        matter: Matter object with chunks

    Returns:
        Estimated page count
    """
    try:
        chunks = getattr(matter, 'chunks', None)
        if not chunks:
            return 0
    except (AttributeError, TypeError):
        return 0

    page_numbers = set()

    for chunk in chunks:
        try:
            page_num = getattr(chunk, 'page_num', None)
            if page_num:
                # Extract numeric part
                match = re.search(r'\d+', str(page_num))
                if match:
                    page_numbers.add(int(match.group()))
        except (AttributeError, TypeError, ValueError):
            continue

    # Return max page number or estimate
    if page_numbers:
        return max(page_numbers)

    try:
        return max(1, len(chunks) // 2)
    except (TypeError, AttributeError):
        return 0


def generate_document_summary(matter) -> Dict[str, Any]:
    """Generate comprehensive document summary.

    Args:
        matter: Matter object with all metadata and chunks

    Returns:
        Dict with document metadata
    """
    try:
        name = getattr(matter, 'name', 'Unknown')
        file_type = getattr(matter, 'file_type', 'unknown')
        status = getattr(matter, 'status', 'unknown')
        updated_at = getattr(matter, 'updated_at', None)

        # Handle updated_at if it's a datetime or None
        processed_at = None
        if updated_at:
            try:
                processed_at = updated_at.isoformat() if hasattr(updated_at, 'isoformat') else str(updated_at)
            except (AttributeError, TypeError):
                processed_at = None

        return {
            "filename": name,
            "file_type": file_type,
            "key_concepts": extract_key_concepts(matter),
            "legal_significance": classify_legal_document_type(matter),
            "total_pages": calculate_page_count(matter),
            "processing_status": status,
            "processed_at": processed_at
        }
    except Exception as e:
        logger.warning(f"Error generating document summary: {str(e)}")
        return {
            "filename": "Unknown",
            "file_type": "unknown",
            "key_concepts": [],
            "legal_significance": "Legal Document",
            "total_pages": 0,
            "processing_status": "unknown",
            "processed_at": None
        }
