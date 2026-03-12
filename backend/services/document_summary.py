"""Generate document summaries and enrichment via Gemini."""
import re
import logging
from collections import Counter
from typing import List, Dict, Any, Optional

import google.generativeai as genai

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Ingestion-time enrichment (async, called from tasks.py)
# ------------------------------------------------------------------

async def generate_doc_summary(extracted_text: str) -> Optional[str]:
    """Generate a 1-2 sentence document summary using Gemini.

    Called once per document at ingestion time. The summary is stored
    on the Document model and prepended to chunks for embedding (SAC).

    Args:
        extracted_text: Full extracted document text

    Returns:
        Summary string (max 200 chars), or None on failure
    """
    settings = get_settings()
    if not settings.google_api_key:
        return None

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)

    # First ~30k chars covers preamble, TOC, key sections
    truncated = extracted_text[:30000]

    prompt = (
        "Summarize this legal document in exactly 1-2 sentences. "
        "Include the document type, subject matter, and key parties if identifiable. "
        "Be factual and specific.\n\n"
        f"{truncated}"
    )

    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=100,
            ),
        )
        summary = response.text.strip()
        return summary[:200] if summary else None
    except Exception as e:
        logger.warning(f"Summary generation failed (graceful degradation): {e}")
        return None


async def classify_document(extracted_text: str) -> Dict[str, str]:
    """Classify document type and jurisdiction using Gemini.

    Args:
        extracted_text: Full extracted document text

    Returns:
        Dict with keys: document_type, jurisdiction
    """
    settings = get_settings()
    if not settings.google_api_key:
        return {"document_type": "other", "jurisdiction": "unknown"}

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)

    truncated = extracted_text[:10000]

    prompt = (
        "Classify this legal document. Respond with ONLY two lines:\n"
        "TYPE: <one of: statute, contract, judgment, regulation, pleading, policy, other>\n"
        "JURISDICTION: <one of: US, UK, EU, AU, CA, SG, IN, UN, other>\n\n"
        f"{truncated}"
    )

    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=50,
            ),
        )
        text = response.text.strip()
        doc_type = "other"
        jurisdiction = "unknown"
        for line in text.split("\n"):
            if line.upper().startswith("TYPE:"):
                doc_type = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("JURISDICTION:"):
                jurisdiction = line.split(":", 1)[1].strip().upper()
        return {"document_type": doc_type, "jurisdiction": jurisdiction}
    except Exception as e:
        logger.warning(f"Document classification failed (graceful degradation): {e}")
        return {"document_type": "other", "jurisdiction": "unknown"}


# ------------------------------------------------------------------
# Query-time helpers (read pre-computed metadata from DB)
# ------------------------------------------------------------------


def extract_key_concepts(matter) -> List[str]:
    """Extract top concepts from pre-computed YAKE keywords stored on chunks.

    Aggregates concepts across all documents' chunks in the matter,
    returning the most frequent terms.

    Args:
        matter: Matter object with documents->chunks relationships

    Returns:
        List of top 10 concepts
    """
    concept_counts = Counter()

    for doc in matter.documents:
        for chunk in doc.chunks:
            for concept in (chunk.concepts or []):
                concept_counts[concept] += 1

    return [c for c, _ in concept_counts.most_common(10)]


def calculate_page_count(matter) -> int:
    """Calculate total pages from chunk metadata.

    Args:
        matter: Matter object with chunks relationship

    Returns:
        Estimated page count
    """
    page_numbers = set()

    for chunk in matter.chunks:
        if chunk.page_num:
            match = re.search(r'\d+', str(chunk.page_num))
            if match:
                page_numbers.add(int(match.group()))

    if page_numbers:
        return max(page_numbers)

    return max(1, len(matter.chunks) // 2)


def generate_document_summary(matter) -> Dict[str, Any]:
    """Generate comprehensive document summary from pre-computed metadata.

    Reads YAKE-extracted concepts from chunks and Gemini-generated
    metadata from documents. No regex computation at query time.

    Args:
        matter: Matter object with all metadata and relationships

    Returns:
        Dict with document metadata
    """
    primary_doc = matter.documents[0] if matter.documents else None

    return {
        "filename": matter.name,
        "file_type": matter.file_type,
        "key_concepts": extract_key_concepts(matter),
        "legal_significance": primary_doc.document_type or "Legal Document" if primary_doc else "Legal Document",
        "total_pages": calculate_page_count(matter),
        "processing_status": matter.status,
        "processed_at": matter.updated_at.isoformat() if matter.updated_at else None,
    }
