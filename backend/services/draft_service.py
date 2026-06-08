"""Legal document draft generation via Google Gemini and RAG.

Generates formal legal document drafts using matter context retrieved
from the vector store and guided by user instructions.
"""
import logging
from typing import Dict, Any, List

try:
    from backend.config import get_settings
    from backend.services.embeddings import embed_query
    from backend.services.vector_store import search_vectors
    from backend.services import llm
except ImportError:
    try:
        from config import get_settings
        from services.embeddings import embed_query
        from services.vector_store import search_vectors
        from services import llm
    except ImportError:
        from ..config import get_settings
        from .embeddings import embed_query
        from .vector_store import search_vectors
        from . import llm

logger = logging.getLogger(__name__)

# Number of relevant chunks to retrieve for context
RAG_CONTEXT_LIMIT = 10

DRAFT_PROMPT = """\
You are an expert legal drafting assistant. Generate a formal legal document based on the instructions below.

DOCUMENT TYPE: {document_type}

USER INSTRUCTIONS:
{instructions}

REFERENCE CONTEXT (from the matter's existing documents):
{context}

DRAFTING GUIDELINES:
1. Write in formal legal language appropriate for a {document_type}.
2. Structure the document with appropriate headings, numbered sections, and sub-sections.
3. Include inline source references where you rely on the reference context, formatted as [Source: document name, Page X].
4. Flag any areas where additional information is needed with [NOTE: description of what is needed].
5. Ensure the document is comprehensive and legally sound.
6. Do not fabricate specific case citations, statutes, or legal authorities that are not present in the reference context.

Generate the document now:
"""


def _build_context_text(search_results: List[Dict]) -> str:
    """Build formatted context text from vector search results.

    Args:
        search_results: List of dicts from search_vectors

    Returns:
        Formatted context string with source metadata
    """
    if not search_results:
        return "(No reference context available for this matter.)"

    parts: list[str] = []
    for i, result in enumerate(search_results, 1):
        doc_name = result.get("document_name", "Unknown Document")
        page = result.get("page_num", "N/A")
        section = result.get("section_name", "")
        content = result.get("content", "")

        header = f"--- Source {i}: {doc_name}"
        if page and page != "N/A":
            header += f", Page {page}"
        if section:
            header += f", Section: {section}"
        header += " ---"

        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


def _extract_sources(search_results: List[Dict]) -> List[Dict[str, Any]]:
    """Extract source metadata from search results for the response.

    Args:
        search_results: List of dicts from search_vectors

    Returns:
        List of source dicts with document_name, page_num, section_name, excerpt
    """
    sources = []
    for result in search_results:
        content = result.get("content", "")
        # Provide a short excerpt (first 200 chars)
        excerpt = content[:200].strip()
        if len(content) > 200:
            excerpt += "..."

        sources.append({
            "document_name": result.get("document_name", "Unknown Document"),
            "page_num": result.get("page_num", ""),
            "section_name": result.get("section_name", ""),
            "excerpt": excerpt,
        })

    return sources


async def generate_draft(
    matter_id: str,
    document_type: str,
    instructions: str,
    db: "Session",
) -> Dict[str, Any]:
    """Generate a legal document draft using Gemini with RAG context.

    Embeds the user instructions, retrieves relevant chunks from the
    matter's vector store, and sends everything to Gemini for drafting.

    Args:
        matter_id: UUID of the matter to pull context from
        document_type: Type of legal document to generate
            (e.g. "Non-Disclosure Agreement", "Employment Contract")
        instructions: User-provided drafting instructions
        db: SQLAlchemy database session (reserved for future use)

    Returns:
        Dict with keys:
            - content: the generated document text
            - sources: list of source references used
        Returns a default error result on failure (graceful degradation).
    """
    default_result: Dict[str, Any] = {
        "content": "",
        "sources": [],
    }

    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("Google API key not configured; returning empty draft")
        return default_result

    # ------------------------------------------------------------------
    # 1. Embed instructions and retrieve relevant context via RAG
    # ------------------------------------------------------------------
    search_results: List[Dict] = []
    try:
        query_embedding = embed_query(instructions)
        search_results = search_vectors(
            matter_id=matter_id,
            query_embedding=query_embedding,
            limit=RAG_CONTEXT_LIMIT,
        )
        logger.info(
            f"Retrieved {len(search_results)} context chunks for draft generation "
            f"(matter={matter_id})"
        )
    except Exception as e:
        logger.warning(
            f"RAG context retrieval failed, proceeding without context: {e}"
        )
        # Continue without context rather than failing entirely

    # ------------------------------------------------------------------
    # 2. Build context and prompt
    # ------------------------------------------------------------------
    context_text = _build_context_text(search_results)

    prompt = DRAFT_PROMPT.format(
        document_type=document_type,
        instructions=instructions,
        context=context_text,
    )

    # ------------------------------------------------------------------
    # 3. Call Gemini for draft generation
    # ------------------------------------------------------------------
    try:
        generated_content = await llm.agenerate(
            prompt,
            json=False,
            temperature=0.3,
            max_output_tokens=8192,
            provider="gemini",
            fallback=False,
        )
    except Exception as e:
        logger.warning(f"Gemini draft generation failed (graceful degradation): {e}")
        return default_result

    if not generated_content:
        logger.warning("Gemini returned empty draft content")
        return default_result

    # ------------------------------------------------------------------
    # 4. Build and return result
    # ------------------------------------------------------------------
    sources = _extract_sources(search_results)

    logger.info(
        f"Draft generation complete: type={document_type}, "
        f"length={len(generated_content)} chars, "
        f"sources={len(sources)}"
    )

    return {
        "content": generated_content,
        "sources": sources,
    }
