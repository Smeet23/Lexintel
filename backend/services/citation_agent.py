"""Citation Verification Agent — Main orchestrator.

Runs after Gemini generates a response. Verifies every legal citation
in the answer is real, quotes are accurate, and cases are still good law.

Pipeline:
1. Extract citations (regex + eyecite + optional LLM)
2. Assign [1], [2], [3] indices
3. Lookup each citation in jurisdiction-specific databases
4. Cross-reference with RAG source chunks
5. Verify quotes (hybrid: cosine + LLM fallback)
6. Check case status (good law / overruled)
7. Produce verification report
"""
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _extract_context_around_citation(answer: str, citation_text: str, window: int = 200) -> str:
    """Extract the text surrounding a citation to find what claim it supports."""
    idx = answer.find(citation_text)
    if idx < 0:
        return ""
    start = max(0, idx - window)
    end = min(len(answer), idx + len(citation_text) + window)
    return answer[start:end]


def _match_citation_to_source(
    citation: Dict,
    sources: List[Dict],
) -> Optional[int]:
    """
    Try to match a citation to one of the RAG source chunks.

    Matches by:
    1. Exact citation text appearing in source content
    2. Case name appearing in document_name
    3. Source type being case_law with matching citation

    Returns source index or None.
    """
    raw_text = citation.get("raw_text", "").lower()
    case_name = (citation.get("case_name") or "").lower()

    for i, source in enumerate(sources):
        source_content = (source.get("content") or "").lower()
        source_doc_name = (source.get("document_name") or "").lower()
        source_page = (source.get("page_num") or "").lower()

        # Check if citation text appears in source content
        if raw_text and raw_text in source_content:
            return i

        # Check if citation matches source page_num (for case law sources)
        if raw_text and raw_text in source_page:
            return i

        # Check case name match
        if case_name and len(case_name) > 5 and case_name in source_doc_name:
            return i

    return None


async def verify_response_citations(
    answer: str,
    sources: List[Dict],
    enable_quote_verification: bool = False,
    enable_llm_extraction: bool = True,
) -> Optional[Dict]:
    """
    Main entry point — verify all legal citations in a Gemini response.

    Args:
        answer: The raw answer text from Gemini
        sources: The RAG source chunks used to generate the answer
        enable_quote_verification: Whether to verify quotes against opinion text
            (adds latency from fetching opinions + embedding)
        enable_llm_extraction: Whether to use LLM for citation extraction

    Returns:
        Dict with annotated_answer, verified_citations, and summary.
        Returns None if no citations found or verification disabled.
    """
    try:
        from backend.config import get_settings
    except ImportError:
        try:
            from config import get_settings
        except ImportError:
            from ..config import get_settings

    settings = get_settings()
    if not getattr(settings, "citation_verification_enabled", True):
        return None

    # Import services
    try:
        from backend.services.citation_extractor import extract_all_citations, annotate_answer_with_indices
        from backend.services.citation_lookup import lookup_citations_batch, check_case_status
        from backend.services.citation_verifier import verify_quote, fetch_opinion_text
    except ImportError:
        try:
            from services.citation_extractor import extract_all_citations, annotate_answer_with_indices
            from services.citation_lookup import lookup_citations_batch, check_case_status
            from services.citation_verifier import verify_quote, fetch_opinion_text
        except ImportError:
            from .citation_extractor import extract_all_citations, annotate_answer_with_indices
            from .citation_lookup import lookup_citations_batch, check_case_status
            from .citation_verifier import verify_quote, fetch_opinion_text

    # ──────────────────────────────────────────────
    # Step 1: Extract all legal citations from answer
    # ──────────────────────────────────────────────
    citations = await extract_all_citations(answer, use_llm=enable_llm_extraction)

    if not citations:
        logger.debug("No legal citations found in answer")
        return None

    # Cap citations to prevent API abuse / cost explosion
    MAX_CITATIONS = 25
    if len(citations) > MAX_CITATIONS:
        logger.warning(f"Capping citations from {len(citations)} to {MAX_CITATIONS}")
        citations = citations[:MAX_CITATIONS]

    logger.info(f"Found {len(citations)} citations to verify")

    # ──────────────────────────────────────────────
    # Step 2: Map citations — the LLM already produces [N] markers via system prompt.
    # We DON'T re-annotate to avoid collision. We only annotate if the LLM
    # used full citation text without [N] markers (e.g., case law citations).
    # ──────────────────────────────────────────────
    import re as _re
    existing_markers = set(int(m.group(1)) for m in _re.finditer(r'\[(\d+)\]', answer))
    if existing_markers:
        # LLM already has [N] markers — don't double-annotate
        annotated_answer = answer
    else:
        # No [N] markers in answer — add them next to citations
        annotated_answer = annotate_answer_with_indices(answer, citations)

    # ──────────────────────────────────────────────
    # Step 3: Batch lookup — verify citations exist
    # ──────────────────────────────────────────────
    lookup_results = await lookup_citations_batch(citations)

    # ──────────────────────────────────────────────
    # Step 4-7: Build verification results
    # ──────────────────────────────────────────────
    verified_citations = []
    summary = {
        "total": len(citations),
        "verified": 0,
        "partial": 0,
        "unverified": 0,
        "not_found": 0,
        "unverifiable": 0,
    }

    for cite in citations:
        idx = cite["index"]
        raw_text = cite["raw_text"]
        jurisdiction = cite.get("jurisdiction", "OTHER")
        lookup = lookup_results.get(idx)

        # Step 4: Match to RAG source
        matched_source_idx = _match_citation_to_source(cite, sources)

        # Determine base verification status
        if lookup and lookup.get("found"):
            # Citation exists in database
            base_status = "verified"
            case_name = lookup.get("case_name", "") or cite.get("case_name", "")
            court = lookup.get("court", "")
            date_filed = lookup.get("date_filed", "")
            source_url = lookup.get("url", "")
            citation_count = lookup.get("citation_count", 0)
            cluster_id = str(lookup.get("cluster_id", ""))
            verification_method = lookup.get("source", "api")
        elif matched_source_idx is not None:
            # Not in external DB but matches a RAG source chunk
            base_status = "verified"
            source = sources[matched_source_idx]
            case_name = cite.get("case_name", "") or source.get("document_name", "")
            court = source.get("section_name", "")
            date_filed = ""
            source_url = source.get("url", "")
            citation_count = 0
            cluster_id = ""
            verification_method = "source_match"
        else:
            # Not found anywhere
            base_status = "not_found"
            case_name = cite.get("case_name", "")
            court = ""
            date_filed = ""
            source_url = ""
            citation_count = 0
            cluster_id = ""
            verification_method = "not_found"

        # Step 5: Quote verification (optional, expensive)
        quote_match = None
        if enable_quote_verification and base_status == "verified" and cluster_id:
            try:
                opinion_text = await fetch_opinion_text("", cluster_id=cluster_id)
                if opinion_text:
                    context = _extract_context_around_citation(answer, raw_text)
                    if context:
                        quote_result = await verify_quote(context, opinion_text)
                        quote_match = quote_result.get("match_score", 0)
                        # Downgrade if quote doesn't match or couldn't be verified.
                        # "unverifiable" (LLM inconclusive) must NOT remain "verified".
                        if quote_result.get("status") in ("unverified", "partial", "unverifiable"):
                            base_status = "partial"
            except Exception as e:
                logger.debug(f"Quote verification failed for citation {idx}: {e}")

        # Step 6: Case status (US only, uses CourtListener)
        case_status = "unknown"
        if jurisdiction == "US" and cluster_id and base_status in ("verified", "partial"):
            try:
                case_status = await check_case_status(cluster_id)
                # Downgrade if case is bad law
                if case_status == "bad_law":
                    base_status = "partial" if base_status == "verified" else base_status
            except Exception as e:
                logger.debug(f"Case status check failed for cluster {cluster_id}: {e}")

        # Step 7: Compute confidence score
        confidence = _compute_confidence(
            base_status, quote_match, citation_count,
            verification_method, case_status
        )

        # Build result
        verified_citation = {
            "index": idx,
            "raw_text": raw_text,
            "case_name": case_name,
            "court": court,
            "date": date_filed,
            "jurisdiction": jurisdiction,
            "status": base_status,
            "case_status": case_status,
            "confidence": round(confidence, 2),
            "source_url": source_url,
            "quote_match": round(quote_match, 2) if quote_match is not None else None,
            "matched_source_idx": matched_source_idx,
            "citation_count": citation_count,
            "verification_method": verification_method,
        }
        verified_citations.append(verified_citation)

        # Update summary
        summary[base_status] = summary.get(base_status, 0) + 1

    logger.info(
        f"Citation verification complete: {summary['verified']} verified, "
        f"{summary['partial']} partial, {summary['not_found']} not found, "
        f"{summary['unverified']} unverified"
    )

    return {
        "annotated_answer": annotated_answer,
        "verified_citations": verified_citations,
        "summary": summary,
    }


def _compute_confidence(
    status: str,
    quote_match: Optional[float],
    citation_count: int,
    verification_method: str,
    case_status: str,
) -> float:
    """Compute a confidence score (0.0-1.0) for a citation verification."""
    # Base score from verification status
    base_scores = {
        "verified": 0.85,
        "partial": 0.55,
        "unverified": 0.20,
        "not_found": 0.05,
        "unverifiable": 0.50,
    }
    score = base_scores.get(status, 0.30)

    # Boost for API-verified (vs pattern match)
    if verification_method in ("courtlistener", "courtlistener_search", "bailii", "indiankanoon"):
        score = min(1.0, score + 0.10)
    elif verification_method == "source_match":
        score = min(1.0, score + 0.05)

    # Boost for highly-cited cases
    if citation_count >= 100:
        score = min(1.0, score + 0.05)
    elif citation_count >= 10:
        score = min(1.0, score + 0.02)

    # Quote match adjustment
    if quote_match is not None:
        if quote_match >= 0.80:
            score = min(1.0, score + 0.05)
        elif quote_match < 0.50:
            score = max(0.0, score - 0.15)

    # Case status adjustment
    if case_status == "bad_law":
        score = max(0.0, score - 0.30)
    elif case_status == "caution":
        score = max(0.0, score - 0.10)

    return score
