"""CourtListener API integration for on-demand legal research"""
import logging
import re
from typing import List, Dict

import httpx

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)

COURTLISTENER_BASE_URL = "https://www.courtlistener.com"
COURTLISTENER_SEARCH_URL = f"{COURTLISTENER_BASE_URL}/api/rest/v4/search/"
REQUEST_TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def _get_headers() -> Dict[str, str]:
    """Build request headers with optional auth token."""
    settings = get_settings()
    headers = {}
    if settings.courtlistener_api_token:
        headers["Authorization"] = f"Token {settings.courtlistener_api_token}"
    return headers


def _strip_html(text: str) -> str:
    """Strip HTML tags (like <mark>) from CourtListener snippets."""
    return re.sub(r"<[^>]+>", "", text) if text else ""


def _extract_snippet(result: Dict) -> str:
    """Extract the best snippet from a CourtListener search result.

    Snippets are inside opinions[].snippet in v4.
    Falls back to caseName if no snippet available.
    """
    opinions = result.get("opinions", [])
    for opinion in opinions:
        snippet = opinion.get("snippet", "")
        if snippet:
            return _strip_html(snippet)
    return ""


def _normalize_result(result: Dict) -> Dict:
    """Normalize a single CourtListener search result into a standard format."""
    # citation is an array of strings
    citation_list = result.get("citation", [])
    citation = citation_list[0] if citation_list else ""

    absolute_url = result.get("absolute_url", "")
    full_url = f"{COURTLISTENER_BASE_URL}{absolute_url}" if absolute_url else ""

    snippet = _extract_snippet(result)

    return {
        "case_name": result.get("caseName", "Unknown Case"),
        "court": result.get("court", ""),
        "court_id": result.get("court_id", ""),
        "date_filed": result.get("dateFiled", ""),
        "snippet": snippet,
        "url": full_url,
        "citation": citation,
        "status": result.get("status", ""),
        "courtlistener_id": result.get("cluster_id", result.get("id", "")),
        "cite_count": result.get("citeCount", 0),
    }


async def search_cases(
    query: str,
    max_results: int = 5,
) -> List[Dict]:
    """
    Search CourtListener for relevant case law.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (1-10)

    Returns:
        List of normalized case dicts. Empty list on any error.
    """
    max_results = min(max(1, max_results), 10)

    params = {
        "q": query,
        "type": "o",  # opinions only
        "format": "json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            response = await client.get(
                COURTLISTENER_SEARCH_URL,
                params=params,
                headers=_get_headers(),
            )
            response.raise_for_status()

        data = response.json()
        raw_results = data.get("results", [])
        # CourtListener doesn't have page_size, so we slice client-side
        return [_normalize_result(r) for r in raw_results[:max_results]]

    except httpx.TimeoutException:
        logger.warning("CourtListener API timed out")
        return []
    except httpx.HTTPStatusError as e:
        logger.warning(f"CourtListener API HTTP error: {e.response.status_code}")
        return []
    except Exception as e:
        logger.warning(f"CourtListener API error: {e}")
        return []


def format_as_context(cases: List[Dict]) -> List[Dict]:
    """
    Convert CourtListener results into chunk-like dicts compatible with the RAG pipeline.

    Each case becomes a "chunk" with content, metadata, and a synthetic score
    based on its search rank position.

    Args:
        cases: List of normalized case dicts from search_cases()

    Returns:
        List of chunk-like dicts compatible with rag_engine's format_legal_context()
    """
    if not cases:
        return []

    chunks = []
    for i, case in enumerate(cases):
        # Build readable content from case metadata
        parts = []
        if case.get("citation"):
            parts.append(f"Citation: {case['citation']}")
        parts.append(f"Case: {case.get('case_name', 'Unknown')}")
        if case.get("court"):
            parts.append(f"Court: {case['court']}")
        if case.get("date_filed"):
            parts.append(f"Date: {case['date_filed']}")
        if case.get("status"):
            parts.append(f"Status: {case['status']}")
        if case.get("cite_count"):
            parts.append(f"Cited by: {case['cite_count']} cases")
        if case.get("snippet"):
            parts.append(f"\n{case['snippet']}")

        content = "\n".join(parts)

        # Assign descending score based on rank (1st = 0.90, 2nd = 0.85, etc.)
        score = max(0.50, 0.90 - (i * 0.05))

        chunk = {
            "content": content,
            "page_num": case.get("citation", "Case Law"),
            "section_name": case.get("court", ""),
            "score": score,
            "document_name": case.get("case_name", "Unknown Case"),
            "document_id": f"courtlistener-{case.get('courtlistener_id', '')}",
            "chunk_id": f"cl-{case.get('courtlistener_id', '')}",
            "source_type": "case_law",
            "url": case.get("url", ""),
        }
        chunks.append(chunk)

    return chunks
