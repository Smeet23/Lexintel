"""Hybrid quote verification — Cohere cosine + Gemini LLM fallback.

Tier 1 (API-only): Cohere embeddings cosine similarity
  - Score >= 0.80 → verified (no LLM)
  - Score < 0.50 → unverified (no LLM)
  - Score 0.50-0.80 → Tier 2

Tier 2 (LLM fallback): Gemini judges whether the quote accurately
represents the case holding/reasoning.
"""
import asyncio
import logging
from collections import OrderedDict
from typing import Dict, List, Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Verification thresholds
VERIFIED_THRESHOLD = 0.80
UNVERIFIED_THRESHOLD = 0.50

# Opinion text cache (process-local, bounded to prevent memory leaks)
class _BoundedStrCache:
    """True LRU bounded cache for large string values (opinion texts).

    Uses OrderedDict with move_to_end on get (most-recently-used at the tail)
    and popitem(last=False) to evict the least-recently-used entry.
    """
    def __init__(self, maxsize: int = 200):
        self._cache: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> Optional[str]:
        if key not in self._cache:
            return None
        # Move to end (most recently used).
        self._cache.move_to_end(key)
        return self._cache[key]

    def set(self, key: str, value: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._cache[key] = value
            return
        if len(self._cache) >= self._maxsize:
            # Evict least-recently-used (head of the OrderedDict).
            self._cache.popitem(last=False)
        self._cache[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._cache

_opinion_cache = _BoundedStrCache(maxsize=200)


async def fetch_opinion_text(
    url: str,
    cluster_id: str = "",
) -> Optional[str]:
    """
    Fetch full opinion text from CourtListener.

    Args:
        url: CourtListener opinion URL
        cluster_id: CourtListener cluster ID (used for API endpoint)

    Returns:
        Opinion text string, or None if unavailable
    """
    cache_key = cluster_id or url
    cached = _opinion_cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        from backend.config import get_settings
    except ImportError:
        try:
            from config import get_settings
        except ImportError:
            from ..config import get_settings

    settings = get_settings()

    # Try cluster API to get opinions
    if cluster_id:
        try:
            headers = {}
            if settings.courtlistener_api_token:
                headers["Authorization"] = f"Token {settings.courtlistener_api_token}"

            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0)) as client:
                # Get opinions for this cluster
                response = await client.get(
                    f"https://www.courtlistener.com/api/rest/v4/clusters/{cluster_id}/",
                    params={"format": "json"},
                    headers=headers,
                )
                if response.status_code == 200:
                    data = response.json()
                    opinion_urls = data.get("sub_opinions", [])

                    # Fetch first opinion's text (validate URL to prevent SSRF)
                    from urllib.parse import urlparse
                    for op_url in opinion_urls[:1]:
                        if isinstance(op_url, str):
                            parsed_url = urlparse(op_url)
                            if parsed_url.hostname and parsed_url.hostname != "www.courtlistener.com":
                                logger.warning(f"Refusing sub_opinion URL to non-CourtListener host: {parsed_url.hostname}")
                                continue
                            op_response = await client.get(
                                op_url,
                                params={"format": "json"},
                                headers=headers,
                            )
                            if op_response.status_code == 200:
                                op_data = op_response.json()
                                # Try different text fields
                                text = (
                                    op_data.get("plain_text")
                                    or op_data.get("html_with_citations")
                                    or op_data.get("html")
                                    or ""
                                )
                                if text:
                                    # Strip HTML tags with a real parser (bs4)
                                    # rather than a brittle `<[^>]+>` regex, which
                                    # mishandles entities and leaves <script>/<style>
                                    # bodies in the text. get_text() drops markup
                                    # and collapses to readable text.
                                    from bs4 import BeautifulSoup
                                    soup = BeautifulSoup(text, "html.parser")
                                    for tag in soup(["script", "style"]):
                                        tag.decompose()
                                    clean_text = soup.get_text(separator=" ", strip=True)
                                    _opinion_cache.set(cache_key, clean_text)
                                    return clean_text
        except Exception as e:
            logger.debug(f"Failed to fetch opinion via API: {e}")

    return None


def _cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(vec_a)
    b = np.array(vec_b)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks for comparison."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


async def _tier1_cosine_verify(
    claimed_text: str,
    opinion_text: str,
) -> Dict:
    """
    Tier 1: Cohere cosine similarity between claimed text and opinion chunks.

    Returns dict with match_score and best_matching_passage.
    """
    try:
        from backend.services.embeddings import embed_text
    except ImportError:
        try:
            from services.embeddings import embed_text
        except ImportError:
            from .embeddings import embed_text

    # Embed the claimed quote — embed_text is blocking; run in a thread pool.
    try:
        claim_embedding = await asyncio.to_thread(embed_text, claimed_text)
    except Exception as e:
        logger.warning(f"Failed to embed claimed text: {e}")
        return {"match_score": 0.0, "best_matching_passage": "", "tier": 1}

    # Chunk the opinion text and embed chunks
    chunks = _chunk_text(opinion_text, chunk_size=500, overlap=100)
    if not chunks:
        return {"match_score": 0.0, "best_matching_passage": "", "tier": 1}

    # Embed chunks (reuse batch embedding) — blocking call, run in thread pool.
    try:
        from backend.services.embeddings import embed_chunks as embed_batch
    except ImportError:
        try:
            from services.embeddings import embed_chunks as embed_batch
        except ImportError:
            from .embeddings import embed_chunks as embed_batch

    try:
        chunk_embeddings = await asyncio.to_thread(embed_batch, chunks)
    except Exception as e:
        logger.warning(f"Failed to embed opinion chunks: {e}")
        return {"match_score": 0.0, "best_matching_passage": "", "tier": 1}

    # Find best matching chunk
    best_score = 0.0
    best_passage = ""
    for chunk_text, chunk_emb in zip(chunks, chunk_embeddings):
        score = _cosine_similarity(claim_embedding, chunk_emb)
        if score > best_score:
            best_score = score
            best_passage = chunk_text

    return {
        "match_score": best_score,
        "best_matching_passage": best_passage[:500],
        "tier": 1,
    }


async def _tier2_llm_verify(
    claimed_text: str,
    opinion_passage: str,
) -> Dict:
    """
    Tier 2: Gemini LLM judges whether the quote accurately represents the case.

    Only called when Tier 1 gives an ambiguous score (0.50-0.80).
    """
    try:
        from backend.services import llm
    except ImportError:
        try:
            from services import llm
        except ImportError:
            from . import llm

    prompt = f"""You are a legal citation verification expert. Determine if the following quote or claim accurately represents the content of the legal opinion excerpt.

CLAIMED TEXT:
"{claimed_text}"

ACTUAL OPINION EXCERPT:
"{opinion_passage[:3000]}"

Respond with EXACTLY one word:
- SUPPORTED — the claimed text accurately represents what the opinion says
- PARTIALLY_SUPPORTED — the claimed text captures the general idea but has inaccuracies or missing nuance
- NOT_SUPPORTED — the claimed text does not accurately represent the opinion

Your answer:"""

    try:
        raw = await llm.agenerate(
            prompt,
            temperature=0.0,
            max_output_tokens=20,
            json=False,
            provider="gemini",
            fallback=False,
        )
        verdict = raw.strip().upper()

        # Normalize response
        if "SUPPORTED" in verdict and "NOT" not in verdict and "PARTIALLY" not in verdict:
            return {"llm_verdict": "supported", "tier": 2}
        elif "PARTIALLY" in verdict:
            return {"llm_verdict": "partially_supported", "tier": 2}
        elif "NOT" in verdict:
            return {"llm_verdict": "not_supported", "tier": 2}
        else:
            return {"llm_verdict": "unknown", "tier": 2}

    except Exception as e:
        logger.warning(f"LLM verification failed: {e}")
        return {"llm_verdict": "unknown", "tier": 2}


async def verify_quote(
    claimed_text: str,
    opinion_text: str,
) -> Dict:
    """
    Hybrid quote verification — Tier 1 cosine, Tier 2 LLM fallback.

    Args:
        claimed_text: The text/quote the LLM claims comes from this case
        opinion_text: The actual opinion text from the court

    Returns:
        Dict with:
        - match_score: float (0.0-1.0)
        - is_verified: bool
        - status: "verified" | "partial" | "unverified"
        - verification_tier: 1 | 2
        - best_matching_passage: str
    """
    if not claimed_text or not opinion_text:
        return {
            "match_score": 0.0,
            "is_verified": False,
            "status": "unverifiable",
            "verification_tier": 0,
            "best_matching_passage": "",
        }

    # Tier 1: Cosine similarity
    tier1 = await _tier1_cosine_verify(claimed_text, opinion_text)
    score = tier1["match_score"]

    if score >= VERIFIED_THRESHOLD:
        return {
            "match_score": score,
            "is_verified": True,
            "status": "verified",
            "verification_tier": 1,
            "best_matching_passage": tier1["best_matching_passage"],
        }

    if score < UNVERIFIED_THRESHOLD:
        return {
            "match_score": score,
            "is_verified": False,
            "status": "unverified",
            "verification_tier": 1,
            "best_matching_passage": tier1["best_matching_passage"],
        }

    # Tier 2: Ambiguous score (0.50-0.80) — invoke LLM
    logger.info(f"Ambiguous cosine score ({score:.2f}), invoking LLM verification")
    tier2 = await _tier2_llm_verify(claimed_text, tier1["best_matching_passage"])
    verdict = tier2.get("llm_verdict", "unknown")

    if verdict == "supported":
        return {
            "match_score": max(score, 0.85),  # Boost score for LLM-verified
            "is_verified": True,
            "status": "verified",
            "verification_tier": 2,
            "best_matching_passage": tier1["best_matching_passage"],
        }
    elif verdict == "partially_supported":
        return {
            "match_score": score,
            "is_verified": False,
            "status": "partial",
            "verification_tier": 2,
            "best_matching_passage": tier1["best_matching_passage"],
        }
    elif verdict == "not_supported":
        return {
            "match_score": min(score, 0.40),  # Penalize for LLM-rejected
            "is_verified": False,
            "status": "unverified",
            "verification_tier": 2,
            "best_matching_passage": tier1["best_matching_passage"],
        }
    else:
        # Unknown verdict — LLM failed or was unavailable; do not present as
        # a genuine partial verification.  Return "unverifiable" so callers
        # can distinguish a real LLM result from an error/timeout path.
        return {
            "match_score": score,
            "is_verified": False,
            "status": "unverifiable",
            "verification_tier": 2,
            "best_matching_passage": tier1["best_matching_passage"],
        }
