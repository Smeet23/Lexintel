"""Unsupervised keyword extraction using YAKE."""
import logging
from typing import List

import yake

logger = logging.getLogger(__name__)

# YAKE configuration optimized for legal text
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM = 3        # Capture multi-word legal terms ("force majeure", "intellectual property")
YAKE_DEDUP_THRESHOLD = 0.9  # Deduplicate similar phrases
YAKE_TOP_KEYWORDS = 10     # Extract top 10 per chunk
YAKE_SCORE_THRESHOLD = 0.2  # Lower = more relevant (YAKE inverts scores)

_extractor = None


def _get_extractor() -> yake.KeywordExtractor:
    """Singleton YAKE extractor (reused across chunks)."""
    global _extractor
    if _extractor is None:
        _extractor = yake.KeywordExtractor(
            lan=YAKE_LANGUAGE,
            n=YAKE_MAX_NGRAM,
            dedupLim=YAKE_DEDUP_THRESHOLD,
            top=YAKE_TOP_KEYWORDS,
            features=None,
        )
    return _extractor


def extract_chunk_keywords(text: str) -> List[str]:
    """Extract keywords from a single chunk using YAKE.

    Args:
        text: Chunk text content

    Returns:
        List of lowercased keyword strings, filtered by relevance score
    """
    if not text or len(text.strip()) < 30:
        return []
    try:
        extractor = _get_extractor()
        keywords = extractor.extract_keywords(text)
        # YAKE score: lower = more relevant. Filter by threshold.
        return [kw.lower() for kw, score in keywords if score < YAKE_SCORE_THRESHOLD]
    except Exception as e:
        logger.warning(f"YAKE extraction failed: {e}")
        return []
