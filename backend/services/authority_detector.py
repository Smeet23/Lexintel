"""Authority hierarchy detection for legal documents via Gemini.

Called once per document at ingestion time (alongside summary generation
and document classification). Returns structured authority metadata that
is stored on every chunk from that document.

Scoring Formula:
    authority_score = (court_level_weight * 0.5)
                    + (jurisdiction_weight * 0.3)
                    + (binding_bonus * 0.2)

At ingestion, jurisdiction_weight defaults to 0.5 (sister-state baseline).
At query time, compute_query_time_authority_score() recalculates with the
actual jurisdiction relationship.
"""
import json
import logging
from typing import Dict, Optional

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

try:
    from backend.services import llm
except ImportError:
    try:
        from services import llm
    except ImportError:
        from . import llm

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# WEIGHT TABLES
# ═══════════════════════════════════════════════════════════════

COURT_LEVEL_WEIGHTS: Dict[str, float] = {
    "supreme": 1.00,
    "appellate": 0.85,
    "trial": 0.70,
    "administrative": 0.55,
    "unknown": 0.50,
}

# Source type defaults — non-court documents get baseline scores
SOURCE_TYPE_DEFAULTS: Dict[str, Dict] = {
    "statute": {
        "court_level": "supreme",
        "binding_authority": True,
        "default_score": 0.90,
    },
    "regulation": {
        "court_level": "administrative",
        "binding_authority": True,
        "default_score": 0.70,
    },
    "contract": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.40,
    },
    "commentary": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.30,
    },
    "other": {
        "court_level": "unknown",
        "binding_authority": False,
        "default_score": 0.35,
    },
}

# Valid court levels (for normalization)
VALID_COURT_LEVELS = frozenset(COURT_LEVEL_WEIGHTS.keys())

# Aliases that Gemini might return instead of canonical court levels
COURT_LEVEL_ALIASES: Dict[str, str] = {
    "circuit": "appellate",
    "appeals": "appellate",
    "high_court": "appellate",
    "district": "trial",
    "magistrate": "trial",
}

VALID_SOURCE_TYPES = frozenset(
    ["case_law", "statute", "regulation", "contract", "commentary", "other"]
)


# ═══════════════════════════════════════════════════════════════
# GEMINI PROMPT
# ═══════════════════════════════════════════════════════════════

AUTHORITY_EXTRACTION_PROMPT = """\
Analyze this legal document excerpt and classify its authority.
Respond with ONLY valid JSON (no markdown, no explanation).

{
  "source_type": "<statute|case_law|regulation|contract|commentary|other>",
  "court_level": "<supreme|appellate|trial|administrative|unknown>",
  "court_name": "<exact court name or 'unknown'>",
  "jurisdiction_code": "<ISO-style code: US.federal, US.state.CA, US.state.NY, UK, UK.england, IN, AU, EU, SG, CA.federal, CA.province.ON, etc. Use 'unknown' if uncertain>",
  "binding_authority": <true|false>,
  "confidence": <0.0 to 1.0>
}

Rules:
- For statutes/regulations: court_level reflects the legislative body level.
- For case law: court_level is the issuing court's tier.
- For contracts/commentary: court_level is "unknown", binding_authority is false.
- jurisdiction_code format: COUNTRY[.subdivision[.specific]]
  Examples: "US.federal", "US.state.CA", "UK.england", "IN", "AU.state.NSW"
- confidence reflects how certain you are about the classification.

Document excerpt:
"""


# ═══════════════════════════════════════════════════════════════
# SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

def _compute_authority_score(
    court_level: str,
    binding_authority: Optional[bool],
) -> float:
    """Compute the static portion of the authority score.

    The jurisdiction component (0.3 weight) is resolved at query time
    because it depends on the user's target jurisdiction.  At ingestion
    we compute:

        partial_score = (court_level_weight * 0.5) + (binding_bonus * 0.2)

    and store it.  At query time the full score is:

        authority_score = partial_score + (jurisdiction_weight * 0.3)

    However, for Qdrant payload filtering and rough pre-ranking we store
    a *default* authority_score that assumes ``jurisdiction_weight = 0.5``
    (sister-state baseline).  The query-time reranker overrides this.
    """
    cl_weight = COURT_LEVEL_WEIGHTS.get(court_level, 0.50)

    if binding_authority is True:
        binding_val = 1.0
    elif binding_authority is False:
        binding_val = 0.0
    else:
        binding_val = 0.30  # unknown

    # Default jurisdiction_weight for storage (overridden at query time)
    default_jurisdiction = 0.50

    score = (cl_weight * 0.5) + (default_jurisdiction * 0.3) + (binding_val * 0.2)
    return round(score, 4)


# ═══════════════════════════════════════════════════════════════
# MAIN DETECTION FUNCTION
# ═══════════════════════════════════════════════════════════════

async def detect_authority(extracted_text: str) -> Dict:
    """Classify a document's legal authority using Gemini.

    Args:
        extracted_text: Full extracted document text (will be truncated
            to first 15,000 chars to stay within fast-response budget).

    Returns:
        Dict with keys: source_type, court_level, court_name,
        jurisdiction_code, binding_authority, authority_score, confidence.
        Falls back to safe defaults on any failure.
    """
    default_result = {
        "source_type": "other",
        "court_level": "unknown",
        "court_name": "unknown",
        "jurisdiction_code": "unknown",
        "binding_authority": None,
        "authority_score": _compute_authority_score("unknown", None),
        "confidence": 0.0,
    }

    settings = get_settings()

    # Feature gate: skip if authority scoring is disabled
    if not settings.authority_scoring_enabled:
        logger.debug("Authority scoring disabled via config")
        return default_result

    if not (settings.google_api_key or getattr(settings, "groq_api_key", "")):
        logger.warning("No LLM provider key — skipping authority detection")
        return default_result

    if not extracted_text or not extracted_text.strip():
        logger.warning("Empty text provided — returning default authority")
        return default_result

    # Use first 15k chars — enough to identify court, jurisdiction, parties
    truncated = extracted_text[:15_000]
    prompt = AUTHORITY_EXTRACTION_PROMPT + truncated

    try:
        # Honor configured provider + fall back to the other backend so a single
        # provider outage (Gemini 429) doesn't degrade every chunk's authority.
        data = await llm.agenerate(
            prompt,
            json=True,
            temperature=0.0,
            max_output_tokens=200,
            provider=(getattr(settings, "llm_answer_provider", "gemini") or "gemini").lower(),
            fallback=True,
        )

        # --- Normalize source_type ---
        source_type = data.get("source_type", "other").lower().strip()
        if source_type not in VALID_SOURCE_TYPES:
            source_type = "other"

        # --- Normalize court_level ---
        court_level = data.get("court_level", "unknown").lower().strip()
        # Map aliases to canonical values
        court_level = COURT_LEVEL_ALIASES.get(court_level, court_level)
        if court_level not in VALID_COURT_LEVELS:
            court_level = "unknown"

        # --- Normalize court_name ---
        court_name = str(data.get("court_name", "unknown")).strip() or "unknown"

        # --- Normalize jurisdiction_code ---
        jurisdiction_code = str(data.get("jurisdiction_code", "unknown")).strip() or "unknown"

        # --- Normalize confidence ---
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        # --- Normalize binding_authority to bool or None ---
        ba_raw = data.get("binding_authority")
        if isinstance(ba_raw, bool):
            binding_authority = ba_raw
        elif isinstance(ba_raw, str):
            ba_lower = ba_raw.lower().strip()
            if ba_lower == "true":
                binding_authority = True
            elif ba_lower == "false":
                binding_authority = False
            else:
                binding_authority = None
        else:
            binding_authority = None

        # For non-case-law types, apply source-type defaults if Gemini
        # returned "unknown" for court_level
        if source_type != "case_law" and court_level == "unknown":
            defaults = SOURCE_TYPE_DEFAULTS.get(source_type, SOURCE_TYPE_DEFAULTS["other"])
            court_level = defaults["court_level"]
            if binding_authority is None:
                binding_authority = defaults["binding_authority"]

        authority_score = _compute_authority_score(court_level, binding_authority)

        result = {
            "source_type": source_type,
            "court_level": court_level,
            "court_name": court_name,
            "jurisdiction_code": jurisdiction_code,
            "binding_authority": binding_authority,
            "authority_score": authority_score,
            "confidence": round(confidence, 2),
        }

        logger.info(
            f"Authority detected: type={source_type}, court={court_level}, "
            f"jurisdiction={jurisdiction_code}, score={authority_score}, "
            f"confidence={confidence:.2f}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"Authority detection JSON parse failed: {e}")
        return default_result
    except Exception as e:
        logger.warning(f"Authority detection failed (graceful degradation): {e}")
        return default_result


# ═══════════════════════════════════════════════════════════════
# QUERY-TIME AUTHORITY SCORING
# ═══════════════════════════════════════════════════════════════

def compute_query_time_authority_score(
    chunk_authority: Dict,
    target_jurisdiction: Optional[str] = None,
) -> float:
    """Recompute authority score with query-specific jurisdiction weight.

    Called during reranking to replace the default jurisdiction_weight=0.5
    with an accurate weight based on the relationship between the chunk's
    jurisdiction and the user's target jurisdiction.

    Args:
        chunk_authority: The authority_metadata dict from the chunk.
        target_jurisdiction: The jurisdiction the user is asking about
            (e.g. "US.state.CA").  If None, uses the default 0.5 weight.

    Returns:
        Float authority score in [0.0, 1.0].
    """
    court_level = chunk_authority.get("court_level", "unknown")
    binding_authority = chunk_authority.get("binding_authority")
    chunk_jurisdiction = chunk_authority.get("jurisdiction_code", "unknown")

    cl_weight = COURT_LEVEL_WEIGHTS.get(court_level, 0.50)

    # Binding bonus
    if binding_authority is True:
        binding_val = 1.0
    elif binding_authority is False:
        binding_val = 0.0
    else:
        binding_val = 0.30

    # Jurisdiction weight
    jw = _resolve_jurisdiction_weight(chunk_jurisdiction, target_jurisdiction)

    score = (cl_weight * 0.5) + (jw * 0.3) + (binding_val * 0.2)
    return round(score, 4)


def _resolve_jurisdiction_weight(
    chunk_jurisdiction: str,
    target_jurisdiction: Optional[str],
) -> float:
    """Determine the jurisdiction relationship weight.

    Hierarchy:
      exact_match  -> 1.0   (chunk="US.state.CA", target="US.state.CA")
      federal      -> 0.85  (chunk="US.federal",   target="US.state.*")
      sister_state -> 0.50  (chunk="US.state.NY",  target="US.state.CA")
      foreign      -> 0.30  (chunk="UK",           target="US.*")
      unknown      -> 0.40  (either side unknown)
    """
    if not target_jurisdiction or target_jurisdiction == "unknown":
        return 0.40
    if not chunk_jurisdiction or chunk_jurisdiction == "unknown":
        return 0.40

    c = chunk_jurisdiction.lower()
    t = target_jurisdiction.lower()

    # Exact match
    if c == t:
        return 1.00

    c_parts = c.split(".")
    t_parts = t.split(".")

    # Same country?
    if c_parts[0] == t_parts[0]:
        # Federal covers all states in same country
        if len(c_parts) >= 2 and c_parts[1] == "federal":
            return 0.85
        # Both are sub-national but different -> sister state
        if len(c_parts) >= 2 and len(t_parts) >= 2:
            return 0.50
        # One is country-level, other is sub-national
        return 0.60

    # Different countries -> foreign
    return 0.30
