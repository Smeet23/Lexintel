"""Problem Formulation Engine.

Identifies legal issues from natural language queries/facts using LLM-based
issue spotting. Runs in parallel with query embedding for zero added latency.

Uses Groq Llama 3.3 70B (preferred) with Gemini fallback.
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

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Groq client (optional — graceful degradation to Gemini if absent)
# ---------------------------------------------------------------------------

_GROQ_AVAILABLE = False
_groq_client = None

try:
    from groq import Groq

    _groq_key = getattr(settings, "groq_api_key", "")
    if _groq_key:
        _groq_client = Groq(api_key=_groq_key)
        _GROQ_AVAILABLE = True
    else:
        logger.info("GROQ_API_KEY not set. Problem formulation will use Gemini fallback.")
except ImportError:
    logger.info("groq SDK not installed. Problem formulation will use Gemini fallback.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 600  # Enough for 5 issues with details

ISSUE_SPOTTER_SYSTEM_PROMPT = """\
You are a legal issue spotter. Given facts or a legal question, identify ALL legal issues.

INSTRUCTIONS:
1. Read the facts. Identify every legally significant fact.
2. Map facts to legal triggers (e.g., "fired" + "pregnant" = employment discrimination).
3. Identify the PRIMARY issue (highest confidence, most direct).
4. Identify SECONDARY issues (plausible but lower confidence).
5. For each issue, state the precise legal question to research.
6. Flag missing information that would change classification.
7. It is BETTER to over-identify than under-identify.
8. Cap at 5 issues total.

Do NOT fabricate legal concepts. Only identify issues genuinely supported by the facts.

Return ONLY valid JSON with this structure:
{
  "issues": [
    {
      "domain": "<free-text domain: contract, tort, corporate, employment, etc.>",
      "issue": "<short issue label, e.g. duty of care standard>",
      "legal_question": "<precise research question>",
      "confidence": <0.0 to 1.0>,
      "key_facts": ["<fact 1>", "<fact 2>"]
    }
  ],
  "missing_information": ["<what info would help classification>"]
}"""


# ---------------------------------------------------------------------------
# Internal LLM helper (Groq + Gemini fallback, synchronous)
# ---------------------------------------------------------------------------

def _call_issue_spotter(query: str) -> Optional[Dict]:
    """Call the issue-spotting LLM and return parsed JSON. Returns None on failure."""
    user_prompt = f"Facts/Question:\n{query}"

    # Groq-primary (fast/free) with Gemini fallback, JSON mode + system prompt —
    # all handled by the unified llm client (configure/fence-strip/blocked-guard).
    try:
        try:
            from backend.services import llm
        except ImportError:
            try:
                from services import llm
            except ImportError:
                from . import llm
        return llm.generate(
            user_prompt,
            system=ISSUE_SPOTTER_SYSTEM_PROMPT,
            json=True,
            temperature=0.0,
            max_output_tokens=GROQ_MAX_TOKENS,
            provider="groq",
            fallback=True,
        )
    except Exception as e:
        logger.error(f"Issue spotter failed (Groq + Gemini): {e}")
        return None


# ---------------------------------------------------------------------------
# Pydantic validation (inline, avoids circular imports with schemas.py)
# ---------------------------------------------------------------------------

def _parse_issue_analysis(raw: Dict) -> Optional[Dict]:
    """Validate and normalise the LLM response into IssueAnalysis dict.

    Returns a plain dict (matching IssueAnalysis schema) or None if invalid.
    """
    try:
        raw_issues = raw.get("issues", [])
        if not isinstance(raw_issues, list):
            return None

        issues = []
        for item in raw_issues[:5]:  # Hard cap at 5
            if not isinstance(item, dict):
                continue

            # Normalise confidence
            try:
                confidence = float(item.get("confidence", 0.0))
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = 0.0

            # Normalise key_facts
            kf = item.get("key_facts", [])
            if not isinstance(kf, list):
                kf = []
            kf = [str(f) for f in kf if f]

            issues.append({
                "domain": str(item.get("domain", "")).strip(),
                "issue": str(item.get("issue", "")).strip(),
                "legal_question": str(item.get("legal_question", "")).strip(),
                "confidence": confidence,
                "key_facts": kf,
            })

        missing = raw.get("missing_information", [])
        if not isinstance(missing, list):
            missing = []
        missing = [str(m) for m in missing if m]

        return {
            "issues": issues,
            "missing_information": missing,
        }
    except Exception as e:
        logger.warning(f"Issue analysis parse/validation failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def identify_issues(query: str) -> Optional[Dict]:
    """Identify legal issues in a query using LLM-based issue spotting.

    Runs as an async coroutine so it can be gathered with embed_query in
    asyncio.gather() for zero added latency on the critical path.

    Args:
        query: Raw user query or fact pattern.

    Returns:
        Dict matching IssueAnalysis schema, or None on any failure.
        Never raises — failures are logged and None is returned.
    """
    if not query or not query.strip():
        return None

    if not getattr(settings, "problem_formulation_enabled", True):
        return None

    try:
        # _call_issue_spotter is synchronous; run in thread to avoid blocking the
        # event loop (asyncio.to_thread requires Python 3.9+, which this project uses)
        import asyncio

        raw = await asyncio.to_thread(_call_issue_spotter, query)
        if raw is None:
            return None

        result = _parse_issue_analysis(raw)
        if result is None:
            logger.warning("Problem formulation returned unparseable response — discarding")
            return None

        issue_count = len(result.get("issues", []))
        logger.info(
            f"Problem formulation: {issue_count} issue(s) identified for query "
            f"({len(query)} chars)"
        )
        return result

    except Exception as e:
        logger.warning(f"identify_issues failed (non-blocking): {e}")
        return None
