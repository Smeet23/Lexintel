"""Extract temporal metadata (effective dates, version info, supersession) from legal text.

Strategy:
  1. Regex patterns detect WHERE dates appear in common legal expressions
  2. python-dateutil parses WHAT the date is (robust, handles many formats)
  3. Gemini LLM fallback when regex yields no results
  4. All dates normalized to UTC-aware datetime objects

Called from tasks.py during ingestion, after text extraction and before chunking.
"""
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

try:
    from backend.services import llm
except ImportError:
    try:
        from services import llm
    except ImportError:
        from . import llm

from dateutil import parser as dateutil_parser
from dateutil.parser import ParserError

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TemporalMetadata:
    """Result of temporal extraction for a single document."""
    effective_date: Optional[datetime] = None
    superseded_date: Optional[datetime] = None
    version_number: Optional[str] = None
    document_status: str = "unknown"
    supersedes_references: List[str] = field(default_factory=list)
        # e.g. ["Data Protection Act 2018", "Regulation (EU) 2016/679"]
    amends_references: List[str] = field(default_factory=list)
        # e.g. ["Section 5 of the Privacy Act"]
    repealed_by_reference: Optional[str] = None
    confidence: float = 0.0  # 0.0-1.0, how confident we are in extracted dates
    extraction_method: str = "none"  # "regex", "llm", "none"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Serialize datetimes to ISO format for JSON storage
        if d["effective_date"]:
            d["effective_date"] = d["effective_date"].isoformat()
        if d["superseded_date"]:
            d["superseded_date"] = d["superseded_date"].isoformat()
        return d


# ----------------------------------------------
# Date parsing utilities (dateutil-based)
# ----------------------------------------------

# Ordinal suffixes: "1st", "2nd", "3rd", "4th" etc. -- strip before parsing
_ORDINAL_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)")

# Bare year pattern: "2024" alone
_BARE_YEAR_RE = re.compile(r"^(\d{4})$")

# Month + year only: "March 2024", "Jan 2024"
_MONTH_YEAR_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|"
    r"september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)"
    r"\s+(\d{4})$",
    re.IGNORECASE,
)


def _parse_date_string(text: str) -> Optional[datetime]:
    """Parse a date string into a timezone-aware UTC datetime.

    Uses python-dateutil for robust parsing. Regex is only used for
    pre-processing (stripping ordinal suffixes) and detecting special
    cases (bare year, month+year).

    Handles formats:
      - "25 March 2024", "March 25, 2024", "25th March 2024"
      - "2024-03-25", "03/25/2024", "25/03/2024"
      - "March 2024" (defaults to 1st of month)
      - "2024" (defaults to January 1st)
    """
    text = text.strip().rstrip(".")

    # Strip ordinal suffixes: "25th" -> "25", "1st" -> "1"
    text = _ORDINAL_RE.sub(r"\1", text)

    # Special case: bare year "2024"
    bare_year_match = _BARE_YEAR_RE.match(text.strip())
    if bare_year_match:
        year = int(bare_year_match.group(1))
        if 1900 <= year <= 2100:
            return datetime(year, 1, 1, tzinfo=timezone.utc)
        return None

    # Special case: month + year only "March 2024" -> defaults to 1st
    month_year_match = _MONTH_YEAR_RE.match(text.strip())
    if month_year_match:
        try:
            parsed = dateutil_parser.parse(text, default=datetime(2000, 1, 1))
            return parsed.replace(day=1, tzinfo=timezone.utc)
        except (ParserError, ValueError, OverflowError):
            return None

    # General case: use dateutil for robust parsing.
    # Default dayfirst=False (US MM/DD/YYYY is the dominant jurisdiction).
    try:
        parsed = dateutil_parser.parse(text, dayfirst=False, fuzzy=False)
        # Ensure timezone-aware (UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ParserError, ValueError, OverflowError):
        pass

    # Fallback: try fuzzy parsing for embedded dates
    try:
        parsed = dateutil_parser.parse(text, dayfirst=False, fuzzy=True)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except (ParserError, ValueError, OverflowError):
        return None


# ----------------------------------------------
# Regex-based detection patterns
# Regex finds WHERE dates appear; dateutil parses WHAT the date is.
# ----------------------------------------------

# Date capture fragment -- matches common date formats in legal text
_DATE_FRAGMENT = (
    r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}"
    r"|\w+\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{4})"
)

# Effective date patterns
_EFFECTIVE_DATE_PATTERNS = [
    # "effective [date]" / "effective from|as of [date]" / "effective date: [date]"
    # / "effective date is [date]"
    re.compile(
        r"effective\s+(?:date\s*(?:is\s+|of\s+|:\s*)?|from\s+|as\s+of\s+)?" + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "comes into force on [date]" / "shall come into force on [date]"
    re.compile(
        r"(?:shall\s+)?come[s]?\s+into\s+(?:force|effect|operation)\s+(?:on|from)\s+"
        + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "entered into force on [date]"
    re.compile(
        r"entered?\s+into\s+(?:force|effect)\s+(?:on\s+)?" + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "enacted on [date]" / "enacted [date]"
    re.compile(
        r"enacted\s+(?:on\s+)?" + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "dated [date]" (contracts, agreements)
    re.compile(
        r"(?:^|\n)\s*dated\s+" + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "this agreement is made on [date]"
    re.compile(
        r"this\s+(?:agreement|contract|deed|act|regulation)\s+(?:is\s+)?made\s+(?:on\s+|as\s+of\s+)?"
        + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
]

# Supersession / repeal patterns
_SUPERSESSION_PATTERNS = [
    # "supersedes [document name]"
    re.compile(
        r"(?:supersedes|replaces|revokes\s+and\s+replaces)\s+(?:the\s+)?"
        r"([A-Z][A-Za-z\s,()]+(?:Act|Regulation|Order|Rule|Directive|Law|Code|Statute)"
        r"(?:\s+\d{4})?)",
        re.IGNORECASE,
    ),
    # "amends [statute]"
    re.compile(
        r"(?:amends|amending|amendment\s+to)\s+(?:the\s+)?"
        r"([A-Z][A-Za-z\s,()]+(?:Act|Regulation|Order|Rule|Directive|Law|Code|Statute)"
        r"(?:\s+\d{4})?)",
        re.IGNORECASE,
    ),
]

# Repeal patterns
_REPEAL_PATTERNS = [
    # "repealed on [date]"
    re.compile(
        r"(?:repealed|revoked|rescinded|abrogated)\s+(?:on\s+|with\s+effect\s+from\s+)?"
        + _DATE_FRAGMENT,
        re.IGNORECASE,
    ),
    # "this act is repealed" (no date)
    re.compile(
        r"this\s+(?:act|regulation|order|rule|section)\s+is\s+(?:hereby\s+)?repealed",
        re.IGNORECASE,
    ),
]

# Version number patterns
_VERSION_PATTERNS = [
    # "Version 2.0", "Version 3"
    re.compile(r"version\s+([\d]+(?:\.[\d]+)?)", re.IGNORECASE),
    # "Revision 3", "Rev. 2"
    re.compile(r"rev(?:ision)?\.?\s+([\d]+(?:\.[\d]+)?)", re.IGNORECASE),
    # "Amendment No. 5", "Amendment 3"
    re.compile(r"amendment\s+(?:no\.?\s*)?([\d]+)", re.IGNORECASE),
    # "Third Edition", "2nd Edition"  (convert to number)
    re.compile(r"(\d+(?:st|nd|rd|th)?|\w+)\s+edition", re.IGNORECASE),
]


def _extract_via_regex(text: str) -> TemporalMetadata:
    """Extract temporal metadata from document text using regex patterns.

    Scans the first 15,000 characters (title pages, preamble, commencement
    sections) where dates and version info are most likely to appear.

    Regex detects the pattern context (e.g., "effective from ...").
    dateutil.parser handles the actual date parsing for robustness.

    Args:
        text: Full extracted document text

    Returns:
        TemporalMetadata with whatever could be extracted
    """
    # Focus on preamble / front matter (first ~15k chars)
    search_text = text[:15000]
    result = TemporalMetadata()
    found_anything = False

    # --- Effective date ---
    for pattern in _EFFECTIVE_DATE_PATTERNS:
        match = pattern.search(search_text)
        if match:
            parsed = _parse_date_string(match.group(1))
            if parsed:
                result.effective_date = parsed
                found_anything = True
                break  # Take first match (most prominent)

    # --- Version number ---
    for pattern in _VERSION_PATTERNS:
        match = pattern.search(search_text)
        if match:
            result.version_number = match.group(1).strip()
            found_anything = True
            break

    # --- Supersession references ---
    for pattern in _SUPERSESSION_PATTERNS:
        for match in pattern.finditer(search_text):
            ref = match.group(1).strip().rstrip(",. ")
            if len(ref) > 5:  # Filter out noise
                if "amend" in pattern.pattern.lower():
                    result.amends_references.append(ref)
                else:
                    result.supersedes_references.append(ref)
                found_anything = True

    # --- Repeal detection ---
    for pattern in _REPEAL_PATTERNS:
        match = pattern.search(search_text)
        if match:
            # If the pattern captured a date group
            if match.lastindex and match.lastindex >= 1:
                parsed = _parse_date_string(match.group(1))
                if parsed:
                    result.superseded_date = parsed
                    result.document_status = "repealed"
                    found_anything = True
            else:
                # "this act is repealed" -- no date
                result.document_status = "repealed"
                found_anything = True
            break

    # --- Infer document_status ---
    if result.document_status != "repealed":
        now = datetime.now(timezone.utc)
        if result.effective_date and result.effective_date > now:
            result.document_status = "draft"  # Future effective date
        elif result.effective_date:
            result.document_status = "current"  # Has effective date, not repealed
        # else: stays "unknown"

    if found_anything:
        result.extraction_method = "regex"
        # Confidence: higher if we found an effective_date, lower if only version/refs
        result.confidence = 0.85 if result.effective_date else 0.50

    return result


# ----------------------------------------------
# LLM-based extraction (Gemini fallback)
# ----------------------------------------------

_LLM_EXTRACTION_PROMPT = """Analyze this legal document excerpt and extract temporal metadata.
Respond with EXACTLY the following lines (use "none" if not found):

EFFECTIVE_DATE: <date in YYYY-MM-DD format, or "none">
SUPERSEDED_DATE: <date the document was repealed/superseded in YYYY-MM-DD, or "none">
VERSION: <version number like "2.0" or "Amendment No. 3", or "none">
STATUS: <one of: current, superseded, repealed, draft, unknown>
SUPERSEDES: <comma-separated list of document names this supersedes, or "none">
AMENDS: <comma-separated list of document names this amends, or "none">

Rules:
- Only extract dates explicitly stated in the text. Do NOT infer or estimate.
- If a document says "effective January 1, 2024" use "2024-01-01".
- If the document is an amendment, list what it amends in the AMENDS field.
- STATUS should be "current" if the document appears to be in force.
- STATUS should be "superseded" if the text says it has been replaced.
- STATUS should be "repealed" if the text says it has been repealed/revoked.
- STATUS should be "draft" if the effective date is in the future.

Document text (first 10,000 characters):
{text}"""


async def _extract_via_llm(text: str) -> TemporalMetadata:
    """Extract temporal metadata using Gemini when regex yields nothing.

    Uses the same Gemini model configured for the rest of the app.
    Cost: ~1 API call per document (~10k input tokens, ~50 output tokens).

    Args:
        text: Full extracted document text

    Returns:
        TemporalMetadata parsed from LLM response
    """
    settings = get_settings()
    if not (settings.google_api_key or getattr(settings, "groq_api_key", "")):
        logger.warning("No LLM provider key, skipping LLM temporal extraction")
        return TemporalMetadata()

    if not getattr(settings, 'temporal_llm_fallback', True):
        logger.info("LLM temporal fallback disabled by config")
        return TemporalMetadata()

    truncated = text[:10000]
    prompt = _LLM_EXTRACTION_PROMPT.format(text=truncated)

    try:
        response_text = await llm.agenerate(
            prompt,
            temperature=0.0,
            max_output_tokens=200,
            provider=(getattr(settings, "llm_answer_provider", "gemini") or "gemini").lower(),
            fallback=True,
        )
        return _parse_llm_response(response_text)
    except Exception as e:
        logger.warning(f"LLM temporal extraction failed (graceful degradation): {e}")
        return TemporalMetadata()


def _parse_llm_response(response_text: str) -> TemporalMetadata:
    """Parse structured LLM response into TemporalMetadata.

    Args:
        response_text: Raw text from Gemini

    Returns:
        TemporalMetadata populated from parsed fields
    """
    result = TemporalMetadata()
    result.extraction_method = "llm"

    for line in response_text.strip().split("\n"):
        line = line.strip()
        if ":" not in line:
            continue

        key, _, value = line.partition(":")
        key = key.strip().upper()
        value = value.strip()

        if value.lower() == "none" or not value:
            continue

        if key == "EFFECTIVE_DATE":
            result.effective_date = _parse_date_string(value)
        elif key == "SUPERSEDED_DATE":
            result.superseded_date = _parse_date_string(value)
        elif key == "VERSION":
            result.version_number = value
        elif key == "STATUS":
            valid_statuses = {"current", "superseded", "repealed", "draft", "unknown"}
            if value.lower() in valid_statuses:
                result.document_status = value.lower()
        elif key == "SUPERSEDES":
            result.supersedes_references = [
                ref.strip() for ref in value.split(",") if ref.strip()
            ]
        elif key == "AMENDS":
            result.amends_references = [
                ref.strip() for ref in value.split(",") if ref.strip()
            ]

    # Set confidence based on what was extracted
    if result.effective_date:
        result.confidence = 0.70  # LLM extraction is less certain than regex
    elif result.version_number or result.supersedes_references:
        result.confidence = 0.45
    else:
        result.confidence = 0.20

    return result


# ----------------------------------------------
# Public API
# ----------------------------------------------

async def extract_temporal_metadata(
    text: str,
    document_name: str = "",
) -> TemporalMetadata:
    """Extract temporal metadata from a legal document.

    Two-phase approach:
      1. Regex detection + dateutil parsing (fast, high confidence for common patterns)
      2. Gemini LLM fallback (if regex found nothing meaningful)

    This function is called once per document at ingestion time, after
    text extraction and before chunking (same phase as summary generation
    and document classification in tasks.py).

    Args:
        text: Full extracted document text
        document_name: Original filename (used for version heuristics)

    Returns:
        TemporalMetadata with best-effort extraction results
    """
    settings = get_settings()
    if not getattr(settings, 'temporal_extraction_enabled', True):
        logger.info("Temporal extraction disabled by config")
        return TemporalMetadata()

    # Phase 1: Regex detection + dateutil parsing
    result = _extract_via_regex(text)

    if result.extraction_method == "regex" and result.effective_date:
        logger.info(
            f"Regex temporal extraction successful for '{document_name}': "
            f"effective={result.effective_date}, status={result.document_status}, "
            f"version={result.version_number}"
        )
        return result

    # Phase 2: LLM fallback (only if regex found no effective_date)
    logger.info(f"Regex extraction insufficient for '{document_name}', trying LLM fallback")
    llm_result = await _extract_via_llm(text)

    # Merge: prefer regex results where available, fill gaps from LLM
    if not result.effective_date and llm_result.effective_date:
        result.effective_date = llm_result.effective_date
    if not result.superseded_date and llm_result.superseded_date:
        result.superseded_date = llm_result.superseded_date
    if not result.version_number and llm_result.version_number:
        result.version_number = llm_result.version_number
    if result.document_status == "unknown" and llm_result.document_status != "unknown":
        result.document_status = llm_result.document_status
    if not result.supersedes_references and llm_result.supersedes_references:
        result.supersedes_references = llm_result.supersedes_references
    if not result.amends_references and llm_result.amends_references:
        result.amends_references = llm_result.amends_references

    # Update extraction method
    if llm_result.extraction_method == "llm":
        if result.extraction_method == "regex":
            result.extraction_method = "regex+llm"
        else:
            result.extraction_method = "llm"

    # Recalculate confidence from merged result
    if result.effective_date:
        result.confidence = max(result.confidence, llm_result.confidence)
    elif result.version_number:
        result.confidence = 0.45
    else:
        result.confidence = 0.20

    logger.info(
        f"Final temporal extraction for '{document_name}': "
        f"method={result.extraction_method}, effective={result.effective_date}, "
        f"status={result.document_status}, confidence={result.confidence:.2f}"
    )
    return result


# ----------------------------------------------
# Filename-based version heuristics
# ----------------------------------------------

def extract_version_from_filename(filename: str) -> Optional[str]:
    """Extract version number from filename if present.

    Examples:
      "Contract_v2.0.pdf" -> "2.0"
      "Policy_Rev3.docx" -> "3"
      "Data Protection Act 2024 (Amendment No. 2).pdf" -> "2"

    Args:
        filename: Original upload filename

    Returns:
        Version string or None
    """
    patterns = [
        re.compile(r"_v([\d]+(?:\.[\d]+)?)", re.IGNORECASE),
        re.compile(r"_rev([\d]+(?:\.[\d]+)?)", re.IGNORECASE),
        re.compile(r"\(amendment\s+no\.?\s*(\d+)\)", re.IGNORECASE),
        re.compile(r"version\s*([\d]+(?:\.[\d]+)?)", re.IGNORECASE),
    ]
    for pattern in patterns:
        match = pattern.search(filename)
        if match:
            return match.group(1)
    return None
