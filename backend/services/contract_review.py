"""Contract risk analysis via Google Gemini.

Analyzes legal contracts for risks, missing clauses, and overall health score.
"""
import json
import logging
from typing import Dict, Any

try:
    from backend.config import get_settings
    from backend.models import Chunk, Document
    from backend.services import llm
except ImportError:
    try:
        from config import get_settings
        from models import Chunk, Document
        from services import llm
    except ImportError:
        from ..config import get_settings
        from ..models import Chunk, Document
        from . import llm

logger = logging.getLogger(__name__)

# Maximum characters to send to Gemini (stay within token limits)
MAX_DOCUMENT_CHARS = 50000

# Default result returned on any failure (graceful degradation)
DEFAULT_RESULT: Dict[str, Any] = {
    "risks": [],
    "summary": {
        "total_clauses": 0,
        "high_risk": 0,
        "medium_risk": 0,
        "low_risk": 0,
    },
    "missing_clauses": [],
    "overall_score": 0,
}

CONTRACT_REVIEW_PROMPT = """\
You are an expert legal contract reviewer. Analyze the following contract text and provide a comprehensive risk assessment.

Return your analysis as JSON with exactly these keys:
- "risks": an array of objects, each with:
    - "clause": the clause name or section reference
    - "risk_level": one of "high", "medium", or "low"
    - "explanation": a concise explanation of the risk
    - "remedy": a suggested remedy or improvement
- "summary": an object with:
    - "total_clauses": total number of clauses analyzed (integer)
    - "high_risk": count of high-risk items (integer)
    - "medium_risk": count of medium-risk items (integer)
    - "low_risk": count of low-risk items (integer)
- "missing_clauses": an array of strings naming standard contract clauses that are absent from this document (e.g. "Indemnification", "Force Majeure", "Governing Law", "Dispute Resolution", "Confidentiality", "Termination", "Limitation of Liability", "Assignment", "Severability", "Entire Agreement")
- "overall_score": an integer from 0 to 100 representing the contract's overall health (100 = excellent, 0 = extremely risky)

Be thorough but concise. Focus on legally significant risks.

CONTRACT TEXT:
{document_text}
"""


def _validate_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the parsed Gemini response.

    Ensures all expected keys exist with correct types so downstream
    consumers never encounter missing fields.

    Args:
        data: Parsed JSON dict from Gemini

    Returns:
        Validated and normalized dict
    """
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = []

    # Normalize each risk entry
    validated_risks = []
    for risk in risks:
        if not isinstance(risk, dict):
            continue
        validated_risks.append({
            "clause": str(risk.get("clause", "Unknown")),
            "risk_level": str(risk.get("risk_level", "low")).lower(),
            "explanation": str(risk.get("explanation", "")),
            "remedy": str(risk.get("remedy", "")),
        })

    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    validated_summary = {
        "total_clauses": int(summary.get("total_clauses", len(validated_risks))),
        "high_risk": int(summary.get("high_risk", 0)),
        "medium_risk": int(summary.get("medium_risk", 0)),
        "low_risk": int(summary.get("low_risk", 0)),
    }

    missing_clauses = data.get("missing_clauses", [])
    if not isinstance(missing_clauses, list):
        missing_clauses = []
    missing_clauses = [str(c) for c in missing_clauses]

    overall_score = data.get("overall_score", 0)
    try:
        overall_score = max(0, min(100, int(overall_score)))
    except (TypeError, ValueError):
        overall_score = 0

    return {
        "risks": validated_risks,
        "summary": validated_summary,
        "missing_clauses": missing_clauses,
        "overall_score": overall_score,
    }


async def analyze_contract(
    matter_id: str, document_id: str, db: "Session"
) -> Dict[str, Any]:
    """Analyze a contract document for risks using Google Gemini.

    Fetches all chunks for the given document from the database,
    reconstructs the document text, and sends it to Gemini for
    contract risk analysis.

    Args:
        matter_id: UUID of the matter
        document_id: UUID of the document to analyze
        db: SQLAlchemy database session

    Returns:
        Dict with keys: risks, summary, missing_clauses, overall_score.
        Returns DEFAULT_RESULT on any failure (graceful degradation).
    """
    settings = get_settings()
    if not settings.google_api_key:
        logger.warning("Google API key not configured; returning default result")
        return dict(DEFAULT_RESULT)

    # ------------------------------------------------------------------
    # 1. Fetch chunks for the document, ordered by sequence
    # ------------------------------------------------------------------
    try:
        chunks = (
            db.query(Chunk)
            .filter(
                Chunk.document_id == document_id,
                Chunk.matter_id == matter_id,
            )
            .order_by(Chunk.chunk_sequence)
            .all()
        )
    except Exception as e:
        logger.error(f"Failed to fetch chunks for document {document_id}: {e}")
        return dict(DEFAULT_RESULT)

    if not chunks:
        logger.warning(f"No chunks found for document {document_id}")
        return dict(DEFAULT_RESULT)

    # ------------------------------------------------------------------
    # 2. Build document text from chunks (cap at MAX_DOCUMENT_CHARS)
    # ------------------------------------------------------------------
    parts: list[str] = []
    total_len = 0

    for chunk in chunks:
        section_header = ""
        if chunk.section_name:
            section_header = f"\n[Section: {chunk.section_name}]\n"

        segment = section_header + (chunk.content or "")

        if total_len + len(segment) > MAX_DOCUMENT_CHARS:
            remaining = MAX_DOCUMENT_CHARS - total_len
            if remaining > 0:
                parts.append(segment[:remaining])
            break

        parts.append(segment)
        total_len += len(segment)

    document_text = "".join(parts)

    if not document_text.strip():
        logger.warning(f"Empty document text for document {document_id}")
        return dict(DEFAULT_RESULT)

    # ------------------------------------------------------------------
    # 3. Call Gemini for contract review
    # ------------------------------------------------------------------
    prompt = CONTRACT_REVIEW_PROMPT.format(document_text=document_text)

    try:
        parsed = await llm.agenerate(
            prompt,
            json=True,
            temperature=0.2,
            max_output_tokens=4096,
            provider="gemini",
            fallback=False,
        )
    except Exception as e:
        logger.warning(f"Gemini contract review call failed (graceful degradation): {e}")
        return dict(DEFAULT_RESULT)

    # ------------------------------------------------------------------
    # 4. Validate JSON response
    # ------------------------------------------------------------------
    try:
        result = _validate_result(parsed)
        logger.info(
            f"Contract review complete for document {document_id}: "
            f"score={result['overall_score']}, "
            f"risks={len(result['risks'])}, "
            f"missing={len(result['missing_clauses'])}"
        )
        return result
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        logger.warning(
            f"Failed to parse Gemini contract review response "
            f"(graceful degradation): {e}"
        )
        return dict(DEFAULT_RESULT)
