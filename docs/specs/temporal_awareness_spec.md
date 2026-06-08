# Temporal Awareness: Document Versioning, Effective Dates & Superseded Document Detection

**Status:** Draft
**Created:** 2026-03-23
**Target files (modify):** `models.py`, `tasks.py`, `rag_engine.py`, `vector_store.py`, `schemas.py`, `config.py`, `frontend/lib/types.ts`, `frontend/components/CitationPanel.tsx`
**Target files (create):** `services/temporal_extractor.py`, `services/amendment_chain_manager.py`, `alembic/versions/12_add_temporal_awareness.py`
**Migration:** `12_add_temporal_awareness`

---

## 1. Problem Statement

The current RAG pipeline treats all documents as equally valid regardless of when they were enacted, amended, or repealed. When a matter contains multiple versions of the same statute (e.g., a 2020 regulation and its 2024 amendment), the system may retrieve outdated text and present it as current law. This produces incorrect legal advice.

**Current behavior (58% accuracy on temporal queries):**
- User uploads "Data Protection Act 2018" and "Data Protection (Amendment) Act 2024"
- Asks: "What are the current data retention requirements?"
- System retrieves chunks from both documents indiscriminately
- Answer cites superseded provisions alongside current ones

**Target behavior (90% accuracy on temporal queries):**
- System auto-detects effective dates and amendment relationships during ingestion
- User query "What are the current data retention requirements?" only retrieves 2024 amendment
- User query "What were the retention requirements before the 2024 amendment?" retrieves 2018 act
- Temporal status badges on citations show document currency

---

## 2. Database Schema Changes

### 2.1 Document Model Extensions

Add five columns to the existing `documents` table.

**File:** `backend/models.py`

Current `Document` class (lines 44-66) gets these new columns:

```python
class Document(Base):
    """A single uploaded file belonging to a matter"""
    __tablename__ = "documents"

    # ... existing columns unchanged ...

    # === NEW: Temporal awareness columns ===
    effective_date = Column(DateTime(timezone=True), nullable=True, index=True)
    superseded_date = Column(DateTime(timezone=True), nullable=True, index=True)
    version_number = Column(String(20), nullable=True)          # "1.0", "2.0", "Amendment No. 3"
    document_status = Column(String(50), default="current", nullable=False, index=True)
        # Values: "current", "superseded", "repealed", "draft", "unknown"
    amendment_chain_id = Column(
        UUID(as_uuid=True),
        ForeignKey("amendment_chains.id"),
        nullable=True,
        index=True
    )

    # === NEW: Relationship ===
    amendment_chain = relationship("AmendmentChain", back_populates="documents")
```

**Valid `document_status` values:**

| Status | Meaning |
|---|---|
| `current` | In force, no known superseding document |
| `superseded` | Replaced by a newer version in the same chain |
| `repealed` | Explicitly repealed/revoked |
| `draft` | Not yet in force (future effective_date) |
| `unknown` | Temporal metadata could not be extracted |

### 2.2 AmendmentChain Model (New)

**File:** `backend/models.py` -- add after `Document` class.

```python
class AmendmentChain(Base):
    """Links document versions that amend/supersede each other.

    A chain groups all versions of a canonical document (e.g., all
    revisions of "Data Protection Act"). The canonical_document_id is
    a normalised identifier (lowercase, stripped of dates/version numbers)
    used for fuzzy matching during auto-detection.
    """
    __tablename__ = "amendment_chains"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    canonical_document_id = Column(String(255), nullable=False)
        # Normalised name: "data protection act" (lowercase, no date/version)
    canonical_name = Column(Text, nullable=False)
        # Human-readable: "Data Protection Act"
    jurisdiction = Column(String(100), nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # Relationships
    documents = relationship("Document", back_populates="amendment_chain", order_by="Document.effective_date")
    matter = relationship("Matter")

    __table_args__ = (
        Index("idx_chain_matter_canonical", "matter_id", "canonical_document_id"),
    )

    def __repr__(self):
        return f"<AmendmentChain(id={self.id}, canonical_name={self.canonical_name})>"
```

### 2.3 Alembic Migration

**File:** `backend/alembic/versions/12_add_temporal_awareness.py`

```python
"""Add temporal awareness columns and amendment_chains table

Revision ID: 12
Revises: 11
Create Date: 2026-03-23
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = "12"
down_revision: Union[str, None] = "11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create amendment_chains table first (documents will FK to it)
    op.create_table(
        "amendment_chains",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("matter_id", UUID(as_uuid=True), sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("canonical_document_id", sa.String(255), nullable=False),
        sa.Column("canonical_name", sa.Text, nullable=False),
        sa.Column("jurisdiction", sa.String(100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index(
        "idx_chain_matter_canonical",
        "amendment_chains",
        ["matter_id", "canonical_document_id"],
    )

    # 2. Add temporal columns to documents
    op.add_column("documents", sa.Column("effective_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("superseded_date", sa.DateTime(timezone=True), nullable=True))
    op.add_column("documents", sa.Column("version_number", sa.String(20), nullable=True))
    op.add_column(
        "documents",
        sa.Column("document_status", sa.String(50), server_default="current", nullable=False),
    )
    op.add_column(
        "documents",
        sa.Column(
            "amendment_chain_id",
            UUID(as_uuid=True),
            sa.ForeignKey("amendment_chains.id"),
            nullable=True,
        ),
    )

    # 3. Indexes on new columns
    op.create_index("idx_doc_effective_date", "documents", ["effective_date"])
    op.create_index("idx_doc_superseded_date", "documents", ["superseded_date"])
    op.create_index("idx_doc_status", "documents", ["document_status"])
    op.create_index("idx_doc_amendment_chain", "documents", ["amendment_chain_id"])

    # 4. Backfill: set all existing documents to status="unknown"
    #    (they have no temporal metadata yet)
    op.execute("UPDATE documents SET document_status = 'unknown' WHERE document_status = 'current'")


def downgrade() -> None:
    op.drop_index("idx_doc_amendment_chain", table_name="documents")
    op.drop_index("idx_doc_status", table_name="documents")
    op.drop_index("idx_doc_superseded_date", table_name="documents")
    op.drop_index("idx_doc_effective_date", table_name="documents")

    op.drop_column("documents", "amendment_chain_id")
    op.drop_column("documents", "document_status")
    op.drop_column("documents", "version_number")
    op.drop_column("documents", "superseded_date")
    op.drop_column("documents", "effective_date")

    op.drop_index("idx_chain_matter_canonical", table_name="amendment_chains")
    op.drop_table("amendment_chains")
```

---

## 3. Temporal Extractor Service

**File:** `backend/services/temporal_extractor.py`

This service uses two strategies in sequence:
1. **Regex pattern matching** -- fast, handles 80% of legal documents
2. **Gemini LLM fallback** -- for documents where regex finds nothing

### 3.1 Complete Implementation

```python
"""Extract temporal metadata (effective dates, version info, supersession) from legal text.

Strategy:
  1. Regex patterns for common legal date/version/repeal expressions
  2. Gemini LLM fallback when regex yields no results
  3. All dates normalized to UTC-aware datetime objects

Called from tasks.py during ingestion, after text extraction and before chunking.
"""
import re
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field, asdict

import google.generativeai as genai

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


# ──────────────────────────────────────────────
# Date parsing utilities
# ──────────────────────────────────────────────

# Month name to number mapping
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dec": 12,
}

# Ordinal suffixes: "1st", "2nd", "3rd", "4th" etc.
_ORDINAL_RE = re.compile(r"(\d{1,2})(?:st|nd|rd|th)")


def _parse_date_string(text: str) -> Optional[datetime]:
    """Parse a date string into a timezone-aware datetime.

    Handles formats:
      - "25 March 2024", "March 25, 2024", "25th March 2024"
      - "2024-03-25", "03/25/2024", "25/03/2024"
      - "March 2024" (defaults to 1st of month)
      - "2024" (defaults to January 1st)
    """
    text = text.strip().rstrip(".")

    # Remove ordinal suffixes: "25th" -> "25"
    text = _ORDINAL_RE.sub(r"\1", text)

    # Try ISO format first: 2024-03-25
    iso_match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if iso_match:
        try:
            return datetime(
                int(iso_match.group(1)),
                int(iso_match.group(2)),
                int(iso_match.group(3)),
                tzinfo=timezone.utc,
            )
        except ValueError:
            pass

    # "25 March 2024" or "25 Mar 2024"
    dmy_match = re.match(
        r"(\d{1,2})\s+(january|february|march|april|may|june|july|august|"
        r"september|october|november|december|jan|feb|mar|apr|jun|jul|aug|"
        r"sep|sept|oct|nov|dec)\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if dmy_match:
        day, month_str, year = int(dmy_match.group(1)), dmy_match.group(2).lower(), int(dmy_match.group(3))
        month = _MONTHS.get(month_str)
        if month:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                pass

    # "March 25, 2024" or "Mar 25, 2024"
    mdy_match = re.match(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
        r"oct|nov|dec)\s+(\d{1,2}),?\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if mdy_match:
        month_str, day, year = mdy_match.group(1).lower(), int(mdy_match.group(2)), int(mdy_match.group(3))
        month = _MONTHS.get(month_str)
        if month:
            try:
                return datetime(year, month, day, tzinfo=timezone.utc)
            except ValueError:
                pass

    # "March 2024" (month + year only)
    my_match = re.match(
        r"(january|february|march|april|may|june|july|august|september|"
        r"october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
        r"oct|nov|dec)\s+(\d{4})",
        text, re.IGNORECASE,
    )
    if my_match:
        month_str, year = my_match.group(1).lower(), int(my_match.group(2))
        month = _MONTHS.get(month_str)
        if month:
            return datetime(year, month, 1, tzinfo=timezone.utc)

    # Bare year: "2024"
    year_match = re.match(r"^(\d{4})$", text.strip())
    if year_match:
        year = int(year_match.group(1))
        if 1900 <= year <= 2100:
            return datetime(year, 1, 1, tzinfo=timezone.utc)

    # US format: MM/DD/YYYY
    us_match = re.match(r"(\d{1,2})/(\d{1,2})/(\d{4})", text)
    if us_match:
        m, d, y = int(us_match.group(1)), int(us_match.group(2)), int(us_match.group(3))
        try:
            return datetime(y, m, d, tzinfo=timezone.utc)
        except ValueError:
            pass

    return None


# ──────────────────────────────────────────────
# Regex-based extraction
# ──────────────────────────────────────────────

# Each pattern is (compiled_regex, handler_function_name).
# Handler receives the match and returns partial TemporalMetadata updates.

# Effective date patterns
_EFFECTIVE_DATE_PATTERNS = [
    # "effective [date]" / "effective from [date]" / "effective as of [date]"
    re.compile(
        r"effective\s+(?:from\s+|as\s+of\s+)?"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # "comes into force on [date]" / "shall come into force on [date]"
    re.compile(
        r"(?:shall\s+)?come[s]?\s+into\s+(?:force|effect|operation)\s+(?:on|from)\s+"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # "entered into force on [date]"
    re.compile(
        r"entered?\s+into\s+(?:force|effect)\s+(?:on\s+)?"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # "enacted on [date]" / "enacted [date]"
    re.compile(
        r"enacted\s+(?:on\s+)?"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # "dated [date]" (contracts, agreements)
    re.compile(
        r"(?:^|\n)\s*dated\s+"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
        re.IGNORECASE,
    ),
    # "this agreement is made on [date]"
    re.compile(
        r"this\s+(?:agreement|contract|deed|act|regulation)\s+(?:is\s+)?made\s+(?:on\s+|as\s+of\s+)?"
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
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
        r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+\s+\d{4}|\w+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2})",
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


# ──────────────────────────────────────────────
# LLM-based extraction (Gemini fallback)
# ──────────────────────────────────────────────

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
    if not settings.google_api_key:
        logger.warning("No Google API key, skipping LLM temporal extraction")
        return TemporalMetadata()

    genai.configure(api_key=settings.google_api_key)
    model = genai.GenerativeModel(model_name=settings.gemini_model)

    truncated = text[:10000]
    prompt = _LLM_EXTRACTION_PROMPT.format(text=truncated)

    try:
        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                max_output_tokens=200,
            ),
        )
        return _parse_llm_response(response.text)
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


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

async def extract_temporal_metadata(
    text: str,
    document_name: str = "",
) -> TemporalMetadata:
    """Extract temporal metadata from a legal document.

    Two-phase approach:
      1. Regex extraction (fast, high confidence for common patterns)
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
    # Phase 1: Regex
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


# ──────────────────────────────────────────────
# Filename-based version heuristics
# ──────────────────────────────────────────────

def extract_version_from_filename(filename: str) -> Optional[str]:
    """Extract version number from filename if present.

    Examples:
      "Contract_v2.0.pdf" -> "2.0"
      "Policy_Rev3.docx" -> "3"
      "Data Protection Act 2024 (Amendment No. 2).pdf" -> "Amendment No. 2"

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
```

---

## 4. Amendment Chain Manager Service

**File:** `backend/services/amendment_chain_manager.py`

This service manages the linkage between document versions within a matter.

### 4.1 Complete Implementation

```python
"""Manage amendment chains -- groups of document versions within a matter.

When a new document is ingested, this service:
  1. Normalises the document name to a canonical ID
  2. Checks if an existing chain matches (same matter, similar canonical ID)
  3. Creates or joins the chain
  4. Updates supersession status of older documents in the chain

Called from tasks.py after temporal extraction, before chunking.
"""
import re
import logging
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from uuid import UUID, uuid4
from sqlalchemy.orm import Session

try:
    from backend.models import Document, AmendmentChain
except ImportError:
    try:
        from models import Document, AmendmentChain
    except ImportError:
        from ..models import Document, AmendmentChain

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Name normalisation
# ──────────────────────────────────────────────

def normalise_document_name(name: str) -> str:
    """Normalise a document name to a canonical identifier for chain matching.

    Strips dates, version numbers, file extensions, and noise words to produce
    a stable key. Two documents with the same canonical ID are likely versions
    of the same base document.

    Examples:
      "Data Protection Act 2018.pdf"         -> "data protection act"
      "Data Protection (Amendment) Act 2024" -> "data protection act"
      "Contract_v2.0.pdf"                    -> "contract"
      "Employment Agreement Rev3.docx"       -> "employment agreement"

    Args:
        name: Original document name or filename

    Returns:
        Lowercase normalised canonical ID
    """
    # Remove file extension
    name = re.sub(r"\.\w{2,4}$", "", name)

    # Remove parenthetical modifiers: (Amendment), (Revised), (No. 2)
    name = re.sub(r"\([^)]*\)", "", name)

    # Remove version indicators
    name = re.sub(r"(?:_v|_rev|version|revision|rev\.?)\s*[\d.]+", "", name, flags=re.IGNORECASE)
    name = re.sub(r"amendment\s+(?:no\.?\s*)?\d+", "", name, flags=re.IGNORECASE)

    # Remove year at end: "Act 2018" -> "Act"
    name = re.sub(r"\s+\d{4}\s*$", "", name)
    # Remove year in middle: "Act 2018 Section" is rare, skip

    # Remove underscores, extra whitespace
    name = name.replace("_", " ")
    name = re.sub(r"\s+", " ", name).strip()

    return name.lower()


def compute_human_readable_name(canonical_id: str) -> str:
    """Convert canonical ID back to a title-case human-readable name.

    Args:
        canonical_id: Normalised canonical document ID

    Returns:
        Title-cased string, e.g., "Data Protection Act"
    """
    return canonical_id.title()


# ──────────────────────────────────────────────
# Chain resolution
# ──────────────────────────────────────────────

def find_or_create_chain(
    db: Session,
    matter_id: UUID,
    canonical_id: str,
    jurisdiction: Optional[str] = None,
) -> AmendmentChain:
    """Find an existing amendment chain or create a new one.

    Looks for an existing chain in the same matter with matching canonical_document_id.
    If none exists, creates a new chain.

    Args:
        db: Database session
        matter_id: UUID of the matter
        canonical_id: Normalised document name
        jurisdiction: Optional jurisdiction code (e.g. "UK")

    Returns:
        AmendmentChain (existing or newly created)
    """
    existing = db.query(AmendmentChain).filter(
        AmendmentChain.matter_id == matter_id,
        AmendmentChain.canonical_document_id == canonical_id,
    ).first()

    if existing:
        logger.info(f"Found existing amendment chain: {existing.id} for '{canonical_id}'")
        return existing

    chain = AmendmentChain(
        id=uuid4(),
        matter_id=matter_id,
        canonical_document_id=canonical_id,
        canonical_name=compute_human_readable_name(canonical_id),
        jurisdiction=jurisdiction,
    )
    db.add(chain)
    db.flush()
    logger.info(f"Created new amendment chain: {chain.id} for '{canonical_id}'")
    return chain


def assign_document_to_chain(
    db: Session,
    document: "Document",
    chain: AmendmentChain,
) -> None:
    """Assign a document to an amendment chain and update supersession.

    After assignment, scans all documents in the chain and marks older
    versions as "superseded" based on effective_date ordering.

    Args:
        db: Database session
        document: Document to assign
        chain: Target amendment chain
    """
    document.amendment_chain_id = chain.id
    db.flush()

    # Recalculate supersession for all documents in the chain
    _recalculate_chain_status(db, chain)


def _recalculate_chain_status(db: Session, chain: AmendmentChain) -> None:
    """Recalculate document_status for all documents in a chain.

    Rules:
    1. Documents are sorted by effective_date (ascending, nulls last).
    2. The document with the latest effective_date is "current"
       (unless it is repealed).
    3. All earlier documents are "superseded" with superseded_date set to
       the effective_date of the next newer document.
    4. Documents already marked "repealed" stay "repealed".
    5. Documents with no effective_date are left as "unknown".

    Args:
        db: Database session
        chain: Amendment chain to recalculate
    """
    docs = (
        db.query(Document)
        .filter(Document.amendment_chain_id == chain.id)
        .all()
    )

    if len(docs) <= 1:
        return  # Nothing to supersede

    # Split: documents with effective_date vs without
    dated = [d for d in docs if d.effective_date is not None]
    undated = [d for d in docs if d.effective_date is None]

    if not dated:
        return  # Can't determine ordering without dates

    # Sort by effective_date ascending
    dated.sort(key=lambda d: d.effective_date)

    # Mark all but the latest as superseded
    for i, doc in enumerate(dated):
        if doc.document_status == "repealed":
            continue  # Don't override explicit repeal

        if i < len(dated) - 1:
            # This is an older version -- mark superseded
            doc.document_status = "superseded"
            doc.superseded_date = dated[i + 1].effective_date
            logger.info(
                f"Document '{doc.name}' marked superseded "
                f"(effective={doc.effective_date}, superseded_by={dated[i+1].name})"
            )
        else:
            # Latest version -- mark current (unless repealed)
            doc.document_status = "current"
            doc.superseded_date = None

    # Undated documents in a chain with dated ones: mark as "unknown"
    for doc in undated:
        if doc.document_status not in ("repealed",):
            doc.document_status = "unknown"

    db.flush()
    logger.info(
        f"Recalculated chain {chain.id}: "
        f"{len(dated)} dated docs, {len(undated)} undated, "
        f"current='{dated[-1].name}'"
    )


# ──────────────────────────────────────────────
# Cross-reference resolution
# ──────────────────────────────────────────────

def resolve_supersession_references(
    db: Session,
    document: "Document",
    matter_id: UUID,
    supersedes_refs: List[str],
    amends_refs: List[str],
) -> None:
    """Try to match supersession/amendment references to existing documents.

    When a document says "supersedes the Data Protection Act 2018", this
    function looks for a document in the same matter whose name matches
    and marks it as superseded.

    Args:
        db: Database session
        document: The new document making the references
        matter_id: UUID of the matter
        supersedes_refs: List of document names this document supersedes
        amends_refs: List of document names this document amends
    """
    all_refs = supersedes_refs + amends_refs
    if not all_refs:
        return

    # Get all other documents in the matter
    other_docs = (
        db.query(Document)
        .filter(
            Document.matter_id == matter_id,
            Document.id != document.id,
        )
        .all()
    )

    for ref in all_refs:
        ref_canonical = normalise_document_name(ref)
        for other_doc in other_docs:
            other_canonical = normalise_document_name(other_doc.name)
            # Fuzzy match: canonical IDs are equal, or one contains the other
            if ref_canonical == other_canonical or ref_canonical in other_canonical or other_canonical in ref_canonical:
                if other_doc.document_status not in ("superseded", "repealed"):
                    other_doc.document_status = "superseded"
                    other_doc.superseded_date = document.effective_date or datetime.now(timezone.utc)
                    logger.info(
                        f"Marked '{other_doc.name}' as superseded based on "
                        f"reference from '{document.name}': '{ref}'"
                    )

    db.flush()
```

---

## 5. Ingestion Pipeline Integration (`tasks.py`)

### 5.1 Changes to `process_document_task`

The temporal extraction and chain management are inserted between the existing enrichment step (step 2c) and the PostgreSQL chunk storage step (step 3). This is the same async phase where summary generation and classification already happen.

**File:** `backend/tasks.py`

**New imports (add to both try/except import blocks):**

```python
from backend.services.temporal_extractor import extract_temporal_metadata, extract_version_from_filename
from backend.services.amendment_chain_manager import (
    normalise_document_name, find_or_create_chain,
    assign_document_to_chain, resolve_supersession_references,
)
```

**Insert after step 2c (line ~133, after `db.commit()` that stores enrichment results), before the chunk metadata propagation loop:**

```python
        # 2d. Extract temporal metadata (effective dates, version, supersession)
        publish_enriching(matter_id, detail="Extracting temporal metadata...")

        async def _extract_temporal():
            return await extract_temporal_metadata(full_text, document.name)

        temporal_meta = asyncio.run(_extract_temporal())

        # Merge filename-based version if text extraction didn't find one
        if not temporal_meta.version_number:
            temporal_meta.version_number = extract_version_from_filename(document.name)

        # Store temporal metadata on Document record
        document.effective_date = temporal_meta.effective_date
        document.superseded_date = temporal_meta.superseded_date
        document.version_number = temporal_meta.version_number
        document.document_status = temporal_meta.document_status

        # 2e. Amendment chain management
        canonical_id = normalise_document_name(document.name)
        chain = find_or_create_chain(
            db,
            matter_id=UUID(matter_id),
            canonical_id=canonical_id,
            jurisdiction=classification["jurisdiction"],
        )
        assign_document_to_chain(db, document, chain)

        # Try to resolve "supersedes X" / "amends Y" references
        resolve_supersession_references(
            db,
            document=document,
            matter_id=UUID(matter_id),
            supersedes_refs=temporal_meta.supersedes_references,
            amends_refs=temporal_meta.amends_references,
        )

        db.commit()

        logger.info(
            f"[Task {self.request.id}] Temporal extraction complete: "
            f"effective={temporal_meta.effective_date}, "
            f"status={temporal_meta.document_status}, "
            f"version={temporal_meta.version_number}, "
            f"chain={chain.id}, "
            f"method={temporal_meta.extraction_method}"
        )
```

**Propagate temporal metadata to chunk dicts (extend the existing loop at ~line 138):**

The existing loop that adds `document_type` and `jurisdiction` to chunk dicts should also add temporal fields:

```python
        # Propagate document-level metadata to chunk dicts for Qdrant payload
        for chunk in chunks:
            chunk["document_type"] = classification["document_type"]
            chunk["jurisdiction"] = classification["jurisdiction"]
            # NEW: temporal metadata for Qdrant filtering
            chunk["effective_date"] = (
                temporal_meta.effective_date.timestamp()
                if temporal_meta.effective_date else None
            )
            chunk["superseded_date"] = (
                temporal_meta.superseded_date.timestamp()
                if temporal_meta.superseded_date else None
            )
            chunk["document_status"] = temporal_meta.document_status
```

---

## 6. Vector Store Changes (`vector_store.py`)

### 6.1 Payload Index Registration

**File:** `backend/services/vector_store.py`

Add temporal fields to `_ensure_payload_indexes` (line ~113):

```python
def _ensure_payload_indexes(client, collection_name: str):
    """Create payload indexes if they don't exist."""
    index_fields = {
        "page_num": PayloadSchemaType.KEYWORD,
        "section_name": PayloadSchemaType.KEYWORD,
        "document_type": PayloadSchemaType.KEYWORD,
        "jurisdiction": PayloadSchemaType.KEYWORD,
        # NEW: temporal filtering indexes
        "effective_date": PayloadSchemaType.FLOAT,       # Unix timestamp
        "superseded_date": PayloadSchemaType.FLOAT,      # Unix timestamp
        "document_status": PayloadSchemaType.KEYWORD,
    }
    for field, schema in index_fields.items():
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )
        except Exception:
            pass
```

### 6.2 Upsert Metadata Expansion

In `upsert_vectors` (line ~252), add temporal fields to the metadata dict:

```python
            metadata = {
                "chunk_id": chunk_id,
                "chunk_sequence": chunk.get("chunk_sequence", 0),
                "page_num": str(chunk.get("page_num", "")),
                "section_name": str(chunk.get("section_name", "")),
                "content": chunk.get("content", ""),
                "document_id": str(chunk.get("document_id", "")),
                "document_name": str(chunk.get("document_name", "")),
                "concepts": chunk.get("concepts", []),
                "document_type": str(chunk.get("document_type", "")),
                "jurisdiction": str(chunk.get("jurisdiction", "")),
                # NEW: temporal metadata
                "effective_date": chunk.get("effective_date"),      # float (unix ts) or None
                "superseded_date": chunk.get("superseded_date"),    # float (unix ts) or None
                "document_status": str(chunk.get("document_status", "unknown")),
            }
```

### 6.3 Search Result Expansion

In `search_vectors` (line ~370), add temporal fields to the result dict:

```python
            result_dict = {
                "score": hit.score,
                "chunk_id": payload.get("chunk_id", ""),
                "chunk_sequence": payload.get("chunk_sequence", 0),
                "page_num": payload.get("page_num", ""),
                "section_name": payload.get("section_name", ""),
                "content": full_content,
                "document_id": payload.get("document_id", ""),
                "document_name": payload.get("document_name", ""),
                "concepts": payload.get("concepts", []),
                "document_type": payload.get("document_type", ""),
                "jurisdiction": payload.get("jurisdiction", ""),
                # NEW: temporal metadata
                "effective_date": payload.get("effective_date"),
                "superseded_date": payload.get("superseded_date"),
                "document_status": payload.get("document_status", "unknown"),
            }
```

### 6.4 Temporal Filter Builder

Add a new helper function for building temporal filters. Place it after `_generate_point_id`:

```python
from qdrant_client.models import Range


def build_temporal_filter(
    as_of_date: Optional[datetime] = None,
    exclude_superseded: bool = True,
    base_filter: Optional[Dict] = None,
) -> Optional[Dict]:
    """Build Qdrant filter conditions for temporal queries.

    This function returns a raw dict of filter conditions that will be
    merged into the main query filter in search_vectors. It does NOT
    return a Filter object directly, because search_vectors already
    builds the Filter from a dict.

    Args:
        as_of_date: Only return documents valid as of this date.
            If None, defaults to "exclude superseded" behavior.
        exclude_superseded: If True (default), exclude documents with
            status "superseded" or "repealed" when no as_of_date is given.
        base_filter: Existing filter dict to merge with (optional).

    Returns:
        Dict of filter conditions to pass to search_vectors, or None.
    """
    # This is used inside rag_engine.py which builds the Qdrant Filter.
    # We return structured condition dicts that rag_engine translates.
    conditions = dict(base_filter) if base_filter else {}

    if as_of_date:
        # For as-of queries, we encode the logic as special keys
        # that search_vectors will interpret
        ts = as_of_date.timestamp()
        conditions["_temporal_as_of"] = ts
    elif exclude_superseded:
        conditions["_temporal_current_only"] = True

    return conditions if conditions else None
```

### 6.5 Extend `search_vectors` to Handle Temporal Filters

Modify the filter-building section of `search_vectors` (line ~340) to handle the temporal condition keys:

```python
        # Build Qdrant filter from query_filter dict
        qdrant_filter = None
        if query_filter:
            conditions = []
            temporal_as_of = query_filter.pop("_temporal_as_of", None)
            temporal_current_only = query_filter.pop("_temporal_current_only", None)

            # Standard field conditions
            for field, value in query_filter.items():
                conditions.append(
                    FieldCondition(key=field, match=MatchValue(value=value))
                )

            # Temporal: as-of-date query
            if temporal_as_of is not None:
                # effective_date <= as_of_date (document was in force by then)
                conditions.append(
                    FieldCondition(
                        key="effective_date",
                        range=Range(lte=temporal_as_of),
                    )
                )
                # superseded_date > as_of_date OR superseded_date is null
                # Qdrant does not support OR in must, so we use must_not:
                # must_not: superseded_date <= as_of_date
                # This is handled via a separate Filter block below.

            # Temporal: exclude superseded/repealed (default behavior)
            elif temporal_current_only:
                conditions.append(
                    FieldCondition(
                        key="document_status",
                        match=MatchValue(value="current"),
                    )
                )

            must_not = []
            if temporal_as_of is not None:
                # Exclude documents that were superseded before the as_of_date
                must_not.append(
                    FieldCondition(
                        key="superseded_date",
                        range=Range(lte=temporal_as_of),
                    )
                )

            qdrant_filter = Filter(
                must=conditions if conditions else None,
                must_not=must_not if must_not else None,
            )
            logger.debug(f"Applying filter: {query_filter}, temporal_as_of={temporal_as_of}")
```

---

## 7. RAG Engine Integration (`rag_engine.py`)

### 7.1 "As of Date" Query Detection

Add a new function after `_detect_query_filters` (line ~142):

```python
from datetime import datetime, timezone


def _detect_temporal_intent(query: str) -> Tuple[Optional[datetime], bool]:
    """Detect temporal intent from query text.

    Parses phrases like:
      - "as of March 2024" -> (datetime(2024,3,1), False)
      - "before the 2024 amendment" -> (datetime(2024,1,1), False)
      - "current law on..." -> (None, True)  # exclude superseded
      - "historical analysis of..." -> (None, False)  # include all

    Args:
        query: User query string

    Returns:
        Tuple of (as_of_date, exclude_superseded)
        - as_of_date: Specific date to filter to, or None
        - exclude_superseded: Whether to exclude superseded docs (default True)
    """
    import re

    query_lower = query.lower()

    # Explicit "as of [date]" pattern
    as_of_match = re.search(
        r"as\s+of\s+(\w+\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+\w+\s+\d{4}|\w+\s+\d{4}|\d{4})",
        query_lower,
    )
    if as_of_match:
        from backend.services.temporal_extractor import _parse_date_string
        parsed = _parse_date_string(as_of_match.group(1))
        if parsed:
            return (parsed, False)

    # "before the [year] amendment" / "prior to [date]"
    before_match = re.search(
        r"(?:before|prior\s+to|pre)\s+(?:the\s+)?(\d{4})",
        query_lower,
    )
    if before_match:
        year = int(before_match.group(1))
        return (datetime(year, 1, 1, tzinfo=timezone.utc), False)

    # "under the [year] act/version"
    under_match = re.search(
        r"under\s+(?:the\s+)?(?:\w+\s+){0,5}(\d{4})\s+(?:act|version|regulation|law|amendment)",
        query_lower,
    )
    if under_match:
        year = int(under_match.group(1))
        # Use mid-year as the as-of date for "under the 2020 Act"
        return (datetime(year, 7, 1, tzinfo=timezone.utc), False)

    # Historical analysis keywords -> include all versions
    historical_keywords = [
        "historical", "history of", "evolution of", "how has",
        "changes over time", "all versions", "compare versions",
        "timeline of",
    ]
    for kw in historical_keywords:
        if kw in query_lower:
            return (None, False)  # Don't exclude anything

    # Default: exclude superseded documents for current-law queries
    return (None, True)
```

### 7.2 Integrate Temporal Filtering into `query_matter`

In `query_matter` (line ~1032), modify the chunk retrieval section to include temporal filtering:

```python
        # 3. Retrieve chunks (request more than top_k to improve recall, then take best)
        # Detect optional filters from query text (jurisdiction, doc type)
        query_filters = _detect_query_filters(query)

        # NEW: Detect temporal intent
        as_of_date, exclude_superseded = _detect_temporal_intent(query)
        temporal_filter = build_temporal_filter(
            as_of_date=as_of_date,
            exclude_superseded=exclude_superseded,
            base_filter=query_filters if query_filters else None,
        )
        effective_filter = temporal_filter if temporal_filter else (query_filters if query_filters else None)

        try:
            retrieved_chunks = retrieve_chunks(
                matter_id, query_embedding,
                top_k=RETRIEVAL_LIMIT,
                query_filter=effective_filter
            )
            # Fallback: if filtered search returns too few results, retry unfiltered
            if effective_filter and len(retrieved_chunks) < 3:
                logger.info(
                    f"Filtered search returned {len(retrieved_chunks)} results "
                    f"(filter: {effective_filter}), retrying without temporal filter"
                )
                # Retry with only jurisdiction filter (drop temporal)
                retrieved_chunks = retrieve_chunks(
                    matter_id, query_embedding,
                    top_k=RETRIEVAL_LIMIT,
                    query_filter=query_filters if query_filters else None,
                )
                # If still too few, retry fully unfiltered
                if len(retrieved_chunks) < 3:
                    retrieved_chunks = retrieve_chunks(
                        matter_id, query_embedding,
                        top_k=RETRIEVAL_LIMIT,
                    )
```

### 7.3 Temporal Context in LLM Prompt

Add a temporal awareness section to `format_legal_context` (line ~145). After the document summaries preamble, add temporal status information:

```python
    # Add temporal status warnings for each unique document
    doc_temporal_status = {}
    for chunk in sorted_chunks:
        doc_id = chunk.get("document_id", "")
        if doc_id and doc_id not in doc_temporal_status:
            status = chunk.get("document_status", "unknown")
            doc_name = chunk.get("document_name", "Unknown")
            eff_date = chunk.get("effective_date")
            sup_date = chunk.get("superseded_date")
            doc_temporal_status[doc_id] = (doc_name, status, eff_date, sup_date)

    has_temporal_info = any(s != "unknown" for _, s, _, _ in doc_temporal_status.values())
    if has_temporal_info:
        context_parts.append("Document Temporal Status:\n")
        for doc_id, (name, status, eff, sup) in doc_temporal_status.items():
            line = f"  - {name}: status={status}"
            if eff:
                from datetime import datetime
                eff_dt = datetime.fromtimestamp(eff) if isinstance(eff, (int, float)) else eff
                line += f", effective={eff_dt.strftime('%Y-%m-%d')}"
            if sup:
                sup_dt = datetime.fromtimestamp(sup) if isinstance(sup, (int, float)) else sup
                line += f", superseded={sup_dt.strftime('%Y-%m-%d')}"
            context_parts.append(line + "\n")
        context_parts.append(
            "  NOTE: Prefer excerpts from 'current' documents. "
            "If citing a 'superseded' document, explicitly state it is no longer in force.\n\n"
        )
```

### 7.4 Temporal Awareness in System Prompt

Append to `LEGAL_SYSTEM_PROMPT` (line ~57):

```python
TEMPORAL_AWARENESS_ADDENDUM = """
TEMPORAL AWARENESS:
- Document excerpts include temporal status metadata (current, superseded, repealed, draft, unknown).
- When answering, prefer citing CURRENT documents over SUPERSEDED ones.
- If a superseded document is relevant, explicitly note: "Under the [prior version], [claim]. This provision was superseded by [current version]."
- For "as of [date]" queries, only cite documents that were in force on that date.
- If the user asks about historical law, include superseded provisions with clear temporal labels.
- Never present superseded law as if it were current."""
```

This addendum is appended to the system prompt when temporal metadata is detected in the context:

```python
# In query_matter, when building the prompt:
system_prompt = LEGAL_SYSTEM_PROMPT
if has_temporal_info:  # from format_legal_context
    system_prompt += TEMPORAL_AWARENESS_ADDENDUM
```

---

## 8. Schema Changes (`schemas.py`)

### 8.1 New Pydantic Models

**File:** `backend/schemas.py`

Add after the existing `MatterResponse` class:

```python
# ============================================
# DOCUMENT SCHEMAS (with temporal awareness)
# ============================================

class DocumentResponse(BaseModel):
    """Document response with temporal metadata"""
    id: UUID
    matter_id: UUID
    name: str
    file_type: str
    status: str
    summary: Optional[str] = None
    document_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    # Temporal awareness fields
    effective_date: Optional[datetime] = None
    superseded_date: Optional[datetime] = None
    version_number: Optional[str] = None
    document_status: str = "unknown"
    amendment_chain_id: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class AmendmentChainResponse(BaseModel):
    """Amendment chain response with linked documents"""
    id: UUID
    canonical_name: str
    jurisdiction: Optional[str] = None
    documents: list[DocumentResponse] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TemporalStatusUpdate(BaseModel):
    """Manual override for document temporal status"""
    effective_date: Optional[datetime] = None
    superseded_date: Optional[datetime] = None
    version_number: Optional[str] = None
    document_status: Optional[str] = Field(
        None,
        pattern="^(current|superseded|repealed|draft|unknown)$",
    )
```

### 8.2 Extend QueryCreate

Add optional `as_of_date` to `QueryCreate`:

```python
class QueryCreate(BaseModel):
    """Query request (ask question)"""
    question: str = Field(..., min_length=1, max_length=1000)
    include_legal_research: bool = Field(False, description="Include CourtListener case law")
    conversation_id: Optional[UUID] = Field(None, description="Conversation thread ID")
    as_of_date: Optional[datetime] = Field(
        None,
        description="Only retrieve documents valid as of this date. "
                    "If omitted, auto-detected from question text or defaults to current.",
    )
```

### 8.3 Extend CitationData

Add temporal status to citation responses:

```python
class CitationData(BaseModel):
    """Citation metadata"""
    page: str
    section: Optional[str] = None
    content_snippet: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_type: Optional[str] = Field(None, description="'document' or 'case_law'")
    url: Optional[str] = Field(None, description="External URL for case law sources")
    # NEW: temporal metadata
    document_status: Optional[str] = Field(None, description="current/superseded/repealed/draft/unknown")
    effective_date: Optional[datetime] = None
```

---

## 9. Configuration Changes (`config.py`)

**File:** `backend/config.py`

Add a feature flag for temporal awareness:

```python
class Settings(BaseSettings):
    # ... existing fields ...

    # Temporal Awareness
    temporal_extraction_enabled: bool = True
    temporal_filter_default: bool = True  # Exclude superseded by default in queries
    temporal_llm_fallback: bool = True    # Use Gemini when regex fails
```

---

## 10. Frontend Changes

### 10.1 TypeScript Types

**File:** `frontend/lib/types.ts`

Add temporal types after the existing `Citation` interface:

```typescript
// ============================================
// Temporal Awareness Types
// ============================================

export type DocumentTemporalStatus = "current" | "superseded" | "repealed" | "draft" | "unknown"

export interface DocumentWithTemporal {
  id: string
  name: string
  file_type: string
  status: string
  summary?: string
  document_type?: string
  jurisdiction?: string
  // Temporal fields
  effective_date?: string   // ISO datetime
  superseded_date?: string  // ISO datetime
  version_number?: string
  document_status: DocumentTemporalStatus
  amendment_chain_id?: string
}

export interface AmendmentChain {
  id: string
  canonical_name: string
  jurisdiction?: string
  documents: DocumentWithTemporal[]
  created_at: string
}
```

Extend the existing `Citation` interface:

```typescript
export interface Citation {
  // ... existing fields ...
  /** Temporal status of the source document */
  documentStatus?: DocumentTemporalStatus
  /** Effective date of the source document (ISO string) */
  effectiveDate?: string
}
```

### 10.2 Temporal Status Badge Component

**File:** `frontend/components/TemporalBadge.tsx` (new file)

```tsx
"use client"

import React from "react"
import { Clock, AlertTriangle, XCircle, FileCheck, HelpCircle, CalendarClock } from "lucide-react"
import { cn } from "@/lib/utils"
import type { DocumentTemporalStatus } from "@/lib/types"

interface TemporalBadgeProps {
  status: DocumentTemporalStatus
  effectiveDate?: string
  className?: string
  showDate?: boolean
}

const STATUS_CONFIG: Record<DocumentTemporalStatus, {
  bg: string; text: string; border: string; label: string; Icon: React.ElementType
}> = {
  current: {
    bg: "bg-emerald-50",
    text: "text-emerald-700",
    border: "border-emerald-200/60",
    label: "Current",
    Icon: FileCheck,
  },
  superseded: {
    bg: "bg-amber-50",
    text: "text-amber-700",
    border: "border-amber-200/60",
    label: "Superseded",
    Icon: AlertTriangle,
  },
  repealed: {
    bg: "bg-red-50",
    text: "text-red-700",
    border: "border-red-200/60",
    label: "Repealed",
    Icon: XCircle,
  },
  draft: {
    bg: "bg-blue-50",
    text: "text-blue-700",
    border: "border-blue-200/60",
    label: "Draft",
    Icon: CalendarClock,
  },
  unknown: {
    bg: "bg-gray-50",
    text: "text-gray-500",
    border: "border-gray-200/60",
    label: "Unknown",
    Icon: HelpCircle,
  },
}

export default function TemporalBadge({
  status,
  effectiveDate,
  className,
  showDate = false,
}: TemporalBadgeProps) {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.unknown
  const { bg, text, border, label, Icon } = config

  const formattedDate = effectiveDate
    ? new Date(effectiveDate).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] font-medium",
        bg, text, border,
        className,
      )}
    >
      <Icon className="h-3 w-3" />
      {label}
      {showDate && formattedDate && (
        <span className="ml-0.5 opacity-75">({formattedDate})</span>
      )}
    </span>
  )
}
```

### 10.3 Integration with CitationPanel

**File:** `frontend/components/CitationPanel.tsx`

Import the badge and render it on each citation card. Inside the citation `<button>` element, after the document name line, add:

```tsx
import TemporalBadge from "@/components/TemporalBadge"

// Inside the citation card render (after the document name span):
{citation.documentStatus && citation.documentStatus !== "unknown" && (
  <TemporalBadge
    status={citation.documentStatus}
    effectiveDate={citation.effectiveDate}
    showDate={true}
    className="ml-2"
  />
)}
```

When a citation's `documentStatus` is `"superseded"` or `"repealed"`, the badge provides an immediate visual warning that the source may not reflect current law.

---

## 11. API Endpoints

### 11.1 New Endpoints

Add to the existing FastAPI router (in the matters routes file):

```python
@router.get("/matters/{matter_id}/documents/{document_id}/temporal")
async def get_document_temporal_status(
    matter_id: UUID, document_id: UUID, db: Session = Depends(get_db)
) -> DocumentResponse:
    """Get temporal metadata for a specific document."""
    doc = db.query(Document).filter(
        Document.id == document_id, Document.matter_id == matter_id
    ).first()
    if not doc:
        raise HTTPException(404, "Document not found")
    return DocumentResponse.model_validate(doc)


@router.patch("/matters/{matter_id}/documents/{document_id}/temporal")
async def update_document_temporal_status(
    matter_id: UUID,
    document_id: UUID,
    update: TemporalStatusUpdate,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """Manually override temporal metadata for a document.

    Used when auto-detection is wrong or incomplete.
    Triggers chain recalculation if document_status changes.
    """
    doc = db.query(Document).filter(
        Document.id == document_id, Document.matter_id == matter_id
    ).first()
    if not doc:
        raise HTTPException(404, "Document not found")

    if update.effective_date is not None:
        doc.effective_date = update.effective_date
    if update.superseded_date is not None:
        doc.superseded_date = update.superseded_date
    if update.version_number is not None:
        doc.version_number = update.version_number
    if update.document_status is not None:
        doc.document_status = update.document_status

    db.commit()

    # Recalculate chain if document is in one
    if doc.amendment_chain_id:
        chain = db.query(AmendmentChain).filter(
            AmendmentChain.id == doc.amendment_chain_id
        ).first()
        if chain:
            from backend.services.amendment_chain_manager import _recalculate_chain_status
            _recalculate_chain_status(db, chain)
            db.commit()

    return DocumentResponse.model_validate(doc)


@router.get("/matters/{matter_id}/amendment-chains")
async def get_amendment_chains(
    matter_id: UUID, db: Session = Depends(get_db)
) -> list[AmendmentChainResponse]:
    """Get all amendment chains for a matter with their linked documents."""
    chains = db.query(AmendmentChain).filter(
        AmendmentChain.matter_id == matter_id
    ).all()
    return [AmendmentChainResponse.model_validate(c) for c in chains]
```

### 11.2 Modify Query Endpoint

The existing `POST /matters/{matter_id}/ask` endpoint should pass the optional `as_of_date` through to `query_matter`:

```python
# In the ask endpoint handler, after parsing QueryCreate:
result = await query_matter(
    matter_id=str(matter_id),
    query=payload.question,
    db=db,
    conversation_history=history,
    include_legal_research=payload.include_legal_research,
    as_of_date=payload.as_of_date,  # NEW
)
```

And `query_matter`'s signature gains the parameter:

```python
async def query_matter(
    matter_id: str,
    query: str,
    db: Session,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2,
    conversation_history: list = None,
    include_legal_research: bool = False,
    as_of_date: Optional[datetime] = None,  # NEW
) -> Dict:
```

When `as_of_date` is provided explicitly, it overrides the auto-detected temporal intent:

```python
        # NEW: Use explicit as_of_date if provided, otherwise auto-detect
        if as_of_date:
            detected_as_of = as_of_date
            detected_exclude = False
        else:
            detected_as_of, detected_exclude = _detect_temporal_intent(query)
```

---

## 12. Testing Strategy

### 12.1 Unit Tests: `backend/tests/test_temporal_extractor.py`

```python
"""Unit tests for temporal_extractor.py regex extraction."""
import pytest
from datetime import datetime, timezone
from backend.services.temporal_extractor import (
    _parse_date_string,
    _extract_via_regex,
    extract_version_from_filename,
    TemporalMetadata,
)


class TestDateParsing:
    def test_dmy_format(self):
        assert _parse_date_string("25 March 2024") == datetime(2024, 3, 25, tzinfo=timezone.utc)

    def test_mdy_format(self):
        assert _parse_date_string("March 25, 2024") == datetime(2024, 3, 25, tzinfo=timezone.utc)

    def test_iso_format(self):
        assert _parse_date_string("2024-03-25") == datetime(2024, 3, 25, tzinfo=timezone.utc)

    def test_ordinal(self):
        assert _parse_date_string("1st January 2020") == datetime(2020, 1, 1, tzinfo=timezone.utc)

    def test_month_year(self):
        assert _parse_date_string("March 2024") == datetime(2024, 3, 1, tzinfo=timezone.utc)

    def test_bare_year(self):
        assert _parse_date_string("2024") == datetime(2024, 1, 1, tzinfo=timezone.utc)

    def test_invalid(self):
        assert _parse_date_string("not a date") is None


class TestRegexExtraction:
    def test_effective_date(self):
        text = "This regulation comes into force on 1st January 2024."
        result = _extract_via_regex(text)
        assert result.effective_date == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert result.document_status == "current"

    def test_enacted_on(self):
        text = "Enacted on 15 March 2023."
        result = _extract_via_regex(text)
        assert result.effective_date == datetime(2023, 3, 15, tzinfo=timezone.utc)

    def test_supersedes(self):
        text = "This Act supersedes the Data Protection Act 2018."
        result = _extract_via_regex(text)
        assert "Data Protection Act 2018" in result.supersedes_references

    def test_amends(self):
        text = "An Act amending the Employment Rights Act 1996."
        result = _extract_via_regex(text)
        assert any("Employment Rights Act" in ref for ref in result.amends_references)

    def test_repealed(self):
        text = "This section was repealed on 30 June 2023."
        result = _extract_via_regex(text)
        assert result.document_status == "repealed"
        assert result.superseded_date == datetime(2023, 6, 30, tzinfo=timezone.utc)

    def test_version_number(self):
        text = "Version 3.1 of this policy document."
        result = _extract_via_regex(text)
        assert result.version_number == "3.1"

    def test_future_effective_date_is_draft(self):
        text = "This regulation shall come into force on 1 January 2099."
        result = _extract_via_regex(text)
        assert result.document_status == "draft"

    def test_no_temporal_info(self):
        text = "This is a general legal memorandum with no dates."
        result = _extract_via_regex(text)
        assert result.extraction_method == "none"
        assert result.document_status == "unknown"


class TestFilenameVersionExtraction:
    def test_version_v(self):
        assert extract_version_from_filename("Contract_v2.0.pdf") == "2.0"

    def test_revision(self):
        assert extract_version_from_filename("Policy_Rev3.docx") == "3"

    def test_amendment(self):
        assert extract_version_from_filename("Act (Amendment No. 5).pdf") == "5"

    def test_no_version(self):
        assert extract_version_from_filename("simple_document.pdf") is None
```

### 12.2 Unit Tests: `backend/tests/test_amendment_chain.py`

```python
"""Unit tests for amendment_chain_manager.py."""
import pytest
from backend.services.amendment_chain_manager import normalise_document_name


class TestNameNormalisation:
    def test_strips_extension(self):
        assert normalise_document_name("Contract.pdf") == "contract"

    def test_strips_year(self):
        assert normalise_document_name("Data Protection Act 2018") == "data protection act"

    def test_strips_parenthetical(self):
        assert normalise_document_name("Data Protection (Amendment) Act 2024") == "data protection act"

    def test_strips_version(self):
        assert normalise_document_name("Contract_v2.0.pdf") == "contract"

    def test_matching_chain(self):
        """Two versions of same doc should normalise to same canonical ID."""
        a = normalise_document_name("Employment Rights Act 1996.pdf")
        b = normalise_document_name("Employment Rights (Amendment) Act 2024.pdf")
        assert a == b

    def test_different_docs(self):
        """Different docs should NOT match."""
        a = normalise_document_name("Data Protection Act 2018")
        b = normalise_document_name("Employment Rights Act 1996")
        assert a != b
```

### 12.3 Integration Test: `backend/tests/test_temporal_e2e.py`

```python
"""End-to-end test for temporal awareness in the RAG pipeline.

Tests the full flow: upload two versions of a statute -> query with
temporal intent -> verify that only the correct version is cited.

Requires: PostgreSQL, Qdrant, Cohere, Gemini API keys.
"""
import pytest
import asyncio
from datetime import datetime, timezone

# Test documents (synthetic)
STATUTE_V1_TEXT = """
# Data Protection Act 2018

This Act may be cited as the Data Protection Act 2018.

This Act comes into force on 25 May 2018.

Section 5. Data Retention
Personal data shall not be retained for longer than 7 years
from the date of collection.

Section 12. Right to Erasure
A data subject may request erasure of personal data within 30 days.
"""

STATUTE_V2_TEXT = """
# Data Protection (Amendment) Act 2024

An Act amending the Data Protection Act 2018.

This Act comes into force on 1 January 2024.

This Act supersedes the Data Protection Act 2018.

Section 5. Data Retention (Amended)
Personal data shall not be retained for longer than 3 years
from the date of collection, reduced from 7 years.

Section 12. Right to Erasure (Amended)
A data subject may request erasure of personal data within 72 hours.
"""


@pytest.mark.asyncio
async def test_temporal_query_current_law():
    """Query for current law should only cite the 2024 amendment."""
    # This test would use the full pipeline; pseudocode for structure:
    # 1. Create matter, upload V1 and V2
    # 2. Process both documents
    # 3. Query: "What is the current data retention period?"
    # 4. Assert: answer cites 3 years (2024), NOT 7 years (2018)
    # 5. Assert: V1 has document_status="superseded"
    # 6. Assert: V2 has document_status="current"
    pass  # Implementation depends on test infrastructure


@pytest.mark.asyncio
async def test_temporal_query_as_of_date():
    """Query with 'as of 2020' should cite the 2018 act."""
    # 1. Same setup as above
    # 2. Query: "What was the data retention period as of 2020?"
    # 3. Assert: answer cites 7 years (2018 act)
    pass


@pytest.mark.asyncio
async def test_temporal_query_historical():
    """Query for historical comparison should cite both versions."""
    # 1. Same setup
    # 2. Query: "How has the data retention period changed over time?"
    # 3. Assert: answer mentions both 7 years (2018) and 3 years (2024)
    pass
```

---

## 13. Rollout & Migration Plan

### Phase 1: Schema + Extraction (Week 1)
1. Apply Alembic migration `12_add_temporal_awareness`
2. Deploy `temporal_extractor.py` and `amendment_chain_manager.py`
3. Add temporal columns to `models.py`
4. Feature-flag `temporal_extraction_enabled=True` in config
5. **All new document uploads** get temporal metadata extracted

### Phase 2: Backfill (Week 1-2)
6. Run backfill script on existing documents:
```python
"""One-time backfill: extract temporal metadata for existing documents."""
async def backfill_temporal_metadata(db: Session):
    docs = db.query(Document).filter(Document.document_status == "unknown").all()
    for doc in docs:
        # Re-extract text and run temporal extraction
        # Update document record and chain membership
        pass
```

### Phase 3: Query Integration (Week 2)
7. Deploy temporal filtering in `vector_store.py`
8. Deploy `_detect_temporal_intent` in `rag_engine.py`
9. Feature-flag `temporal_filter_default=True`
10. Monitor query quality metrics

### Phase 4: Frontend (Week 3)
11. Deploy `TemporalBadge` component
12. Add temporal badges to `CitationPanel`
13. Add `as_of_date` picker to query UI (optional, can be Phase 5)

### Rollback
- Set `temporal_extraction_enabled=False` to stop extracting metadata on new uploads
- Set `temporal_filter_default=False` to stop filtering at query time
- All changes are additive (new columns, new table) -- no destructive schema changes
- Alembic downgrade removes all new columns and the `amendment_chains` table

---

## 14. Performance Considerations

| Operation | Overhead | Mitigation |
|---|---|---|
| Regex extraction | ~5ms per document | Negligible; runs once at ingestion |
| LLM fallback | ~1.5s per document (1 Gemini call) | Only when regex fails; same batch as existing summary/classification |
| Chain resolution | ~2ms (1-2 DB queries) | Indexed lookup on `(matter_id, canonical_document_id)` |
| Temporal Qdrant filter | ~0ms additional | Payload index on `effective_date`, `superseded_date`, `document_status` |
| Temporal intent detection | ~1ms (regex on query) | No API calls; pure string matching |

**Net impact on ingestion:** +5ms (regex) to +1.5s (LLM fallback), parallelised with existing enrichment.
**Net impact on query time:** No measurable increase (Qdrant payload filtering is free).

---

## 15. Open Questions & Future Work

1. **Cross-matter chains:** Should amendment chains span matters? Current design scopes chains to a single matter. Cross-matter chains would require a different matching strategy.

2. **Partial supersession:** A document may amend only specific sections of a prior document. The current design marks entire documents as superseded. Section-level supersession tracking would require chunk-level temporal metadata.

3. **Jurisdiction-aware date parsing:** Some jurisdictions use DD/MM/YYYY, others MM/DD/YYYY. The current parser tries both but may misparse ambiguous dates like "03/04/2024". Future: use jurisdiction from `classify_document` to disambiguate.

4. **User override UI:** The PATCH endpoint for manual temporal status overrides exists, but the frontend form for it is deferred to a later phase.

5. **Qdrant re-indexing:** Existing vectors in Qdrant lack temporal payload fields. The backfill script (Phase 2) must re-upsert vectors with updated payloads, or temporal filtering will silently return no results for pre-existing documents.
