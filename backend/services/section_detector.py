"""Legal document section boundary detection.

Detects structural boundaries in legal documents (Articles, Sections,
Exhibits, contract headers, litigation sections) using regex patterns.
Used for plain-text documents that lack markdown header formatting.
"""
import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

# Section patterns ordered by specificity (most specific first)
SECTION_PATTERNS = [
    # Articles
    (r'^(?:ARTICLE|Art\.)\s+([IVXLCDM]+|\d+)[.:\s—–-]*(.*)$', 'article'),
    # Numbered sections (e.g., "Section 1.2", "§ 3")
    (r'^(?:SECTION|Section|Sec\.|§)\s*(\d+(?:\.\d+)*)[.:\s—–-]*(.*)$', 'section'),
    # Contract headers (all-caps)
    (r'^(RECITALS|WHEREAS|DEFINITIONS|REPRESENTATIONS\s+AND\s+WARRANTIES|'
     r'COVENANTS|CONDITIONS\s+PRECEDENT|INDEMNIFICATION|TERMINATION|'
     r'GOVERNING\s+LAW|MISCELLANEOUS|GENERAL\s+PROVISIONS|'
     r'CONFIDENTIALITY|NON-COMPETITION|NON-SOLICITATION|'
     r'LIMITATION\s+OF\s+LIABILITY|DISPUTE\s+RESOLUTION|'
     r'FORCE\s+MAJEURE|ASSIGNMENT|NOTICES|SEVERABILITY|'
     r'ENTIRE\s+AGREEMENT|AMENDMENTS?|WAIVER|'
     r'INTELLECTUAL\s+PROPERTY|PAYMENT\s+TERMS?)\s*$', 'contract_header'),
    # Exhibits, Schedules, Annexes
    (r'^(?:EXHIBIT|SCHEDULE|ANNEX|APPENDIX)\s+([A-Z]|\d+)[.:\s—–-]*(.*)$', 'exhibit'),
    # Litigation sections
    (r'^(?:COUNT|CLAIM|CAUSE\s+OF\s+ACTION)\s+([IVXLCDM]+|\d+)[.:\s—–-]*(.*)$', 'litigation_count'),
    (r'^(PRAYER\s+FOR\s+RELIEF|RELIEF\s+REQUESTED|WHEREFORE)\s*$', 'litigation_relief'),
    (r'^(STATEMENT\s+OF\s+FACTS?|FACTUAL\s+BACKGROUND|BACKGROUND)\s*$', 'statement_of_facts'),
    (r'^(ARGUMENT|LEGAL\s+ANALYSIS|DISCUSSION|CONCLUSIONS?\s+OF\s+LAW)\s*$', 'legal_argument'),
    # UK/EU/Commonwealth patterns
    (r'^(?:CLAUSE|Clause)\s+(\d+(?:\.\d+)*)[.:\s—–-]*(.*)$', 'clause'),
    (r'^(?:PART|Part)\s+([IVXLCDM]+|\d+)[.:\s—–-]*(.*)$', 'part'),
    (r'^(?:REGULATION|Regulation)\s+(\d+(?:\.\d+)*)[.:\s—–-]*(.*)$', 'regulation'),
    # Chapters and Divisions (AU, CA, UK - used in long legislation)
    (r'^(?:CHAPTER|Chapter)\s+([IVXLCDM]+|\d+)[.:\s—–-]*(.*)$', 'chapter'),
    (r'^(?:DIVISION|Division)\s+(\d+(?:\.\d+)*)[.:\s—–-]*(.*)$', 'division'),
    (r'^(?:SUBDIVISION|Subdivision)\s+([A-Z]|\d+(?:\.\d+)*)[.:\s—–-]*(.*)$', 'subdivision'),
    # US Title/Subtitle patterns (used in major legislation like Dodd-Frank)
    (r'^(?:TITLE|Title)\s+([IVXLCDM]+|\d+)[.:\s—–-]*(.*)$', 'title'),
    (r'^(?:SUBTITLE|Subtitle)\s+([A-Z]|\d+)[.:\s—–-]*(.*)$', 'subtitle'),
    # Numbered list headers (e.g., "1. DEFINITIONS", "2.1 Scope")
    (r'^(\d+(?:\.\d+)*)\s+([A-Z][A-Z\s]{2,})\s*$', 'numbered_header'),
]

# Compiled patterns for performance
# numbered_header must NOT use IGNORECASE — it relies on [A-Z] to match only ALL-CAPS labels
_COMPILED_PATTERNS = [
    (re.compile(p, re.MULTILINE) if t == 'numbered_header'
     else re.compile(p, re.MULTILINE | re.IGNORECASE), t)
    for p, t in SECTION_PATTERNS
]


def detect_sections(text: str) -> List[Dict]:
    """Detect legal section boundaries in text.

    Splits text into sections based on legal document structure patterns.
    Each section includes the header, detected type, and content.

    Args:
        text: Plain text content of a legal document

    Returns:
        List of dicts with keys:
        - header: Section header text
        - section_type: Type classification (article, section, exhibit, etc.)
        - content: Full section content including header
        - start_pos: Character position where section starts

    If no sections are detected, returns the entire text as a single section.
    """
    if not text or not text.strip():
        return []

    # Find all section boundaries
    boundaries = []
    for pattern, section_type in _COMPILED_PATTERNS:
        for match in pattern.finditer(text):
            header = match.group(0).strip()
            boundaries.append({
                "header": header,
                "section_type": section_type,
                "start_pos": match.start(),
            })

    if not boundaries:
        # No sections detected — return entire text as one section
        return [{
            "header": "",
            "section_type": "body",
            "content": text.strip(),
            "start_pos": 0,
        }]

    # Sort by position and deduplicate overlapping matches
    boundaries.sort(key=lambda x: x["start_pos"])
    deduplicated = []
    for b in boundaries:
        if not deduplicated or b["start_pos"] > deduplicated[-1]["start_pos"] + len(deduplicated[-1]["header"]):
            deduplicated.append(b)
    boundaries = deduplicated

    # Extract content between boundaries
    sections = []
    for i, boundary in enumerate(boundaries):
        start = boundary["start_pos"]
        end = boundaries[i + 1]["start_pos"] if i + 1 < len(boundaries) else len(text)
        content = text[start:end].strip()

        if content:
            sections.append({
                "header": boundary["header"],
                "section_type": boundary["section_type"],
                "content": content,
                "start_pos": start,
            })

    # Merge tiny sections (header-only) into neighbors
    MIN_SECTION_SIZE = 50
    merged = []
    i = 0
    while i < len(sections):
        section = sections[i]
        if len(section["content"]) < MIN_SECTION_SIZE and i + 1 < len(sections):
            # Merge forward into next section: prepend header as context
            next_sec = sections[i + 1]
            next_sec["content"] = section["content"] + "\n\n" + next_sec["content"]
            # Keep the parent header (e.g., ARTICLE) as the section name
            next_sec["header"] = section["header"]
            next_sec["section_type"] = section["section_type"]
            next_sec["start_pos"] = section["start_pos"]
            i += 1  # Skip to the merged section
        elif len(section["content"]) < MIN_SECTION_SIZE and merged:
            # Last tiny section — merge backward into previous
            merged[-1]["content"] = merged[-1]["content"] + "\n\n" + section["content"]
            i += 1
        else:
            merged.append(section)
            i += 1
    sections = merged

    # If there's content before the first section, prepend it
    if boundaries and boundaries[0]["start_pos"] > 0:
        preamble = text[:boundaries[0]["start_pos"]].strip()
        if preamble:
            sections.insert(0, {
                "header": "",
                "section_type": "preamble",
                "content": preamble,
                "start_pos": 0,
            })

    logger.info(f"Detected {len(sections)} sections in document")
    return sections
