"""Amendment chain management for versioned legal documents.

Links documents that are versions of the same statute/regulation
(e.g. Clean Air Act v1.0 -> v1.1 -> v2.0) into chains so the
temporal filter can prefer the latest version.

Detection uses name-based matching (canonical name + jurisdiction),
not LLM analysis.
"""
import logging
import re
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

try:
    from backend.models import AmendmentChain, Document
except ImportError:
    try:
        from models import AmendmentChain, Document
    except ImportError:
        from ..models import AmendmentChain, Document

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# NAME NORMALIZATION
# ═══════════════════════════════════════════════════════════════

# Patterns stripped when computing canonical name for matching
_VERSION_PATTERNS = [
    r'\bv(?:ersion)?\s*\d+(?:\.\d+)*',   # v1.0, version 2.3
    r'\b\d{4}\s*(?:amendment|revision)',    # 2024 amendment
    r'\((?:as\s+)?amended[^)]*\)',          # (as amended 2024)
    r'\b(?:amended|revised|updated)\b',     # standalone keywords
    r'\breprint\b',
    r'\bconsolidated\b',
]

_VERSION_RE = re.compile(
    '|'.join(_VERSION_PATTERNS),
    re.IGNORECASE,
)


def _normalize_name(name: str) -> str:
    """Strip version/amendment qualifiers to get a canonical name for matching.

    Examples:
        "Clean Air Act v2.0"          -> "clean air act"
        "GDPR (as amended 2024)"      -> "gdpr"
        "Companies Act 2006 Revision" -> "companies act 2006"
    """
    normalized = _VERSION_RE.sub('', name)
    # Collapse whitespace
    normalized = re.sub(r'\s+', ' ', normalized).strip().lower()
    return normalized


# ═══════════════════════════════════════════════════════════════
# CHAIN CRUD
# ═══════════════════════════════════════════════════════════════

def find_or_create_chain(
    matter_id: UUID,
    canonical_doc_id: str,
    canonical_name: str,
    jurisdiction: Optional[str],
    db: Session,
) -> AmendmentChain:
    """Find an existing amendment chain or create a new one.

    Lookup is by (matter_id, canonical_document_id). If no chain exists,
    a new one is created.

    Args:
        matter_id: The matter this chain belongs to.
        canonical_doc_id: A stable identifier for the document lineage
            (e.g. statute short-cite or normalized title).
        canonical_name: Human-readable canonical name for the chain.
        jurisdiction: Jurisdiction code (e.g. 'US.federal', 'UK').
        db: SQLAlchemy session.

    Returns:
        The existing or newly created AmendmentChain.
    """
    chain = (
        db.query(AmendmentChain)
        .filter(
            AmendmentChain.matter_id == matter_id,
            AmendmentChain.canonical_document_id == canonical_doc_id,
        )
        .first()
    )

    if chain:
        logger.debug(f"Found existing amendment chain: {chain.id} ({canonical_name})")
        return chain

    chain = AmendmentChain(
        matter_id=matter_id,
        canonical_document_id=canonical_doc_id,
        canonical_name=canonical_name,
        jurisdiction=jurisdiction,
    )
    db.add(chain)
    db.flush()  # Assign ID without committing
    logger.info(f"Created amendment chain: {chain.id} ({canonical_name})")
    return chain


def link_document_to_chain(
    document: "Document",
    chain_id: UUID,
    db: Session,
) -> None:
    """Link a document to an amendment chain.

    Args:
        document: The Document ORM instance to link.
        chain_id: The amendment chain ID.
        db: SQLAlchemy session.
    """
    document.amendment_chain_id = chain_id
    db.add(document)
    db.flush()
    logger.info(
        f"Linked document {document.id} ({document.name}) "
        f"to amendment chain {chain_id}"
    )


def detect_supersession(
    new_doc: "Document",
    existing_docs: List["Document"],
    db: Session,
) -> Optional[AmendmentChain]:
    """Check if a new document supersedes any existing documents.

    Uses name-based matching: if the normalized canonical name of the
    new document matches an existing document (within the same
    jurisdiction), they belong to the same amendment chain.

    Args:
        new_doc: The newly uploaded Document.
        existing_docs: Other documents in the same matter.
        db: SQLAlchemy session.

    Returns:
        The AmendmentChain if the new doc was linked to one, else None.
    """
    if not existing_docs:
        return None

    new_canonical = _normalize_name(new_doc.name)
    if not new_canonical:
        return None

    for existing in existing_docs:
        if existing.id == new_doc.id:
            continue

        existing_canonical = _normalize_name(existing.name)
        if not existing_canonical:
            continue

        # Match: same canonical name and (same jurisdiction or either unknown)
        if new_canonical == existing_canonical:
            new_jurisdiction = getattr(new_doc, 'jurisdiction', None) or 'unknown'
            existing_jurisdiction = getattr(existing, 'jurisdiction', None) or 'unknown'

            jurisdictions_compatible = (
                new_jurisdiction == existing_jurisdiction
                or new_jurisdiction == 'unknown'
                or existing_jurisdiction == 'unknown'
            )

            if not jurisdictions_compatible:
                continue

            logger.info(
                f"Detected supersession: '{new_doc.name}' matches "
                f"'{existing.name}' (canonical: '{new_canonical}')"
            )

            # Use existing chain or create one
            chain_jurisdiction = (
                new_jurisdiction if new_jurisdiction != 'unknown'
                else existing_jurisdiction
            )
            canonical_doc_id = new_canonical  # Use normalized name as stable ID

            chain = find_or_create_chain(
                matter_id=new_doc.matter_id,
                canonical_doc_id=canonical_doc_id,
                canonical_name=new_canonical,
                jurisdiction=chain_jurisdiction if chain_jurisdiction != 'unknown' else None,
                db=db,
            )

            # Link both documents to the chain
            if not existing.amendment_chain_id:
                link_document_to_chain(existing, chain.id, db)
            link_document_to_chain(new_doc, chain.id, db)

            # Recompute which versions are current vs superseded.
            recalculate_chain_status(chain.id, db)

            return chain

    return None


def recalculate_chain_status(
    chain_id: UUID,
    db: Session,
) -> None:
    """Mark the latest version in a chain 'current' and older ones 'superseded'.

    Orders the chain chronologically (via get_version_chain) and sets each
    non-latest document's status to 'superseded' with superseded_date equal to
    the effective_date of its successor (falling back to the successor's
    created_at). Explicit 'repealed' / 'draft' statuses are preserved.

    Args:
        chain_id: The amendment chain ID.
        db: SQLAlchemy session.
    """
    docs = get_version_chain(chain_id, db)
    if len(docs) < 2:
        return

    last_index = len(docs) - 1
    for i, doc in enumerate(docs):
        current_status = (getattr(doc, "document_status", None) or "unknown").lower()

        if i == last_index:
            # Latest version is current law (unless explicitly repealed/draft).
            doc.superseded_date = None
            if current_status in ("unknown", "current", "superseded"):
                doc.document_status = "current"
        else:
            successor = docs[i + 1]
            doc.superseded_date = (
                getattr(successor, "effective_date", None)
                or getattr(successor, "created_at", None)
            )
            if current_status != "repealed":
                doc.document_status = "superseded"

        db.add(doc)

    db.flush()
    logger.info(
        f"Recalculated amendment chain {chain_id}: "
        f"{last_index} superseded, 1 current ({len(docs)} versions)"
    )


def get_version_chain(
    chain_id: UUID,
    db: Session,
) -> List["Document"]:
    """Get all documents in an amendment chain ordered by version/date.

    Orders by: effective_date (if available), then version_number,
    then created_at as a fallback.

    Args:
        chain_id: The amendment chain ID.
        db: SQLAlchemy session.

    Returns:
        List of Document instances in chronological order (oldest first).
    """
    documents = (
        db.query(Document)
        .filter(Document.amendment_chain_id == chain_id)
        .all()
    )

    def _sort_key(doc: "Document"):
        # Primary: effective_date (None sorts last)
        effective = getattr(doc, 'effective_date', None)
        has_effective = effective is not None
        # Secondary: version_number (try numeric parse)
        version = getattr(doc, 'version_number', None) or ''
        try:
            version_num = float(version)
        except (ValueError, TypeError):
            version_num = 0.0
        # Tertiary: created_at
        created = getattr(doc, 'created_at', None)
        return (not has_effective, effective, version_num, created)

    documents.sort(key=_sort_key)
    return documents
