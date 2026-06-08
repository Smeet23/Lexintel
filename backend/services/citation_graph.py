"""Citation Knowledge Graph service for Lexintel.

Builds and queries a directed graph of legal citations using SQLAlchemy for
persistence and NetworkX for graph analytics (PageRank authority scoring,
BFS traversal, Jaccard similarity).

Nodes represent individual legal authorities (cases, statutes, regulations).
Edges represent citation relationships with typed treatments (FOLLOWS, OVERRULES,
CITES, etc.) that encode the citing court's attitude toward the cited authority.

Treatment taxonomy:
  Positive   : FOLLOWS, AFFIRMS, APPLIES
  Negative   : OVERRULES, REVERSES, ABROGATES, SUPERSEDES
  Cautionary : QUESTIONS, CRITICIZES, LIMITS
  Neutral    : CITES, DISTINGUISHES, EXPLAINS, HARMONIZES
  Statutory  : AMENDS, REPEALS, CODIFIES, INTERPRETS
"""
import asyncio
import json
import logging
from collections import deque
from typing import Dict, List, Optional, Set, Tuple
import uuid as _uuid

from sqlalchemy.orm import Session

try:
    from backend.models import CitationNode, CitationEdge
    from backend.config import get_settings
except ImportError:
    try:
        from models import CitationNode, CitationEdge
        from config import get_settings
    except ImportError:
        from ..models import CitationNode, CitationEdge
        from ..config import get_settings

try:
    from backend.services.citation_extractor import extract_all_citations
except ImportError:
    try:
        from services.citation_extractor import extract_all_citations
    except ImportError:
        from citation_extractor import extract_all_citations

try:
    from backend.services import llm
except ImportError:
    try:
        from services import llm
    except ImportError:
        from . import llm

# CourtListener enrichment is optional — import lazily so a missing module or
# offline environment never breaks citation indexing.
try:
    from backend.services.citation_lookup import (
        lookup_citation,
        check_case_status,
        get_outbound_case_citations,
    )
except ImportError:
    try:
        from services.citation_lookup import (
            lookup_citation,
            check_case_status,
            get_outbound_case_citations,
        )
    except ImportError:
        try:
            from citation_lookup import (
                lookup_citation,
                check_case_status,
                get_outbound_case_citations,
            )
        except ImportError:
            lookup_citation = None
            check_case_status = None
            get_outbound_case_citations = None

logger = logging.getLogger(__name__)


async def _gather_bounded(factories: List, limit: int) -> List:
    """Run async-coroutine *factories* concurrently, capped at ``limit`` in flight.

    ``factories`` is a list of zero-arg callables each returning a coroutine
    (thunks), so the semaphore bounds the number of coroutines actually started
    — not just awaited. Exceptions are captured per task (``return_exceptions``)
    and returned in position so the caller can count failures without one bad
    call aborting the whole batch. Order of results matches order of factories.

    IMPORTANT — concurrency-safety contract: the ingest runs on a SYNCHRONOUS
    SQLAlchemy session, so this MUST only wrap coroutines that perform pure
    external I/O (LLM / CourtListener / Groq HTTP) and never touch ``db``. Any
    coroutine that reads or writes the session (find_or_create_node, create_edge,
    enrich_node_from_courtlistener, _merge_node_by_cluster) stays sequential:
    those routines interleave their own awaits with db.flush(), so gathering them
    would let one task's flush/autoflush persist another task's half-built ORM
    state. The single-threaded event loop only guarantees atomicity of a
    coroutine's synchronous span BETWEEN awaits — not across them.
    """
    if not factories:
        return []
    sem = asyncio.Semaphore(max(1, limit))

    async def _run(factory):
        async with sem:
            return await factory()

    return await asyncio.gather(
        *(_run(f) for f in factories), return_exceptions=True
    )


def _year_from_date(date_str: Optional[str]) -> Optional[int]:
    """Parse a 4-digit year from an ISO-ish date string (e.g. '1954-05-17').

    Pure-Python (no regex): take the leading 4 characters of the trimmed string
    and interpret them as an integer year. Returns None on any non-numeric input.
    """
    if not date_str or not isinstance(date_str, str):
        return None
    prefix = date_str.strip()[:4]
    if len(prefix) == 4 and prefix.isdigit():
        return int(prefix)
    return None


async def enrich_node_from_courtlistener(db: Session, node: "CitationNode") -> bool:
    """Best-effort enrichment of a case node via CourtListener.

    Populates court, year, source_url, courtlistener_id, is_verified, and an
    authoritative is_good_law (US only). Never raises; returns True if the node
    was updated. Skips non-case nodes and already-enriched nodes.
    """
    if lookup_citation is None or node.node_type != "case" or node.courtlistener_id:
        return False
    jurisdiction = (node.jurisdiction or "US").upper()
    try:
        meta = await lookup_citation(node.citation_text, jurisdiction)
    except Exception as exc:  # pragma: no cover - network best-effort
        logger.debug(f"CourtListener lookup failed for {node.citation_text!r}: {exc}")
        return False
    if not meta:
        return False

    changed = False
    if not node.name and meta.get("case_name"):
        node.name = meta["case_name"]
        changed = True
    if not node.court and meta.get("court"):
        node.court = meta["court"]
        changed = True
    if not node.year:
        year = _year_from_date(meta.get("date_filed"))
        if year:
            node.year = year
            changed = True
    if not node.source_url and meta.get("url"):
        node.source_url = meta["url"]
        changed = True
    cluster_id = meta.get("cluster_id")
    if cluster_id:
        node.courtlistener_id = str(cluster_id)
        node.is_verified = True
        changed = True
        # Authoritative good-law signal (US only) — overrides graph-derived guess.
        if check_case_status is not None and jurisdiction == "US":
            try:
                status = await check_case_status(str(cluster_id))
                if status == "good_law":
                    node.is_good_law = True
                elif status == "bad_law":
                    node.is_good_law = False
            except Exception:  # pragma: no cover - network best-effort
                pass
    if changed:
        db.add(node)
        db.flush()
    return changed


# ═══════════════════════════════════════════════════════════════
# TREATMENT CONSTANTS
# ═══════════════════════════════════════════════════════════════

POSITIVE_TREATMENTS: Set[str] = {"FOLLOWS", "AFFIRMS", "APPLIES"}
NEGATIVE_TREATMENTS: Set[str] = {"OVERRULES", "REVERSES", "ABROGATES", "SUPERSEDES"}
CAUTIONARY_TREATMENTS: Set[str] = {"QUESTIONS", "CRITICIZES", "LIMITS"}
NEUTRAL_TREATMENTS: Set[str] = {"CITES", "DISTINGUISHES", "EXPLAINS", "HARMONIZES"}
STATUTORY_TREATMENTS: Set[str] = {"AMENDS", "REPEALS", "CODIFIES", "INTERPRETS"}

ALL_TREATMENTS: Set[str] = (
    POSITIVE_TREATMENTS
    | NEGATIVE_TREATMENTS
    | CAUTIONARY_TREATMENTS
    | NEUTRAL_TREATMENTS
    | STATUTORY_TREATMENTS
)

# Natural-language / tense / voice variants → canonical treatment token.
# The LLM frequently returns the verb as written in the document ("overruled",
# "distinguished") instead of the canonical uppercase enum. Without this map the
# triple is silently dropped and the case→case edge is lost.
_TREATMENT_ALIASES: Dict[str, str] = {
    "FOLLOWED": "FOLLOWS", "FOLLOWING": "FOLLOWS", "FOLLOW": "FOLLOWS",
    "AFFIRMED": "AFFIRMS", "AFFIRMING": "AFFIRMS", "AFFIRM": "AFFIRMS",
    "APPLIED": "APPLIES", "APPLYING": "APPLIES", "APPLY": "APPLIES",
    "OVERRULED": "OVERRULES", "OVERRULING": "OVERRULES", "OVERRULE": "OVERRULES",
    "REVERSED": "REVERSES", "REVERSING": "REVERSES", "REVERSE": "REVERSES",
    "ABROGATED": "ABROGATES", "ABROGATING": "ABROGATES", "ABROGATE": "ABROGATES",
    "SUPERSEDED": "SUPERSEDES", "SUPERSEDING": "SUPERSEDES", "SUPERSEDE": "SUPERSEDES",
    "QUESTIONED": "QUESTIONS", "QUESTIONING": "QUESTIONS", "QUESTION": "QUESTIONS",
    "CRITICIZED": "CRITICIZES", "CRITICISED": "CRITICIZES", "CRITICIZING": "CRITICIZES",
    "CRITICIZE": "CRITICIZES", "CRITICISE": "CRITICIZES",
    "LIMITED": "LIMITS", "LIMITING": "LIMITS", "LIMIT": "LIMITS",
    "DISTINGUISHED": "DISTINGUISHES", "DISTINGUISHING": "DISTINGUISHES",
    "DISTINGUISH": "DISTINGUISHES",
    "EXPLAINED": "EXPLAINS", "EXPLAINING": "EXPLAINS", "EXPLAIN": "EXPLAINS",
    "HARMONIZED": "HARMONIZES", "HARMONISED": "HARMONIZES", "HARMONIZE": "HARMONIZES",
    "AMENDED": "AMENDS", "AMENDING": "AMENDS", "AMEND": "AMENDS",
    "REPEALED": "REPEALS", "REPEALING": "REPEALS", "REPEAL": "REPEALS",
    "CODIFIED": "CODIFIES", "CODIFYING": "CODIFIES", "CODIFY": "CODIFIES",
    "INTERPRETED": "INTERPRETS", "INTERPRETING": "INTERPRETS", "INTERPRET": "INTERPRETS",
    "CITED": "CITES", "CITING": "CITES", "CITE": "CITES",
}


def _canonical_treatment(value: object) -> Optional[str]:
    """Normalize an LLM/free-text treatment to a canonical ALL_TREATMENTS token.

    Handles tense/voice variants ("overruled" -> "OVERRULES") via _TREATMENT_ALIASES.
    Returns None if the value cannot be mapped to a known treatment.
    """
    token = str(value or "").upper().strip()
    if not token:
        return None
    if token in ALL_TREATMENTS:
        return token
    return _TREATMENT_ALIASES.get(token)


# Edge weight for PageRank — higher weight = stronger positive signal
TREATMENT_WEIGHTS: Dict[str, float] = {
    # Positive
    "FOLLOWS": 1.0,
    "AFFIRMS": 1.0,
    "APPLIES": 0.9,
    # Negative (still creates a citation link, but weakly negative)
    "OVERRULES": 0.1,
    "REVERSES": 0.1,
    "ABROGATES": 0.05,
    "SUPERSEDES": 0.05,
    # Cautionary
    "QUESTIONS": 0.3,
    "CRITICIZES": 0.2,
    "LIMITS": 0.4,
    # Neutral
    "CITES": 0.6,
    "DISTINGUISHES": 0.5,
    "EXPLAINS": 0.7,
    "HARMONIZES": 0.7,
    # Statutory
    "AMENDS": 0.8,
    "REPEALS": 0.1,
    "CODIFIES": 0.8,
    "INTERPRETS": 0.7,
}

# Default treatment when classification is uncertain
_DEFAULT_TREATMENT = "CITES"

# Gemini prompt for treatment classification
_TREATMENT_CLASSIFICATION_PROMPT = """\
You are a legal citation analyst. Given a short text excerpt from a legal document
and the name of the case being cited, classify the treatment applied to that case.

Excerpt:
<EXCERPT>{excerpt}</EXCERPT>

Cited case: {cited_case_name}

Choose exactly ONE treatment from this list:
FOLLOWS, AFFIRMS, APPLIES, OVERRULES, REVERSES, ABROGATES, SUPERSEDES,
QUESTIONS, CRITICIZES, LIMITS, CITES, DISTINGUISHES, EXPLAINS, HARMONIZES,
AMENDS, REPEALS, CODIFIES, INTERPRETS

CITES is the neutral default. If the passage does not CLEARLY indicate how the
citing court treats the cited case, choose CITES and report a low confidence
(<= 0.4). Only choose a stronger treatment (e.g. OVERRULES, FOLLOWS) when the
text explicitly supports it.

Respond ONLY with valid JSON — no markdown, no explanation:
{"treatment": "<TREATMENT>", "confidence": <0.0 to 1.0>}
"""

# Gemini prompt for inter-case relationship (triple) extraction
_RELATIONSHIP_EXTRACTION_PROMPT = """\
You are a legal citation analyst. Read the document excerpt below and extract
INTER-CASE relationships: every place where one legal authority treats another
authority. Examples of relationships:
  - "Brown v. Board overruled Plessy" -> citing=Brown, cited=Plessy, OVERRULES
  - "Dobbs overruled Roe and Casey" -> two triples (Dobbs->Roe, Dobbs->Casey)
  - "Miranda distinguished Escobedo" -> citing=Miranda, cited=Escobedo, DISTINGUISHES

Extract every distinct treated authority. A relationship counts even when the
ACTING authority is the opinion you are reading (the court speaking in the first
person — "we hold", "this Court"): in that case set citing_name to the document's
own case name and citing_citation to its citation if stated, else null. Do NOT
drop a relationship merely because the citing court is unnamed. Do NOT invent
relationships; ignore mere passing mentions with no treatment.

NEGATION & SCOPE DISCIPLINE (critical — read carefully):
  - "distinguished X" or "declined to extend X" -> DISTINGUISHES, never OVERRULES.
  - "declined to follow X" -> a negative treatment (use QUESTIONS/LIMITS as fits).
  - Only use OVERRULES / REVERSES / ABROGATES / SUPERSEDES when the court
    EXPLICITLY nullifies the cited holding.
  - Do NOT emit a treatment that the sentence NEGATES: "we do not overrule X",
    "X remains good law", "were we writing on a blank slate we might overrule X"
    -> these are NOT OVERRULES. Emit no triple (or a neutral one) for them.
  - The treatment attaches ONLY to its grammatical object. In "overruled Roe and
    Casey", BOTH Roe and Casey are overruled (two triples). In "followed A but
    distinguished B", A is FOLLOWS and B is DISTINGUISHES (two different triples).

For each relationship return an object with these keys:
  - citing_citation: the reporter citation of the acting authority if present
    (e.g. "347 U.S. 483"), otherwise its case name
  - citing_name: human-readable name of the acting authority (or null)
  - citing_year: 4-digit decision year of the acting authority if stated (or null)
  - cited_citation: the reporter citation of the authority being treated if
    present, otherwise its case name
  - cited_name: human-readable name of the treated authority (or null)
  - cited_year: 4-digit decision year of the treated authority if stated (or null)
  - treatment: exactly ONE of:
    FOLLOWS, AFFIRMS, APPLIES, OVERRULES, REVERSES, ABROGATES, SUPERSEDES,
    QUESTIONS, CRITICIZES, LIMITS, CITES, DISTINGUISHES, EXPLAINS, HARMONIZES,
    AMENDS, REPEALS, CODIFIES, INTERPRETS
  - evidence: the short phrase/sentence that supports the relationship

Document excerpt:
<DOCUMENT>{document}</DOCUMENT>

Respond ONLY with a valid JSON array — no markdown, no explanation. If there are
no inter-case relationships, return [].
"""


# ═══════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════

def _normalise_text(text: str) -> str:
    """Whitespace-collapse fallback key for citation strings.

    Pure-Python, string-level only — collapses all internal whitespace
    (spaces, tabs, newlines) to a single space and trims the ends. It does NOT
    attempt any reporter-spacing normalisation: that would produce non-canonical
    forms (e.g. 'F.Supp.') that diverge from eyecite's canonical output and
    fragment graph identity. Reporter canonicalisation is handled exclusively by
    ``_canonical_citation_key`` via eyecite; this function only serves as a
    stable fallback for strings eyecite cannot parse.
    """
    return " ".join(text.split())


def _looks_like_case_citation(s: str) -> bool:
    """True iff ``s`` contains a real reporter/statutory citation, per eyecite.

    Library-backed ONLY — no hand-rolled reporter regex and no ``" v. "`` style
    substring guess (both are brittle: a regex/abbreviation list is lossy across
    jurisdictions, and a caption-substring both under-matches real authorities
    with no "v." — ``In re X``, ``Ex parte Y``, foreign neutral cites — and
    over-matches prose containing " v "/" vs "). We accept a string only when
    eyecite/reporters-db extracts a ``FullCaseCitation`` or ``FullLawCitation``
    from it. Real authorities flowing through the pipeline carry their reporter
    (``Bowers v. Hardwick, 478 U.S. 186``) — even inside a multi-line opinion
    header — so eyecite recognises them; procedural-disposition prose ("The
    judgment of the Court of Appeals ... is reversed") yields nothing and is
    dropped, preventing the phantom 'case' node falsely marked BAD LAW.

    Deliberate tradeoff: a citation given as a *reporter-less bare name* returns
    False here. Guessing it is an authority from the string alone is the brittle
    behaviour we are removing; the robust way to admit such a name is to RESOLVE
    it via CourtListener name-search (see the P0 corroboration work), not a
    pattern. Better to omit an unverifiable authority than to assert/deny good
    law on a guess.
    """
    s = (s or "").strip()
    if not s:
        return False
    try:
        from eyecite import get_citations
        from eyecite.models import FullCaseCitation, FullLawCitation

        return any(
            isinstance(c, (FullCaseCitation, FullLawCitation))
            for c in get_citations(s)
        )
    except Exception:  # pragma: no cover - eyecite best-effort
        return False


def _normalize_case_name(name: Optional[str]) -> Optional[str]:
    """Clean a case/authority display name to a single tidy line.

    The relationship-extraction LLM often copies the case name straight from an
    opinion's multi-line header (e.g. ``"LAWRENCE v. TEXAS\\nSupreme Court of the
    United States, 539 U.S. 558 (2003)"``). Storing that verbatim leaks newlines
    and court boilerplate into ``/matters/{id}/graph`` and the RAG inject block.
    Take the first non-empty line and collapse internal whitespace; this is the
    single choke point all node-creation callers pass through.
    """
    if not name:
        return None
    first = next((ln.strip() for ln in name.splitlines() if ln.strip()), "")
    cleaned = " ".join(first.split())
    return cleaned or None


def _canonical_citation_key(citation_text: str) -> str:
    """Return a canonical lookup key for a citation string.

    Layered, library-backed (NO reporter-spacing regex):
      1. eyecite ``FullCaseCitation.corrected_citation()`` — the canonical
         '{volume} {reporter-edition} {page}' form, driven by reporters-db's
         ``edition_guess`` (variation-aware: '410 U. S. 113', '410 US 113' and
         '410 U.S. 113' all canonicalise to '410 U.S. 113'; the page-number
         space is preserved by the library, unlike a hand-written regex).
      2. Fallback to ``_normalise_text`` ONLY for strings eyecite cannot parse
         as a full reporter citation (statutes, bare case names, malformed
         cites). This keeps a stable key for un-parseable input without guessing.

    Note: eyecite alone does NOT merge true PARALLEL citations (different
    reporters for the same case, e.g. '410 U.S. 113' vs '93 S. Ct. 705'). That
    requires CourtListener cluster identity, handled separately in
    ``find_or_create_node`` via ``courtlistener_id``.
    """
    fallback = _normalise_text(citation_text)
    if not fallback:
        return fallback
    try:
        from eyecite import get_citations
        from eyecite.models import FullCaseCitation

        for cite in get_citations(citation_text):
            if isinstance(cite, FullCaseCitation):
                corrected = cite.corrected_citation()
                if corrected and corrected.strip():
                    return corrected.strip()
    except Exception:  # pragma: no cover - eyecite is best-effort
        pass
    return fallback


def _node_to_dict(node: CitationNode) -> Dict:
    """Serialise a CitationNode to a plain dict for API responses."""
    authority_score = 0.0
    try:
        authority_score = float(node.authority_score or 0.0)
    except (TypeError, ValueError):
        pass

    return {
        "id": str(node.id),
        "node_type": node.node_type,
        "citation_text": node.citation_text,
        "name": node.name,
        "court": node.court,
        "year": node.year,
        "jurisdiction": node.jurisdiction,
        "authority_score": round(authority_score, 4),
        "document_id": str(node.document_id) if node.document_id else None,
        "matter_id": str(node.matter_id) if node.matter_id else None,
        "courtlistener_id": node.courtlistener_id,
        "source_url": node.source_url,
        "is_verified": node.is_verified,
        "is_good_law": node.is_good_law,
        "created_at": node.created_at.isoformat() if node.created_at else None,
    }


def _edge_to_dict(edge: CitationEdge) -> Dict:
    """Serialise a CitationEdge to a plain dict for API responses."""
    confidence = None
    try:
        confidence = float(edge.confidence) if edge.confidence is not None else None
    except (TypeError, ValueError):
        pass

    return {
        "id": str(edge.id),
        "source_id": str(edge.source_id),
        "target_id": str(edge.target_id),
        "treatment": edge.treatment,
        "edge_kind": getattr(edge, "edge_kind", None),
        "confidence": confidence,
        "context_excerpt": edge.context_excerpt,
        "matter_id": str(edge.matter_id) if edge.matter_id else None,
        "created_at": edge.created_at.isoformat() if edge.created_at else None,
    }


# ═══════════════════════════════════════════════════════════════
# CORE CRUD OPERATIONS
# ═══════════════════════════════════════════════════════════════

def find_or_create_node(
    db: Session,
    citation_text: str,
    node_type: str = "case",
    *,
    name: Optional[str] = None,
    court: Optional[str] = None,
    year: Optional[int] = None,
    jurisdiction: Optional[str] = None,
    document_id: Optional[_uuid.UUID] = None,
    matter_id: Optional[_uuid.UUID] = None,
    courtlistener_id: Optional[str] = None,
    source_url: Optional[str] = None,
) -> CitationNode:
    """Find an existing CitationNode by citation_text or create a new one.

    If the node already exists and any optional fields are provided, those
    fields are updated on the existing record (non-destructive merge).

    Args:
        db: Active SQLAlchemy session.
        citation_text: Canonical citation string (e.g. "347 U.S. 483").
        node_type: One of "case", "statute", "regulation".
        name: Human-readable case/statute name.
        court: Court name string.
        year: Decision/enactment year.
        jurisdiction: Short jurisdiction code (US/UK/EU/IN/AU/SG).
        document_id: UUID of the source Document row (optional).
        matter_id: UUID of the parent Matter (optional).
        courtlistener_id: CourtListener cluster ID (optional).
        source_url: External URL to the authoritative source (optional).

    Returns:
        The CitationNode (already flushed to the DB). A transient attribute
        ``node._was_created`` is set to True when a new row was inserted and
        False when an existing row was found — callers that need to count
        creations read it via ``getattr(node, "_was_created", False)`` instead
        of issuing an extra COUNT query.
    """
    # Layered node identity (Gap 3 + Gap 6), highest-confidence first:
    #   Layer 1: CourtListener cluster_id — the only key that merges true
    #            PARALLEL citations (different reporters, same case).
    #   Layer 2/3: eyecite/reporters-db canonical form, else normalised string.
    canonical = _canonical_citation_key(citation_text)
    # Normalise the display name once, here, so every caller (doc_cites + the
    # case->case relationship path) stores a clean single-line name instead of a
    # leaked multi-line opinion header.
    name = _normalize_case_name(name)

    existing = None
    if courtlistener_id:
        existing = (
            db.query(CitationNode)
            .filter(CitationNode.courtlistener_id == str(courtlistener_id))
            .first()
        )
    if existing is None:
        existing = (
            db.query(CitationNode)
            .filter(CitationNode.citation_text == canonical)
            .first()
        )

    if existing is not None:
        # Non-destructive update: fill in missing fields if caller provides them
        changed = False
        if name and not existing.name:
            existing.name = name
            changed = True
        if court and not existing.court:
            existing.court = court
            changed = True
        if year and not existing.year:
            existing.year = year
            changed = True
        if jurisdiction and not existing.jurisdiction:
            existing.jurisdiction = jurisdiction
            changed = True
        if document_id and not existing.document_id:
            existing.document_id = document_id
            changed = True
        if matter_id and not existing.matter_id:
            existing.matter_id = matter_id
            changed = True
        if courtlistener_id and not existing.courtlistener_id:
            existing.courtlistener_id = courtlistener_id
            changed = True
        if source_url and not existing.source_url:
            existing.source_url = source_url
            changed = True
        if changed:
            db.flush()
        existing._was_created = False
        return existing

    node = CitationNode(
        citation_text=canonical,
        node_type=node_type,
        name=name,
        court=court,
        year=year,
        jurisdiction=jurisdiction,
        document_id=document_id,
        matter_id=matter_id,
        courtlistener_id=courtlistener_id,
        source_url=source_url,
        authority_score=0.0,
        is_verified=False,
        is_good_law=None,
    )
    db.add(node)
    db.flush()
    node._was_created = True
    logger.debug(f"Created CitationNode: {canonical!r} (type={node_type})")
    return node


def _node_decision_year(node: Optional["CitationNode"]) -> Optional[int]:
    """Best available decision year for a node, or None when unknown.

    ``node.year`` is populated authoritatively from CourtListener ``date_filed``
    (via ``enrich_node_from_courtlistener`` -> ``_year_from_date``). Returns None
    when unknown so callers ABSTAIN rather than fabricate a chronology.
    """
    if node is None:
        return None
    try:
        return int(node.year) if node.year else None
    except (TypeError, ValueError):
        return None


def _chronology_ok(
    citing: Optional["CitationNode"], cited: Optional["CitationNode"]
) -> Optional[bool]:
    """Chronology check for a negative treatment edge.

    A court can only negatively treat precedent that already exists, so a valid
    negative edge requires ``citing_year >= cited_year``. Returns:
      - True  : both years known and the ordering holds
      - False : both years known and the ordering is violated (impossible edge)
      - None  : at least one year unknown -> ABSTAIN (never fabricate ordering)
    """
    cy = _node_decision_year(citing)
    dy = _node_decision_year(cited)
    if cy is None or dy is None:
        return None
    return cy >= dy


def _merge_node_by_cluster(db: Session, node: "CitationNode") -> "CitationNode":
    """Collapse ``node`` onto an existing node sharing its CourtListener cluster.

    Parallel citations (e.g. '410 U.S. 113' and '93 S. Ct. 705') are created as
    separate nodes — no string method can know they are one case — until
    CourtListener enrichment assigns a ``courtlistener_id``. Once two nodes carry
    the SAME cluster id, this merges the just-enriched ``node`` into the older
    canonical node: edges are re-pointed and the duplicate is deleted. Returns
    the surviving (canonical) node. No-op when the node has no cluster id or is
    already the only node for that cluster.
    """
    cid = node.courtlistener_id
    if not cid:
        return node
    canonical = (
        db.query(CitationNode)
        .filter(CitationNode.courtlistener_id == str(cid), CitationNode.id != node.id)
        .order_by(CitationNode.created_at.asc())
        .first()
    )
    if canonical is None:
        return node
    # Re-point any edges that reference the duplicate onto the canonical node.
    db.query(CitationEdge).filter(CitationEdge.source_id == node.id).update(
        {CitationEdge.source_id: canonical.id}, synchronize_session=False
    )
    db.query(CitationEdge).filter(CitationEdge.target_id == node.id).update(
        {CitationEdge.target_id: canonical.id}, synchronize_session=False
    )
    # Backfill any fields the canonical node is missing from the duplicate.
    for field in ("name", "court", "year", "jurisdiction", "source_url"):
        if not getattr(canonical, field, None) and getattr(node, field, None):
            setattr(canonical, field, getattr(node, field))
    db.delete(node)
    db.flush()
    logger.debug(f"Merged parallel-cite node into cluster {cid}")
    return canonical


def create_edge(
    db: Session,
    source_id: _uuid.UUID,
    target_id: _uuid.UUID,
    treatment: str,
    *,
    confidence: Optional[float] = None,
    context_excerpt: Optional[str] = None,
    matter_id: Optional[_uuid.UUID] = None,
    edge_kind: str = "document_cites",
    source_node: Optional["CitationNode"] = None,
    cited_node: Optional["CitationNode"] = None,
) -> Optional[CitationEdge]:
    """Create a directed citation edge between two nodes.

    Skips creation if an identical (source_id, target_id, treatment, edge_kind)
    edge already exists. Validates that treatment is in ALL_TREATMENTS. Refuses
    self-edges. For negative ``case_cites`` treatments, enforces a chronology
    guard when both decision years are known (a court cannot negatively treat a
    future case); abstains (keeps the edge, lowers confidence) when years are
    unknown.

    Args:
        db: Active SQLAlchemy session.
        source_id: UUID of the citing CitationNode.
        target_id: UUID of the cited CitationNode.
        treatment: Treatment type string (must be in ALL_TREATMENTS).
        confidence: Model confidence in the treatment classification [0, 1].
        context_excerpt: Snippet of text that prompted this edge.
        matter_id: UUID of the parent Matter (optional).
        edge_kind: Provenance of the edge — 'document_cites' (a document cites a
            case) or 'case_cites' (one authority treats another). Included in the
            dedup key so both kinds can coexist between the same pair of nodes.
        source_node: The citing CitationNode (optional; enables the chronology
            guard for negative case_cites edges).
        cited_node: The cited CitationNode (optional; enables the chronology
            guard for negative case_cites edges).

    Returns:
        The newly-created CitationEdge, or None if it already exists, is a
        self-edge, is chronologically impossible, or the treatment is invalid.
    """
    # Self-edge guard (Gap 4): a node can never cite itself. Defense-in-depth —
    # covers every path into create_edge, not just the LLM extractor.
    if source_id == target_id:
        logger.debug("Refusing self-edge (source_id == target_id)")
        return None

    treatment_upper = _canonical_treatment(treatment)
    if treatment_upper is None:
        logger.warning(
            f"Invalid treatment {treatment!r}; must be one of {sorted(ALL_TREATMENTS)}"
        )
        return None

    # Chronology guard (Gap 1, CRITICAL): a court cannot negatively treat a
    # case decided AFTER it. Authoritative years come from CourtListener
    # (node.year). Only enforce for negative case_cites treatments, and only
    # when both years are known — otherwise abstain (keep + flag low-confidence).
    settings = get_settings()
    if (
        getattr(settings, "citation_graph_chronology_guard_enabled", True)
        and edge_kind == "case_cites"
        and treatment_upper in NEGATIVE_TREATMENTS
        and source_node is not None
        and cited_node is not None
    ):
        ok = _chronology_ok(source_node, cited_node)
        if ok is False:
            logger.warning(
                f"Chronology violation: {source_node.citation_text!r} "
                f"({source_node.year}) cannot {treatment_upper} "
                f"{cited_node.citation_text!r} ({cited_node.year}); dropping edge."
            )
            return None
        if ok is None:
            # Dates unknown -> ABSTAIN: keep the edge but cap confidence so
            # downstream review logic can flag it as unverified-direction.
            confidence = min(confidence if confidence is not None else 0.5, 0.4)

    # Deduplication check (edge_kind included so document_cites and case_cites
    # between the same nodes can coexist).
    existing = (
        db.query(CitationEdge)
        .filter(
            CitationEdge.source_id == source_id,
            CitationEdge.target_id == target_id,
            CitationEdge.treatment == treatment_upper,
            CitationEdge.edge_kind == edge_kind,
            # Scope dedup to the matter. Nodes are GLOBAL (shared across matters),
            # but edges are per-matter facts: the same relationship asserted in
            # two matters must yield an edge row in EACH, otherwise get_matter_graph
            # (which filters by matter_id) shows it in only the first matter.
            CitationEdge.matter_id == matter_id,
        )
        .first()
    )
    if existing is not None:
        logger.debug(
            f"Edge ({source_id} -[{treatment_upper}/{edge_kind}]-> {target_id}) "
            f"already exists in matter {matter_id}, skipping"
        )
        return None

    edge = CitationEdge(
        source_id=source_id,
        target_id=target_id,
        treatment=treatment_upper,
        edge_kind=edge_kind,
        confidence=confidence,
        context_excerpt=context_excerpt,
        matter_id=matter_id,
    )
    db.add(edge)
    db.flush()
    logger.debug(f"Created CitationEdge: {source_id} -[{treatment_upper}]-> {target_id}")
    return edge


# ═══════════════════════════════════════════════════════════════
# GRAPH ANALYTICS (NetworkX)
# ═══════════════════════════════════════════════════════════════

def build_networkx_graph(
    db: Session,
    matter_id: Optional[_uuid.UUID] = None,
    edge_kind: Optional[str] = "case_cites",
):
    """Load citation nodes and edges from the database into a NetworkX DiGraph.

    Args:
        db: Active SQLAlchemy session.
        matter_id: If provided, restricts the graph to this matter's citations.
        edge_kind: If provided (default 'case_cites'), restricts edges to that
            provenance so analytics run over the REAL case->case graph rather
            than the document-centred star. Pass None to include all edges.

    Returns:
        A ``networkx.DiGraph`` where node attributes carry CitationNode fields
        and edge attributes carry treatment + weight.
    """
    import networkx as nx  # lazy import — not needed for non-graph ops

    node_query = db.query(CitationNode)
    edge_query = db.query(CitationEdge)

    if matter_id is not None:
        node_query = node_query.filter(CitationNode.matter_id == matter_id)
        edge_query = edge_query.filter(CitationEdge.matter_id == matter_id)

    if edge_kind is not None:
        edge_query = edge_query.filter(CitationEdge.edge_kind == edge_kind)

    nodes = node_query.all()
    edges = edge_query.all()

    # Cross-matter node attribution (Gap 5): nodes are global authorities shared
    # across matters (find_or_create_node merges by citation identity), but the
    # matter-scoped node_query above only returns nodes whose own matter_id ==
    # this matter. An edge in this matter can therefore point at a node first
    # created under a DIFFERENT matter and be dropped from PageRank as an orphan.
    # Mirror get_matter_graph's invariant: union in every edge-endpoint node so
    # the graph is fully attributed.
    if matter_id is not None and edges:
        referenced_ids: Set[_uuid.UUID] = set()
        for e in edges:
            referenced_ids.add(e.source_id)
            referenced_ids.add(e.target_id)
        have = {n.id for n in nodes}
        missing = [nid for nid in referenced_ids if nid not in have]
        if missing:
            nodes = nodes + (
                db.query(CitationNode).filter(CitationNode.id.in_(missing)).all()
            )

    graph = nx.DiGraph()

    for node in nodes:
        authority_score = 0.0
        try:
            authority_score = float(node.authority_score or 0.0)
        except (TypeError, ValueError):
            pass

        graph.add_node(
            str(node.id),
            citation_text=node.citation_text,
            node_type=node.node_type,
            name=node.name,
            court=node.court,
            year=node.year,
            jurisdiction=node.jurisdiction,
            authority_score=authority_score,
            is_good_law=node.is_good_law,
        )

    for edge in edges:
        weight = TREATMENT_WEIGHTS.get(edge.treatment, 0.6)
        confidence = 1.0
        try:
            confidence = float(edge.confidence) if edge.confidence is not None else 1.0
        except (TypeError, ValueError):
            pass

        graph.add_edge(
            str(edge.source_id),
            str(edge.target_id),
            treatment=edge.treatment,
            weight=weight * confidence,
            context_excerpt=edge.context_excerpt,
        )

    logger.info(
        f"Built NetworkX graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
        + (f" (matter_id={matter_id})" if matter_id else "")
    )
    return graph


def compute_authority_scores(
    db: Session,
    matter_id: Optional[_uuid.UUID] = None,
) -> Dict[str, float]:
    """Run PageRank on the citation graph and persist scores to CitationNode rows.

    Uses weighted PageRank where edge weights encode treatment strength
    (FOLLOWS=1.0 ... OVERRULES=0.1). Higher-cited authorities score higher.

    Args:
        db: Active SQLAlchemy session.
        matter_id: If provided, restricts computation to this matter.

    Returns:
        Dict mapping node UUID string to PageRank score.
    """
    import networkx as nx

    # Authority is a function of the REAL case->case citation graph, not the
    # document-centred provenance edges.
    graph = build_networkx_graph(db, matter_id=matter_id, edge_kind="case_cites")

    if graph.number_of_nodes() == 0:
        logger.info("Empty graph — no authority scores to compute")
        return {}

    try:
        scores = nx.pagerank(graph, alpha=0.85, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge — using unweighted fallback")
        scores = nx.pagerank(graph, alpha=0.85, weight=None, max_iter=200)

    # Persist scores back to DB
    node_ids = list(scores.keys())
    nodes = (
        db.query(CitationNode)
        .filter(CitationNode.id.in_([_uuid.UUID(nid) for nid in node_ids]))
        .all()
    )

    for node in nodes:
        score = scores.get(str(node.id), 0.0)
        node.authority_score = round(score, 6)

    db.flush()
    logger.info(f"Persisted PageRank scores for {len(nodes)} citation nodes")
    return scores


# ═══════════════════════════════════════════════════════════════
# GOOD LAW CHECK
# ═══════════════════════════════════════════════════════════════

def is_good_law(
    db: Session,
    citation_text: str,
) -> Dict:
    """Determine whether a citation is still good law.

    A citation is considered bad law if it has any incoming NEGATIVE treatment
    edges (OVERRULES, REVERSES, ABROGATES, SUPERSEDES) from other authorities.

    Args:
        db: Active SQLAlchemy session.
        citation_text: The citation to check.

    Returns:
        Dict with keys:
          - status: "good" | "bad" | "caution" | "unknown"
          - is_good_law: bool or None
          - negative_treatments: list of dicts describing adverse citations
          - positive_count: int of affirming citations
          - cautionary_count: int of cautionary citations
    """
    normalised = _canonical_citation_key(citation_text)
    node = (
        db.query(CitationNode)
        .filter(CitationNode.citation_text == normalised)
        .first()
    )

    if node is None:
        return {
            "status": "unknown",
            "is_good_law": None,
            "negative_treatments": [],
            "positive_count": 0,
            "cautionary_count": 0,
            "message": f"Citation {citation_text!r} not found in graph",
        }

    # Load incoming case->case edges only. A document merely *citing* a case
    # (edge_kind='document_cites') must NOT be allowed to flip good-law status;
    # only a real authority treating another authority counts here.
    incoming = (
        db.query(CitationEdge)
        .filter(
            CitationEdge.target_id == node.id,
            CitationEdge.edge_kind == "case_cites",
        )
        .all()
    )

    negative = [e for e in incoming if e.treatment in NEGATIVE_TREATMENTS]
    positive = [e for e in incoming if e.treatment in POSITIVE_TREATMENTS]
    cautionary = [e for e in incoming if e.treatment in CAUTIONARY_TREATMENTS]

    negative_details = []
    if negative:
        source_ids = [e.source_id for e in negative]
        sources_by_id = {
            n.id: n
            for n in db.query(CitationNode)
            .filter(CitationNode.id.in_(source_ids))
            .all()
        }
        for edge in negative:
            source = sources_by_id.get(edge.source_id)
            negative_details.append({
                "treatment": edge.treatment,
                "citing_citation": source.citation_text if source else str(edge.source_id),
                "citing_name": source.name if source else None,
                "context": edge.context_excerpt,
            })

    if negative:
        status = "bad"
        good_law_flag = False
    elif cautionary:
        status = "caution"
        good_law_flag = None  # Debatable — leave as unknown
    else:
        status = "good"
        good_law_flag = True if positive else None

    # GL-006 precedence guard: a CourtListener-authoritative status (set by
    # ``enrich_node_from_courtlistener`` from ``check_case_status``) MUST NOT be
    # silently overwritten by a graph-edge-derived guess. In particular a CL
    # ``False`` (bad law) must never flip back to ``True`` just because the local
    # graph happens to hold positive edges and no negative ones. Only persist a
    # graph-derived flag when the node is NOT authoritatively sourced, OR when
    # the graph itself found a definitive negative treatment (real adverse edges
    # always strengthen a bad-law finding and may legitimately downgrade a stale
    # authoritative "good").
    is_authoritative = bool(node.is_verified or node.courtlistener_id)
    if is_authoritative and not negative:
        # Surface the authoritative flag (set at ingestion by
        # ``enrich_node_from_courtlistener``) rather than the graph-edge guess.
        # A CourtListener ``False`` (bad law) must never be flipped back to True
        # by positive-only local edges.
        good_law_flag = node.is_good_law
        if good_law_flag is False:
            status = "bad"
        elif good_law_flag is True and status != "caution":
            status = "good"

    # NOTE: is_good_law() is READ-ONLY. It is called from a GET endpoint and from
    # the RAG answer path, so it must NEVER write the DB (a write here races
    # concurrent reads of the same row). The persisted ``node.is_good_law`` is
    # owned by the ingestion path (enrich_node_from_courtlistener); this function
    # only computes and returns the current good-law view.

    return {
        "status": status,
        "is_good_law": good_law_flag,
        "negative_treatments": negative_details,
        "positive_count": len(positive),
        "cautionary_count": len(cautionary),
    }


# ═══════════════════════════════════════════════════════════════
# CHAIN AND NETWORK TRAVERSAL
# ═══════════════════════════════════════════════════════════════

def get_citation_chain(
    db: Session,
    citation_text: str,
    depth: int = 2,
    max_nodes: int = 100,
) -> Dict:
    """Return the forward and backward citation chain via BFS traversal.

    Args:
        db: Active SQLAlchemy session.
        citation_text: Starting citation text.
        depth: BFS depth limit (hops from root).
        max_nodes: Maximum total nodes to return across both directions.

    Returns:
        Dict with keys:
          - root: serialised root CitationNode or None
          - backward: list of nodes that cite the root (and their ancestors)
          - forward: list of nodes that the root cites (and their descendants)
          - total_nodes: int
    """
    normalised = _canonical_citation_key(citation_text)
    root = (
        db.query(CitationNode)
        .filter(CitationNode.citation_text == normalised)
        .first()
    )

    if root is None:
        return {
            "root": None,
            "backward": [],
            "forward": [],
            "total_nodes": 0,
            "message": f"Citation {citation_text!r} not found",
        }

    def bfs_edges(start_id: _uuid.UUID, direction: str) -> List[Dict]:
        """BFS in one direction; direction is 'forward' or 'backward'."""
        visited: Set[str] = {str(start_id)}
        queue: deque = deque([(start_id, 0)])
        results: List[Dict] = []
        # Cache of already-fetched nodes to avoid repeated per-node queries.
        node_cache: Dict[str, CitationNode] = {}

        while queue and len(results) < max_nodes // 2:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            if direction == "forward":
                edges = (
                    db.query(CitationEdge)
                    .filter(CitationEdge.source_id == current_id)
                    .all()
                )
                neighbour_ids = [(e.target_id, e) for e in edges]
            else:
                edges = (
                    db.query(CitationEdge)
                    .filter(CitationEdge.target_id == current_id)
                    .all()
                )
                neighbour_ids = [(e.source_id, e) for e in edges]

            # Batch-fetch all new neighbour nodes in one query per BFS level.
            new_ids = [
                nid for nid, _ in neighbour_ids
                if str(nid) not in visited and str(nid) not in node_cache
            ]
            if new_ids:
                fetched = (
                    db.query(CitationNode)
                    .filter(CitationNode.id.in_(new_ids))
                    .all()
                )
                for n in fetched:
                    node_cache[str(n.id)] = n

            for neighbour_id, edge in neighbour_ids:
                nid_str = str(neighbour_id)
                if nid_str in visited:
                    continue
                visited.add(nid_str)

                neighbour_node = node_cache.get(nid_str)
                if neighbour_node is None:
                    continue

                results.append({
                    "node": _node_to_dict(neighbour_node),
                    "edge": _edge_to_dict(edge),
                    "depth": current_depth + 1,
                })
                queue.append((neighbour_id, current_depth + 1))

        return results

    forward_chain = bfs_edges(root.id, "forward")
    backward_chain = bfs_edges(root.id, "backward")

    return {
        "root": _node_to_dict(root),
        "forward": forward_chain,
        "backward": backward_chain,
        "total_nodes": 1 + len(forward_chain) + len(backward_chain),
    }


def get_citation_network(
    db: Session,
    citation_text: str,
    depth: int = 2,
    max_nodes: int = 100,
) -> Dict:
    """Return a visualisation-ready subgraph centred on a citation.

    Collects nodes and edges up to `depth` hops in both directions from the
    root, capped at `max_nodes`.  Returns flat node and edge lists suitable
    for graph visualisation libraries (e.g. D3, Cytoscape, React Flow).

    Args:
        db: Active SQLAlchemy session.
        citation_text: The focal citation text.
        depth: Hop limit from root in any direction.
        max_nodes: Hard cap on returned nodes.

    Returns:
        Dict with keys:
          - nodes: list of serialised CitationNode dicts
          - edges: list of serialised CitationEdge dicts
          - root_id: str UUID of the root node, or None
    """
    normalised = _canonical_citation_key(citation_text)
    root = (
        db.query(CitationNode)
        .filter(CitationNode.citation_text == normalised)
        .first()
    )

    if root is None:
        return {
            "nodes": [],
            "edges": [],
            "root_id": None,
            "message": f"Citation {citation_text!r} not found",
        }

    visited_nodes: Set[str] = {str(root.id)}
    visited_edges: Set[str] = set()
    node_objects: Dict[str, CitationNode] = {str(root.id): root}
    edge_objects: Dict[str, CitationEdge] = {}

    queue: deque = deque([(root.id, 0)])

    while queue and len(visited_nodes) < max_nodes:
        current_id, current_depth = queue.popleft()

        if current_depth >= depth:
            continue

        # Outgoing edges (root cites others)
        outgoing = (
            db.query(CitationEdge)
            .filter(CitationEdge.source_id == current_id)
            .all()
        )
        # Incoming edges (others cite root)
        incoming = (
            db.query(CitationEdge)
            .filter(CitationEdge.target_id == current_id)
            .all()
        )

        all_edges = outgoing + incoming

        # Batch-fetch all new neighbour nodes in one query per BFS step.
        new_neighbour_ids = []
        for edge in all_edges:
            nid = edge.target_id if edge.source_id == current_id else edge.source_id
            nid_str = str(nid)
            if nid_str not in visited_nodes and nid_str not in node_objects:
                new_neighbour_ids.append(nid)
        if new_neighbour_ids:
            fetched = (
                db.query(CitationNode)
                .filter(CitationNode.id.in_(new_neighbour_ids))
                .all()
            )
            for n in fetched:
                node_objects[str(n.id)] = n

        for edge in all_edges:
            eid = str(edge.id)
            if eid not in visited_edges:
                visited_edges.add(eid)
                edge_objects[eid] = edge

            neighbour_id = edge.target_id if edge.source_id == current_id else edge.source_id
            nid_str = str(neighbour_id)

            if nid_str not in visited_nodes and len(visited_nodes) < max_nodes:
                visited_nodes.add(nid_str)
                if nid_str in node_objects:
                    queue.append((neighbour_id, current_depth + 1))

    return {
        "nodes": [_node_to_dict(n) for n in node_objects.values()],
        "edges": [_edge_to_dict(e) for e in edge_objects.values()],
        "root_id": str(root.id),
        "total_nodes": len(node_objects),
        "total_edges": len(edge_objects),
    }


def get_matter_graph(
    db: Session,
    matter_id: _uuid.UUID,
    max_nodes: int = 200,
) -> Dict:
    """Return the full citation graph for a matter, ready for visualisation.

    Refreshes PageRank authority scores, then returns all CitationNode and
    CitationEdge rows belonging to the matter (capped at `max_nodes`).

    Args:
        db: Active SQLAlchemy session.
        matter_id: UUID of the matter.
        max_nodes: Hard cap on returned nodes.

    Returns:
        Dict with keys nodes, edges, stats (matching the visualisation shape).
    """
    # Serve the persisted authority_score values written at ingest time by
    # extract_and_index_citations (which calls compute_authority_scores once
    # after committing). Computing PageRank on every read is O(nodes+edges)
    # and unsafe in a GET endpoint under concurrent load.

    # Top-authority nodes (numeric sort now that authority_score is Float).
    nodes = (
        db.query(CitationNode)
        .filter(CitationNode.matter_id == matter_id)
        .order_by(CitationNode.authority_score.desc())
        .limit(max_nodes)
        .all()
    )
    nodes_by_id: Dict[_uuid.UUID, CitationNode] = {n.id: n for n in nodes}

    # ALL persisted edges for the matter (both document_cites and case_cites).
    edges = (
        db.query(CitationEdge)
        .filter(CitationEdge.matter_id == matter_id)
        .all()
    )

    # Invariant: every returned edge must have BOTH endpoints present in the
    # returned node set, and NO persisted edge may be dropped due to the node
    # cap. So we union the top-authority nodes with every edge endpoint and
    # fetch any missing endpoint nodes.
    referenced_ids: Set[_uuid.UUID] = set()
    for e in edges:
        referenced_ids.add(e.source_id)
        referenced_ids.add(e.target_id)

    missing_ids = [nid for nid in referenced_ids if nid not in nodes_by_id]
    if missing_ids:
        extra_nodes = (
            db.query(CitationNode)
            .filter(CitationNode.id.in_(missing_ids))
            .all()
        )
        for n in extra_nodes:
            nodes_by_id[n.id] = n

    # Defensive: only keep edges whose endpoints both resolved to a node (an
    # endpoint could be missing if a node row was deleted out from under an
    # orphaned edge — keeps the invariant true).
    final_edges = [
        e for e in edges
        if e.source_id in nodes_by_id and e.target_id in nodes_by_id
    ]

    final_nodes = list(nodes_by_id.values())

    # "cases" count excludes provenance document nodes — only real legal
    # authorities (case/statute/regulation) count. A matter whose only node is
    # the document itself (a citation-free upload) reports 0 cases, so the UI
    # shows the empty-state instead of a misleading "1 cases".
    case_nodes = sum(1 for n in final_nodes if n.node_type != "document")

    # LIVE good-law: derive each node's flag from the CURRENT case_cites edges
    # rather than serving a possibly-stale stored ``node.is_good_law``. A node is
    # bad law iff it has an incoming negative case_cites edge in this graph; an
    # authoritative (CourtListener-sourced) stored flag still wins when present.
    # This prevents a stale persisted flag (e.g. left by an old/bad edge that was
    # since removed) from mislabelling a node in the visualisation.
    incoming_negative: Set[_uuid.UUID] = {
        e.target_id for e in final_edges
        if e.edge_kind == "case_cites" and e.treatment in NEGATIVE_TREATMENTS
    }
    # A node that is itself the SOURCE of a negative case_cites edge is the
    # controlling authority that overruled/superseded another case. If nothing
    # adversely treats it in turn, that is a strong positive good-law signal —
    # stronger than "no_adverse" — so we upgrade it to "good" below.
    outgoing_negative: Set[_uuid.UUID] = {
        e.source_id for e in final_edges
        if e.edge_kind == "case_cites" and e.treatment in NEGATIVE_TREATMENTS
    }

    def _is_real_authority(node: CitationNode) -> bool:
        # Defense-in-depth behind the ingest-time precision guard: never let a
        # mis-extracted prose node (no reporter/case-name) be labelled bad/good.
        return _looks_like_case_citation(node.citation_text or "") or _looks_like_case_citation(node.name or "")

    def _live_good_law(node: CitationNode) -> Optional[bool]:
        if node.id in incoming_negative and _is_real_authority(node):
            return False
        if node.is_verified or node.courtlistener_id:
            return node.is_good_law  # authoritative value stands
        if node.id in outgoing_negative and _is_real_authority(node):
            return True  # overruled another case, untouched itself
        return None  # no adverse edge + not authoritative -> not computed

    def _good_law_status(node: CitationNode) -> str:
        """Honest validity label for the UI.

        - "bad"        : an incoming negative treatment edge exists (overruled etc.)
        - "good"       : authoritatively confirmed good law (CourtListener-verified),
                         or a controlling authority that overruled another case and
                         is itself untouched by any adverse treatment
        - "no_adverse" : a real authority with NO adverse treatment found in this
                         matter's graph (a meaningful positive signal — better than
                         "not computed" — but NOT a confident "good law" claim)
        - "unknown"    : provenance/document nodes or otherwise undetermined
        """
        if node.id in incoming_negative and _is_real_authority(node):
            return "bad"
        if node.is_verified or node.courtlistener_id:
            # Authoritative (CourtListener) signal wins over "no local edge".
            if node.is_good_law is True:
                return "good"
            if node.is_good_law is False:
                return "bad"
        if node.id in outgoing_negative and _is_real_authority(node):
            # Controlling authority (it overruled another case) with no incoming
            # adverse treatment — confidently good law.
            return "good"
        if node.node_type != "document":
            return "no_adverse"
        return "unknown"

    node_dicts = []
    for n in final_nodes:
        d = _node_to_dict(n)
        d["is_good_law"] = _live_good_law(n)
        d["good_law_status"] = _good_law_status(n)
        node_dicts.append(d)

    return {
        "nodes": node_dicts,
        "edges": [_edge_to_dict(e) for e in final_edges],
        "stats": {
            "total_nodes": case_nodes,
            "total_edges": len(final_edges),
        },
    }


# ═══════════════════════════════════════════════════════════════
# SIMILARITY
# ═══════════════════════════════════════════════════════════════

def find_similar_cases(
    db: Session,
    citation_text: str,
    limit: int = 10,
) -> List[Dict]:
    """Find cases similar to the given citation using Jaccard similarity.

    Similarity is computed over the set of outgoing citation targets —
    cases that cite many of the same authorities are considered similar.

    Args:
        db: Active SQLAlchemy session.
        citation_text: The reference citation text.
        limit: Maximum number of similar cases to return.

    Returns:
        List of dicts with keys: node (CitationNode dict), jaccard_score.
        Sorted by descending Jaccard similarity.
    """
    normalised = _canonical_citation_key(citation_text)
    root = (
        db.query(CitationNode)
        .filter(CitationNode.citation_text == normalised)
        .first()
    )

    if root is None:
        return []

    # Collect root's outgoing target IDs
    root_targets: Set[str] = {
        str(e.target_id)
        for e in db.query(CitationEdge)
        .filter(CitationEdge.source_id == root.id)
        .all()
    }

    if not root_targets:
        return []

    # Find other nodes that share at least one target with root
    candidates = (
        db.query(CitationEdge.source_id)
        .filter(
            CitationEdge.target_id.in_([_uuid.UUID(tid) for tid in root_targets]),
            CitationEdge.source_id != root.id,
        )
        .distinct()
        .all()
    )

    similar: List[Dict] = []

    if candidates:
        # Batch-fetch all outgoing edges for all candidates in one query.
        candidate_ids = [cid for (cid,) in candidates]
        all_candidate_edges = (
            db.query(CitationEdge)
            .filter(CitationEdge.source_id.in_(candidate_ids))
            .all()
        )
        # Build per-candidate target sets from the single result set.
        targets_by_candidate: Dict[str, Set[str]] = {str(cid): set() for cid in candidate_ids}
        for e in all_candidate_edges:
            src_str = str(e.source_id)
            if src_str in targets_by_candidate:
                targets_by_candidate[src_str].add(str(e.target_id))

        # Batch-fetch all candidate nodes.
        candidate_nodes_by_id: Dict[str, CitationNode] = {
            str(n.id): n
            for n in db.query(CitationNode)
            .filter(CitationNode.id.in_(candidate_ids))
            .all()
        }

        for (candidate_id,) in candidates:
            cid_str = str(candidate_id)
            candidate_targets = targets_by_candidate.get(cid_str, set())

            intersection = root_targets & candidate_targets
            union = root_targets | candidate_targets
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard == 0.0:
                continue

            candidate_node = candidate_nodes_by_id.get(cid_str)
            if candidate_node is None:
                continue

            similar.append({"node": _node_to_dict(candidate_node), "jaccard_score": round(jaccard, 4)})

    similar.sort(key=lambda x: x["jaccard_score"], reverse=True)
    return similar[:limit]


# ═══════════════════════════════════════════════════════════════
# LLM TREATMENT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

async def classify_treatment(
    context_excerpt: str,
    cited_case_name: str,
) -> Dict:
    """Classify the treatment applied to a cited case using Gemini.

    Args:
        context_excerpt: The sentence or short passage containing the citation.
        cited_case_name: Name of the cited case (for context in the prompt).

    Returns:
        Dict with keys:
          - treatment: str (one of ALL_TREATMENTS, falls back to "CITES")
          - confidence: float in [0, 1]
    """
    default = {"treatment": _DEFAULT_TREATMENT, "confidence": 0.5}

    settings = get_settings()
    if not (settings.google_api_key or getattr(settings, "groq_api_key", "")):
        logger.warning("No LLM provider key — defaulting treatment to CITES")
        return default

    # Truncate excerpt to keep cost low
    truncated_excerpt = context_excerpt[:1000]

    prompt = (
        _TREATMENT_CLASSIFICATION_PROMPT
        .replace("{excerpt}", truncated_excerpt)
        .replace("{cited_case_name}", cited_case_name or "unknown")
    )

    try:
        # Honor the configured provider and fall back to the other backend so a
        # single-provider outage (e.g. Gemini 429) doesn't silently drop every
        # treatment to CITES while the other provider sits idle.
        data = await llm.agenerate(
            prompt,
            json=True,
            temperature=0.0,
            max_output_tokens=80,
            provider=(getattr(settings, "llm_answer_provider", "gemini") or "gemini").lower(),
            fallback=True,
        )

        treatment = _canonical_treatment(data.get("treatment", _DEFAULT_TREATMENT))
        if treatment is None:
            logger.debug(f"Unrecognised treatment {data.get('treatment')!r}, falling back to {_DEFAULT_TREATMENT}")
            treatment = _DEFAULT_TREATMENT

        try:
            confidence = float(data.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        return {"treatment": treatment, "confidence": confidence}

    except json.JSONDecodeError as exc:
        logger.warning(f"Treatment classification JSON parse error: {exc}")
        return default
    except (llm.LLMError, llm.LLMBlocked) as exc:
        logger.warning(f"Treatment classification failed: {exc}")
        return default
    except Exception as exc:
        logger.warning(f"Treatment classification failed: {exc}")
        return default


def _sliding_windows(text: str, size: int = 12000, overlap: int = 1500):
    """Yield overlapping character windows so relationships near a window
    boundary are not lost (the old code hard-truncated at 12k chars and silently
    dropped everything after). Overlap keeps boundary-straddling sentences intact.
    """
    if len(text) <= size:
        yield text
        return
    start = 0
    step = max(1, size - overlap)
    while start < len(text):
        yield text[start:start + size]
        start += step


def _parse_year(value: object) -> Optional[int]:
    """Parse a 4-digit year from an int/str, else None."""
    if value is None:
        return None
    try:
        year = int(str(value).strip()[:4])
        return year if 1000 <= year <= 2200 else None
    except (TypeError, ValueError):
        return None


def _parse_relationship_items(data: object) -> List[Dict]:
    """Validate + normalise a list of raw relationship items into triples.

    Pure (no I/O). Applies treatment canonicalisation, the canonical-key
    self-edge guard, and carries through citing_year/cited_year.
    """
    if isinstance(data, dict):
        # Models wrap the array under varying keys (Gemini: bare array or
        # 'relationships'; Groq: 'inter_case_relationships', etc.). Prefer known
        # keys, then fall back to the FIRST list value anywhere in the object so
        # we're robust to whatever wrapper key a provider chooses.
        unwrapped = None
        for key in ("relationships", "triples", "results", "data",
                    "inter_case_relationships", "citations", "items"):
            if isinstance(data.get(key), list):
                unwrapped = data[key]
                break
        if unwrapped is None:
            unwrapped = next((v for v in data.values() if isinstance(v, list)), [])
        data = unwrapped
    if not isinstance(data, list):
        return []

    triples: List[Dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        treatment = _canonical_treatment(item.get("treatment"))
        if treatment is None:
            logger.debug(f"Skipping triple with unmappable treatment {item.get('treatment')!r}")
            continue

        citing_citation = (item.get("citing_citation") or item.get("citing_name") or "").strip()
        cited_citation = (item.get("cited_citation") or item.get("cited_name") or "").strip()
        if not citing_citation or not cited_citation:
            continue
        # Precision guard: BOTH endpoints must look like a real authority. The
        # relationship LLM sometimes mines a procedural disposition ("The
        # judgment of the Court of Appeals ... is reversed") as a REVERSES
        # triple; without this, that phrase becomes a phantom 'case' node falsely
        # flagged BAD LAW and injected into the RAG prompt.
        if not _looks_like_case_citation(citing_citation) or not _looks_like_case_citation(cited_citation):
            logger.debug(
                f"Dropping relationship triple with non-citation endpoint: "
                f"{citing_citation!r} -> {cited_citation!r}"
            )
            continue
        # Self-edge guard using the CANONICAL key (catches parallel/spacing
        # variants of the same cite, not just exact string matches).
        if _canonical_citation_key(citing_citation) == _canonical_citation_key(cited_citation):
            continue

        triples.append({
            "citing_citation": citing_citation,
            "citing_name": item.get("citing_name") or None,
            "citing_year": _parse_year(item.get("citing_year")),
            "cited_citation": cited_citation,
            "cited_name": item.get("cited_name") or None,
            "cited_year": _parse_year(item.get("cited_year")),
            "treatment": treatment,
            "evidence": item.get("evidence") or None,
        })
    return triples


def _strip_json_fences(raw: str) -> str:
    """Strip ```json ... ``` fences a model may wrap JSON in."""
    raw = (raw or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    return raw


async def _extract_relationships_window_gemini(model, window: str) -> Optional[List[Dict]]:
    """Extract relationships from one window via Gemini.

    Returns the parsed triples on success, or None on a PROVIDER failure (e.g.
    quota/429/outage) so the caller can fall back to Groq. An empty list means
    the call succeeded but found no relationships (no fallback needed).
    """
    prompt = _RELATIONSHIP_EXTRACTION_PROMPT.replace("{document}", window)
    try:
        import google.generativeai as genai

        response = await model.generate_content_async(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        return _parse_relationship_items(json.loads(_strip_json_fences(response.text)))
    except json.JSONDecodeError as exc:
        # Model replied but JSON was malformed — treat as "no triples", not a
        # provider failure (Groq would likely repeat the parse issue).
        logger.warning(f"Relationship extraction JSON parse error (Gemini): {exc}")
        return []
    except Exception as exc:
        logger.warning(f"Gemini relationship window failed (will try Groq): {exc}")
        return None


async def _extract_relationships_window_groq(window: str) -> Optional[List[Dict]]:
    """Extract relationships from one window via Groq Llama-3.3-70b.

    Used when Gemini is unavailable (no key) or fails (quota/429). Groq has a
    separate, far larger free quota. Returns parsed triples, [] when none, or
    None when Groq itself is unavailable/fails.
    """
    settings = get_settings()
    groq_key = getattr(settings, "groq_api_key", "") or ""
    if not groq_key:
        return None
    prompt = _RELATIONSHIP_EXTRACTION_PROMPT.replace("{document}", window)

    def _call() -> str:
        from groq import Groq

        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""

    try:
        raw = await asyncio.wait_for(asyncio.to_thread(_call), timeout=45.0)
        data = json.loads(_strip_json_fences(raw))
        return _parse_relationship_items(data)
    except json.JSONDecodeError as exc:
        logger.warning(f"Relationship extraction JSON parse error (Groq): {exc}")
        return []
    except Exception as exc:
        logger.warning(f"Groq relationship window failed: {exc}")
        return None


async def _extract_relationships_window(model, window: str) -> List[Dict]:
    """Run relationship extraction over one window, honoring the configured
    provider order with cross-provider fallback.

    When ``LLM_ANSWER_PROVIDER=groq`` we try Groq FIRST (rather than always
    burning a Gemini call/quota per window before falling back). Either provider
    returning None (missing/quota/429) falls through to the other.
    """
    try:
        provider = (getattr(get_settings(), "llm_answer_provider", "gemini") or "gemini").lower()

        if provider == "groq":
            groq_result = await _extract_relationships_window_groq(window)
            if groq_result is not None:
                return groq_result
            # Groq missing/failed -> Gemini fallback.
            if model is not None:
                gem_result = await _extract_relationships_window_gemini(model, window)
                if gem_result is not None:
                    logger.info("Relationship window extracted via Gemini fallback")
                    return gem_result
            return []

        # Default: Gemini first, Groq fallback.
        result = None
        if model is not None:
            result = await _extract_relationships_window_gemini(model, window)
        if result is not None:
            return result
        groq_result = await _extract_relationships_window_groq(window)
        if groq_result is not None:
            logger.info("Relationship window extracted via Groq fallback")
            return groq_result
        return []
    except Exception as exc:
        logger.warning(f"Relationship extraction window failed: {exc}")
        return []


async def extract_relationships_llm(full_text: str) -> List[Dict]:
    """Extract inter-case relationship triples from a document using Gemini.

    Mines the document for places where one authority treats another (e.g.
    "Brown v. Board overruled Plessy") and returns them as triples — the primary
    mechanism for building REAL case->case edges. The text is processed in
    overlapping sliding windows (not hard-truncated) so long documents do not
    silently lose relationships; triples are deduped across windows by
    (canonical citing, canonical cited, treatment).

    Args:
        full_text: Document text. Windowed internally; bounded by
            ``settings.citation_graph_max_windows``.

    Returns:
        A list of dicts, each with keys: citing_citation, citing_name,
        citing_year, cited_citation, cited_name, cited_year, treatment, evidence.
        Returns [] on any failure, missing API key, or when none are found.
    """
    if not full_text or not full_text.strip():
        return []

    settings = get_settings()

    # Build a Gemini model when a key is present; otherwise rely on the Groq
    # fallback inside _extract_relationships_window (model=None). At least one
    # provider must be configured.
    model = None
    if settings.google_api_key:
        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.google_api_key)
            model = genai.GenerativeModel(model_name=settings.gemini_model)
        except ImportError:
            logger.warning("google-generativeai not installed — using Groq for relationships")

    if model is None and not (getattr(settings, "groq_api_key", "") or ""):
        logger.warning("No Gemini or Groq key — skipping relationship extraction")
        return []

    max_windows = getattr(settings, "citation_graph_max_windows", 6) or 6
    seen: Set[Tuple[str, str, str]] = set()
    triples: List[Dict] = []

    for i, window in enumerate(_sliding_windows(full_text)):
        if i >= max_windows:
            logger.info(f"Relationship extraction hit window cap ({max_windows})")
            break
        for t in await _extract_relationships_window(model, window):
            key = (
                _canonical_citation_key(t["citing_citation"]),
                _canonical_citation_key(t["cited_citation"]),
                t["treatment"],
            )
            if key in seen:
                continue
            seen.add(key)
            triples.append(t)

    logger.info(f"Relationship extraction found {len(triples)} inter-case triple(s)")
    return triples


# ═══════════════════════════════════════════════════════════════
# INGESTION PIPELINE
# ═══════════════════════════════════════════════════════════════

async def extract_and_index_citations(
    db: Session,
    matter_id: _uuid.UUID,
    document_id: _uuid.UUID,
    document_name: str,
    jurisdiction: str,
    chunks: List[Dict],
) -> Dict:
    """Extract citations from document chunks and index them in the knowledge graph.

    Builds a REAL citation graph, not a document-centred star:

    Per chunk (provenance layer):
      1. Regex + LLM extraction via ``extract_all_citations``
      2. Create/update a CitationNode for each found citation
      3. Classify the treatment used in context (async Gemini call)
      4. Create a 'document_cites' CitationEdge from the document's own node to
         each cited node (provenance only — never used for analytics/good-law)

    After the chunk loop (knowledge layer):
      5. Run ``extract_relationships_llm`` over the joined chunk text to mine
         INTER-CASE relationship triples (e.g. "Dobbs overruled Roe")
      6. For each triple, find/create BOTH the citing and cited case nodes and
         create a 'case_cites' CitationEdge between them carrying the real
         treatment. These are the edges authority scoring and good-law analysis
         operate over.
      7. Best-effort: for cited cases that enrich to a CourtListener cluster,
         add a few more 'case_cites' edges from their outbound citations (capped,
         fail-safe).

    The function also creates a "document node" representing the document
    itself (using document_name as citation_text) so that provenance edges have
    a source to point from.

    Args:
        db: Active SQLAlchemy session.
        matter_id: UUID of the parent Matter.
        document_id: UUID of the Document being indexed.
        document_name: Human-readable document name (used as the source node text).
        jurisdiction: Jurisdiction code for the document (US/UK/EU/IN/AU/SG).
        chunks: List of chunk dicts.  Each must have a "content" key; optionally
                "section_name" and "page_num" for richer context.

    Returns:
        Dict with keys:
          - nodes_created: int
          - nodes_updated: int
          - edges_created: int
          - errors: int
    """
    settings = get_settings()
    if not settings.citation_graph_enabled:
        logger.info("Citation graph disabled via config — skipping indexing")
        return {"nodes_created": 0, "nodes_updated": 0, "edges_created": 0, "errors": 0}

    # Coerce defensively: a misconfigured (or test-mocked) setting must never
    # crash the ingest path with a non-numeric value.
    try:
        max_concurrency = int(getattr(settings, "citation_graph_max_concurrency", 8))
    except (TypeError, ValueError):
        max_concurrency = 8
    if max_concurrency < 1:
        max_concurrency = 1

    nodes_created = 0
    nodes_updated = 0
    edges_created = 0
    errors = 0

    # Create/find the source node for this document
    doc_node = find_or_create_node(
        db,
        citation_text=document_name,
        node_type="document",  # provenance source node, NOT a legal authority
        name=document_name,
        jurisdiction=jurisdiction,
        document_id=document_id,
        matter_id=matter_id,
    )
    if getattr(doc_node, "_was_created", False):
        nodes_created += 1

    # ── Phase 1: extract citations from every chunk CONCURRENTLY ──────────────
    # extract_all_citations is a pure-I/O LLM call. Fan them out (bounded) rather
    # than awaiting one chunk at a time. Results stay positional so a failing
    # chunk is counted as one error and the rest still index (parity with the
    # old sequential "continue on error" behaviour).
    indexed_chunks = [
        (idx, content)
        for idx, chunk in enumerate(chunks)
        for content in (chunk.get("content", "") or "",)
        if content.strip()
    ]
    extract_results = await _gather_bounded(
        [(lambda c=content: extract_all_citations(c, use_llm=True))
         for _idx, content in indexed_chunks],
        max_concurrency,
    )

    # ── Dedup citations across ALL chunks: each unique authority is created,
    # enriched and classified exactly ONCE (the core fan-out fix). Keyed by the
    # same canonical key find_or_create_node uses. ────────────────────────────
    unique_cites: Dict[str, Dict] = {}
    for (chunk_idx, content), result in zip(indexed_chunks, extract_results):
        if isinstance(result, Exception):
            logger.warning(f"Citation extraction failed for chunk {chunk_idx}: {result}")
            errors += 1
            continue
        for cite in result or []:
            raw_text = (cite.get("raw_text") or "").strip()
            if not raw_text:
                continue
            key = _canonical_citation_key(raw_text)
            if key in unique_cites:
                continue  # first occurrence wins (dedup across + within chunks)
            citation_pos = content.find(raw_text)
            if citation_pos >= 0:
                start = max(0, citation_pos - 150)
                end = min(len(content), citation_pos + len(raw_text) + 150)
                context_excerpt = content[start:end].strip()
            else:
                context_excerpt = content[:300].strip()
            unique_cites[key] = {
                "raw_text": raw_text,
                "case_name": cite.get("case_name"),
                "jurisdiction": cite.get("jurisdiction", jurisdiction),
                "cite_type": cite.get("type", "case_law"),
                "context_excerpt": context_excerpt,
            }

    # ── Serial DB: find-or-create the cited node for each unique authority ─────
    doc_cites: List[Dict] = []
    for info in unique_cites.values():
        cite_type = info["cite_type"]
        if cite_type in ("statute",):
            node_type = "statute"
        elif cite_type in ("regulation",):
            node_type = "regulation"
        else:
            node_type = "case"
        cite_jur = info["jurisdiction"]
        cited_node = find_or_create_node(
            db,
            citation_text=info["raw_text"],
            node_type=node_type,
            name=info["case_name"],
            jurisdiction=cite_jur if isinstance(cite_jur, str) else None,
            matter_id=matter_id,
        )
        was_created = bool(getattr(cited_node, "_was_created", False))
        if was_created:
            nodes_created += 1
        else:
            nodes_updated += 1
        doc_cites.append({
            "node": cited_node,
            "node_type": node_type,
            "was_created": was_created,
            "name": info["case_name"],
            "raw_text": info["raw_text"],
            "context_excerpt": info["context_excerpt"],
        })

    # ── Phase 2: classify treatment (Gemini) for every unique cite CONCURRENTLY.
    # classify_treatment is pure I/O (no DB), so fanning it out (bounded) is safe
    # under the synchronous session. ───────────────────────────────────────────
    classifications = await _gather_bounded(
        [(lambda dc=dc: classify_treatment(
            context_excerpt=dc["context_excerpt"],
            cited_case_name=dc["name"] or dc["raw_text"],
        )) for dc in doc_cites],
        max_concurrency,
    )

    # Enrich freshly-created case nodes SERIALLY. enrich_node_from_courtlistener
    # mutates the shared session and contains TWO awaits (lookup, then
    # check_case_status) around a db.flush(), so it is NOT safe to gather under a
    # synchronous session — a concurrent sibling's flush/autoflush could persist
    # this node's half-built state (is_verified set, is_good_law not yet). Dedup
    # above already guarantees each unique authority is enriched at most once,
    # which is the actual fan-out fix; the per-call network latency stays serial.
    if settings.citation_graph_enrich_enabled:
        for dc in doc_cites:
            if dc["was_created"] and dc["node_type"] == "case":
                try:
                    await enrich_node_from_courtlistener(db, dc["node"])
                except Exception as exc:  # pragma: no cover - never block indexing
                    logger.debug(f"Node enrichment skipped for {dc['raw_text']!r}: {exc}")

    # ── Serial DB: create the provenance (document_cites) edges ───────────────
    for dc, classification in zip(doc_cites, classifications):
        treatment = _DEFAULT_TREATMENT
        confidence = 0.5
        if isinstance(classification, dict):
            treatment = classification.get("treatment", _DEFAULT_TREATMENT)
            confidence = classification.get("confidence", 0.5)
        elif isinstance(classification, Exception):
            logger.debug(
                f"Treatment classification skipped for {dc['raw_text']!r}: {classification}"
            )
        excerpt = dc["context_excerpt"]
        edge = create_edge(
            db,
            source_id=doc_node.id,
            target_id=dc["node"].id,
            treatment=treatment,
            confidence=confidence,
            context_excerpt=excerpt[:500] if excerpt else None,
            matter_id=matter_id,
            edge_kind="document_cites",
        )
        if edge is not None:
            edges_created += 1

    # ── Build the REAL case->case graph from inter-case relationship triples ──
    # The LLM mines the joined chunk text for treatments between authorities
    # (e.g. "Dobbs overruled Roe"). These are the edges a lawyer actually cares
    # about — they carry treatment + good-law signal, unlike doc->case edges.
    # Join the FULL chunk text (capped generously); extract_relationships_llm
    # windows it internally, so we no longer hard-truncate at 12k and lose tail
    # relationships. Cap at ~60k to bound total LLM windows.
    joined_text = ""
    for chunk in chunks:
        content = chunk.get("content", "")
        if content and content.strip():
            joined_text += content.strip() + "\n\n"
            if len(joined_text) >= 60000:
                break
    joined_text = joined_text[:60000]

    if joined_text.strip():
        try:
            triples = await extract_relationships_llm(joined_text)
        except Exception as exc:  # never let relationship mining break ingestion
            logger.warning(f"Relationship extraction skipped: {exc}")
            triples = []

        # Pre-compute independent NEGATIVE-treatment verifications CONCURRENTLY.
        # verify_negative_treatment is a pure-I/O (Groq) second-pass that depends
        # ONLY on triple data, so it is safe to fan out before the sequential DB
        # loop below (which owns the delicate find_or_create → enrich → merge
        # sequence and must stay serial). Keyed by triple index. The membership
        # test mirrors the in-loop check (raw treatment token) for exact parity.
        verify_results: Dict[int, Tuple] = {}
        if triples and getattr(settings, "treatment_verification_enabled", True):
            try:
                from backend.services.treatment_verifier import verify_negative_treatment
            except ImportError:
                from services.treatment_verifier import verify_negative_treatment
            neg_indices = [
                i for i, t in enumerate(triples)
                if t.get("treatment") in NEGATIVE_TREATMENTS
            ]
            neg_outcomes = await _gather_bounded(
                [(lambda t=triples[i]: verify_negative_treatment(
                    t["treatment"],
                    t.get("evidence") or "",
                    t.get("citing_name") or t["citing_citation"],
                    t.get("cited_name") or t["cited_citation"],
                )) for i in neg_indices],
                max_concurrency,
            )
            for i, outcome in zip(neg_indices, neg_outcomes):
                if not isinstance(outcome, Exception):
                    verify_results[i] = outcome

        for idx, triple in enumerate(triples):
            try:
                citing_node = find_or_create_node(
                    db,
                    citation_text=triple["citing_citation"],
                    node_type="case",
                    name=triple.get("citing_name"),
                    year=triple.get("citing_year"),
                    jurisdiction=jurisdiction,
                    matter_id=matter_id,
                )
                nodes_created += int(getattr(citing_node, "_was_created", False))
                cited_node = find_or_create_node(
                    db,
                    citation_text=triple["cited_citation"],
                    node_type="case",
                    name=triple.get("cited_name"),
                    year=triple.get("cited_year"),
                    jurisdiction=jurisdiction,
                    matter_id=matter_id,
                )
                nodes_created += int(getattr(cited_node, "_was_created", False))

                # Enrich BOTH endpoints so node.year (authoritative CourtListener
                # date_filed) is available for the chronology guard. Best-effort.
                # After enrichment assigns a cluster_id, collapse onto any existing
                # node for that cluster — this is what merges PARALLEL citations
                # (different reporters of the same case) discovered on separate
                # triples into a single authoritative node.
                if settings.citation_graph_enrich_enabled:
                    if not citing_node.courtlistener_id:
                        try:
                            await enrich_node_from_courtlistener(db, citing_node)
                        except Exception:  # pragma: no cover - best-effort
                            pass
                        citing_node = _merge_node_by_cluster(db, citing_node)
                    if not cited_node.courtlistener_id:
                        try:
                            await enrich_node_from_courtlistener(db, cited_node)
                        except Exception:  # pragma: no cover - best-effort
                            pass
                        cited_node = _merge_node_by_cluster(db, cited_node)

                # Independent second-pass verification of NEGATIVE treatments
                # (Gap 2). A different model (Groq) votes on whether the evidence
                # truly asserts the negative treatment; on rejection we DOWNGRADE
                # to neutral CITES rather than assert an unverified "overruled".
                treatment = triple["treatment"]
                confidence = 0.8
                if (
                    treatment in NEGATIVE_TREATMENTS
                    and getattr(settings, "treatment_verification_enabled", True)
                ):
                    # Result was computed concurrently above (verify_results).
                    # Missing entry == the verify call failed/was skipped → keep
                    # the negative treatment as-is (parity with the old fail-safe).
                    outcome = verify_results.get(idx)
                    if outcome is not None:
                        label, vconf = outcome
                        if label == "rejected":
                            treatment = _DEFAULT_TREATMENT  # neutral CITES
                            confidence = 0.4
                        elif label == "confirmed":
                            confidence = vconf

                case_edge = create_edge(
                    db,
                    source_id=citing_node.id,
                    target_id=cited_node.id,
                    treatment=treatment,
                    confidence=confidence,
                    context_excerpt=(triple.get("evidence") or None),
                    matter_id=matter_id,
                    edge_kind="case_cites",
                    source_node=citing_node,  # chronology guard (Gap 1)
                    cited_node=cited_node,
                )
                if case_edge is not None:
                    edges_created += 1

                # Secondary, best-effort: if the cited case resolved to a
                # CourtListener cluster (via the enrich step above), pull a few of
                # the cases IT cites to add more real case->case edges. Strictly
                # capped and fail-safe so it never meaningfully slows ingestion.
                # NOTE: no redundant re-enrich here — the main enrich block above
                # already attempted CourtListener resolution for cited_node.
                if (
                    settings.citation_graph_enrich_enabled
                    and get_outbound_case_citations is not None
                    and cited_node.courtlistener_id
                ):
                    try:
                        outbound = await get_outbound_case_citations(
                            cited_node.courtlistener_id, max_results=5
                        )
                    except Exception:  # pragma: no cover - best-effort
                        outbound = []
                    for ob in outbound[:5]:
                        ob_cite = (ob.get("citation") or ob.get("case_name") or "").strip()
                        if not ob_cite:
                            continue
                        try:
                            ob_node = find_or_create_node(
                                db,
                                citation_text=ob_cite,
                                node_type="case",
                                name=ob.get("case_name") or None,
                                jurisdiction=jurisdiction,
                                matter_id=matter_id,
                                courtlistener_id=(ob.get("cluster_id") or None),
                            )
                            ob_edge = create_edge(
                                db,
                                source_id=cited_node.id,
                                target_id=ob_node.id,
                                treatment=_DEFAULT_TREATMENT,
                                confidence=0.6,
                                context_excerpt=None,
                                matter_id=matter_id,
                                edge_kind="case_cites",
                            )
                            if ob_edge is not None:
                                edges_created += 1
                        except Exception:  # pragma: no cover - best-effort
                            continue
            except Exception as exc:
                logger.debug(f"Failed to index relationship triple {triple!r}: {exc}")
                errors += 1

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error(f"DB commit failed during citation graph indexing: {exc}")
        return {
            "nodes_created": 0,
            "nodes_updated": 0,
            "edges_created": 0,
            "errors": errors + 1,
        }

    # Recompute PageRank authority scores once at ingest time so get_matter_graph
    # can serve the persisted values without re-running PageRank on every read.
    try:
        compute_authority_scores(db, matter_id)
        db.commit()
    except Exception as exc:  # never block the indexing return value
        logger.warning(f"Authority score computation skipped after indexing: {exc}")

    logger.info(
        f"Citation graph indexing complete for document {document_name!r}: "
        f"nodes_created={nodes_created}, nodes_updated={nodes_updated}, "
        f"edges_created={edges_created}, errors={errors}"
    )

    return {
        "nodes_created": nodes_created,
        "nodes_updated": nodes_updated,
        "edges_created": edges_created,
        "errors": errors,
    }
