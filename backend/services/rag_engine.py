"""RAG Query Engine for legal document analysis with comprehensive error handling"""
import logging
import asyncio
import re
import threading
from typing import List, Dict, Tuple, Optional
from uuid import UUID
from sqlalchemy.orm import Session
import google.generativeai as genai

try:
    from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
except ImportError:
    class GoogleAPIError(Exception):
        pass
    class ResourceExhausted(GoogleAPIError):
        pass

try:
    from backend.services.embeddings import embed_query as embed_query_fn
    from backend.services.vector_store import search_vectors
    from backend.models import Matter, Document, Query, Chunk
    from backend.config import get_settings
    from backend.exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
    from backend.services.legal_research import search_cases, format_as_context
    from backend.services.citation_agent import verify_response_citations
    from backend.services.conflict_detector import detect_conflicts, augment_context_with_conflicts
    from backend.services.authority_detector import compute_query_time_authority_score
except ImportError:
    try:
        from services.embeddings import embed_query as embed_query_fn
        from services.vector_store import search_vectors
        from models import Matter, Document, Query, Chunk
        from config import get_settings
        from exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
        from services.legal_research import search_cases, format_as_context
        from services.citation_agent import verify_response_citations
        from services.conflict_detector import detect_conflicts, augment_context_with_conflicts
        from services.authority_detector import compute_query_time_authority_score
    except ImportError:
        from .embeddings import embed_query as embed_query_fn
        from .vector_store import search_vectors
        from ..models import Matter, Document, Query, Chunk
        from ..config import get_settings
        from ..exceptions import QueryProcessingException, EmbeddingException, VectorStoreException
        from .legal_research import search_cases, format_as_context
        from .citation_agent import verify_response_citations
        from .conflict_detector import detect_conflicts, augment_context_with_conflicts
        from .authority_detector import compute_query_time_authority_score

logger = logging.getLogger(__name__)
settings = get_settings()

# Cache for reranker model (lazy loaded)
_RERANKER_MODEL = None
_RERANKER_LOCK = threading.Lock()

# Gemini model name
GEMINI_MODEL = settings.gemini_model

# Groq is used as a generation fallback when Gemini fails (e.g. quota/429).
_GROQ_GEN_MODEL = "llama-3.3-70b-versatile"
_groq_gen_client = None
_GROQ_GEN_AVAILABLE = False
try:
    from groq import Groq as _GroqClient
    _groq_gen_key = getattr(settings, "groq_api_key", "") or ""
    if _groq_gen_key:
        _groq_gen_client = _GroqClient(api_key=_groq_gen_key)
        _GROQ_GEN_AVAILABLE = True
        logger.info("Groq generation fallback enabled (llama-3.3-70b-versatile)")
except Exception as _groq_exc:  # pragma: no cover - optional dependency
    _GROQ_GEN_AVAILABLE = False


async def _groq_generate_answer(
    user_prompt: str,
    include_legal_research: bool,
    temperature: float,
) -> Optional[Tuple[str, int]]:
    """Generate an answer with Groq Llama 3.3 70B (primary when llm_answer_provider=
    groq, else Gemini fallback).

    Returns ``(answer_text, tokens_used)``, or None if Groq is unavailable/fails so
    the caller can surface the original error / try the other provider.
    """
    if not (_GROQ_GEN_AVAILABLE and _groq_gen_client):
        return None
    system_prompt = LEGAL_RESEARCH_SYSTEM_PROMPT if include_legal_research else LEGAL_SYSTEM_PROMPT

    def _call() -> Tuple[str, int]:
        resp = _groq_gen_client.chat.completions.create(
            model=_GROQ_GEN_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content or ""
        tokens = 0
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens = getattr(usage, "total_tokens", 0) or 0
        return text, tokens

    try:
        text, tokens = await asyncio.wait_for(asyncio.to_thread(_call), timeout=45.0)
        logger.info("Answer generated via Groq")
        return text, tokens
    except Exception as e:  # pragma: no cover - network/best-effort
        logger.error(f"Groq generation failed: {e}")
        return None

# Configuration
CONTEXT_TOKEN_BUDGET = 50_000
LEGAL_SYSTEM_PROMPT = """You are LexIntel, a legal research assistant. Synthesize analysis from retrieved documents.

AUTHORITY HIERARCHY:
1. BINDING: Same jurisdiction higher court decisions (must follow)
2. PERSUASIVE: Other jurisdiction courts (should consider)
3. SECONDARY: Treatises, restatements, commentary (reference only)
Lead with binding authority. Explain why it controls. Never rely solely on secondary sources for a legal conclusion.

ANALYSIS FORMAT (CREAC):
Structure every answer as follows:
- CONCLUSION: Answer the question upfront with a confidence level tag. Do not bury the answer.
- RULE: State the binding legal rule with court name and year. If no binding rule exists in the sources, say so explicitly.
- EXPLANATION: Explain how the rule has been interpreted or how it evolved across the provided sources.
- APPLICATION: Apply the rule to the specific facts or issues raised in the question.
- CONTRARY: Address any opposing or distinguishable authorities found in the sources. Explain why they do not control or how they differ.
If the question is simple and a full CREAC is unnecessary, you may abbreviate, but always lead with the conclusion.

CHAIN-OF-THOUGHT (internal reasoning):
Before writing your answer, work through these steps:
Step 1: Identify the legal issue(s) presented in the question.
Step 2: Identify controlling authority from the provided source excerpts.
Step 3: State the binding rule with full citation.
Step 4: Identify the elements or factors of that rule.
Step 5: Apply each element to the facts in the question.
Step 6: Address any contrary authority — distinguish it or explain its lesser weight.
Step 7: Synthesize your conclusion with the appropriate confidence level.
Present the result in CREAC format. Do not show these steps as numbered items in your output.

CITATIONS:
- Use [1], [2], [3] format corresponding to source excerpt numbers in the context.
- Place the citation number immediately after the claim it supports.
- Every factual claim MUST have at least one numbered citation.
- Example: "The court held that the contract was void [1] and damages were not recoverable [2]."
- Do NOT fabricate citations. If no source supports a claim, flag it with [CITATION NEEDED - VERIFY].

CONFLICTS:
- If sources in the context disagree, explicitly state the conflict.
- Explain which source is more authoritative and why (jurisdiction, court level, recency).
- Mark the passage with [CONFLICTING AUTHORITIES] so the reader is alerted.

CITATION VALIDATION:
- Context may include a "CITATION VALIDATION (Knowledge Graph)" block with good-law status.
- If an authority is BAD LAW (overruled/reversed/superseded), do NOT rely on it; warn the reader and mark [BAD LAW].
- If CAUTION, note uncertain validity before relying.
- Weight authorities by stated authority score and jurisdiction when sources conflict.

CONFIDENCE LEVELS:
Open your answer with one of the following tags:
- [HIGH] — Binding authority directly on point. Strong basis for the conclusion.
- [MODERATE] — Binding authority exists but requires inference, analogy, or extension to the facts.
- [LOW] — No binding authority found. Relying on persuasive authority, secondary sources, or limited data.

HALLUCINATION FLAGS:
Insert these inline when applicable:
- [CITATION NEEDED - VERIFY] — You are making a claim not directly supported by provided sources.
- [CONFLICTING AUTHORITIES] — Sources in the context disagree on this point.
- [BAD LAW] — The cited authority has been overruled/reversed/superseded per the citation validation block.
- [INSUFFICIENT DATA] — The provided documents do not contain enough information to fully answer this sub-question.
- [DEVELOPING LAW] — The legal area appears to be in flux based on the sources provided.

PROHIBITED:
- Do NOT fabricate case names, citations, statutes, or any legal authority.
- Do NOT ignore binding authority in favor of weaker sources.
- Do NOT treat all sources as equally authoritative.
- Do NOT present inferences as holdings. Clearly distinguish between what a court held and what you are inferring.
- Do NOT speculate beyond what the documents state. If the answer is not in the sources, say so.

When conversation history is provided, use it to resolve references like "that", "it", "the above", etc. Answer the current question based on the document excerpts, using conversation context only for disambiguation."""

LEGAL_RESEARCH_SYSTEM_PROMPT = """You are LexIntel, a legal research assistant. Synthesize analysis from BOTH the user's uploaded documents AND relevant case law from public legal databases.

SOURCE DISTINCTION:
- Clearly indicate which information comes from the user's uploaded documents vs. external case law.
- When citing case law, include the full legal citation in text:
  e.g., "In Brown v. Board of Education, 347 U.S. 483 (1954) [3], the Court held..."
- When citing user documents, use the excerpt number: "The agreement provides... [1]."

AUTHORITY HIERARCHY:
1. BINDING: Same jurisdiction higher court decisions (must follow)
2. PERSUASIVE: Other jurisdiction courts, lower court decisions (should consider)
3. SECONDARY: Treatises, restatements, commentary (reference only)
4. USER DOCUMENTS: Contracts, filings, agreements — these are facts, not authority. Analyze them in light of the law.
Lead with binding authority. Explain why it controls. Use case law to support, distinguish, or contextualize the user's documents.

ANALYSIS FORMAT (CREAC):
Structure every answer as follows:
- CONCLUSION: Answer the question upfront with a confidence level tag. Do not bury the answer.
- RULE: State the binding legal rule with court name and year. If case law provides the rule, cite the case with full legal citation.
- EXPLANATION: Explain how courts have interpreted the rule. Reference specific case holdings from the provided case law.
- APPLICATION: Apply the rule to the user's specific documents and facts. Quote or reference the user's documents where relevant.
- CONTRARY: Address any opposing case law or distinguishable precedent. Explain why it does not control or how the facts differ.
If the question is simple and a full CREAC is unnecessary, you may abbreviate, but always lead with the conclusion.

CHAIN-OF-THOUGHT (internal reasoning):
Before writing your answer, work through these steps:
Step 1: Identify the legal issue(s) presented in the question.
Step 2: Identify controlling authority from the provided case law and document excerpts.
Step 3: State the binding rule with full legal citation.
Step 4: Identify the elements or factors of that rule.
Step 5: Apply each element to the facts found in the user's documents.
Step 6: Address any contrary case law — distinguish it or explain its lesser weight.
Step 7: Synthesize your conclusion with the appropriate confidence level.
Present the result in CREAC format. Do not show these steps as numbered items in your output.

CITATIONS:
- Use [1], [2], [3] format corresponding to source excerpt numbers in the context.
- Place the citation number immediately after the claim it supports.
- For case law, ALSO include the full legal citation in text:
  e.g., "In Smith v. Jones, 500 F.3d 200 (2d Cir. 2007) [4], the court reasoned..."
- Every factual claim MUST have at least one numbered citation.
- Do NOT invent case names or citations. Only cite sources from the provided context.
- If no source supports a claim, flag it with [CITATION NEEDED - VERIFY].

JURISDICTION-AWARE SYNTHESIS:
- When case law comes from multiple jurisdictions, note the jurisdiction of each case.
- Give greater weight to cases from the same jurisdiction as the user's matter.
- If no same-jurisdiction authority is available, state this explicitly and explain the persuasive value of the cited cases.

CONFLICTS:
- If case law and user documents suggest different outcomes, analyze both.
- If cases from different jurisdictions conflict, explain the split and which view is majority/minority.
- Mark passages with [CONFLICTING AUTHORITIES] so the reader is alerted.

CITATION VALIDATION:
- Context may include a "CITATION VALIDATION (Knowledge Graph)" block with good-law status.
- If an authority is BAD LAW (overruled/reversed/superseded), do NOT rely on it; warn the reader and mark [BAD LAW].
- If CAUTION, note uncertain validity before relying.
- Weight authorities by stated authority score and jurisdiction when sources conflict.

CONFIDENCE LEVELS:
Open your answer with one of the following tags:
- [HIGH] — Binding case law directly on point. Strong basis for the conclusion.
- [MODERATE] — Case law exists but requires inference, analogy, or extension to the user's facts.
- [LOW] — No binding authority found. Relying on persuasive case law, secondary sources, or limited data.

HALLUCINATION FLAGS:
Insert these inline when applicable:
- [CITATION NEEDED - VERIFY] — You are making a claim not directly supported by provided sources.
- [CONFLICTING AUTHORITIES] — Sources in the context disagree on this point.
- [BAD LAW] — The cited authority has been overruled/reversed/superseded per the citation validation block.
- [INSUFFICIENT DATA] — The provided documents and case law do not contain enough information to fully answer this sub-question.
- [DEVELOPING LAW] — The legal area appears to be in flux; recent changes may affect this analysis.

PROHIBITED:
- Do NOT fabricate case names, citations, statutes, or any legal authority.
- Do NOT ignore binding authority in favor of weaker sources.
- Do NOT treat all sources as equally authoritative.
- Do NOT present inferences as holdings. Clearly distinguish between what a court held and what you are inferring.
- Do NOT speculate beyond what the documents and case law state. If the answer is not in the sources, say so.
- Do NOT invent case names or citations. Only cite sources from the provided context.

When conversation history is provided, use it to resolve references like "that", "it", "the above", etc. Answer the current question based on the document excerpts and case law, using conversation context only for disambiguation."""

MIN_QUERY_LENGTH = 3
MIN_CONFIDENCE_SCORE = 0.30  # Require 30% semantic similarity minimum (tuned for Cohere embed-v3)
RETRIEVAL_TOP_K = 15
RETRIEVAL_LIMIT = 30  # Request more from vector store, then take top_k (improves recall)
FINAL_CHUNK_COUNT = 8


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count for text (roughly 4 characters per token).

    Args:
        text: Text to count tokens for

    Returns:
        Estimated number of tokens

    Raises:
        ValueError: If text is empty
    """
    if not text or not text.strip():
        return 0

    # Rough estimate: 1 token ~ 4 characters
    return len(text) // 4


def _detect_query_filters(query: str) -> Dict:
    """Detect optional Qdrant filters from query text.

    Only applies explicit jurisdiction mentions. Returns empty dict
    if no strong signal is detected (empty dict = no filters applied).
    """
    filters = {}
    query_lower = query.lower()

    jurisdiction_hints = {
        "UK": ["uk ", "united kingdom", "english law", "british"],
        "US": ["us ", "united states", "american", "federal law"],
        "EU": ["eu ", "european union", "gdpr", "directive"],
        "AU": ["australia", "australian"],
        "CA": ["canada", "canadian"],
        "SG": ["singapore", "singaporean"],
        "IN": ["india", "indian law"],
    }
    for code, hints in jurisdiction_hints.items():
        if any(h in query_lower for h in hints):
            filters["jurisdiction"] = code
            break

    return filters


# Year band guards against matching citations / section numbers (e.g. "s. 1998").
_TEMPORAL_YEAR_RE = __import__("re").compile(r"\b(19\d{2}|20\d{2})\b")
_AS_OF_RE = __import__("re").compile(
    r"\bas\s+of\s+([A-Za-z0-9,\s/\-]+?)(?:[.?!]|$|\bfor\b|\bunder\b|\bin\b)",
    __import__("re").IGNORECASE,
)


def _detect_temporal_intent(query: str):
    """Infer as-of-date temporal intent from natural-language query text.

    Returns a tuple ``(as_of_date, exclude_superseded)``:
      - ``as_of_date``: timezone-aware ``datetime`` to evaluate the law as-of, or None.
      - ``exclude_superseded``: whether to restrict to current-only docs when no
        as-of date applies.

    Precedence of patterns (case-insensitive, first match wins):
      - "as of <date>"                              → (parsed_date, False)
      - "before|prior to|pre <year>"                → (Jan 1 of year, False)
      - "in <year>" / "under the <year> act"        → (Jul 1 of year, False)
      - "current law|currently|in force|today"      → (None, True)
      - "historical|history of|evolution of|all versions|over time" → (None, False)
      - default                                     → (None, True)

    Explicit ``as_of_date`` params (from the API) take precedence over this and
    should be checked by the caller before invoking intent detection.
    """
    from datetime import datetime, timezone

    try:
        from backend.services.temporal_extractor import _parse_date_string
    except ImportError:
        try:
            from services.temporal_extractor import _parse_date_string
        except ImportError:
            from .temporal_extractor import _parse_date_string

    if not query:
        return None, True

    q = query.strip()
    q_lower = q.lower()

    def _year_in(text):
        m = _TEMPORAL_YEAR_RE.search(text)
        if m:
            year = int(m.group(1))
            if 1900 <= year <= 2100:
                return year
        return None

    # 1. Explicit "as of <date>"
    as_of_match = _AS_OF_RE.search(q)
    if as_of_match:
        candidate = as_of_match.group(1).strip().rstrip(",. ")
        if candidate:
            parsed = _parse_date_string(candidate)
            if parsed is not None:
                return parsed, False

    # 2. "before / prior to / pre <year>" — law before the start of that year
    before_idx = None
    for kw in ("before ", "prior to ", "pre-", "pre "):
        idx = q_lower.find(kw)
        if idx != -1:
            before_idx = idx + len(kw)
            break
    if before_idx is not None:
        year = _year_in(q_lower[before_idx:before_idx + 12])
        if year:
            return datetime(year, 1, 1, tzinfo=timezone.utc), False

    # 3. "in <year>" / "under the <year> act" — mid-year snapshot
    #    Only accept a year that appears within ~12 characters AFTER the matched
    #    keyword so that "...in the 5th Amendment ... 1983" does not
    #    misclassify as as-of-1983.  Scanning the whole query was too greedy.
    _TEMPORAL_KWS_3 = (" in ", "under the ", "during ", "back in ")
    for _kw3 in _TEMPORAL_KWS_3:
        _idx3 = q_lower.find(_kw3)
        if _idx3 != -1:
            _window_start = _idx3 + len(_kw3)
            year = _year_in(q_lower[_window_start:_window_start + 12])
            if year:
                return datetime(year, 7, 1, tzinfo=timezone.utc), False
            # No year in this keyword's window — continue to next keyword

    # 4. Explicit current-law intent → current-only
    if any(kw in q_lower for kw in ("current law", "currently", "in force", "as it stands today", "right now")):
        return None, True

    # 5. Historical / all-versions intent → keep superseded docs
    if any(
        kw in q_lower
        for kw in ("historical", "history of", "evolution of", "all versions", "over time", "previous version")
    ):
        return None, False

    # Default: current-law, no as-of date (zero regression)
    return None, True


def _detect_target_jurisdiction(query: str) -> Optional[str]:
    """Extract a jurisdiction code from the query for authority scoring.

    Returns an ISO-style jurisdiction code (e.g. 'US.federal', 'UK')
    suitable for compute_query_time_authority_score(), or None if no
    jurisdiction signal is detected.
    """
    query_lower = query.lower()
    # Map natural-language hints to ISO-style jurisdiction codes used by authority_detector
    jurisdiction_hints = {
        "US.federal": ["federal law", "federal court", "scotus", "supreme court of the united states"],
        "US": ["us ", "united states", "american"],
        "UK": ["uk ", "united kingdom", "english law", "british"],
        "EU": ["eu ", "european union", "gdpr", "directive"],
        "AU": ["australia", "australian"],
        "CA": ["canada", "canadian"],
        "SG": ["singapore", "singaporean"],
        "IN": ["india", "indian law"],
    }
    for code, hints in jurisdiction_hints.items():
        if any(h in query_lower for h in hints):
            return code
    return None


def format_legal_context(chunks: List[Dict], matter_name: str, doc_summaries: Dict = None) -> str:
    """
    Format retrieved chunks into structured legal context with metadata.

    Args:
        chunks: List of chunk dicts with content, page_num, section_name, score
        matter_name: Name of the matter
        doc_summaries: Optional dict mapping document_id -> {"name": str, "summary": str}

    Returns:
        Formatted context string with metadata

    Raises:
        ValueError: If chunks list is empty
    """
    if not chunks:
        raise ValueError("Chunks list cannot be empty")

    # Sort by score (highest first)
    sorted_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)

    context_parts = [f"Matter: {matter_name}\n", "=" * 60, "\n"]

    # Add document summaries as preamble (once per document, not per chunk)
    if doc_summaries:
        context_parts.append("Document Summaries:\n")
        for doc_id, info in doc_summaries.items():
            context_parts.append(f"  - {info['name']}: {info['summary']}\n")
        context_parts.append("\n")

    for i, chunk in enumerate(sorted_chunks, 1):
        location = chunk.get("page_num", "Unknown")
        section = chunk.get("section_name", "")
        score = chunk.get("score", 0)
        content = chunk.get("content", "")
        source_type = chunk.get("source_type", "document")

        # Authority and temporal labels for LLM awareness
        authority_label = ""
        court_level = chunk.get("court_level", "unknown")
        jurisdiction_code = chunk.get("jurisdiction_code", "unknown")
        binding = chunk.get("binding_authority", False)
        if court_level != "unknown":
            authority_label = f", Authority: {court_level.upper()}"
            if binding:
                authority_label += " (BINDING)"
            else:
                authority_label += " (PERSUASIVE)"
            if jurisdiction_code != "unknown":
                authority_label += f", Jurisdiction: {jurisdiction_code}"

        temporal_label = ""
        doc_status = chunk.get("document_status", "unknown")
        if doc_status and doc_status not in ("unknown", "current"):
            temporal_label = f", Status: {doc_status.upper()}"

        if source_type == "case_law":
            header = f"--- CASE LAW {i} (Case: {chunk.get('document_name', 'Unknown')}"
            if section:
                header += f", Court: {section}"
            header += f"{authority_label}{temporal_label}, Score: {score:.2f}) ---\n"
        else:
            if location.startswith("para"):
                location_label = f"Paragraph {location[5:]}"
            elif location.startswith("line"):
                location_label = f"Lines {location[5:]}"
            else:
                location_label = f"Page {location}"

            header = f"--- EXCERPT {i} ({location_label}"
            if section:
                header += f", Section: {section}"
            doc_name = chunk.get("document_name", "")
            if doc_name:
                header += f", Document: {doc_name}"
            header += f"{authority_label}{temporal_label}, Score: {score:.2f}) ---\n"

        context_parts.append(header)
        context_parts.append(content)
        context_parts.append("\n\n")

    return "".join(context_parts)


# ─────────────────────────────────────────────────────────────────
# CITATION GRAPH VALIDATION CONTEXT
# ─────────────────────────────────────────────────────────────────

_GRAPH_BLOCK_MAX_CHARS = 2000


def _authority_bucket(score: float) -> str:
    """Map a PageRank authority score to a coarse human-readable bucket."""
    if score >= 0.05:
        return "high"
    if score >= 0.01:
        return "medium"
    return "low"


def _build_graph_context(
    db: Session,
    matter_id: str,
    citations: List[Dict],
    max_citations: int = 5,
) -> Optional[str]:
    """Build a "CITATION VALIDATION (Knowledge Graph)" context block.

    For each unique citation found in the retrieved chunks / query, resolve its
    node in the citation knowledge graph and surface good/bad-law signals so the
    LLM can warn about overruled authority and weight precedent.

    This is the *sync core* — DB work (``is_good_law`` etc.) is synchronous and
    should be wrapped in ``asyncio.to_thread`` at the call site. Never raises;
    returns ``None`` when no usable validity signal is found (caller skips
    cleanly, leaving ``formatted_context`` untouched).

    Args:
        db: Active SQLAlchemy session.
        matter_id: UUID string of the parent matter (matter-scoped lookup first).
        citations: List of citation dicts (from ``extract_all_citations``); each
            should carry a ``raw_text`` key.
        max_citations: Cap on the number of validity lines emitted.

    Returns:
        A formatted context block string, or ``None`` if nothing to emit.
    """
    # Lazy, triple-fallback imports so a missing graph module never breaks RAG.
    try:
        from backend.services.citation_graph import is_good_law, _canonical_citation_key
        from backend.models import CitationNode
    except ImportError:
        try:
            from services.citation_graph import is_good_law, _canonical_citation_key
            from models import CitationNode
        except ImportError:
            from .citation_graph import is_good_law, _canonical_citation_key
            from ..models import CitationNode

    # 1. Dedup raw_text case-insensitively (union of query + chunks needs re-dedup).
    seen: set = set()
    unique_raw: List[str] = []
    for cite in citations or []:
        raw = (cite.get("raw_text") or "").strip()
        if not raw:
            continue
        # Dedup on the canonical key so spacing/dot variants of the same
        # authority collapse to one entry (matches how find_or_create_node WRITES).
        key = _canonical_citation_key(raw).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_raw.append(raw)

    if not unique_raw:
        return None

    # 2. Resolve nodes. Two-tier: matter-scoped first, global fallback (an
    #    overruling authority may live outside this matter). Skip unresolved
    #    (→ unknown, no value) and self-citations (the document's own node).
    matter_uuid = None
    try:
        matter_uuid = UUID(str(matter_id))
    except (ValueError, TypeError, AttributeError):
        matter_uuid = None

    resolved: List[Tuple[str, "CitationNode"]] = []
    for raw in unique_raw:
        # Use the SAME canonical key the writer (find_or_create_node) stores in
        # citation_text, so good-law lookups actually hit the node instead of
        # being silently dropped by a key mismatch.
        normalised = _canonical_citation_key(raw)
        node = None
        if matter_uuid is not None:
            node = (
                db.query(CitationNode)
                .filter(
                    CitationNode.citation_text == normalised,
                    CitationNode.matter_id == matter_uuid,
                )
                .first()
            )
        if node is None:
            node = (
                db.query(CitationNode)
                .filter(CitationNode.citation_text == normalised)
                .first()
            )
        if node is None:
            continue
        # Defensive: skip ONLY true self-references — i.e. nodes whose
        # citation_text equals the document's own name (a document citing
        # itself).  The original guard skipped every citation node that
        # merely lived in this matter, which wrongly dropped third-party
        # cases cited within briefs/filings.  We no longer inspect
        # document_id/matter_id here; is_good_law() will return "unknown"
        # for any node without enough graph data, which is filtered below.
        pass  # no skip — proceed to is_good_law validation
        resolved.append((raw, node))

    if not resolved:
        return None

    # 3. Cap top-N by authority score descending.
    def _score(item: Tuple[str, "CitationNode"]) -> float:
        try:
            return float(item[1].authority_score or 0.0)
        except (TypeError, ValueError):
            return 0.0

    resolved.sort(key=_score, reverse=True)
    resolved = resolved[: max(1, max_citations)]

    # 4. Validate each citation and 5. format a compact line per non-unknown status.
    lines: List[str] = []
    for raw, node in resolved:
        try:
            result = is_good_law(db, raw)
        except Exception as exc:  # never let one bad lookup break the block
            logger.debug(f"is_good_law failed for {raw!r}: {exc}")
            continue

        status = (result.get("status") or "unknown").lower()
        if status == "unknown":
            continue

        display = node.name or raw
        # Avoid duplicating the name when it is already embedded in the raw
        # citation (raw='Bowers v. Hardwick, 478 U.S. 186 (1986)',
        # name='Bowers v. Hardwick' must not become 'Bowers v. Hardwick, Bowers
        # v. Hardwick, 478 U.S. 186 (1986)'). Prefer the richer string.
        if node.name and raw:
            if node.name in raw:
                display = raw
            elif raw in node.name:
                display = node.name
            else:
                display = f"{node.name}, {raw}"

        authority = _authority_bucket(_score((raw, node)))
        jurisdiction = node.jurisdiction or "?"

        if status == "bad":
            negatives = result.get("negative_treatments") or []
            overruler = ""
            if negatives:
                first = negatives[0]
                # Collapse any leaked newlines/tabs (e.g. a multi-line opinion
                # header) so the one-bullet-per-authority block stays single-line.
                citing = " ".join(
                    (first.get("citing_name") or first.get("citing_citation") or "").split()
                )
                # Past-tense for natural prose ("overruled by", not "overrules by").
                _past = {
                    "OVERRULES": "overruled", "REVERSES": "reversed",
                    "SUPERSEDES": "superseded", "ABROGATES": "abrogated",
                }
                treatment = _past.get((first.get("treatment") or "").upper(), "overruled")
                if citing:
                    overruler = f" ({treatment} by {citing})"
            lines.append(
                f"• {display} — BAD LAW{overruler}. "
                f"Authority: {authority}. {jurisdiction}."
            )
        elif status == "caution":
            count = result.get("cautionary_count") or 0
            detail = f" (questioned/limited by {count} authorities)" if count else ""
            lines.append(
                f"• {display} — CAUTION{detail}. "
                f"Authority: {authority}. {jurisdiction}."
            )
        elif status == "good":
            count = result.get("positive_count") or 0
            detail = f" (followed by {count} authorities)" if count else ""
            lines.append(
                f"• {display} — GOOD LAW{detail}. "
                f"Authority: {authority}. {jurisdiction}."
            )

    if not lines:
        return None

    header = (
        "--- CITATION VALIDATION (Knowledge Graph) ---\n"
        "Heed these validity signals. If an authority is BAD LAW, do not rely on it "
        "for a conclusion; warn the reader and prefer the controlling authority.\n\n"
    )
    block = header + "\n".join(lines)
    return block[:_GRAPH_BLOCK_MAX_CHARS]


def embed_query(query: str) -> List[float]:
    """
    Embed user query into vector space (uses Cohere search_query input type).

    Args:
        query: User query string

    Returns:
        1024-dimensional embedding vector
    """
    return embed_query_fn(query)


def _get_reranker():
    """
    Get or initialize cross-encoder reranker model.
    Uses sentence-transformers cross-encoder for efficiency.

    Returns:
        CrossEncoder model or None if not available
    """
    global _RERANKER_MODEL

    if _RERANKER_MODEL is not None:
        return _RERANKER_MODEL

    with _RERANKER_LOCK:
        if _RERANKER_MODEL is not None:  # double-check after acquiring lock
            return _RERANKER_MODEL
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Initializing cross-encoder reranker")
            # Pin device (default CPU): Apple-MPS in a Celery prefork child crashes
            # the worker; this tiny model runs fine on CPU. Override LOCAL_MODEL_DEVICE.
            import os as _os
            _device = _os.environ.get("LOCAL_MODEL_DEVICE", "cpu")
            _RERANKER_MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=_device)
            return _RERANKER_MODEL
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Reranking disabled. Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to initialize reranker: {str(e)}")
            return None


def rerank_chunks(
    query: str,
    chunks: List[Dict],
    top_k: int = FINAL_CHUNK_COUNT,
    target_jurisdiction: Optional[str] = None,
) -> List[Dict]:
    """
    Rerank retrieved chunks using cross-encoder for relevance.

    Better chunks = better answers. Uses cross-encoder to rerank chunks
    by comparing query relevance (more accurate than vector similarity alone).

    Args:
        query: User query string
        chunks: List of chunks from vector search
        top_k: Number of top chunks to return after reranking
        target_jurisdiction: Optional jurisdiction code from query context
            (e.g. 'US.federal', 'UK'). When provided, authority scores are
            recalculated to reflect jurisdiction relevance.

    Returns:
        List of reranked chunks sorted by relevance
    """
    if not chunks:
        logger.debug("No chunks to rerank")
        return []

    # Auto-detect target jurisdiction from query if not explicitly provided
    if target_jurisdiction is None:
        target_jurisdiction = _detect_target_jurisdiction(query)

    reranker = _get_reranker()

    if reranker is None:
        logger.debug("Reranker not available, returning chunks as-is")
        return chunks[:top_k]

    try:
        # Prepare pairs for reranking (query, chunk_content)
        pairs = []

        for chunk in chunks:
            content = chunk.get("content", "")[:512]  # Cross-encoder handles up to ~512 tokens
            pairs.append([query, content])

        logger.debug(f"Reranking {len(chunks)} chunks")

        # Get reranking scores
        scores = reranker.predict(pairs)

        # Add rerank scores to chunks
        for i, chunk in enumerate(chunks):
            # Combine vector similarity score (original) with rerank score
            original_score = chunk.get("score", 0)
            rerank_score = float(scores[i])  # 0-1 normalized score

            # Authority-aware reranking: combine vector + cross-encoder + authority
            # Only compute authority scores when the feature is enabled
            authority_score = 0.5
            if settings.authority_scoring_enabled:
                authority_score = chunk.get("authority_score", 0.5)
                if target_jurisdiction:
                    chunk_authority = {
                        "court_level": chunk.get("court_level", "unknown"),
                        "jurisdiction_code": chunk.get("jurisdiction_code", "unknown"),
                        "binding_authority": chunk.get("binding_authority"),
                        "authority_score": authority_score,
                    }
                    try:
                        authority_score = compute_query_time_authority_score(
                            chunk_authority, target_jurisdiction
                        )
                    except Exception as e:
                        logger.debug(f"Query-time authority scoring failed, using default: {e}")

            chunk["rerank_score"] = rerank_score
            chunk["authority_score"] = authority_score
            if settings.authority_scoring_enabled:
                chunk["combined_score"] = (
                    (original_score * 0.15) +
                    (rerank_score * 0.45) +
                    (authority_score * 0.40)
                )
            else:
                chunk["combined_score"] = (
                    (original_score * 0.30) +
                    (rerank_score * 0.70)
                )

        # Sort by combined score
        reranked = sorted(chunks, key=lambda x: x.get("combined_score", 0), reverse=True)

        # Log top result
        if reranked:
            top = reranked[0]
            logger.debug(
                f"Top reranked chunk: "
                f"vector_score={top.get('score', 0):.3f}, "
                f"rerank_score={top.get('rerank_score', 0):.3f}, "
                f"combined={top.get('combined_score', 0):.3f}"
            )

        return reranked[:top_k]

    except Exception as e:
        logger.warning(f"Reranking failed, using original chunks: {str(e)}")
        return chunks[:top_k]


def retrieve_chunks(matter_id: str, query_embedding: List[float], top_k: int = RETRIEVAL_TOP_K, query_filter: Dict = None) -> List[Dict]:
    """
    Retrieve similar chunks from vector store.

    Args:
        matter_id: Matter identifier
        query_embedding: Query embedding vector
        top_k: Number of top results to retrieve
        query_filter: Optional payload filter dict for Qdrant

    Returns:
        List of chunk dicts with score, page_num, content, etc.

    Raises:
        ValueError: If top_k is not positive
        Exception: If vector search fails
    """
    # Validate top_k parameter (Issue 6)
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    try:
        results = search_vectors(matter_id, query_embedding, limit=top_k, query_filter=query_filter)
        return results
    except VectorStoreException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve chunks: {str(e)}")
        raise VectorStoreException(
            "Failed to retrieve chunks from vector store",
            detail=str(e)
        ) from e


def extract_citations(answer: str, chunks: List[Dict]) -> Tuple[str, List[Dict], bool]:
    """
    Extract citations from answer, match to retrieved chunks, and remove hallucinations.

    Handles citations for multiple document formats:
    - [Page X] for PDFs
    - [Paragraph X] for Word documents
    - [Lines X-Y] for text files

    Args:
        answer: Generated answer with location citations
        chunks: List of retrieved chunks

    Returns:
        Tuple of (cleaned_answer, citations_list, has_hallucinations)
        - cleaned_answer: Answer with hallucinated citations removed
        - citations_list: List of valid citation dicts
        - has_hallucinations: Bool indicating if any hallucinations were detected

    Raises:
        None - gracefully handles citation mismatches
    """
    import re

    citations = []

    # Define patterns for all citation types
    citation_patterns = [
        (r'\[Page\s+(\d+)(?:,\s*Section[:\s]+[^\]]+)?\]', 'page'),  # [Page 3] or [Page 3, Section: 4.1]
        (r'\[Paragraph\s+(\d+)\]', 'paragraph'),
        (r'\[Lines?\s+(\d+-\d+)(?:,\s*Section[:\s]+[^\]]+)?\]', 'line_range'),  # [Lines 1-5] or [Lines 1-5, Section: 9]
        (r'\[Section[:\s]+"?([^"\]]+)"?\]', 'section'),
    ]

    # Fallback patterns for less structured LLM citations
    fallback_patterns = [
        (r'(?:on |see |from |per )page\s+(\d+)', 'page'),
        (r'(?:in |see |per )section\s+(\d+(?:\.\d+)*)', 'section'),
        (r'(?:in |see |per )paragraph\s+(\d+)', 'paragraph'),
    ]

    def _line_range_overlaps(cited_range: str, chunk_range: str) -> bool:
        """Check if a cited line range (e.g. '1-5') overlaps with a chunk range (e.g. 'line 1-50')."""
        try:
            # Parse cited range: "1-5" or "101-148"
            cited_parts = cited_range.split("-")
            cited_start, cited_end = int(cited_parts[0]), int(cited_parts[1])
            # Parse chunk range: "line 1-50" or "line 51-100"
            chunk_nums = chunk_range.replace("line ", "").split("-")
            chunk_start, chunk_end = int(chunk_nums[0]), int(chunk_nums[1])
            return cited_start <= chunk_end and cited_end >= chunk_start
        except (ValueError, IndexError):
            return False

    # Create location to chunk mapping
    location_to_chunks = {}
    section_to_chunks = {}
    valid_locations = set()
    line_range_chunks = []  # For overlap matching
    for chunk in chunks:
        location = str(chunk.get("page_num", ""))
        if location not in location_to_chunks:
            location_to_chunks[location] = chunk
            valid_locations.add(location)
        # Track line-range chunks for overlap matching
        if location.startswith("line "):
            line_range_chunks.append((location, chunk))
        # Also map section names for [Section "X"] citations
        section_name = str(chunk.get("section_name", ""))
        if section_name and section_name not in section_to_chunks:
            section_to_chunks[section_name] = chunk

    def _find_chunk_for_citation(location: str, citation_type: str):
        """Find the best matching chunk for a citation, using overlap for line ranges."""
        # Exact match first
        if location in location_to_chunks:
            return location_to_chunks[location]
        # For line ranges, check overlap with chunk ranges
        if citation_type == 'line_range' and location.startswith("line "):
            cited_nums = location.replace("line ", "")
            best_chunk = None
            best_score = 0
            for chunk_loc, chunk in line_range_chunks:
                if _line_range_overlaps(cited_nums, chunk_loc):
                    score = chunk.get("score", 0)
                    if score > best_score:
                        best_score = score
                        best_chunk = chunk
            return best_chunk
        return None

    # Extract citations and match to chunks
    unmatched_citations = []
    valid_matches = []

    for pattern, citation_type in citation_patterns:
        matches = list(re.finditer(pattern, answer))

        for match in matches:
            if citation_type == 'page':
                location = match.group(1)  # Page number
            elif citation_type == 'paragraph':
                location = f"para {match.group(1)}"  # Convert to "para X" format
            elif citation_type == 'line_range':
                location = f"line {match.group(1)}"  # Convert to "line X-Y" format
            elif citation_type == 'section':
                location = match.group(1).strip()  # Section name
            else:
                continue

            # Match section citations against section_name metadata
            if citation_type == 'section' and location in section_to_chunks:
                chunk = section_to_chunks[location]
                citation = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "location": location,
                    "citation_type": citation_type,
                    "relevance_score": chunk.get("score", 0)
                }
                if citation not in citations:
                    citations.append(citation)
                valid_matches.append(match)
                continue

            matched_chunk = _find_chunk_for_citation(location, citation_type)
            if matched_chunk:
                citation = {
                    "chunk_id": matched_chunk.get("chunk_id", ""),
                    "location": location,
                    "citation_type": citation_type,
                    "relevance_score": matched_chunk.get("score", 0)
                }
                if citation not in citations:  # Avoid duplicates
                    citations.append(citation)
                valid_matches.append(match)
            else:
                unmatched_citations.append(match.group(0))

    # Bare numbered citations [1], [2], [3] — the format the system prompt
    # actually mandates ("Use [1], [2], [3] ... corresponding to source excerpt
    # numbers in the context"). These map POSITIONALLY to the chunks as numbered
    # in the context, which format_legal_context orders by score descending —
    # mirror that exact ordering so [n] resolves to the right excerpt. Without
    # this, grounded `citations` were always empty (only [Page N]/[Section X]
    # were matched), so historical queries lost their supporting authorities.
    numbered_chunks = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
    for match in re.finditer(r'\[(\d+)\]', answer):
        n = int(match.group(1))
        if 1 <= n <= len(numbered_chunks):
            chunk = numbered_chunks[n - 1]
            citation = {
                "chunk_id": chunk.get("chunk_id", ""),
                "location": str(chunk.get("page_num", "")),
                "citation_type": "excerpt",
                "relevance_score": chunk.get("score", 0),
            }
            if citation not in citations:
                citations.append(citation)
            valid_matches.append(match)
        else:
            # References an excerpt number that does not exist → hallucinated.
            unmatched_citations.append(match.group(0))

    # Try fallback patterns if no citations found via primary patterns
    if not citations:
        for pattern, citation_type in fallback_patterns:
            matches = list(re.finditer(pattern, answer, re.IGNORECASE))
            for match in matches:
                if citation_type == 'page':
                    location = match.group(1)
                elif citation_type == 'paragraph':
                    location = f"para {match.group(1)}"
                elif citation_type == 'section':
                    location = match.group(1)
                else:
                    continue

                if location in location_to_chunks:
                    chunk = location_to_chunks[location]
                    citation = {
                        "chunk_id": chunk.get("chunk_id", ""),
                        "location": location,
                        "citation_type": citation_type,
                        "relevance_score": chunk.get("score", 0)
                    }
                    if citation not in citations:
                        citations.append(citation)
                    valid_matches.append(match)

    # Remove hallucinated citations from answer
    cleaned_answer = answer
    has_hallucinations = False

    if unmatched_citations:
        has_hallucinations = True
        logger.warning(
            f"Hallucination detected: answer references locations {unmatched_citations} "
            f"not in retrieved chunks {list(valid_locations)}. "
            f"Removing hallucinated citations from response."
        )

        # Remove all invalid citation patterns from answer
        for citation_text in unmatched_citations:
            cleaned_answer = cleaned_answer.replace(citation_text, "").strip()

        # Clean up any extra spaces
        cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

    return cleaned_answer, citations, has_hallucinations


def _best_excerpt(content: str, query: str = "", width: int = 480) -> str:
    """Return the ~``width``-char window of ``content`` most relevant to ``query``.

    The cited chunk can be 1-2k chars; blindly taking the first 500 often shows a
    passage that does NOT contain the cited fact. Slide a window over the chunk
    and centre on the region with the most query-term hits, so the displayed
    supporting excerpt actually contains what the user asked about. Falls back to
    the leading window when there is no query overlap (or no query).
    """
    content = (content or "").strip()
    if len(content) <= width:
        return content
    q_terms = {w for w in re.findall(r"[a-z0-9]{3,}", (query or "").lower())}
    if not q_terms:
        return content[:width].strip()
    # Find the sentence with the most query-term hits, then return a window
    # CENTRED on it (so the supporting sentence isn't clipped at the boundary).
    best_center, best_hits = None, 0
    for m in re.finditer(r"[^.!?]+[.!?]?", content):
        low = m.group(0).lower()
        hits = sum(1 for t in q_terms if t in low)
        if hits > best_hits:
            best_hits = hits
            best_center = (m.start() + m.end()) // 2
    if best_center is None:  # no overlap at all
        return content[:width].strip()
    start = max(0, best_center - width // 2)
    end = min(len(content), start + width)
    start = max(0, end - width)  # re-clamp when near the tail
    snippet = content[start:end].strip()
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(content) else ""
    return f"{prefix}{snippet}{suffix}"


def ground_citations_in_source(
    citations: List[Dict],
    chunks: List[Dict],
    query: str = "",
) -> Tuple[List[Dict], List[Dict], bool]:
    """
    Validate that citations are actually supported by source chunks.
    Extract supporting text excerpts for each citation.

    Args:
        citations: List of citation dicts from extract_citations
        chunks: List of retrieved chunks with content
        query: The user query — used to pick the most relevant excerpt window
            from each cited chunk (so the snippet actually contains the fact).

    Returns:
        Tuple of (grounded_citations, unsupported_claims, has_unsupported)
        - grounded_citations: Citations with supporting text excerpts
        - unsupported_claims: Claims that couldn't be grounded
        - has_unsupported: Bool indicating if any claims were unsupported
    """
    grounded_citations = []
    unsupported_claims = []

    # Map by both chunk_id (exact — used by bare [n] citations) and page_num
    # (used by [Page N]/[Section X] citations). Prefer chunk_id so a numbered
    # citation grounds against the EXACT excerpt it referenced, not just any
    # chunk that happens to share a page number.
    location_to_chunk = {str(c.get("page_num", "")): c for c in chunks}
    id_to_chunk = {str(c.get("chunk_id", "")): c for c in chunks if c.get("chunk_id")}

    for citation in citations:
        location = citation.get("location", "")
        chunk = id_to_chunk.get(str(citation.get("chunk_id", ""))) or location_to_chunk.get(location)

        if not chunk:
            # Citation location not found in chunks
            unsupported_claims.append({
                "location": location,
                "reason": "Location not found in retrieved chunks"
            })
            continue

        # Extract supporting text from chunk
        chunk_content = chunk.get("content", "")

        grounded_citation = {
            "location": location,
            "citation_type": citation.get("citation_type", "page"),
            "relevance_score": citation.get("relevance_score", 0),
            "chunk_id": chunk.get("chunk_id", ""),
            "supporting_excerpt": _best_excerpt(chunk_content, query),
            "is_grounded": True
        }

        grounded_citations.append(grounded_citation)
        logger.debug(f"Citation grounded: {location}")

    has_unsupported = len(unsupported_claims) > 0

    if has_unsupported:
        logger.warning(f"Found {len(unsupported_claims)} unsupported citations")

    return grounded_citations, unsupported_claims, has_unsupported


def _calculate_citation_coverage(answer: str, citations: List[Dict]) -> float:
    """Calculate percentage of answer with citation support (0.0-1.0)"""
    if not answer or not citations:
        return 0.0

    import re
    sentences = re.split(r'[.!?]+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    cited_sentences = 0
    for sentence in sentences:
        for citation in citations:
            location = citation.get("location", "")
            if not location:
                continue
            # Match formatted citation tokens to avoid bare-number false positives
            # (e.g. "3" matching inside any sentence containing that digit).
            # These formats mirror the citation tokens used in the system prompt.
            formatted_tokens = [
                f"[Page {location}]",
                f"[Paragraph {location}]",
                f"[Section {location}]",
                f"[¶ {location}]",
            ]
            if any(tok in sentence for tok in formatted_tokens):
                cited_sentences += 1
                break

    return min(1.0, cited_sentences / len(sentences))


def _calculate_average_relevance(citations: List[Dict]) -> float:
    """Calculate average relevance score of citations (0.0-1.0)"""
    if not citations:
        return 0.0

    scores = [c.get("relevance_score", 0.0) for c in citations]
    return sum(scores) / len(scores) if scores else 0.0


def _format_relevance_level(score: float) -> str:
    """Convert relevance score to descriptive level"""
    if score >= 0.85:
        return "excellent"
    elif score >= 0.70:
        return "good"
    elif score >= 0.55:
        return "moderate"
    else:
        return "weak"


def _generate_confidence_summary(
    overall_score: float,
    citation_coverage: float,
    avg_relevance: float,
    citation_count: int,
    has_hallucinations: bool
) -> str:
    """Generate human-readable summary of confidence factors"""
    rating = "high" if overall_score >= 0.75 else "medium" if overall_score >= 0.60 else "low"

    factors = []

    if citation_coverage > 0.8:
        factors.append("strong citation coverage")
    elif citation_coverage > 0.5:
        factors.append("partial citation coverage")
    else:
        factors.append("limited citation coverage")

    if avg_relevance >= 0.85:
        factors.append("highly relevant sources")
    elif avg_relevance >= 0.70:
        factors.append("moderately relevant sources")
    else:
        factors.append("weak source relevance")

    if citation_count >= 3:
        factors.append("multiple supporting citations")
    elif citation_count > 0:
        factors.append("single source citation")

    if has_hallucinations:
        factors.append("potential unsupported claims")

    if rating == "high":
        return f"High confidence answer with {', '.join(factors)}."
    elif rating == "medium":
        return f"Medium confidence answer with {', '.join(factors)}."
    else:
        return f"Low confidence answer due to {', '.join(factors)}."


def explain_confidence_score(
    answer: str,
    citations: List[Dict],
    has_hallucinations: bool,
    confidence_score: float
) -> Dict:
    """
    Generate structured explanation for confidence score with factor breakdown.

    Args:
        answer: Generated answer text
        citations: Grounded citations with scores
        has_hallucinations: Whether hallucinations were detected
        confidence_score: Overall confidence score (0.0-1.0)

    Returns:
        Dict with overall_score, rating, factors, and summary
    """
    # Calculate individual factors
    citation_coverage = _calculate_citation_coverage(answer, citations)
    avg_relevance = _calculate_average_relevance(citations)
    citation_count = len(citations)

    # Hallucination risk (1.0 if no hallucinations, lower if hallucinations present)
    hallucination_risk = 0.0 if has_hallucinations else 1.0

    # Citation quantity factor (scales with number of citations)
    citation_quantity_score = min(1.0, citation_count / 3.0)

    # Determine rating
    if confidence_score >= 0.75:
        rating = "high"
    elif confidence_score >= 0.60:
        rating = "medium"
    else:
        rating = "low"

    # Build factor explanations
    factors = {
        "citation_coverage": {
            "score": citation_coverage,
            "explanation": f"{int(citation_coverage * 100)}% of answer is supported by citations"
        },
        "source_relevance": {
            "score": avg_relevance,
            "explanation": f"Source relevance is {_format_relevance_level(avg_relevance)}"
        },
        "hallucination_risk": {
            "score": hallucination_risk,
            "explanation": "Potential unsupported claims detected" if has_hallucinations else "No unsupported claims detected"
        },
        "citation_quantity": {
            "score": citation_quantity_score,
            "explanation": f"{citation_count} supporting citations provided"
        }
    }

    # Generate summary
    summary = _generate_confidence_summary(
        confidence_score,
        citation_coverage,
        avg_relevance,
        citation_count,
        has_hallucinations
    )

    return {
        "overall_score": confidence_score,
        "rating": rating,
        "factors": factors,
        "summary": summary
    }


def calculate_answer_confidence(
    answer: str,
    citations: List[Dict],
    chunks: List[Dict],
    has_hallucinations: bool
) -> float:
    """
    Calculate confidence score for generated answer (0.0-1.0).

    Factors:
    - Citation coverage: % of answer sentences with citations
    - Chunk similarity: Average relevance score of cited chunks
    - Hallucination presence: Penalize for hallucinations
    - Citation count: More citations = higher confidence

    Args:
        answer: Generated answer text
        citations: Grounded citations with scores
        chunks: Retrieved chunks
        has_hallucinations: Whether hallucinations were detected

    Returns:
        Confidence score between 0.0 and 1.0
    """
    if not answer or not citations:
        return 0.0

    import re

    # Count logical sentences — split on sentence-ending punctuation that is NOT inside brackets
    # Remove bracketed citations first to get clean sentence boundaries, then check originals
    clean_for_splitting = re.sub(r'\[[^\]]*\]', ' [CITE] ', answer.strip())
    sentences = re.split(r'(?<=[.!?])\s+', clean_for_splitting)
    sentences = [s.strip() for s in sentences if s.strip() and s.strip() != '[CITE]']

    if not sentences:
        return 0.0

    # For citation coverage, check how many sentences in the ORIGINAL answer have citations nearby
    # Split original answer the same way and check for bracket patterns
    original_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\[])', answer.strip())
    original_sentences = [s.strip() for s in original_sentences if s.strip()]
    if not original_sentences:
        original_sentences = sentences

    citation_pattern = re.compile(r'\[')  # Any bracket citation in proximity
    cited_sentences = sum(1 for s in original_sentences if citation_pattern.search(s))
    citation_coverage = cited_sentences / len(original_sentences) if original_sentences else 0

    # Average citation relevance
    if citations:
        avg_relevance = sum(c.get("relevance_score", 0) for c in citations) / len(citations)
    else:
        avg_relevance = 0

    # Penalize for hallucinations
    hallucination_penalty = 0.2 if has_hallucinations else 0

    # Bonus for multiple citations (up to 0.15)
    citation_bonus = min(0.15, len(citations) * 0.03)

    # Recalibrated weights for realistic distribution
    # avg_relevance typically 0.6-0.8, citation_coverage typically 0.1-0.5
    confidence = (
        (citation_coverage * 0.25) +
        (avg_relevance * 0.45) +
        (citation_bonus) +
        0.15  # Base score for having any grounded response
    ) - hallucination_penalty

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    logger.debug(
        f"Confidence calculation: coverage={citation_coverage:.2f}, "
        f"relevance={avg_relevance:.2f}, hallucinations={has_hallucinations}, "
        f"final={confidence:.2f}"
    )

    return confidence


def classify_confidence_level(confidence_score: float) -> str:
    """
    Classify confidence score into categorical level.

    Args:
        confidence_score: Float between 0.0 and 1.0

    Returns:
        "high", "medium", "low", or "none"
    """
    if confidence_score >= 0.75:
        return "high"
    elif confidence_score >= 0.6:
        return "medium"
    elif confidence_score >= 0.4:
        return "low"
    else:
        return "none"


def _get_gemini_model(include_legal_research: bool = False):
    """Configure and return Gemini model instance."""
    if not settings.google_api_key:
        raise ValueError("GOOGLE_API_KEY is not configured.")

    genai.configure(api_key=settings.google_api_key)

    prompt = LEGAL_RESEARCH_SYSTEM_PROMPT if include_legal_research else LEGAL_SYSTEM_PROMPT

    return genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=prompt,
    )


async def generate_answer(
    query: str,
    context: str,
    temperature: float = 0.2,
    conversation_history: Optional[list] = None,
    include_legal_research: bool = False,
) -> Tuple[str, int]:
    """
    Generate answer using Google Gemini API.

    Args:
        query: User query
        context: Formatted context from retrieval
        temperature: Temperature parameter (0.2 for legal precision)
        conversation_history: Optional list of previous Q&A turns for follow-up support
        include_legal_research: Whether case law context is included

    Returns:
        Tuple of (answer, tokens_used)

    Raises:
        ValueError: If API key is not configured
        QueryProcessingException: If API call fails
    """
    try:
        # Build prompt with conversation history for follow-up support
        prompt_parts = [f"Context:\n{context}\n"]

        if conversation_history:
            prompt_parts.append("Previous conversation:")
            for turn in conversation_history[-5:]:  # Last 5 exchanges max
                prompt_parts.append(f"User: {turn['question']}")
                answer_preview = turn['answer'][:500] if turn.get('answer') else ''
                prompt_parts.append(f"Assistant: {answer_preview}")
            prompt_parts.append("")

        prompt_parts.append(f"Question: {query}")
        prompt = "\n".join(prompt_parts)

        # Provider routing: when configured for Groq (fast/free), generate with
        # Groq first and only fall through to Gemini if Groq is unavailable. The
        # legal system prompt is applied inside _groq_generate_answer.
        provider = (getattr(settings, "llm_answer_provider", "gemini") or "gemini").lower()
        if provider == "groq":
            groq_result = await _groq_generate_answer(prompt, include_legal_research, temperature)
            if groq_result:
                return groq_result  # (answer_text, tokens_used)
            logger.warning("llm_answer_provider=groq but Groq unavailable; falling back to Gemini")

        model = _get_gemini_model(include_legal_research=include_legal_research)

        # Retry transient API errors (rate limit / 5xx) with exponential backoff.
        # Timeouts are NOT retried (would compound latency); they fail fast.
        response = None
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = await asyncio.wait_for(
                    model.generate_content_async(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=2000,
                        ),
                    ),
                    timeout=45.0,
                )
                break
            except asyncio.TimeoutError:
                logger.error("LLM answer generation timed out after 45s")
                raise QueryProcessingException(
                    "Answer generation timed out",
                    detail="LLM did not respond within 45 seconds"
                )
            except (ResourceExhausted, GoogleAPIError) as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        f"Gemini transient error (attempt {attempt + 1}/{max_attempts}), "
                        f"retrying in {backoff}s: {e}"
                    )
                    await asyncio.sleep(backoff)
                    continue
                # Gemini exhausted (e.g. quota) — fall back to Groq before giving up.
                logger.warning(f"Gemini unavailable after {max_attempts} attempts ({e}); trying Groq fallback")
                groq_result = await _groq_generate_answer(prompt, include_legal_research, temperature)
                if groq_result:
                    return groq_result  # (answer_text, tokens_used)
                raise  # Groq unavailable too — outer handler wraps it

        if not getattr(response, "candidates", None):
            raise QueryProcessingException(
                "LLM response blocked (safety filter or empty candidates)",
                detail="Gemini returned no candidates — likely a safety filter rejection"
            )
        answer = response.text
        tokens_used = 0
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            tokens_used = getattr(response.usage_metadata, 'total_token_count', 0)

        return answer, tokens_used

    except ValueError:
        raise
    except QueryProcessingException:
        raise
    except (ResourceExhausted, GoogleAPIError) as e:
        logger.error(f"Google AI API error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Failed to generate answer with Google AI",
            detail=f"Google AI error: {str(e)}"
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error generating answer: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error during answer generation",
            detail=str(e)
        ) from e


async def query_matter(
    matter_id: str,
    query: str,
    db: Session,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2,
    conversation_history: Optional[list] = None,
    include_legal_research: bool = False,
    as_of_date=None,
) -> Dict:
    """
    Main RAG query orchestration function.

    Implements full RAG pipeline:
    1. Validate query
    2. Embed query
    3. Retrieve similar chunks
    4. Filter by confidence
    5. Format context
    6. Generate answer with LLM
    7. Extract citations
    8. Return structured response

    Args:
        matter_id: Matter identifier (UUID string)
        query: User query string
        db: Database session
        top_k: Number of chunks to include in context
        temperature: LLM temperature for response

    Returns:
        Dict with keys:
        - answer: Generated answer string (or None if error)
        - sources: List of source dicts with citation info
        - matter_id: Matter identifier
        - query: Original query
        - model: Model name used
        - tokens_used: Total tokens consumed
        - confidence: "high|medium|low|none"
        - error: Error message (or None)
    """
    error_response = {
        "answer": None,
        "sources": [],
        "citations": [],
        "matter_id": matter_id,
        "query": query,
        "model": GEMINI_MODEL,
        "tokens_used": 0,
        "confidence": {"level": "none", "score": 0.0, "factors": {}},
        "citation_verification": None,
        "claim_verification": None,
        "conflict_analysis": None,
        "error": None
    }

    # 1. Validate query
    if not query or len(query.strip()) < MIN_QUERY_LENGTH:
        error_response["error"] = f"Please enter a question with at least {MIN_QUERY_LENGTH} characters."
        return error_response

    try:
        # Get matter from database
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        if not matter:
            error_response["error"] = "This matter could not be found."
            return error_response

        # 2. Embed query + identify legal issues (parallel, zero added latency)
        issue_analysis = None
        if settings.problem_formulation_enabled:
            try:
                from backend.services.problem_formulation import identify_issues
            except ImportError:
                try:
                    from services.problem_formulation import identify_issues
                except ImportError:
                    from .problem_formulation import identify_issues

            try:
                query_embedding, issue_analysis = await asyncio.gather(
                    asyncio.to_thread(embed_query, query),
                    identify_issues(query),
                )
                if issue_analysis:
                    logger.info(
                        f"Problem formulation: {len(issue_analysis.get('issues', []))} issues identified"
                    )
            except (EmbeddingException, ValueError) as e:
                logger.error(f"Query embedding failed: {str(e)}")
                error_response["error"] = "We couldn't process your question right now. Please try again."
                return error_response
            except Exception as e:
                logger.warning(f"Problem formulation failed, falling back to standard embedding: {e}")
                try:
                    query_embedding = await asyncio.to_thread(embed_query, query)
                except (EmbeddingException, ValueError) as emb_e:
                    logger.error(f"Query embedding failed: {str(emb_e)}")
                    error_response["error"] = "We couldn't process your question right now. Please try again."
                    return error_response
                except Exception as emb_e:
                    logger.error(f"Unexpected error embedding query: {str(emb_e)}")
                    raise QueryProcessingException(
                        "Unexpected error during query embedding",
                        detail=str(emb_e)
                    ) from emb_e
        else:
            try:
                query_embedding = await asyncio.to_thread(embed_query, query)
            except (EmbeddingException, ValueError) as e:
                logger.error(f"Query embedding failed: {str(e)}")
                error_response["error"] = "We couldn't process your question right now. Please try again."
                return error_response
            except Exception as e:
                logger.error(f"Unexpected error embedding query: {str(e)}")
                raise QueryProcessingException(
                    "Unexpected error during query embedding",
                    detail=str(e)
                ) from e

        # 3. Retrieve chunks (request more than top_k to improve recall, then take best)
        # Detect optional filters from query text (jurisdiction, doc type)
        query_filters = _detect_query_filters(query)

        # 3.05 Resolve as-of-date temporal intent. Explicit param wins over NL.
        temporal_as_of = None
        temporal_exclude_superseded = True
        temporal_filter_relaxed = False
        if getattr(settings, "temporal_as_of_enabled", True):
            if as_of_date is not None:
                temporal_as_of, temporal_exclude_superseded = as_of_date, False
            else:
                temporal_as_of, temporal_exclude_superseded = _detect_temporal_intent(query)
            if temporal_as_of is not None:
                logger.info(
                    f"Temporal as-of filtering active: as_of={temporal_as_of}, "
                    f"source={'explicit' if as_of_date is not None else 'nl-intent'}"
                )

        # Build the combined (field + temporal) Qdrant filter via the shared builder.
        try:
            from backend.services.vector_store import build_temporal_filter
        except ImportError:
            try:
                from services.vector_store import build_temporal_filter
            except ImportError:
                from .vector_store import build_temporal_filter

        combined_filter = build_temporal_filter(
            as_of_date=temporal_as_of,
            exclude_superseded=temporal_exclude_superseded,
            base_filter=query_filters if query_filters else None,
        )

        try:
            retrieved_chunks = retrieve_chunks(
                matter_id, query_embedding,
                top_k=RETRIEVAL_LIMIT,
                query_filter=combined_filter,
            )
            # Progressive fallback: temporal+jurisdiction → jurisdiction-only →
            # unfiltered. Relax in steps so the UI can warn the result is broader
            # than requested.
            if len(retrieved_chunks) < 3:
                # Step 1: drop temporal constraint, keep jurisdiction.
                if temporal_as_of is not None or temporal_exclude_superseded:
                    logger.info(
                        f"Temporal-filtered search returned {len(retrieved_chunks)} results, "
                        "relaxing temporal constraint"
                    )
                    relaxed_filter = (
                        dict(query_filters) if query_filters else None
                    )
                    retrieved_chunks = retrieve_chunks(
                        matter_id, query_embedding,
                        top_k=RETRIEVAL_LIMIT,
                        query_filter=relaxed_filter,
                    )
                    temporal_filter_relaxed = True
                # Step 2: drop jurisdiction too → fully unfiltered.
                if query_filters and len(retrieved_chunks) < 3:
                    logger.info(
                        f"Jurisdiction-filtered search returned {len(retrieved_chunks)} results, "
                        "retrying unfiltered"
                    )
                    retrieved_chunks = retrieve_chunks(
                        matter_id, query_embedding, top_k=RETRIEVAL_LIMIT
                    )
        except (VectorStoreException, ValueError) as e:
            logger.error(f"Chunk retrieval failed: {str(e)}")
            error_response["error"] = "No documents are available to answer this question yet. Please upload a document or wait for processing to finish."
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error retrieving chunks: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during chunk retrieval",
                detail=str(e)
            ) from e

        # 3.25 Hybrid search: BM25 sparse retrieval + RRF fusion (optional, non-blocking)
        if getattr(settings, 'hybrid_search_enabled', True):
            try:
                from backend.services.hybrid_search import (
                    generate_sparse_vector, get_rrf_weights, reciprocal_rank_fusion
                )
                from backend.services.vector_store import search_sparse_vectors

                sparse_query = generate_sparse_vector(query)
                if sparse_query:
                    bm25_weight, dense_weight = get_rrf_weights(query)
                    # Mirror the filter the dense path actually used: full
                    # temporal+field filter unless it was progressively relaxed.
                    sparse_filter = (
                        (dict(query_filters) if query_filters else None)
                        if temporal_filter_relaxed
                        else combined_filter
                    )
                    bm25_results = search_sparse_vectors(
                        matter_id,
                        sparse_query,
                        limit=RETRIEVAL_LIMIT,
                        query_filter=sparse_filter,
                    )
                    if bm25_results:
                        retrieved_chunks = reciprocal_rank_fusion(
                            bm25_results=bm25_results,
                            dense_results=retrieved_chunks,
                            bm25_weight=bm25_weight,
                            dense_weight=dense_weight,
                            top_n=RETRIEVAL_LIMIT,
                        )
                        logger.info(
                            f"Hybrid search: merged {len(bm25_results)} BM25 + dense results "
                            f"(weights: bm25={bm25_weight}, dense={dense_weight})"
                        )
                    else:
                        logger.debug("BM25 search returned no results, using dense-only")
                else:
                    logger.debug("Sparse query generation returned None, using dense-only")
            except ImportError:
                logger.debug("Hybrid search modules not available, using dense-only retrieval")
            except Exception as e:
                logger.warning(f"Hybrid search failed (non-blocking, using dense-only): {e}")

        # 3.5 On-demand legal research (CourtListener)
        external_chunks = []
        if include_legal_research:
            try:
                case_results = await search_cases(query, max_results=5)
                if case_results:
                    external_chunks = format_as_context(case_results)
                    logger.info(f"CourtListener returned {len(external_chunks)} case law results")
            except Exception as e:
                logger.warning(f"Legal research failed (non-blocking): {e}")

        # Merge external chunks with retrieved chunks
        if external_chunks:
            retrieved_chunks = retrieved_chunks + external_chunks

        # Check for empty retrieval
        if not retrieved_chunks:
            error_response["error"] = "I couldn't find anything in the uploaded documents relevant to that question. Try rephrasing or adding more detail."
            return error_response

        # Log scores for debugging
        scores = [c.get("score", 0) for c in retrieved_chunks]
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks, scores: {[f'{s:.4f}' for s in scores]}")

        # 4. Filter by confidence and select top k; fallback to best available if all below threshold
        high_confidence_chunks = [c for c in retrieved_chunks if c.get("score", 0) >= MIN_CONFIDENCE_SCORE]
        low_confidence_fallback = False
        retrieval_avg_score = sum(scores) / len(scores) if scores else 0.0

        if not high_confidence_chunks:
            # Use best available chunks and mark as low confidence (still answer the question)
            low_confidence_fallback = True
            logger.warning(
                f"Low confidence retrieval - average score: {retrieval_avg_score:.2f}; "
                "using top chunks and marking answer as low confidence"
            )
            initial_chunks = sorted(retrieved_chunks, key=lambda x: x.get("score", 0), reverse=True)[:RETRIEVAL_TOP_K]
        else:
            initial_chunks = sorted(high_confidence_chunks, key=lambda x: x.get("score", 0), reverse=True)[:RETRIEVAL_TOP_K]

        # 4.5. Rerank chunks for better relevance
        # Separate case law chunks so they aren't dropped by reranking
        doc_chunks = [c for c in initial_chunks if c.get("source_type") != "case_law"]
        case_law_chunks = [c for c in initial_chunks if c.get("source_type") == "case_law"]

        try:
            reranked_docs = await asyncio.to_thread(rerank_chunks, query, doc_chunks, top_k=top_k)
            logger.debug(f"Reranking improved chunks: {len(doc_chunks)} → {len(reranked_docs)}")
        except Exception as e:
            logger.warning(f"Reranking failed, using initial chunks: {str(e)}")
            reranked_docs = doc_chunks[:top_k]

        # Merge: document chunks + case law chunks (case law always included)
        final_chunks = reranked_docs + case_law_chunks

        # 4.55. Temporal filtering — AUTHORITATIVE, DB-driven (Document table is the
        # source of truth). Qdrant payload status/dates can be stale (a doc may be
        # superseded AFTER its chunks were embedded), so we resolve document_status,
        # effective_date and superseded_date from the Document rows here and apply:
        #   - as-of date set  → keep docs in force on that date:
        #       (effective_date is null OR effective_date <= as_of)
        #       AND (superseded_date is null OR superseded_date > as_of)
        #   - no as-of (current-law) → exclude superseded/repealed.
        # This is what actually makes amendment-chain supersession work at query time.
        if getattr(settings, 'temporal_filter_default', True):
            doc_temporal: dict = {}  # doc_id -> (status, effective_date, superseded_date)
            try:
                temporal_doc_ids = set()
                for c in final_chunks:
                    raw_id = c.get("document_id")
                    if raw_id:
                        try:
                            temporal_doc_ids.add(UUID(str(raw_id)))
                        except (ValueError, TypeError):
                            pass
                if temporal_doc_ids:
                    rows = (
                        db.query(
                            Document.id, Document.document_status,
                            Document.effective_date, Document.superseded_date,
                        )
                        .filter(Document.id.in_(temporal_doc_ids))
                        .all()
                    )
                    doc_temporal = {
                        str(r[0]): ((r[1] or "unknown"), r[2], r[3]) for r in rows
                    }
            except Exception as temporal_exc:
                logger.warning(f"Temporal lookup failed, skipping temporal filter: {temporal_exc}")

            def _keep_chunk(chunk: dict) -> bool:
                doc_id = str(chunk.get("document_id", "") or "")
                status, eff, sup = doc_temporal.get(doc_id, ("unknown", None, None))
                if temporal_as_of is not None:
                    # In force on as_of date?
                    if eff is not None and eff > temporal_as_of:
                        return False  # not yet effective
                    if sup is not None and sup <= temporal_as_of:
                        return False  # already superseded by then
                    return True
                # current-law default: drop superseded/repealed
                return status not in ("superseded", "repealed")

            active_chunks = [c for c in final_chunks if _keep_chunk(c)]
            if active_chunks:
                dropped = len(final_chunks) - len(active_chunks)
                if dropped > 0:
                    logger.info(
                        f"Temporal filter ({'as-of '+str(temporal_as_of.date()) if temporal_as_of else 'current-law'}): "
                        f"excluded {dropped} out-of-force chunks"
                    )
                    if temporal_as_of is None:
                        temporal_filter_relaxed = False  # genuine filter applied, not a relax
                final_chunks = active_chunks
            else:
                # All chunks filtered out → keep them (better than no results), flag relaxed.
                temporal_filter_relaxed = True

        # 4.55b. CANONICAL CITATION ORDERING (correctness-critical).
        # The inline [n] markers the LLM emits must resolve to the SAME source
        # everywhere: format_legal_context numbers excerpts by score-desc, and the
        # citation/claim verifiers index sources[n-1] positionally. If final_chunks
        # is left in rerank-merge order, [n] in the context points at one chunk but
        # the verifier checks a different one — so a claim gets "verified" against
        # the wrong authority. Sort once here into the canonical (score-desc) order
        # so format_legal_context, extract_citations, verify_* and the sources list
        # all agree on what [n] means.
        final_chunks = sorted(final_chunks, key=lambda c: c.get("score", 0), reverse=True)

        # 4.6. Cross-source conflict detection
        conflict_analysis = None
        if settings.conflict_detection_enabled:
            try:
                conflict_analysis = await asyncio.to_thread(
                    detect_conflicts, final_chunks, query=query
                )
                if conflict_analysis:
                    logger.info(
                        f"Conflict detection: {conflict_analysis['summary']['total_conflicts']} "
                        f"conflicts found ({conflict_analysis['summary']['high_severity']} high)"
                    )
            except Exception as e:
                logger.warning(f"Conflict detection failed (non-blocking): {e}")

        # 5. Format context with token budgeting
        try:
            # Fetch document summaries for retrieved chunks
            doc_ids = set()
            for c in final_chunks:
                if c.get("document_id"):
                    try:
                        doc_ids.add(UUID(c["document_id"]))
                    except (ValueError, AttributeError):
                        pass  # Skip non-UUID IDs (e.g. CourtListener chunks)
            doc_summaries = {}
            if doc_ids:
                docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
                doc_summaries = {
                    str(d.id): {"name": d.name, "summary": d.summary}
                    for d in docs if d.summary
                }

            formatted_context = format_legal_context(final_chunks, matter.name, doc_summaries)

            # Augment context with conflict notices (if any)
            if conflict_analysis:
                formatted_context = augment_context_with_conflicts(
                    formatted_context, conflict_analysis
                )

            # 4.7. Citation graph validation — inject good/bad-law signals for the
            #      authorities the answer will actually cite (union of query + top
            #      chunks). Non-blocking: any failure leaves formatted_context intact.
            # Keep graph_block in outer scope so the token-budget path can re-apply it.
            _saved_graph_block: Optional[str] = None
            if settings.citation_graph_enabled and getattr(
                settings, "citation_graph_rag_injection_enabled", True
            ):
                try:
                    try:
                        from backend.services.citation_extractor import extract_all_citations
                    except ImportError:
                        try:
                            from services.citation_extractor import extract_all_citations
                        except ImportError:
                            from .citation_extractor import extract_all_citations

                    cite_text = query + "\n" + "\n".join(
                        c.get("content", "") for c in final_chunks[:5]
                    )
                    # use_llm=False → regex + eyecite only, no network, ~0 latency.
                    found_citations = await extract_all_citations(cite_text, use_llm=False)
                    _saved_graph_block = await asyncio.to_thread(
                        _build_graph_context, db, matter_id, found_citations
                    )
                    if _saved_graph_block:
                        # PREPEND so the validation survives any later token-budget trim.
                        formatted_context = _saved_graph_block + "\n\n" + formatted_context
                except Exception as _graph_err:
                    logger.warning(
                        f"Graph validation context failed (non-blocking): {_graph_err}"
                    )

            # Inject identified issues into context for per-issue CREAC structuring.
            # Keep issue_context in outer scope so the token-budget path can re-apply it.
            _saved_issue_context: Optional[str] = None
            if issue_analysis and issue_analysis.get("issues"):
                issue_lines = []
                for iss in issue_analysis["issues"]:
                    domain = iss.get("domain", "general").upper()
                    question = iss.get("legal_question", iss.get("issue", ""))
                    conf = iss.get("confidence", 0)
                    issue_lines.append(f"- {domain}: {question} (confidence: {conf:.0%})")
                _saved_issue_context = "\n".join(issue_lines)
                formatted_context = (
                    f"IDENTIFIED LEGAL ISSUES (structure your CREAC analysis per-issue):\n"
                    f"{_saved_issue_context}\n\n{formatted_context}"
                )

            # Count tokens in context (estimate)
            context_tokens = count_tokens_estimate(formatted_context)
            query_tokens = count_tokens_estimate(query)

            # Check token budget (reserve buffer for response).
            # Strategy: trim chunks first, then re-apply ALL enrichment blocks
            # (conflict, graph, issue-context) so no signals are silently dropped.
            # Only as a last resort drop enrichment WITH an explicit warning log.
            estimated_total = context_tokens + query_tokens + 500  # 500 token buffer
            if estimated_total > CONTEXT_TOKEN_BUDGET:
                # Trim to 2 chunks and rebuild context with all enrichment blocks.
                logger.warning(
                    f"Token budget exceeded ({estimated_total} > {CONTEXT_TOKEN_BUDGET}): "
                    f"trimming to 2 chunks and re-applying enrichment blocks"
                )
                final_chunks = final_chunks[:2]
                formatted_context = format_legal_context(final_chunks, matter.name, doc_summaries)

                # Re-apply conflict augmentation
                if conflict_analysis:
                    formatted_context = augment_context_with_conflicts(
                        formatted_context, conflict_analysis
                    )

                # Re-apply graph block
                if _saved_graph_block:
                    formatted_context = _saved_graph_block + "\n\n" + formatted_context

                # Re-apply issue context
                if _saved_issue_context:
                    formatted_context = (
                        f"IDENTIFIED LEGAL ISSUES (structure your CREAC analysis per-issue):\n"
                        f"{_saved_issue_context}\n\n{formatted_context}"
                    )

                context_tokens = count_tokens_estimate(formatted_context)
                estimated_total = context_tokens + query_tokens + 500

                if estimated_total > CONTEXT_TOKEN_BUDGET:
                    # Still over budget: drop enrichment blocks and warn explicitly.
                    logger.warning(
                        "Token budget exceeded even after chunk trim: "
                        "graph/issue/conflict context stripped from prompt to fit budget"
                    )
                    formatted_context = format_legal_context(final_chunks, matter.name, doc_summaries)
                    context_tokens = count_tokens_estimate(formatted_context)
                    estimated_total = context_tokens + query_tokens + 500

                    if estimated_total > CONTEXT_TOKEN_BUDGET:
                        error_response["error"] = "Your question matched too much material to process at once. Please ask something more specific."
                        return error_response

        except ValueError as e:
            logger.error(f"Context formatting validation failed: {str(e)}")
            error_response["error"] = "We couldn't prepare an answer for that question. Please try again."
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error formatting context: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during context formatting",
                detail=str(e)
            ) from e

        # 6. Generate answer
        try:
            answer, tokens_used = await generate_answer(query, formatted_context, temperature, conversation_history, include_legal_research)
        except QueryProcessingException as e:
            logger.error(f"Answer generation failed: {str(e)}")
            error_response["error"] = "The answer service is temporarily unavailable. Please try again in a moment."
            return error_response
        except Exception as e:
            logger.error(f"Unexpected error generating answer: {str(e)}")
            raise QueryProcessingException(
                "Unexpected error during answer generation",
                detail=str(e)
            ) from e

        # 7. Extract citations and detect hallucinations
        cleaned_answer, citations, has_hallucinations = extract_citations(answer, final_chunks)

        # 7.5. Ground citations in source text
        grounded_citations, unsupported_claims, has_unsupported = ground_citations_in_source(citations, final_chunks, query)

        if has_unsupported:
            logger.warning(f"Found {len(unsupported_claims)} unsupported citation(s)")
            # Mark unsupported claims in response
            for claim in unsupported_claims:
                logger.warning(f"  - Unsupported: {claim['location']} ({claim['reason']})")

        # 7.6 + 7.65: Citation verification + Claim verification (parallel)
        citation_verification = None
        claim_verification = None

        async def _verify_citations():
            if not settings.citation_verification_enabled:
                return None
            try:
                result = await verify_response_citations(cleaned_answer, final_chunks)
                if result:
                    logger.info(f"Citation verification: {result.get('summary', {})}")
                return result
            except Exception as e:
                logger.warning(f"Citation verification failed (non-blocking): {e}")
                return None

        async def _verify_claims():
            if not settings.claim_verification_enabled:
                return None
            try:
                from backend.services.claim_verifier import verify_claims
            except ImportError:
                try:
                    from services.claim_verifier import verify_claims
                except ImportError:
                    from .claim_verifier import verify_claims
            try:
                result = await verify_claims(cleaned_answer, final_chunks)
                if result:
                    logger.info(f"Claim verification: {result.get('summary', {})}")
                return result
            except Exception as e:
                logger.warning(f"Claim verification failed (non-blocking): {e}")
                return None

        citation_verification, claim_verification = await asyncio.gather(
            _verify_citations(),
            _verify_claims(),
        )

        # Use annotated answer from citation verification if available
        if citation_verification and citation_verification.get("annotated_answer"):
            cleaned_answer = citation_verification["annotated_answer"]

        # 7.7. Calculate answer confidence score (incorporate verification results)
        if low_confidence_fallback:
            answer_confidence_score = max(0.0, min(1.0, retrieval_avg_score))
            answer_confidence_level = "low"
        else:
            answer_confidence_score = calculate_answer_confidence(
                cleaned_answer,
                grounded_citations,
                final_chunks,
                has_hallucinations
            )

            # Adjust confidence based on claim verification results
            if claim_verification and claim_verification.get("summary"):
                cv_summary = claim_verification["summary"]
                total = cv_summary.get("total", 0)
                if total > 0:
                    supported_ratio = cv_summary.get("supported", 0) / total
                    unsupported_count = cv_summary.get("unsupported", 0)
                    # Penalize if many claims are unsupported
                    if unsupported_count > 0:
                        penalty = min(0.3, unsupported_count * 0.1)
                        answer_confidence_score = max(0.0, answer_confidence_score - penalty)
                    # Boost slightly if all claims verified
                    elif supported_ratio >= 0.9:
                        answer_confidence_score = min(1.0, answer_confidence_score + 0.05)

            answer_confidence_level = classify_confidence_level(answer_confidence_score)

        logger.info(
            f"Answer quality: confidence={answer_confidence_level} "
            f"(score={answer_confidence_score:.2f}), "
            f"grounded_citations={len(grounded_citations)}, "
            f"unsupported={len(unsupported_claims)}, "
            f"hallucinations={has_hallucinations}"
        )

        # 8. Prepare sources list with FULL content from database (batch fetch)
        sources = []
        # Batch-fetch all chunk content in a single query
        chunk_uuids = []
        for chunk in final_chunks:
            chunk_id = chunk.get("chunk_id", "")
            if chunk_id:
                try:
                    chunk_uuids.append(UUID(chunk_id))
                except (ValueError, AttributeError):
                    pass

        db_chunks_map = {}
        if chunk_uuids:
            db_chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_uuids)).all()
            db_chunks_map = {str(c.id): c for c in db_chunks}

        for chunk in final_chunks:
            chunk_id = chunk.get("chunk_id", "")
            db_chunk = db_chunks_map.get(chunk_id)
            full_content = db_chunk.content if db_chunk else chunk.get("content", "")

            source = {
                "chunk_id": chunk_id,
                "page_num": chunk.get("page_num", ""),
                "section_name": chunk.get("section_name", ""),
                "relevance_score": chunk.get("score", 0),
                "content": full_content,
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("document_name", ""),
                "source_type": chunk.get("source_type", "document"),
                "url": chunk.get("url", ""),
                "court_level": chunk.get("court_level"),
                "jurisdiction_code": chunk.get("jurisdiction_code"),
                "authority_score": chunk.get("authority_score"),
                "binding_authority": chunk.get("binding_authority"),
                "effective_date": chunk.get("effective_date"),
                "document_status": chunk.get("document_status"),
            }
            sources.append(source)

        # Use calculated confidence score (more sophisticated than simple averaging)
        confidence = answer_confidence_level

        # Prepare successful response (using cleaned_answer without hallucinated citations)
        return {
            "answer": cleaned_answer,
            "sources": sources,
            "citations": grounded_citations,  # Citations with supporting excerpts
            "matter_id": matter_id,
            "query": query,
            "model": GEMINI_MODEL,
            "tokens_used": tokens_used,
            "confidence": {
                "level": confidence,  # "high", "medium", "low", "none"
                "score": answer_confidence_score,  # 0.0-1.0
                "factors": {
                    "has_hallucinations": has_hallucinations,
                    "unsupported_claims": len(unsupported_claims),
                    "grounded_citations": len(grounded_citations),
                    "avg_citation_relevance": (
                        sum(c.get("relevance_score", 0) for c in grounded_citations) / len(grounded_citations)
                        if grounded_citations else 0.0
                    )
                }
            },
            "citation_verification": citation_verification,
            "claim_verification": claim_verification,
            "conflict_analysis": conflict_analysis,
            "issue_analysis": issue_analysis,
            "temporal_filter_relaxed": temporal_filter_relaxed,
            "as_of_date": temporal_as_of.isoformat() if temporal_as_of is not None else None,
            "error": None
        }

    except QueryProcessingException as e:
        logger.error(f"Query processing error in query_matter: {str(e)}")
        error_response["error"] = "Something went wrong while answering your question. Please try again."
        return error_response
    except Exception as e:
        logger.error(f"Unexpected error in query_matter: {str(e)}")
        raise QueryProcessingException(
            "Unexpected error in query processing pipeline",
            detail=str(e)
        ) from e
