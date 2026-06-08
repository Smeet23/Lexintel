"""Agentic RAG orchestrator using LangGraph.

Multi-step query pipeline with routing, clarification, CRAG-style evaluation,
and query rewriting. Uses Groq (Llama 3.3 70B) for fast LLM operations and
existing Gemini integration for answer generation.

Fast path: simple queries bypass the graph and call query_matter() directly.
Slow path: complex queries go through clarify -> retrieve -> evaluate -> generate -> verify.

Feature-gated via settings.agentic_rag_enabled. Falls back to single-pass RAG on failure.
"""

import asyncio
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from typing_extensions import TypedDict

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
# Optional imports -- degrade gracefully
# ---------------------------------------------------------------------------

_LANGGRAPH_AVAILABLE = False
try:
    from langgraph.graph import START, END, StateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    logger.warning("langgraph not installed. Agentic RAG disabled.")

_GROQ_AVAILABLE = False
_groq_client = None
try:
    from groq import Groq
    _groq_key = getattr(settings, "groq_api_key", "")
    if _groq_key:
        _groq_client = Groq(api_key=_groq_key)
        _GROQ_AVAILABLE = True
    else:
        logger.info("GROQ_API_KEY not set. Fast LLM ops will use Gemini fallback.")
except ImportError:
    logger.info("groq SDK not installed. Fast LLM ops will use Gemini fallback.")

# Existing services (triple-import pattern)
try:
    from backend.services.rag_engine import (
        query_matter, embed_query, retrieve_chunks, rerank_chunks,
        format_legal_context, generate_answer, extract_citations,
        ground_citations_in_source, RETRIEVAL_LIMIT, FINAL_CHUNK_COUNT,
        MIN_CONFIDENCE_SCORE, classify_confidence_level, GEMINI_MODEL,
    )
    from backend.services.hybrid_search import (
        classify_query_type, generate_sparse_vector,
        get_rrf_weights, reciprocal_rank_fusion,
    )
    from backend.services.vector_store import search_sparse_vectors
    from backend.services.citation_agent import verify_response_citations
    from backend.services.claim_verifier import verify_claims
    from backend.services.conflict_detector import detect_conflicts
    from backend.services.legal_research import search_cases, format_as_context
    from backend.models import Matter, Document
except ImportError:
    try:
        from services.rag_engine import (
            query_matter, embed_query, retrieve_chunks, rerank_chunks,
            format_legal_context, generate_answer, extract_citations,
            ground_citations_in_source, RETRIEVAL_LIMIT, FINAL_CHUNK_COUNT,
            MIN_CONFIDENCE_SCORE, classify_confidence_level, GEMINI_MODEL,
        )
        from services.hybrid_search import (
            classify_query_type, generate_sparse_vector,
            get_rrf_weights, reciprocal_rank_fusion,
        )
        from services.vector_store import search_sparse_vectors
        from services.citation_agent import verify_response_citations
        from services.claim_verifier import verify_claims
        from services.conflict_detector import detect_conflicts
        from services.legal_research import search_cases, format_as_context
        from models import Matter, Document
    except ImportError:
        from .rag_engine import (
            query_matter, embed_query, retrieve_chunks, rerank_chunks,
            format_legal_context, generate_answer, extract_citations,
            ground_citations_in_source, RETRIEVAL_LIMIT, FINAL_CHUNK_COUNT,
            MIN_CONFIDENCE_SCORE, classify_confidence_level, GEMINI_MODEL,
        )
        from .hybrid_search import (
            classify_query_type, generate_sparse_vector,
            get_rrf_weights, reciprocal_rank_fusion,
        )
        from .vector_store import search_sparse_vectors
        from .citation_agent import verify_response_citations
        from .claim_verifier import verify_claims
        from .conflict_detector import detect_conflicts
        from .legal_research import search_cases, format_as_context
        from ..models import Matter, Document

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_REWRITE_ITERATIONS = 3
RELEVANCE_THRESHOLD = 0.35  # Minimum average score to consider retrieval sufficient

# Sufficiency / strategy-selection threshold defaults (overridable via settings).
SUFFICIENCY_HIGH = 0.70
COVERAGE_HIGH = 0.80
RELEVANCE_LOW = 0.45
COVERAGE_LOW = 0.50
MIN_FILTERED_CHUNKS = 3  # below this, relax filters


def _setting_num(name: str, default):
    """Read a numeric setting, coercing non-numeric values (e.g. test MagicMocks)
    to the provided default. Keeps thresholds robust under mocked settings."""
    val = getattr(settings, name, default)
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return default
    return val

# Stopword set for the sub-question coverage overlap heuristic.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is", "are",
    "was", "were", "be", "been", "what", "which", "who", "whom", "how", "does",
    "do", "did", "can", "could", "would", "should", "this", "that", "these",
    "those", "with", "by", "as", "at", "from", "it", "its", "their", "there",
})


# ═══════════════════════════════════════════════════════════════════════════
# State definition
# ═══════════════════════════════════════════════════════════════════════════

class AgenticState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    # Inputs
    matter_id: str
    query: str
    db: Any  # SQLAlchemy Session (not serialisable, kept in-process only)
    conversation_history: Optional[list]
    conversation_summary: Optional[str]  # rolling summary of older turns (context mgmt)
    include_legal_research: bool
    temperature: float

    # Router
    complexity: str  # "simple" | "complex"
    query_type: str  # "citation" | "conceptual" | "mixed"

    # Clarification
    jurisdiction: Optional[str]
    entities: List[str]
    refined_query: str
    sub_questions: List[str]
    requires_external_research: bool
    temporal_scope: Optional[str]

    # Planning
    research_plan: Optional[Dict]  # last-writer-wins, no reducer

    # Retrieval
    chunks: List[Dict]
    retrieval_score: float
    rewrite_count: int
    filters_relaxed: bool
    force_external: bool
    external_done: bool

    # Sufficiency / strategy
    sufficiency_signals: Optional[Dict]
    next_strategy: str
    strategy_log: List[str]

    # Generation
    answer: str
    sources: List[Dict]
    citations: List[Dict]          # grounded citations (parity with query_matter)
    issue_analysis: Optional[Dict] # problem-formulation issues (parity)
    tokens_used: int

    # Verification
    citation_verification: Optional[Dict]
    claim_verification: Optional[Dict]
    conflict_analysis: Optional[Dict]

    # Final
    result: Optional[Dict]
    error: Optional[str]


# ═══════════════════════════════════════════════════════════════════════════
# Fast LLM helper (Groq w/ Gemini fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _fast_llm_json(system_prompt: str, user_prompt: str) -> Optional[Dict]:
    """Call a fast LLM and parse JSON response. Returns None on failure."""
    # Try Groq first
    if _GROQ_AVAILABLE and _groq_client:
        try:
            resp = _groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Groq fast LLM failed, falling back to Gemini: {e}")

    # Gemini fallback (synchronous, heavier but always available)
    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(settings.gemini_model)
        combined = f"{system_prompt}\n\nRespond ONLY with valid JSON.\n\n{user_prompt}"
        resp = model.generate_content(combined)
        text = resp.text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        logger.error(f"Fast LLM (Gemini fallback) also failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Graph nodes
# ═══════════════════════════════════════════════════════════════════════════

async def context_manager_node(state: AgenticState) -> dict:
    """Context management / summarisation — the FIRST node in the loop.

    Long multi-turn conversations would otherwise grow the prompt unbounded. Once
    the history exceeds ``agentic_context_summarize_after`` turns, summarise the
    OLDER turns into a single rolling brief (via the fast/free Groq LLM) and keep
    only the most-recent ``agentic_context_keep_recent`` turns verbatim. The
    managed history (``[summary turn] + recent turns``) replaces
    ``conversation_history`` in state so every downstream node — and the fast-path
    delegation — works from bounded context. Best-effort: any failure leaves the
    original history untouched and never blocks the loop.
    """
    history = state.get("conversation_history") or []
    summarize_after = int(_setting_num("agentic_context_summarize_after", 6))
    keep_recent = int(_setting_num("agentic_context_keep_recent", 4))
    if not isinstance(history, list) or len(history) <= summarize_after:
        return {}

    older = history[:-keep_recent] if keep_recent > 0 else history
    recent = history[-keep_recent:] if keep_recent > 0 else []
    convo = "\n\n".join(
        f"User: {(t.get('question') or '').strip()}\n"
        f"Assistant: {((t.get('answer') or '').strip())[:600]}"
        for t in older if isinstance(t, dict)
    ).strip()
    if not convo:
        return {}

    max_chars = int(_setting_num("agentic_context_summary_max_chars", 1500))
    prompt = (
        "Summarise the following legal Q&A conversation into a concise factual "
        "brief (key facts, parties, legal issues, and prior conclusions) that "
        "preserves the context needed to answer follow-up questions. Keep it under "
        f"{max_chars} characters. Do NOT invent information.\n\nCONVERSATION:\n{convo}"
    )
    try:
        try:
            from backend.services import llm
        except ImportError:  # pragma: no cover - layout fallback
            from services import llm
        summary = await llm.agenerate(
            prompt, provider="groq", fallback=True,
            temperature=0.0, max_output_tokens=600,
        )
    except Exception as exc:  # never block the loop on summarisation
        logger.warning(f"Context summarisation skipped: {exc}")
        return {}

    if not summary or not str(summary).strip():
        return {}
    summary = str(summary).strip()[:max_chars]
    managed = [{"question": "[Summary of earlier conversation]", "answer": summary}] + recent
    logger.info(
        f"Context manager: summarised {len(older)} older turns "
        f"(history {len(history)} → {len(managed)} turns)"
    )
    return {"conversation_history": managed, "conversation_summary": summary}


def router_node(state: AgenticState) -> dict:
    """Classify query complexity. Simple queries take the fast path."""
    query = state["query"]
    query_type = classify_query_type(query)

    # Heuristic: short queries or pure citation lookups are simple
    word_count = len(query.split())
    is_simple = (
        word_count < 15
        and query_type == "citation"
    ) or word_count < 8

    # For borderline cases, ask the LLM
    if not is_simple and word_count < 25:
        result = _fast_llm_json(
            "You classify legal queries. Respond with JSON: {\"complexity\": \"simple\" or \"complex\"}",
            f"Query: {query}",
        )
        if result and result.get("complexity") == "simple":
            is_simple = True

    complexity = "simple" if is_simple else "complex"
    logger.info(f"Router: complexity={complexity}, query_type={query_type}, words={word_count}")
    return {"complexity": complexity, "query_type": query_type}


def clarify_node(state: AgenticState) -> dict:
    """Extract jurisdiction, entities, sub-questions, external-research need, and
    temporal scope; produce a refined query for retrieval."""
    query = state["query"]
    result = _fast_llm_json(
        (
            "You are a legal query analyst. Extract structured information from the query.\n"
            "Decompose the query into atomic sub-questions only if it genuinely asks\n"
            "more than one thing; otherwise return a single-element list.\n"
            "Respond with JSON: {\n"
            '  "jurisdiction": "US" | "UK" | "EU" | "IN" | "AU" | "CA" | null,\n'
            '  "entities": ["entity1", "entity2"],\n'
            '  "refined_query": "improved search query",\n'
            '  "sub_questions": ["sub-question 1", "sub-question 2"],\n'
            '  "requires_external_research": true | false,\n'
            '  "temporal_scope": "current" | "as_of_date" | "historical" | null\n'
            "}\n"
            "Set requires_external_research=true only when answering needs external\n"
            "case law / statutes beyond the matter's own documents."
        ),
        f"Query: {query}",
    )
    if result:
        sub_questions = result.get("sub_questions") or [query]
        # Defensive: ensure a non-empty list of strings.
        if not isinstance(sub_questions, list) or not sub_questions:
            sub_questions = [query]
        sub_questions = [str(s).strip() for s in sub_questions if str(s).strip()]
        if not sub_questions:
            sub_questions = [query]
        return {
            "jurisdiction": result.get("jurisdiction"),
            "entities": result.get("entities", []),
            "refined_query": result.get("refined_query", query),
            "sub_questions": sub_questions,
            "requires_external_research": bool(result.get("requires_external_research", False)),
            "temporal_scope": result.get("temporal_scope"),
        }
    # Fallback: pass through unchanged.
    return {
        "jurisdiction": None,
        "entities": [],
        "refined_query": query,
        "sub_questions": [query],
        "requires_external_research": False,
        "temporal_scope": None,
    }


def plan_node(state: AgenticState) -> dict:
    """Derive a research plan from the clarification output.

    Deterministic for <=1 sub-question (no LLM call). For genuine multi-hop
    queries, ask the fast LLM for an ordered strategy, falling back to the
    deterministic plan if the LLM is unavailable.
    """
    sub_questions = state.get("sub_questions") or [state["query"]]
    requires_external = bool(state.get("requires_external_research", False))
    max_cap = int(_setting_num("agentic_max_iterations", 4))
    max_iterations = min(len(sub_questions) + 1, max_cap)

    if len(sub_questions) <= 1:
        complexity = "single_hop"
        strategy_hint = "direct_retrieval"
        plan: Dict[str, Any] = {
            "estimated_complexity": complexity,
            "max_iterations": max_iterations,
            "strategy_hint": strategy_hint,
            "sub_questions": sub_questions,
            "requires_external_research": requires_external,
            "ordered_steps": [{"sub_question": sub_questions[0], "strategy": strategy_hint}],
            "source": "deterministic",
        }
        logger.info(f"Plan: deterministic single-hop, max_iterations={max_iterations}")
        return {"research_plan": plan}

    # Multi-hop: ask the fast LLM for an ordered plan.
    result = _fast_llm_json(
        (
            "You are a legal research planner. Given sub-questions, produce an ordered\n"
            "research plan. Respond with JSON: {\n"
            '  "estimated_complexity": "multi_hop",\n'
            '  "strategy_hint": "decompose" | "sequential" | "parallel",\n'
            '  "ordered_steps": [{"sub_question": "...", "strategy": "targeted"}]\n'
            "}"
        ),
        "Sub-questions:\n" + "\n".join(f"- {s}" for s in sub_questions),
    )
    if result and isinstance(result.get("ordered_steps"), list) and result["ordered_steps"]:
        plan = {
            "estimated_complexity": result.get("estimated_complexity", "multi_hop"),
            "max_iterations": max_iterations,
            "strategy_hint": result.get("strategy_hint", "decompose"),
            "sub_questions": sub_questions,
            "requires_external_research": requires_external,
            "ordered_steps": result["ordered_steps"],
            "source": "llm",
        }
        logger.info(
            f"Plan: LLM multi-hop, steps={len(plan['ordered_steps'])}, "
            f"max_iterations={max_iterations}"
        )
        return {"research_plan": plan}

    # LLM unavailable / malformed -> deterministic multi-hop fallback.
    plan = {
        "estimated_complexity": "multi_hop",
        "max_iterations": max_iterations,
        "strategy_hint": "decompose",
        "sub_questions": sub_questions,
        "requires_external_research": requires_external,
        "ordered_steps": [{"sub_question": s, "strategy": "targeted"} for s in sub_questions],
        "source": "deterministic_fallback",
    }
    logger.info(f"Plan: deterministic multi-hop fallback, max_iterations={max_iterations}")
    return {"research_plan": plan}


async def retrieve_node(state: AgenticState) -> dict:
    """Execute hybrid retrieval using existing vector_store + BM25 pipeline.

    Async node: embed_query (Cohere HTTP), retrieve_chunks / search_sparse_vectors
    (Qdrant HTTP) and rerank_chunks (CPU CrossEncoder) are all blocking, so they
    run in a thread to keep the asyncio event loop responsive under concurrency.
    """
    matter_id = state["matter_id"]
    search_query = state.get("refined_query") or state["query"]
    rewrite_count = state.get("rewrite_count", 0)

    try:
        query_embedding = await asyncio.to_thread(embed_query, search_query)
        chunks = await asyncio.to_thread(retrieve_chunks, matter_id, query_embedding, RETRIEVAL_LIMIT)

        # Hybrid search (BM25 + RRF fusion) if enabled
        if getattr(settings, "hybrid_search_enabled", False):
            try:
                sparse_query = generate_sparse_vector(search_query)
                if sparse_query:
                    bm25_weight, dense_weight = get_rrf_weights(search_query)
                    bm25_results = await asyncio.to_thread(search_sparse_vectors, matter_id, sparse_query, RETRIEVAL_LIMIT)
                    if bm25_results:
                        chunks = reciprocal_rank_fusion(
                            bm25_results=bm25_results,
                            dense_results=chunks,
                            bm25_weight=bm25_weight,
                            dense_weight=dense_weight,
                            top_n=RETRIEVAL_LIMIT,
                        )
            except Exception as e:
                logger.warning(f"Hybrid search in agentic retrieve failed (non-blocking): {e}")

        # Filter + rerank. When a prior strategy relaxed filters, skip the
        # confidence cutoff (mirrors rag_engine's filter-relax fallback).
        if state.get("filters_relaxed", False):
            candidates = sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        else:
            high_conf = [c for c in chunks if c.get("score", 0) >= MIN_CONFIDENCE_SCORE]
            candidates = high_conf if high_conf else sorted(chunks, key=lambda x: x.get("score", 0), reverse=True)
        candidates = candidates[:15]  # top 15 for reranker

        try:
            reranked = await asyncio.to_thread(rerank_chunks, search_query, candidates, FINAL_CHUNK_COUNT)
        except Exception as e:
            logger.warning(f"Reranker failed in retrieve_node, falling back to top-k: {e}")
            reranked = candidates[:FINAL_CHUNK_COUNT]

        # Compute average retrieval score
        scores = [c.get("score", 0) for c in reranked]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        logger.info(f"Retrieve: {len(reranked)} chunks, avg_score={avg_score:.3f}, rewrite={rewrite_count}")
        return {"chunks": reranked, "retrieval_score": avg_score, "rewrite_count": rewrite_count}

    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {"chunks": [], "retrieval_score": 0.0, "rewrite_count": rewrite_count,
                "error": "We couldn't search the documents for that question. Please try again."}


async def external_research_node(state: AgenticState) -> dict:
    """Gated CourtListener research step. Merges external case-law chunks into
    the retrieved set, deduped by chunk_id. Non-blocking — any failure leaves the
    existing chunks untouched.

    Runs only when external research is enabled AND the clarifier (or a later
    `external` strategy via force_external) requested it.
    """
    if not getattr(settings, "agentic_external_research_enabled", False):
        return {}

    wants_external = (
        state.get("requires_external_research", False)
        or state.get("include_legal_research", False)
        or state.get("force_external", False)
    )
    if not wants_external:
        return {}

    search_query = state.get("refined_query") or state["query"]
    existing = list(state.get("chunks", []))  # copy — never mutate input
    existing_ids = {c.get("chunk_id") for c in existing if c.get("chunk_id")}

    try:
        cases = await search_cases(search_query, max_results=5)
        case_chunks = format_as_context(cases)
    except Exception as e:
        logger.warning(f"External research failed (non-blocking): {e}")
        return {"external_done": True}

    merged = list(existing)
    added = 0
    for ch in case_chunks:
        cid = ch.get("chunk_id")
        if cid and cid in existing_ids:
            continue
        if cid:
            existing_ids.add(cid)
        merged.append(ch)
        added += 1

    logger.info(f"External research: +{added} case-law chunks (query='{search_query[:60]}')")
    return {"chunks": merged, "external_done": True, "force_external": False}


def _tokens(text: str) -> set:
    """Lowercase content tokens (>=3 chars, non-stopword) for overlap heuristics."""
    import re
    raw = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {t for t in raw if len(t) >= 3 and t not in _STOPWORDS}


def _compute_sufficiency_signals(state: AgenticState) -> Dict:
    """Compute retrieval-sufficiency signals from current chunks + clarification.

    Pure function over state (no I/O). Returns a dict of signals used by the
    strategy chooser. Defensive against missing/partial chunk metadata.
    """
    chunks = state.get("chunks", []) or []
    sub_questions = state.get("sub_questions") or [state.get("query", "")]
    state_jurisdiction = state.get("jurisdiction")

    scores = [c.get("score", 0) or 0 for c in chunks]
    avg_relevance = sum(scores) / len(scores) if scores else 0.0
    chunk_count = len(chunks)

    # Coverage: a sub-question is "covered" if >=3 of its content tokens appear
    # across the union of chunk-content tokens.
    chunk_token_union: set = set()
    for c in chunks:
        chunk_token_union |= _tokens(c.get("content", ""))

    covered, uncovered = [], []
    for sq in sub_questions:
        sq_tokens = _tokens(sq)
        overlap = len(sq_tokens & chunk_token_union)
        if overlap >= 3 or (sq_tokens and sq_tokens <= chunk_token_union):
            covered.append(sq)
        else:
            uncovered.append(sq)
    coverage = len(covered) / len(sub_questions) if sub_questions else 0.0

    # Jurisdiction match: unknown chunk jurisdiction counts as a match (no penalty).
    jurisdiction_match = True
    if state_jurisdiction:
        mismatches = 0
        checked = 0
        for c in chunks:
            meta = c.get("authority_metadata") or {}
            cj = meta.get("jurisdiction")
            if cj:
                checked += 1
                if str(cj).upper() != str(state_jurisdiction).upper():
                    mismatches += 1
        # Only flag a mismatch if we actually saw jurisdiction metadata and the
        # majority disagrees.
        if checked and mismatches > checked / 2:
            jurisdiction_match = False

    # Binding authority: any chunk with an authority tier, or a good-law case.
    has_binding_authority = False
    for c in chunks:
        meta = c.get("authority_metadata") or {}
        tier = meta.get("tier") or meta.get("authority_tier")
        if tier:
            has_binding_authority = True
            break
        if c.get("source_type") == "case_law" and c.get("is_good_law") is True:
            has_binding_authority = True
            break

    return {
        "avg_relevance": round(avg_relevance, 4),
        "coverage": round(coverage, 4),
        "sub_questions_covered": covered,
        "sub_questions_uncovered": uncovered,
        "jurisdiction_match": jurisdiction_match,
        "has_binding_authority": has_binding_authority,
        "chunk_count": chunk_count,
    }


def _select_strategy(state: AgenticState, signals: Dict) -> str:
    """Choose the next strategy from sufficiency signals. Pure function.

    Returns one of: "sufficient", "reformulate", "targeted", "filter_relax",
    "external", "expand".
    """
    rewrite_count = state.get("rewrite_count", 0)
    avg = signals["avg_relevance"]
    coverage = signals["coverage"]
    chunk_count = signals["chunk_count"]

    sufficiency_high = _setting_num("agentic_sufficiency_high", SUFFICIENCY_HIGH)
    coverage_high = _setting_num("agentic_coverage_high", COVERAGE_HIGH)
    relevance_low = _setting_num("agentic_relevance_low", RELEVANCE_LOW)
    coverage_low = _setting_num("agentic_coverage_low", COVERAGE_LOW)

    # 1. Clearly sufficient.
    if avg >= sufficiency_high and coverage >= coverage_high:
        return "sufficient"

    # 2. Hard cap reached -> stop looping.
    if rewrite_count >= MAX_REWRITE_ITERATIONS:
        return "sufficient"

    # 3. Too few chunks (over-filtered) -> relax filters once.
    if chunk_count < MIN_FILTERED_CHUNKS and not state.get("filters_relaxed", False):
        return "filter_relax"

    # 4. Relevance too low -> reformulate the query.
    if avg < relevance_low:
        return "reformulate"

    # 5. Coverage too low -> targeted search of uncovered sub-questions.
    if coverage < coverage_low and signals["sub_questions_uncovered"]:
        return "targeted"

    # 6. No binding authority and external research is allowed -> pull case law.
    external_allowed = bool(getattr(settings, "agentic_external_research_enabled", False))
    if (
        not signals["has_binding_authority"]
        and external_allowed
        and not state.get("external_done", False)
    ):
        return "external"

    # 7. Marginal — broaden the search.
    return "expand"


def evaluate_node(state: AgenticState) -> dict:
    """Compute sufficiency signals and select the next strategy.

    Replaces the previous shallow top-1 grade. Keeps the rewrite loop bounded by
    MAX_REWRITE_ITERATIONS (enforced in _select_strategy and the router).
    """
    chunks = state.get("chunks", [])
    strategy_log = list(state.get("strategy_log", []))  # copy — immutable update

    if not chunks:
        # Nothing retrieved — reformulate if we still have budget, else give up.
        rewrite_count = state.get("rewrite_count", 0)
        strategy = "reformulate" if rewrite_count < MAX_REWRITE_ITERATIONS else "sufficient"
        strategy_log.append(strategy)
        logger.info(f"Evaluate: no chunks -> strategy={strategy}")
        return {
            "retrieval_score": 0.0,
            "sufficiency_signals": {"avg_relevance": 0.0, "coverage": 0.0, "chunk_count": 0},
            "next_strategy": strategy,
            "strategy_log": strategy_log,
        }

    signals = _compute_sufficiency_signals(state)
    strategy = _select_strategy(state, signals)
    strategy_log.append(strategy)

    logger.info(
        f"Evaluate: avg={signals['avg_relevance']:.3f} coverage={signals['coverage']:.2f} "
        f"binding={signals['has_binding_authority']} -> strategy={strategy}"
    )
    return {
        "retrieval_score": signals["avg_relevance"],
        "sufficiency_signals": signals,
        "next_strategy": strategy,
        "strategy_log": strategy_log,
    }


def strategy_node(state: AgenticState) -> dict:
    """Apply the selected non-terminal strategy before the next retrieval pass.

    Increments rewrite_count (loop counter) and adjusts state so the next
    retrieve pass behaves differently. Terminal strategies ("sufficient") never
    reach this node — routing sends them straight to generation.
    """
    strategy = state.get("next_strategy", "expand")
    rewrite_count = state.get("rewrite_count", 0) + 1
    query = state["query"]
    updates: Dict[str, Any] = {"rewrite_count": rewrite_count}

    if strategy == "reformulate":
        result = _fast_llm_json(
            (
                "You rewrite legal search queries to improve document retrieval.\n"
                "Make the query more specific, add relevant legal terms, expand abbreviations.\n"
                'Respond with JSON: {"rewritten_query": "improved query"}'
            ),
            f"Original query: {query}\nThis is rewrite attempt {rewrite_count}.",
        )
        rewritten = result.get("rewritten_query", query) if result else query
        updates["refined_query"] = rewritten
        logger.info(f"Strategy reformulate ({rewrite_count}): '{query}' -> '{rewritten}'")

    elif strategy == "targeted":
        uncovered = (state.get("sufficiency_signals") or {}).get("sub_questions_uncovered", [])
        if uncovered:
            updates["refined_query"] = " ".join(uncovered[:2])
            logger.info(f"Strategy targeted: searching uncovered -> '{updates['refined_query']}'")
        else:
            updates["refined_query"] = state.get("refined_query") or query

    elif strategy == "filter_relax":
        updates["filters_relaxed"] = True
        logger.info("Strategy filter_relax: dropping confidence filters on next pass")

    elif strategy == "external":
        updates["force_external"] = True
        logger.info("Strategy external: forcing CourtListener research on next pass")

    else:  # expand
        result = _fast_llm_json(
            (
                "You broaden a legal search query to surface more relevant documents.\n"
                'Respond with JSON: {"rewritten_query": "broader query"}'
            ),
            f"Original query: {query}",
        )
        broadened = result.get("rewritten_query", query) if result else query
        updates["refined_query"] = broadened
        logger.info(f"Strategy expand ({rewrite_count}): '{query}' -> '{broadened}'")

    return updates


async def generate_node(state: AgenticState) -> dict:
    """Generate answer using existing Gemini pipeline."""
    chunks = state.get("chunks", [])
    query = state["query"]
    db = state["db"]
    temperature = state.get("temperature", 0.2)
    conversation_history = state.get("conversation_history")
    include_legal_research = state.get("include_legal_research", False)

    matter_id = state["matter_id"]
    if not chunks:
        return {"answer": "", "sources": [], "tokens_used": 0, "error": "No chunks available for generation"}

    from uuid import UUID
    # CANONICAL CITATION ORDERING — [n] must resolve to the same source in the
    # context, extraction, grounding AND verification (mirror query_matter). We
    # also persist the sorted list back to state so verify_node indexes it identically.
    chunks = sorted(chunks, key=lambda c: c.get("score", 0), reverse=True)

    try:
        matter = db.query(Matter).filter(Matter.id == UUID(matter_id)).first()
        matter_name = matter.name if matter else "Unknown Matter"
    except Exception:
        matter_name = "Unknown Matter"

    # Document summaries (parity with query_matter)
    doc_summaries = {}
    try:
        doc_ids = set()
        for c in chunks:
            if c.get("document_id"):
                try:
                    doc_ids.add(UUID(c["document_id"]))
                except (ValueError, AttributeError):
                    pass
        if doc_ids:
            docs = db.query(Document).filter(Document.id.in_(doc_ids)).all()
            doc_summaries = {str(d.id): {"name": d.name, "summary": d.summary} for d in docs if d.summary}
    except Exception as e:
        logger.debug(f"Agentic doc-summary fetch skipped: {e}")

    formatted_context = format_legal_context(chunks, matter_name, doc_summaries)

    # Conflict detection — compute here so we can augment the prompt; reused by verify_node.
    conflict_analysis = None
    if getattr(settings, "conflict_detection_enabled", True):
        try:
            conflict_analysis = await asyncio.to_thread(detect_conflicts, chunks, query=query)
        except Exception as e:
            logger.warning(f"Agentic conflict detection failed (non-blocking): {e}")
    if conflict_analysis:
        try:
            try:
                from backend.services.rag_engine import augment_context_with_conflicts
            except ImportError:
                from services.rag_engine import augment_context_with_conflicts
            formatted_context = augment_context_with_conflicts(formatted_context, conflict_analysis)
        except Exception as e:
            logger.debug(f"Conflict augmentation skipped: {e}")

    # Citation-graph good/bad-law injection (parity with query_matter)
    if getattr(settings, "citation_graph_enabled", True) and getattr(
        settings, "citation_graph_rag_injection_enabled", True
    ):
        try:
            try:
                from backend.services.rag_engine import _build_graph_context
                from backend.services.citation_extractor import extract_all_citations
            except ImportError:
                from services.rag_engine import _build_graph_context
                from services.citation_extractor import extract_all_citations
            top_text = " ".join((c.get("content") or "")[:500] for c in chunks[:3])
            found = await extract_all_citations(f"{query}\n{top_text}", use_llm=False)
            graph_block = await asyncio.to_thread(_build_graph_context, db, matter_id, found)
            if graph_block:
                # PREPEND (matching query_matter) so the citation-validation block
                # leads the context and survives any later token-budget trim.
                formatted_context = f"{graph_block}\n\n{formatted_context}"
        except Exception as e:
            logger.debug(f"Agentic graph-context injection skipped: {e}")

    # Issue analysis (parity)
    issue_analysis = None
    if getattr(settings, "problem_formulation_enabled", True):
        try:
            try:
                from backend.services.problem_formulation import identify_issues
            except ImportError:
                from services.problem_formulation import identify_issues
            issue_analysis = await identify_issues(query)
        except Exception as e:
            logger.debug(f"Agentic issue analysis skipped: {e}")

    answer, tokens_used = await generate_answer(
        query, formatted_context, temperature, conversation_history, include_legal_research,
    )
    cleaned_answer, citations, _ = extract_citations(answer, chunks)
    grounded_citations, _, _ = ground_citations_in_source(citations, chunks, query)

    # Rich sources (full parity with query_matter's source dict)
    sources = []
    for chunk in chunks:
        sources.append({
            "chunk_id": chunk.get("chunk_id", ""),
            "page_num": chunk.get("page_num", ""),
            "section_name": chunk.get("section_name", ""),
            "relevance_score": chunk.get("score", 0),
            "content": chunk.get("content", ""),
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
        })

    return {
        "answer": cleaned_answer,
        "sources": sources,
        "citations": grounded_citations,
        "tokens_used": tokens_used,
        "conflict_analysis": conflict_analysis,  # reused by verify_node
        "issue_analysis": issue_analysis,
        "chunks": chunks,  # canonical-sorted; verify_node indexes the SAME order
    }


async def verify_node(state: AgenticState) -> dict:
    """Run verification pipeline: citations + claims + conflicts (all non-blocking)."""
    answer = state.get("answer", "")
    chunks = state.get("chunks", [])
    query = state.get("query", "")

    citation_verification = None
    claim_verification_result = None
    conflict_analysis = None

    if not answer or not chunks:
        return {}

    # Citation verification
    if getattr(settings, "citation_verification_enabled", True):
        try:
            citation_verification = await verify_response_citations(answer, chunks)
        except Exception as e:
            logger.warning(f"Agentic citation verification failed (non-blocking): {e}")

    # Claim verification
    if getattr(settings, "claim_verification_enabled", True):
        try:
            claim_verification_result = await verify_claims(answer, chunks)
        except Exception as e:
            logger.warning(f"Agentic claim verification failed (non-blocking): {e}")

    # Conflict detection — reuse the analysis already computed in generate_node
    # (it augmented the prompt there); only compute if missing.
    conflict_analysis = state.get("conflict_analysis")
    if conflict_analysis is None and getattr(settings, "conflict_detection_enabled", True):
        try:
            conflict_analysis = await asyncio.to_thread(detect_conflicts, chunks, query=query)
        except Exception as e:
            logger.warning(f"Agentic conflict detection failed (non-blocking): {e}")

    return {
        "citation_verification": citation_verification,
        "claim_verification": claim_verification_result,
        "conflict_analysis": conflict_analysis,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Routing functions (conditional edges)
# ═══════════════════════════════════════════════════════════════════════════

def route_by_complexity(state: AgenticState) -> str:
    """After router: simple -> fast_path, complex -> clarify."""
    return "fast_path" if state.get("complexity") == "simple" else "clarify"


def route_by_strategy(state: AgenticState) -> str:
    """After evaluate: terminal strategy -> generate, else -> strategy node.

    The MAX_REWRITE_ITERATIONS cap is enforced in _select_strategy (which returns
    "sufficient" once the cap is hit), so this router only translates the chosen
    strategy into a graph edge.
    """
    strategy = state.get("next_strategy", "sufficient")
    if strategy == "sufficient":
        return "generate"
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count >= MAX_REWRITE_ITERATIONS:
        logger.warning(f"Max rewrites ({MAX_REWRITE_ITERATIONS}) reached, generating with best available")
        return "generate"
    return "strategy"


# ═══════════════════════════════════════════════════════════════════════════
# Graph construction
# ═══════════════════════════════════════════════════════════════════════════

def _build_graph() -> Optional[Any]:
    """Build and compile the LangGraph StateGraph. Returns None if unavailable."""
    if not _LANGGRAPH_AVAILABLE:
        return None

    planning_enabled = getattr(settings, "agentic_planning_enabled", True)

    graph = StateGraph(AgenticState)

    # Nodes
    graph.add_node("context_manager", context_manager_node)
    graph.add_node("router", router_node)
    graph.add_node("clarify", clarify_node)
    if planning_enabled:
        graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    # external_research self-gates on agentic_external_research_enabled; keeping
    # it in the graph unconditionally lets the "external" strategy re-trigger it.
    graph.add_node("external_research", external_research_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_node("strategy", strategy_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    # Edges: START -> context_manager -> router
    # Context management runs FIRST so conversation history is summarised/bounded
    # before any routing, retrieval, generation or the fast-path delegation.
    graph.add_edge(START, "context_manager")
    graph.add_edge("context_manager", "router")

    # router -> fast_path (END, handled externally) or clarify
    graph.add_conditional_edges(
        "router",
        route_by_complexity,
        {"fast_path": END, "clarify": "clarify"},
    )

    # clarify -> plan -> retrieve  (or clarify -> retrieve when planning disabled)
    if planning_enabled:
        graph.add_edge("clarify", "plan")
        graph.add_edge("plan", "retrieve")
    else:
        graph.add_edge("clarify", "retrieve")

    # retrieve -> external_research -> evaluate
    graph.add_edge("retrieve", "external_research")
    graph.add_edge("external_research", "evaluate")

    # evaluate -> generate (terminal) or strategy (loop)
    graph.add_conditional_edges(
        "evaluate",
        route_by_strategy,
        {"generate": "generate", "strategy": "strategy"},
    )

    # strategy -> retrieve (loop)
    graph.add_edge("strategy", "retrieve")

    # generate -> verify
    graph.add_edge("generate", "verify")

    # verify -> END
    graph.add_edge("verify", END)

    return graph.compile()


# Singleton compiled graph
_COMPILED_GRAPH = None
_GRAPH_LOCK = threading.Lock()


def _get_graph():
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        with _GRAPH_LOCK:
            if _COMPILED_GRAPH is None:
                _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ═══════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════

async def agentic_query_matter(
    matter_id: str,
    query: str,
    db: Any,
    top_k: int = FINAL_CHUNK_COUNT,
    temperature: float = 0.2,
    conversation_history: Optional[list] = None,
    include_legal_research: bool = False,
) -> Dict:
    """
    Agentic RAG entry point. Same signature and return shape as query_matter().

    Routes simple queries to the existing single-pass pipeline. Complex queries
    go through the LangGraph orchestrator with clarify/retrieve/evaluate/rewrite
    cycles, then generation and verification.

    Falls back to query_matter() if agentic pipeline is unavailable or fails.
    """
    # Gate check
    if not getattr(settings, "agentic_rag_enabled", False):
        logger.debug("Agentic RAG disabled, using single-pass pipeline")
        return await query_matter(
            matter_id, query, db, top_k, temperature,
            conversation_history, include_legal_research,
        )

    graph = _get_graph()
    if graph is None:
        logger.warning("LangGraph unavailable, falling back to single-pass pipeline")
        return await query_matter(
            matter_id, query, db, top_k, temperature,
            conversation_history, include_legal_research,
        )

    try:
        # Run the graph (router decides fast vs slow path)
        initial_state: AgenticState = {
            "matter_id": matter_id,
            "query": query,
            "db": db,
            "conversation_history": conversation_history,
            "include_legal_research": include_legal_research,
            "temperature": temperature,
            "rewrite_count": 0,
            "chunks": [],
            "retrieval_score": 0.0,
            "sources": [],
            "entities": [],
            "sub_questions": [],
            "requires_external_research": False,
            "filters_relaxed": False,
            "force_external": False,
            "external_done": False,
            "strategy_log": [],
        }

        # Hard backstop against infinite loops: bound the number of graph steps.
        # Each rewrite iteration touches ~4 nodes; give generous headroom over
        # MAX_REWRITE_ITERATIONS while still preventing runaway recursion.
        recursion_limit = (MAX_REWRITE_ITERATIONS + 1) * 6 + 8
        final_state = await graph.ainvoke(
            initial_state, config={"recursion_limit": recursion_limit}
        )

        # If router sent to fast_path (END), the graph exited early.
        # Delegate to existing single-pass pipeline.
        if final_state.get("complexity") == "simple" and not final_state.get("answer"):
            logger.info("Fast path: delegating to single-pass query_matter()")
            # Use the context-managed history (context_manager_node ran first, even
            # on the fast path) so simple queries also benefit from summarisation.
            managed_history = final_state.get("conversation_history", conversation_history)
            return await query_matter(
                matter_id, query, db, top_k, temperature,
                managed_history, include_legal_research,
            )

        # Build response in the same shape as query_matter()
        return _build_response(final_state, matter_id, query)

    except Exception as e:
        logger.error(f"Agentic RAG failed, falling back to single-pass: {e}", exc_info=True)
        return await query_matter(
            matter_id, query, db, top_k, temperature,
            conversation_history, include_legal_research,
        )


def _build_response(state: AgenticState, matter_id: str, query: str) -> Dict:
    """Convert AgenticState into the standard query_matter() response dict."""

    answer = state.get("answer", "")
    sources = state.get("sources", [])
    grounded_citations = state.get("citations", [])

    return {
        "answer": answer or None,
        "sources": sources,
        "citations": grounded_citations,  # grounded citations (parity with query_matter)
        "matter_id": matter_id,
        "query": query,
        "model": GEMINI_MODEL,
        "tokens_used": state.get("tokens_used", 0),
        "confidence": {
            "level": classify_confidence_level(state.get("retrieval_score", 0.0)) if answer else "none",
            "score": state.get("retrieval_score", 0.0),
            "factors": {
                "has_hallucinations": False,
                "unsupported_claims": 0,
                "grounded_citations": len(grounded_citations),
                "avg_citation_relevance": state.get("retrieval_score", 0.0),
            },
        },
        "confidence_explanation": None,
        "source_document": None,
        "citation_verification": state.get("citation_verification"),
        "claim_verification": state.get("claim_verification"),
        "conflict_analysis": state.get("conflict_analysis"),
        "issue_analysis": state.get("issue_analysis"),
        "agentic_metadata": {
            "complexity": state.get("complexity"),
            "query_type": state.get("query_type"),
            "jurisdiction": state.get("jurisdiction"),
            "entities": state.get("entities", []),
            "rewrite_count": state.get("rewrite_count", 0),
            "retrieval_score": state.get("retrieval_score", 0.0),
            "sub_questions": state.get("sub_questions", []),
            "requires_external_research": state.get("requires_external_research"),
            "temporal_scope": state.get("temporal_scope"),
            "research_plan": state.get("research_plan"),
            "strategy_log": state.get("strategy_log", []),
            "sufficiency_signals": state.get("sufficiency_signals"),
        },
        "error": state.get("error"),
    }
