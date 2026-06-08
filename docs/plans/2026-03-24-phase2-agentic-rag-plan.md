---
title: "Phase 2: Agentic RAG Layer — Multi-Agent Legal Research Pipeline"
type: feat
status: planned
date: 2026-03-24
priority: P0
estimated_effort: 6 weeks
depends_on: phase1-domain-embeddings (optional, enhances results but not blocking)
---

# Phase 2: Agentic RAG Layer

## 1. Problem Statement

### Why Single-Pass RAG Fails Lawyers

Lexintel's current pipeline (`query_matter()` in `backend/services/rag_engine.py`) executes
a single linear pass: embed query, retrieve top-k chunks, rerank, format context, generate
answer. This works for simple lookup queries but systematically fails at what lawyers
actually need.

**The gap between "retrieve and answer" and "legal research" is enormous:**

| Capability | Single-Pass RAG (Current) | What Lawyers Need |
|-----------|---------------------------|-------------------|
| Query understanding | Literal embedding match | Jurisdiction extraction, entity recognition, query decomposition |
| Retrieval strategy | One-shot top-k | Iterative: search, evaluate, refine, search again |
| Multi-hop reasoning | None | "Find the statute, then find cases interpreting it, then check if those cases still hold" |
| Self-correction | None | Detect when retrieval is insufficient and try different search strategies |
| Answer structure | Flat markdown | CREAC with hierarchical citations, confidence per section |
| Verification | Post-hoc (bolted on) | Integrated into the reasoning loop, not an afterthought |

**Concrete failures we see today:**

1. **Ambiguous queries get literal answers.** "What are the liability implications?" retrieves
   chunks about "liability" but does not ask: which jurisdiction? Statutory or tort? What type
   of entity? The current system cannot clarify.

2. **Complex questions get shallow answers.** "Compare the treatment of force majeure in our
   contracts against recent case law" requires: (a) finding force majeure clauses in uploaded
   documents, (b) retrieving relevant case law, (c) analyzing differences, (d) synthesizing.
   Single-pass RAG does step (a) at best.

3. **Insufficient retrieval is invisible.** When the top-8 chunks do not contain the answer,
   the LLM hallucinates or gives a generic response. There is no mechanism to detect this
   and try a different retrieval strategy.

4. **Verification is disconnected.** Citation verification, claim verification, and conflict
   detection run as post-processing steps. They cannot feed back into the answer to fix
   problems they find.

**The solution: an agentic layer that reasons about retrieval strategy, self-corrects,
and integrates verification into the generation loop.**

### Industry Context

Thomson Reuters CoCounsel uses a multi-agent "Deep Research" architecture that decomposes
complex queries into sub-tasks, executes them iteratively, and synthesizes results. Harvey
AI uses iterative retrieval with self-reflection. Both report 30-50% improvement in answer
quality on complex queries versus single-pass RAG.

---

## 2. Architecture

### 5-Agent System with LangGraph

```
                                   User Query
                                       |
                                       v
                            +---------------------+
                            |    ROUTER (fast)     |
                            | gemini-2.5-flash-lite|
                            +---------------------+
                               /              \
                    simple (40%)            complex (60%)
                             /                    \
                            v                      v
                  +------------------+   +---------------------+
                  | DIRECT RAG PATH  |   | 1. CLARIFICATION    |
                  | (existing pipeline|   |    AGENT            |
                  |  < 3 seconds)    |   | gemini-2.5-flash-lite|
                  +------------------+   +---------------------+
                            |                      |
                            v                      v
                       Return answer      +---------------------+
                                          | 2. PLANNING AGENT   |
                                          | gemini-2.5-flash-lite|
                                          +---------------------+
                                                   |
                                                   v
                                          +---------------------+
                                          | 3. EXECUTION AGENT  |<----+
                                          | (iterative retrieval)|    |
                                          | gemini-2.5-flash     |    |
                                          +---------------------+    |
                                              |           |          |
                                    retrieval |    self-   |  loop   |
                                    tools     |  reflection| (max 5) |
                                              v           +----------+
                                          +---------------------+
                                          | 4. VERIFICATION     |
                                          |    AGENT            |
                                          | (reuse existing)    |
                                          +---------------------+
                                                   |
                                                   v
                                          +---------------------+
                                          | 5. SYNTHESIS AGENT  |
                                          | gemini-2.5-flash    |
                                          +---------------------+
                                                   |
                                                   v
                                            Final Response
                                       (CREAC + citations +
                                        verification badges)
```

### LangGraph State Machine

```
START --> router
router --[simple]--> direct_rag --> END
router --[complex]--> clarify --> plan --> execute
execute --> reflect
reflect --[sufficient]--> verify --> synthesize --> END
reflect --[insufficient, iterations < 5]--> execute
reflect --[max_iterations]--> verify --> synthesize --> END
```

---

## 3. Agent Specifications

### 3.1 Router (Fast Path / Slow Path Decision)

| Attribute | Value |
|-----------|-------|
| **Role** | Classify query complexity to route simple queries to fast path |
| **LLM** | `gemini-2.5-flash-lite` (cheapest, fastest) |
| **Latency** | < 300ms |
| **Input** | `AgenticState.query`, `AgenticState.conversation_history` |
| **Output** | `AgenticState.route` = `"simple"` or `"complex"` |
| **Tools** | None |

**Classification heuristics (before LLM call for zero-cost routing):**

```python
def classify_query_complexity(query: str, conversation_history: list) -> str:
    """Rule-based pre-classifier. Falls through to LLM only for ambiguous cases."""
    query_lower = query.lower()

    # Simple: single-entity lookup, definition, yes/no
    simple_patterns = [
        r"^what (?:is|are|does) ",           # "What is force majeure?"
        r"^(?:define|explain)\b",             # "Define consideration"
        r"^(?:does|is|can|will|should)\b.*\?$",  # Yes/no questions
        r"^(?:when|where) (?:is|was|did)\b",  # Simple factual
    ]
    if any(re.match(p, query_lower) for p in simple_patterns):
        if len(query.split()) < 15 and not conversation_history:
            return "simple"

    # Complex: multi-part, comparative, analytical
    complex_signals = [
        "compare", "contrast", "analyze", "implications",
        "what are the risks", "how does.*relate to",
        "in light of", "considering", "given that",
        "step by step", "all the", "comprehensive",
    ]
    if any(signal in query_lower for signal in complex_signals):
        return "complex"

    # Follow-up questions in conversation are typically complex
    if conversation_history and len(conversation_history) >= 2:
        return "complex"

    return "ambiguous"  # LLM decides
```

### 3.2 Clarification Agent

| Attribute | Value |
|-----------|-------|
| **Role** | Extract structured metadata from query: jurisdiction, entities, query type, temporal scope |
| **LLM** | `gemini-2.5-flash-lite` |
| **Latency** | ~500ms |
| **Input** | `AgenticState.query`, `AgenticState.conversation_history`, `AgenticState.matter_metadata` |
| **Output** | `AgenticState.clarification` (structured dict) |
| **Tools** | None (pure LLM extraction) |

**Output schema:**

```python
class Clarification(TypedDict):
    jurisdiction: str             # "US.federal", "US.state.CA", "UK", "unknown"
    query_type: str               # "legal_analysis", "factual_lookup", "comparison",
                                  # "risk_assessment", "drafting_support"
    entities: list[str]           # ["force majeure", "ABC Corp", "Section 230"]
    temporal_scope: str           # "current", "historical", "2020-2024"
    sub_questions: list[str]      # Decomposed sub-questions for multi-hop queries
    requires_external_research: bool  # Should we query CourtListener?
    ambiguities: list[str]        # Things the system is unsure about
```

### 3.3 Planning Agent

| Attribute | Value |
|-----------|-------|
| **Role** | Create a multi-step research plan based on clarification output |
| **LLM** | `gemini-2.5-flash-lite` |
| **Latency** | ~500ms |
| **Input** | `AgenticState.clarification`, `AgenticState.matter_metadata` |
| **Output** | `AgenticState.research_plan` (ordered list of retrieval steps) |
| **Tools** | None |

**Output schema:**

```python
class ResearchStep(TypedDict):
    step_id: int
    action: str          # "vector_search", "hybrid_search", "courtlistener_search",
                         # "filter_by_jurisdiction", "filter_by_date", "expand_query"
    query: str           # The specific search query for this step
    filters: dict        # Qdrant payload filters: {"jurisdiction": "US", "source_type": "statute"}
    depends_on: list[int]  # Step IDs this depends on
    rationale: str       # Why this step is needed

class ResearchPlan(TypedDict):
    steps: list[ResearchStep]
    estimated_complexity: str   # "low", "medium", "high"
    max_iterations: int         # 1-5, based on complexity
```

**Example plan for "Compare force majeure treatment in our contracts vs recent case law":**

```
Step 1: vector_search("force majeure clause", filters={"source_type": "contract"})
Step 2: vector_search("force majeure interpretation court", filters={"source_type": "case_law"})
Step 3: courtlistener_search("force majeure enforceability recent")
Step 4: vector_search("force majeure exception limitation", depends_on=[1])
```

### 3.4 Execution Agent (Iterative Retrieval)

| Attribute | Value |
|-----------|-------|
| **Role** | Execute research plan steps, retrieve chunks, evaluate sufficiency, self-correct |
| **LLM** | `gemini-2.5-flash` (needs reasoning for self-evaluation) |
| **Latency** | 2-8s per iteration (1-5 iterations) |
| **Input** | `AgenticState.research_plan`, `AgenticState.retrieved_chunks`, `AgenticState.iteration` |
| **Output** | Updated `AgenticState.retrieved_chunks`, `AgenticState.retrieval_metadata` |
| **Tools** | `vector_search`, `hybrid_search`, `sparse_search`, `courtlistener_search`, `expand_query` |

**Tools bound to this agent:**

```python
from langchain_core.tools import tool

@tool
def vector_search(query: str, matter_id: str, filters: dict = None, top_k: int = 15) -> list[dict]:
    """Search the matter's vector store for semantically similar chunks.

    Args:
        query: Search query text
        matter_id: UUID of the matter to search
        filters: Optional Qdrant payload filters (e.g. {"jurisdiction": "US"})
        top_k: Number of results to return
    """
    from backend.services.embeddings import embed_query
    from backend.services.vector_store import search_vectors
    embedding = embed_query(query)
    return search_vectors(matter_id, embedding, limit=top_k, query_filter=filters)


@tool
def hybrid_search(query: str, matter_id: str, top_k: int = 15) -> list[dict]:
    """Search using both semantic (dense) and keyword (BM25 sparse) retrieval with RRF fusion.

    Best for citation-heavy queries or when exact legal terms matter.
    """
    from backend.services.hybrid_search import (
        generate_sparse_vector, get_rrf_weights, reciprocal_rank_fusion
    )
    from backend.services.vector_store import search_vectors, search_sparse_vectors
    from backend.services.embeddings import embed_query

    embedding = embed_query(query)
    dense_results = search_vectors(matter_id, embedding, limit=top_k)
    sparse_query = generate_sparse_vector(query)
    if sparse_query:
        bm25_results = search_sparse_vectors(matter_id, sparse_query, limit=top_k)
        bm25_w, dense_w = get_rrf_weights(query)
        return reciprocal_rank_fusion(bm25_results, dense_results, bm25_w, dense_w, top_n=top_k)
    return dense_results


@tool
async def courtlistener_search(query: str, max_results: int = 5) -> list[dict]:
    """Search CourtListener for relevant case law. Use for external legal research."""
    from backend.services.legal_research import search_cases, format_as_context
    results = await search_cases(query, max_results=max_results)
    return format_as_context(results) if results else []


@tool
def expand_query(original_query: str, context_so_far: str) -> str:
    """Generate an expanded/reformulated query based on what has been found so far.

    Use when initial retrieval is insufficient and you need to try a different angle.
    """
    # LLM-based query expansion — handled by the execution agent's own reasoning
    return original_query  # Placeholder; actual expansion done in agent prompt
```

### 3.5 Verification Agent

| Attribute | Value |
|-----------|-------|
| **Role** | Verify citations, ground claims, detect conflicts. **Reuses existing verification layer.** |
| **LLM** | Inherits from existing services (Gemini for Tier 2 fallback) |
| **Latency** | 1-3s (mostly local NLI inference) |
| **Input** | `AgenticState.draft_answer`, `AgenticState.retrieved_chunks` |
| **Output** | `AgenticState.verification_result` |
| **Tools** | None (calls existing services directly) |

**Integration with existing services:**

```python
async def verification_node(state: AgenticState) -> dict:
    """Reuse all existing verification infrastructure."""
    from backend.services.citation_agent import verify_response_citations
    from backend.services.claim_verifier import verify_claims
    from backend.services.conflict_detector import detect_conflicts

    draft = state["draft_answer"]
    chunks = state["retrieved_chunks"]

    # Run all three verifications in parallel
    citation_result, claim_result, conflict_result = await asyncio.gather(
        verify_response_citations(draft, chunks),
        verify_claims(draft, chunks),
        asyncio.to_thread(detect_conflicts, chunks, state["query"]),
    )

    # Compute verification score
    issues = []
    if claim_result:
        summary = claim_result.get("summary", {})
        unsupported = summary.get("unsupported", 0)
        if unsupported > 0:
            issues.append(f"{unsupported} unsupported claims")
    if conflict_result and conflict_result.get("has_conflicts"):
        high = conflict_result["summary"]["high_severity"]
        if high > 0:
            issues.append(f"{high} high-severity conflicts")
    if citation_result:
        not_found = citation_result.get("summary", {}).get("not_found", 0)
        if not_found > 0:
            issues.append(f"{not_found} unverified citations")

    return {
        "citation_verification": citation_result,
        "claim_verification": claim_result,
        "conflict_analysis": conflict_result,
        "verification_issues": issues,
        "verification_passed": len(issues) == 0,
    }
```

### 3.6 Synthesis Agent

| Attribute | Value |
|-----------|-------|
| **Role** | Generate final CREAC-structured answer with traceable citations and verification badges |
| **LLM** | `gemini-2.5-flash` (strong model for final output quality) |
| **Latency** | 1-3s |
| **Input** | `AgenticState.retrieved_chunks`, `AgenticState.verification_result`, `AgenticState.clarification`, `AgenticState.research_plan` |
| **Output** | `AgenticState.final_answer`, `AgenticState.sources`, `AgenticState.confidence` |
| **Tools** | None |

The synthesis agent receives the full context of what was retrieved, what was verified, and
what conflicts exist. Its system prompt includes the existing CREAC instructions from
`LEGAL_SYSTEM_PROMPT` in `rag_engine.py`, enhanced with:

- Verification results injected into context (so the LLM knows which claims are supported)
- Conflict notices with recommended sources
- Authority hierarchy metadata per source
- Explicit instruction to mark unsupported claims with `[CITATION NEEDED - VERIFY]`

---

## 4. Fast Path vs Slow Path

### Query Routing Logic

```
                     User Query
                         |
                         v
              +---------------------+
              | Rule-based classifier|
              +---------------------+
               /       |          \
          simple    ambiguous    complex
             |         |            |
             |         v            |
             |   +----------+      |
             |   | LLM router|     |
             |   +----------+      |
             |    /        \       |
             |  simple   complex   |
             |   /           \     |
             v  v             v    v
        +-----------+    +----------------+
        | FAST PATH |    |   SLOW PATH    |
        | direct_rag|    | 5-agent pipeline|
        | < 3 sec   |    | 8-15 sec       |
        +-----------+    +----------------+
```

**Fast path criteria (any of these skip to direct RAG):**

1. Query is < 15 words AND matches simple pattern (definition, yes/no, single-entity lookup)
2. No conversation history (not a follow-up)
3. Matter has only 1 document (limited research scope)
4. User explicitly requests quick answer (future: UI toggle)

**Slow path triggers (any of these force full pipeline):**

1. Query contains comparative language ("compare", "contrast", "vs")
2. Query asks for risk assessment or implications
3. Query references multiple documents or jurisdictions
4. Conversation depth >= 3 turns (complex thread)
5. LLM classifier returns "complex" for ambiguous queries

**Performance targets:**

| Path | P50 Latency | P95 Latency | Cost |
|------|-------------|-------------|------|
| Fast path | 2.0s | 3.5s | ~$0.02 |
| Slow path (1 iteration) | 6.0s | 10.0s | ~$0.08 |
| Slow path (3 iterations) | 10.0s | 15.0s | ~$0.13 |
| Slow path (5 iterations) | 14.0s | 20.0s | ~$0.18 |

---

## 5. Self-Reflection Loop

### How the System Evaluates Answer Quality

After each execution iteration, a reflection step evaluates whether the retrieved context
is sufficient to answer the query. This is the core mechanism that differentiates agentic
RAG from single-pass.

```python
async def reflection_node(state: AgenticState) -> dict:
    """Evaluate retrieval quality and decide whether to iterate."""
    chunks = state["retrieved_chunks"]
    plan = state["research_plan"]
    iteration = state["iteration"]
    max_iter = plan.get("max_iterations", 3)

    # Hard stop: max iterations reached
    if iteration >= max_iter:
        return {"route_after_reflect": "verify", "reflection_notes": "Max iterations reached"}

    # Compute sufficiency signals
    signals = _compute_sufficiency_signals(state)

    # Decision matrix
    if signals["avg_relevance"] >= 0.70 and signals["sub_questions_covered"] >= 0.80:
        return {"route_after_reflect": "verify", "reflection_notes": "Sufficient coverage"}

    if signals["avg_relevance"] < 0.45:
        # Very low relevance — reformulate query entirely
        return {
            "route_after_reflect": "execute",
            "iteration": iteration + 1,
            "reflection_notes": f"Low relevance ({signals['avg_relevance']:.2f}), reformulating",
            "next_strategy": "reformulate",
        }

    if signals["sub_questions_covered"] < 0.50:
        # Missing sub-question coverage — targeted retrieval
        missing = signals["uncovered_sub_questions"]
        return {
            "route_after_reflect": "execute",
            "iteration": iteration + 1,
            "reflection_notes": f"Missing coverage for: {missing}",
            "next_strategy": "targeted",
            "target_sub_questions": missing,
        }

    # Marginal — one more iteration with expanded query
    return {
        "route_after_reflect": "execute",
        "iteration": iteration + 1,
        "reflection_notes": "Marginal coverage, expanding",
        "next_strategy": "expand",
    }


def _compute_sufficiency_signals(state: AgenticState) -> dict:
    """Compute numeric signals for reflection decision."""
    chunks = state["retrieved_chunks"]
    clarification = state["clarification"]
    sub_questions = clarification.get("sub_questions", [state["query"]])

    # Average relevance of retrieved chunks
    scores = [c.get("score", 0) for c in chunks]
    avg_relevance = sum(scores) / len(scores) if scores else 0.0

    # Sub-question coverage: how many sub-questions have at least 1 relevant chunk
    covered = 0
    uncovered = []
    for sq in sub_questions:
        # Check if any chunk is relevant to this sub-question
        # (lightweight: keyword overlap check, not a full embedding comparison)
        sq_words = set(sq.lower().split())
        has_relevant = any(
            len(sq_words & set(c.get("content", "").lower().split()[:100])) >= 3
            for c in chunks
        )
        if has_relevant:
            covered += 1
        else:
            uncovered.append(sq)

    coverage_ratio = covered / len(sub_questions) if sub_questions else 1.0

    # Authority coverage: do we have binding authority?
    has_binding = any(c.get("binding_authority") for c in chunks)

    # Jurisdiction match
    target_jurisdiction = clarification.get("jurisdiction", "unknown")
    jurisdiction_match = any(
        c.get("jurisdiction_code", "unknown") == target_jurisdiction
        for c in chunks
    ) if target_jurisdiction != "unknown" else True

    return {
        "avg_relevance": avg_relevance,
        "sub_questions_covered": coverage_ratio,
        "uncovered_sub_questions": uncovered,
        "has_binding_authority": has_binding,
        "jurisdiction_match": jurisdiction_match,
        "chunk_count": len(chunks),
    }
```

### Iteration Strategies

| Strategy | When Used | What Happens |
|----------|-----------|-------------|
| `reformulate` | avg_relevance < 0.45 | LLM rewrites the query using different terminology |
| `targeted` | sub_questions_covered < 0.50 | Searches specifically for uncovered sub-questions |
| `expand` | Marginal coverage | Adds synonyms, broader terms, related concepts |
| `filter_relax` | Filtered search returned < 3 results | Removes jurisdiction/type filters and retries |
| `external` | No binding authority found | Triggers CourtListener search for case law |

---

## 6. Integration with Existing Systems

### Existing Services Reused (Not Rebuilt)

The agentic layer is an **orchestration wrapper** around existing services. Nothing is
rebuilt from scratch. The value is in the control flow, not new capabilities.

| Existing Service | File | How the Agentic Layer Uses It |
|-----------------|------|-------------------------------|
| Vector search | `services/vector_store.py` | Execution agent's `vector_search` tool calls `search_vectors()` |
| Hybrid search | `services/hybrid_search.py` | Execution agent's `hybrid_search` tool calls `reciprocal_rank_fusion()` |
| BM25 sparse | `services/hybrid_search.py` | Execution agent uses `generate_sparse_vector()` + `search_sparse_vectors()` |
| Embeddings | `services/embeddings.py` | All search tools call `embed_query()` |
| Reranker | `services/rag_engine.py` | `rerank_chunks()` called after each retrieval iteration |
| Citation verification | `services/citation_agent.py` | Verification agent calls `verify_response_citations()` |
| Claim verification | `services/claim_verifier.py` | Verification agent calls `verify_claims()` |
| Conflict detection | `services/conflict_detector.py` | Verification agent calls `detect_conflicts()` |
| Authority hierarchy | `services/authority_detector.py` | Reranking uses `compute_query_time_authority_score()` |
| CourtListener | `services/legal_research.py` | Execution agent's `courtlistener_search` tool |
| Context formatting | `services/rag_engine.py` | Synthesis agent reuses `format_legal_context()` |
| Confidence scoring | `services/rag_engine.py` | Final response reuses `calculate_answer_confidence()` |

### Integration Points in `rag_engine.py`

The existing `query_matter()` function remains intact as the fast-path handler. A new
`agentic_query_matter()` function is the slow-path entry point. The router decides which
to call.

```python
# In backend/services/rag_engine.py (modified)

async def query_matter_routed(
    matter_id: str,
    query: str,
    db: Session,
    conversation_history: list = None,
    include_legal_research: bool = False,
    force_agentic: bool = False,
) -> Dict:
    """Top-level entry point that routes to fast or agentic path."""
    if force_agentic:
        route = "complex"
    else:
        route = classify_query_complexity(query, conversation_history or [])
        if route == "ambiguous":
            route = await _llm_classify_query(query)

    if route == "simple":
        # Existing pipeline — no changes
        return await query_matter(
            matter_id, query, db,
            conversation_history=conversation_history,
            include_legal_research=include_legal_research,
        )
    else:
        # New agentic pipeline
        from backend.services.agentic.pipeline import agentic_query_matter
        return await agentic_query_matter(
            matter_id, query, db,
            conversation_history=conversation_history,
            include_legal_research=include_legal_research,
        )
```

---

## 7. Implementation Steps

### Week 1: Foundation — State Schema + LangGraph Skeleton

**Goal:** LangGraph graph compiles, runs end-to-end with stub nodes, tests pass.

**Files to create:**

| File | Purpose |
|------|---------|
| `backend/services/agentic/__init__.py` | Package init |
| `backend/services/agentic/state.py` | `AgenticState` TypedDict + reducers |
| `backend/services/agentic/graph.py` | LangGraph `StateGraph` definition, all edges |
| `backend/services/agentic/pipeline.py` | `agentic_query_matter()` entry point |
| `backend/tests/test_agentic_graph.py` | Graph compilation + stub execution tests |

**Key implementation — `state.py`:**

```python
"""Agentic RAG state schema for LangGraph."""
from typing import Annotated, Optional
from typing_extensions import TypedDict


def _merge_chunks(existing: list, new: list) -> list:
    """Reducer: merge retrieved chunks, deduplicate by chunk_id."""
    seen = {c.get("chunk_id") for c in existing}
    merged = list(existing)
    for chunk in new:
        if chunk.get("chunk_id") not in seen:
            merged.append(chunk)
            seen.add(chunk.get("chunk_id"))
    return merged


def _append_list(existing: list, new: list) -> list:
    """Reducer: append to list."""
    return existing + new


class AgenticState(TypedDict):
    """Shared state passed between all nodes in the agentic RAG graph."""
    # Input
    query: str
    matter_id: str
    conversation_history: list
    matter_metadata: dict              # {name, document_count, jurisdictions, ...}
    include_legal_research: bool

    # Router
    route: str                         # "simple" | "complex"

    # Clarification
    clarification: dict                # Clarification TypedDict

    # Planning
    research_plan: dict                # ResearchPlan TypedDict

    # Execution (iterative)
    retrieved_chunks: Annotated[list, _merge_chunks]
    retrieval_log: Annotated[list, _append_list]  # Trace of what was searched
    iteration: int
    next_strategy: str                 # "reformulate" | "targeted" | "expand" | ...
    target_sub_questions: list

    # Reflection
    reflection_notes: str
    route_after_reflect: str           # "execute" | "verify"

    # Draft (generated by execution agent for self-eval, or by synthesis for final)
    draft_answer: str

    # Verification
    citation_verification: Optional[dict]
    claim_verification: Optional[dict]
    conflict_analysis: Optional[dict]
    verification_issues: list
    verification_passed: bool

    # Final output
    final_answer: str
    sources: list
    confidence: dict
    error: Optional[str]
```

**Key implementation — `graph.py`:**

```python
"""LangGraph graph definition for agentic RAG pipeline."""
from langgraph.graph import StateGraph, START, END
from backend.services.agentic.state import AgenticState


def build_agentic_graph():
    """Build and return the compiled agentic RAG graph."""
    from backend.services.agentic.nodes import (
        router_node,
        direct_rag_node,
        clarification_node,
        planning_node,
        execution_node,
        reflection_node,
        verification_node,
        synthesis_node,
    )

    builder = StateGraph(AgenticState)

    # Add nodes
    builder.add_node("router", router_node)
    builder.add_node("direct_rag", direct_rag_node)
    builder.add_node("clarify", clarification_node)
    builder.add_node("plan", planning_node)
    builder.add_node("execute", execution_node)
    builder.add_node("reflect", reflection_node)
    builder.add_node("verify", verification_node)
    builder.add_node("synthesize", synthesis_node)

    # Edges
    builder.add_edge(START, "router")

    # Router -> fast or slow path
    builder.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"simple": "direct_rag", "complex": "clarify"},
    )

    # Fast path
    builder.add_edge("direct_rag", END)

    # Slow path: linear until execute/reflect loop
    builder.add_edge("clarify", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "reflect")

    # Reflect -> loop or proceed
    builder.add_conditional_edges(
        "reflect",
        lambda state: state["route_after_reflect"],
        {"execute": "execute", "verify": "verify"},
    )

    # Verification -> synthesis -> done
    builder.add_edge("verify", "synthesize")
    builder.add_edge("synthesize", END)

    # Compile (no checkpointer for now — stateless per-request)
    return builder.compile()


# Singleton compiled graph
_GRAPH = None

def get_agentic_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_agentic_graph()
    return _GRAPH
```

**Files to modify:**

| File | Change |
|------|--------|
| `backend/requirements.txt` | Add `langgraph>=0.4.0`, `langgraph-checkpoint-postgres>=2.0.0` |
| `backend/config.py` | Add `agentic_enabled: bool = False`, `agentic_max_iterations: int = 5` |

---

### Week 2: Router + Clarification + Planning Agents

**Goal:** Query routing works. Clarification extracts jurisdiction/entities. Planning produces research plans.

**Files to create:**

| File | Purpose |
|------|---------|
| `backend/services/agentic/nodes/__init__.py` | Node package init, re-exports |
| `backend/services/agentic/nodes/router.py` | Rule-based + LLM router |
| `backend/services/agentic/nodes/clarification.py` | Clarification agent |
| `backend/services/agentic/nodes/planning.py` | Planning agent |
| `backend/tests/test_agentic_router.py` | Router classification tests |
| `backend/tests/test_agentic_clarification.py` | Clarification extraction tests |

**Files to modify:**

| File | Change |
|------|--------|
| `backend/services/rag_engine.py` | Add `query_matter_routed()` wrapper function |

---

### Week 3: Execution Agent + Tool Bindings

**Goal:** Execution agent can call vector_search, hybrid_search, courtlistener_search tools. Iterative retrieval works.

**Files to create:**

| File | Purpose |
|------|---------|
| `backend/services/agentic/nodes/execution.py` | Execution agent with tool calls |
| `backend/services/agentic/nodes/reflection.py` | Self-reflection node |
| `backend/services/agentic/tools.py` | LangChain `@tool` definitions wrapping existing services |
| `backend/tests/test_agentic_execution.py` | Tool invocation + iteration tests |

---

### Week 4: Verification + Synthesis Agents

**Goal:** Full pipeline runs end-to-end. Verification agent reuses existing services. Synthesis agent produces CREAC output.

**Files to create:**

| File | Purpose |
|------|---------|
| `backend/services/agentic/nodes/verification.py` | Calls existing verification services |
| `backend/services/agentic/nodes/synthesis.py` | CREAC-structured answer generation |
| `backend/services/agentic/nodes/direct_rag.py` | Thin wrapper around existing `query_matter()` |
| `backend/tests/test_agentic_e2e.py` | Full agentic pipeline end-to-end test |

**Files to modify:**

| File | Change |
|------|--------|
| `backend/schemas.py` | Add `AgenticQueryResponse` schema with routing metadata |
| `frontend/lib/types.ts` | Add `agenticMetadata` to `QueryMessage` type |

---

### Week 5: Frontend Integration + Streaming

**Goal:** Frontend shows agentic progress (which agent is running), routing decision, iteration count.

**Files to modify:**

| File | Change |
|------|--------|
| `frontend/components/ChatPanel.tsx` | Show "Researching..." with agent step indicators |
| `frontend/lib/api-services.ts` | Support SSE streaming from agentic endpoint |
| `backend/services/agentic/pipeline.py` | Add SSE progress events per node |
| `backend/services/progress.py` | Add `publish_agentic_step()` events |

**New SSE event format:**

```json
{
  "stage": "agentic",
  "step": 3,
  "total_steps": 6,
  "agent": "execution",
  "iteration": 2,
  "message": "Retrieving case law for force majeure...",
  "detail": "CourtListener search: 5 results"
}
```

---

### Week 6: Testing, Optimization, Shadow Mode

**Goal:** Agentic pipeline is tested, optimized for latency, deployed in shadow mode.

**Files to create:**

| File | Purpose |
|------|---------|
| `backend/tests/test_agentic_quality.py` | Comparative quality tests: agentic vs single-pass |
| `backend/services/agentic/metrics.py` | Latency + cost tracking per agent node |
| `backend/alembic/versions/13_add_agentic_metadata.py` | Add `routing_decision`, `agent_iterations` columns to `queries` table |

**Files to modify:**

| File | Change |
|------|--------|
| `backend/config.py` | Add `agentic_shadow_mode: bool = True` (log but don't serve) |
| `backend/models.py` | Add `routing_decision`, `agent_iterations`, `agent_latency_ms` to `Query` model |

---

## 8. Latency Optimization

### Target: Complex queries in 6-10 seconds (from naive 15+ seconds)

**Optimization 1: Parallel Clarification + Matter Metadata Fetch**

The clarification agent and matter metadata fetch are independent. Run them concurrently.

```python
async def parallel_init(state: AgenticState) -> dict:
    clarification_task = asyncio.create_task(clarify(state))
    metadata_task = asyncio.create_task(fetch_matter_metadata(state["matter_id"]))
    clarification, metadata = await asyncio.gather(clarification_task, metadata_task)
    return {**clarification, "matter_metadata": metadata}
```

**Savings: ~400ms** (metadata fetch overlaps with clarification LLM call)

**Optimization 2: Batch Retrieval in Execution**

Instead of sequential search calls, execute independent retrieval steps in parallel.

```python
# Before (sequential): 3 searches x 300ms = 900ms
# After (parallel): max(300ms, 300ms, 300ms) = 300ms

async def execute_parallel_searches(steps: list[ResearchStep]) -> list:
    independent_steps = [s for s in steps if not s["depends_on"]]
    tasks = [execute_search_step(s) for s in independent_steps]
    results = await asyncio.gather(*tasks)
    return [chunk for result in results for chunk in result]
```

**Savings: ~600ms** for plans with 3+ independent steps

**Optimization 3: Google AI Prompt Caching**

The system prompt (LEGAL_SYSTEM_PROMPT) is ~2000 tokens and identical across all requests.
Use Gemini's context caching to avoid re-processing it.

```python
# Cache the system prompt (lives for 1 hour)
from google.generativeai.caching import CachedContent

cached_content = CachedContent.create(
    model=settings.gemini_model,
    system_instruction=LEGAL_SYSTEM_PROMPT,
    ttl=datetime.timedelta(hours=1),
)
```

**Savings: ~200ms per LLM call** (system prompt tokens processed once)

**Optimization 4: Skip Verification for High-Confidence Results**

If the execution agent's self-reflection indicates very high confidence (avg_relevance > 0.85,
all sub-questions covered, binding authority found), skip the full verification pass and
run only lightweight citation extraction.

**Savings: ~1.5s** for ~30% of complex queries

**Optimization 5: Streaming First Token**

Start streaming the synthesis agent's output to the frontend as soon as the first token
is generated, rather than waiting for the full response. The verification badges can be
appended as a post-stream update.

**Savings: Perceived latency reduced by 2-4s** (user sees text appearing while verification runs)

### Latency Budget (Optimized)

| Stage | Naive | Optimized | Notes |
|-------|-------|-----------|-------|
| Router | 300ms | 50ms | Rule-based for 80% of cases |
| Clarification | 500ms | 400ms | Parallel with metadata |
| Planning | 500ms | 400ms | Cached system prompt |
| Execution (iter 1) | 2000ms | 1200ms | Parallel retrieval + cached prompt |
| Reflection | 300ms | 200ms | Local heuristics, no LLM |
| Execution (iter 2) | 2000ms | 1200ms | Only if needed (~60% of complex) |
| Verification | 2000ms | 1000ms | Parallel verification services |
| Synthesis | 2000ms | 1500ms | Cached prompt + streaming |
| **Total (1 iter)** | **7600ms** | **4750ms** | |
| **Total (2 iter)** | **9900ms** | **6150ms** | |
| **Total (3 iter)** | **12200ms** | **7550ms** | |

---

## 9. Cost Model

### Per-Query Token Breakdown

**Fast path (simple query):**

| Component | Input Tokens | Output Tokens | Model | Cost |
|-----------|-------------|---------------|-------|------|
| Embed query | N/A | N/A | Cohere | $0.0001 |
| Generate answer | ~12,000 | ~800 | gemini-2.5-flash-lite | ~$0.015 |
| **Total** | | | | **~$0.015** |

**Slow path (complex query, 2 iterations):**

| Component | Input Tokens | Output Tokens | Model | Cost |
|-----------|-------------|---------------|-------|------|
| Router (LLM, 20% of queries) | ~200 | ~10 | gemini-2.5-flash-lite | $0.0003 |
| Clarification | ~500 | ~200 | gemini-2.5-flash-lite | $0.001 |
| Planning | ~800 | ~300 | gemini-2.5-flash-lite | $0.001 |
| Embed queries (3 searches) | N/A | N/A | Cohere | $0.0003 |
| Execution self-eval | ~2,000 | ~100 | gemini-2.5-flash | $0.003 |
| Reflection | ~500 | ~50 | local heuristics | $0.000 |
| Execution iter 2 | ~2,000 | ~100 | gemini-2.5-flash | $0.003 |
| NLI verification (local) | N/A | N/A | CPU | $0.000 |
| Citation verification (Tier 2 LLM, 10%) | ~800 | ~20 | gemini-2.5-flash-lite | $0.001 |
| Claim verification (LLM, 10%) | ~1,500 | ~300 | gemini-2.5-flash-lite | $0.002 |
| Synthesis | ~15,000 | ~1,200 | gemini-2.5-flash | $0.020 |
| **Total** | | | | **~$0.032** |

**Weighted average (40% simple, 60% complex):**

```
$0.015 * 0.40 + $0.032 * 0.60 = $0.025 per query
```

**At scale (1,000 queries/day):** ~$25/day = ~$750/month

**Cost optimization levers:**
- Prompt caching: -30% on repeated system prompts (~$0.007 savings)
- Routing more queries to fast path: target 50% simple (from 40%)
- Shorter context windows for planning/clarification (< 1000 tokens)

### Comparison with Current System

| Metric | Current (single-pass) | Agentic (estimated) |
|--------|----------------------|---------------------|
| Cost per query | ~$0.015 | ~$0.025 |
| Cost increase | baseline | +67% |
| Quality improvement (complex) | baseline | +30-50% estimated |
| Latency (simple) | 2-3s | 2-3s (unchanged) |
| Latency (complex) | 2-3s | 6-10s |

---

## 10. Testing Strategy

### How to Validate Agentic Answers Are Better

**10.1 Evaluation Dataset**

Build a test suite of 50 queries across 5 categories:

| Category | Count | Example |
|----------|-------|---------|
| Simple factual | 10 | "What is the effective date of the NDA?" |
| Multi-document comparison | 10 | "How do the indemnification clauses differ between the two contracts?" |
| Legal analysis | 10 | "What are the risks of the non-compete clause under California law?" |
| Multi-hop reasoning | 10 | "Find the governing law clause, then identify case law on its enforceability" |
| Ambiguous / underspecified | 10 | "What about the liability?" (needs context from conversation) |

**10.2 Quality Metrics (Per-Query)**

| Metric | How Measured | Target |
|--------|-------------|--------|
| Correctness | Expert lawyer review (1-5 scale) | >= 4.0 |
| Citation accuracy | % of citations that are real and relevant | >= 90% |
| Coverage | % of relevant documents/sections cited | >= 80% |
| CREAC structure | Does answer follow CREAC? (binary) | >= 90% |
| Hallucination rate | % of claims not supported by sources | <= 5% |
| Conflict handling | Conflicts acknowledged when present? | >= 95% |

**10.3 A/B Comparison Protocol**

For each test query, run BOTH pipelines and compare:

```python
async def ab_compare(matter_id: str, query: str, db: Session):
    """Run both pipelines and compare results."""
    single_pass = await query_matter(matter_id, query, db)
    agentic = await agentic_query_matter(matter_id, query, db)

    return {
        "query": query,
        "single_pass_answer": single_pass["answer"],
        "agentic_answer": agentic["answer"],
        "single_pass_confidence": single_pass["confidence"]["score"],
        "agentic_confidence": agentic["confidence"]["score"],
        "single_pass_sources": len(single_pass["sources"]),
        "agentic_sources": len(agentic["sources"]),
        "agentic_iterations": agentic.get("agentic_metadata", {}).get("iterations", 1),
        "agentic_latency_ms": agentic.get("agentic_metadata", {}).get("total_latency_ms", 0),
    }
```

**10.4 Automated Regression Tests**

```python
# backend/tests/test_agentic_quality.py

class TestAgenticQuality:
    """Compare agentic pipeline quality against single-pass baseline."""

    def test_complex_query_coverage(self):
        """Agentic pipeline should cite more relevant sources for complex queries."""
        # ... assert agentic sources > single_pass sources for multi-doc queries

    def test_multi_hop_reasoning(self):
        """Agentic pipeline should answer multi-hop queries that single-pass cannot."""
        # ... assert agentic answer is not "Insufficient data" when documents exist

    def test_fast_path_unchanged(self):
        """Simple queries should produce identical results via fast path."""
        # ... assert fast path answer == single_pass answer

    def test_iteration_improves_relevance(self):
        """Each iteration should increase average retrieval relevance."""
        # ... assert relevance(iter_2) >= relevance(iter_1)

    def test_verification_integration(self):
        """Verification results should be present in agentic response."""
        # ... assert response has citation_verification and claim_verification
```

---

## 11. Rollout Plan

### Progressive Deployment with Quality Gates

```
Week 6      Week 7      Week 8      Week 9      Week 10     Week 11
Shadow   -> 10% A/B  -> 25% A/B  -> 50% A/B  -> 100%     -> Fast path
Mode        Traffic     Traffic     Traffic     Complex      optimization
```

**Stage 1: Shadow Mode (Week 6-7)**

- `agentic_shadow_mode = True` in config
- Every complex query runs BOTH pipelines
- Single-pass result is served to the user
- Agentic result is logged for comparison
- No user-facing impact

**Quality gate to exit shadow mode:**
- Agentic answer quality >= single-pass on 80%+ of shadow comparisons
- P95 latency <= 15 seconds
- No error rate increase

**Stage 2: 10% A/B (Week 7-8)**

- 10% of complex queries served by agentic pipeline
- Remaining 90% use single-pass
- Monitor: latency, error rate, user feedback (thumbs up/down if available)

**Quality gate:**
- Error rate < 1%
- No increase in "low confidence" answers
- Latency P95 < 12 seconds

**Stage 3: 25% Traffic (Week 8-9)**

- Scale to 25% of complex queries
- Begin collecting structured feedback

**Stage 4: 50% Traffic (Week 9-10)**

- Half of complex queries go agentic
- Compare aggregate metrics

**Stage 5: 100% Complex (Week 10-11)**

- All complex queries use agentic pipeline
- Simple queries still use fast path
- Full monitoring in place

**Stage 6: Optimize Fast Path (Week 11+)**

- Tune router to maximize fast-path usage (target: 50% of queries)
- Reduce agentic latency via prompt caching
- Consider LangGraph checkpointing for conversation continuity

---

## 12. Risks & Mitigations

### Technical Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **LangGraph version instability** | Breaking API changes | Medium | Pin to exact version. Wrap all LangGraph calls in thin adapter layer. |
| **Infinite loop in reflect->execute cycle** | Resource exhaustion, timeout | Low | Hard cap at `max_iterations` (default 5). Timeout at 30s total. |
| **LLM routing errors** | Simple queries take 10s, complex queries get shallow answers | Medium | Rule-based pre-classifier handles 80% of cases. LLM only for ambiguous 20%. |
| **Token budget explosion** | Cost spike from iterative retrieval | Medium | Cap total tokens per agentic run at 100k. Log and alert on anomalies. |
| **Reranker + NLI model memory** | OOM on constrained deployments | Low | Models already loaded (singleton). Agentic adds no new models. |
| **Concurrent agentic requests** | CPU/memory contention from parallel LLM calls | Medium | Semaphore limiting concurrent agentic executions (default: 4). |

### Product Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Users perceive slowness** | Abandonment of complex queries | High | Streaming progress indicators. "Researching..." with agent step names. |
| **Over-iteration produces worse answers** | More iterations but diminishing returns | Medium | Reflection heuristics tuned conservatively. Always show iteration count. |
| **Complex path answers differently than simple** | User confusion on identical queries | Low | Router determinism: same query always routes the same way. |

### Operational Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|------------|
| **Shadow mode doubles API costs** | 2x LLM spend during shadow period | Certain | Shadow mode is temporary (2 weeks max). Budget accordingly. |
| **Debugging agentic failures** | Multi-step pipeline harder to trace | High | Full execution trace logged per request (agent, duration, inputs, outputs). |
| **Rollback complexity** | Cannot easily revert if agentic has bugs | Medium | Feature flag `agentic_enabled`. One config change to disable entirely. |

---

## 13. Dependencies

### New Python Dependencies

| Package | Version | Purpose | Size |
|---------|---------|---------|------|
| `langgraph` | `>=0.4.0,<1.0.0` | Graph-based agent orchestration | ~2MB |
| `langgraph-checkpoint-postgres` | `>=2.0.0` | PostgreSQL state persistence (future) | ~500KB |

**Note:** LangGraph depends on `langchain-core`, which Lexintel already has via `langchain>=0.3.0`.

### Existing Dependencies (Already Installed)

| Package | Used By |
|---------|---------|
| `google-generativeai` | All LLM calls (router, clarification, planning, execution, synthesis) |
| `cohere` | Embedding for all search tools |
| `sentence-transformers` | Reranker (execution agent), NLI models (verification agent) |
| `qdrant-client` | Vector search + sparse search tools |
| `fastembed` | BM25 sparse encoder for hybrid search tool |
| `langchain` | Tool definitions (`@tool` decorator) |

### Infrastructure Requirements

| Component | Current | Additional Needs |
|-----------|---------|-----------------|
| PostgreSQL | Running | Add `routing_decision`, `agent_iterations`, `agent_latency_ms` columns to `queries` table |
| Redis | Running (Celery broker) | No changes |
| Qdrant | Running | No changes |
| Celery | Running | No changes (agentic runs in FastAPI async, not Celery) |

### Multi-Model Routing

| Agent | Model | Why |
|-------|-------|-----|
| Router | `gemini-2.5-flash-lite` | Cheapest, fastest. Classification is easy. |
| Clarification | `gemini-2.5-flash-lite` | Structured extraction is well within flash-lite capabilities. |
| Planning | `gemini-2.5-flash-lite` | Plan generation from structured input. |
| Execution (self-eval) | `gemini-2.5-flash` | Needs reasoning to evaluate retrieval quality. |
| Verification | Existing (local NLI + flash-lite fallback) | No change from current pipeline. |
| Synthesis | `gemini-2.5-flash` | Final output quality matters most. Strongest model. |

**Future consideration:** When Lexintel scales beyond single-user, consider
`gemini-2.5-pro` for synthesis on high-stakes queries (flagged by the user or
by the system based on matter sensitivity).

### PostgreSQL State Persistence (Future)

For conversation-aware agentic sessions, LangGraph's PostgreSQL checkpointer
enables resuming research across multiple queries in the same conversation.

```python
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# Uses Lexintel's existing PostgreSQL connection
DB_URI = settings.database_url
async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    graph = build_agentic_graph().compile(checkpointer=checkpointer)
    # Thread ID = conversation ID for continuity
    config = {"configurable": {"thread_id": str(conversation_id)}}
    result = await graph.ainvoke(initial_state, config)
```

This is deferred to a future phase because:
1. It requires LangGraph checkpoint tables in PostgreSQL (new migration)
2. Conversation continuity is a separate feature from agentic quality
3. The current system already passes `conversation_history` as context

---

## Appendix: File Tree (New Files)

```
backend/
  services/
    agentic/
      __init__.py
      state.py              # AgenticState TypedDict + reducers
      graph.py              # LangGraph StateGraph definition
      pipeline.py           # agentic_query_matter() entry point
      tools.py              # @tool definitions wrapping existing services
      metrics.py            # Latency + cost tracking
      nodes/
        __init__.py          # Re-exports all node functions
        router.py            # Query complexity router
        clarification.py     # Jurisdiction/entity extraction
        planning.py          # Research plan generation
        execution.py         # Iterative retrieval with tools
        reflection.py        # Self-evaluation + loop decision
        verification.py      # Reuses existing verification services
        synthesis.py         # CREAC answer generation
        direct_rag.py        # Fast-path wrapper around query_matter()
  tests/
    test_agentic_graph.py    # Graph compilation + stub tests
    test_agentic_router.py   # Router classification tests
    test_agentic_clarification.py
    test_agentic_execution.py
    test_agentic_e2e.py      # Full pipeline end-to-end
    test_agentic_quality.py  # A/B quality comparison
  alembic/
    versions/
      13_add_agentic_metadata.py  # Migration for Query table columns
```
