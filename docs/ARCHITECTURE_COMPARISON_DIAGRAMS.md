# Legal AI Architecture Comparison: Visual Reference

**Date:** March 24, 2026

---

## ARCHITECTURE 1: HARVEY AI - Agent-Based Orchestration

```
┌─────────────────────────────────────────────────────────────────┐
│                      USER QUERY (Legal Task)                     │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  TASK CLASSIFIER AGENT     │
        │  ├─ Research task?         │
        │  ├─ Drafting task?         │
        │  ├─ Analysis task?         │
        │  └─ Validation task?       │
        └────────┬───────────────────┘
                 │
    ┌────────────┼────────────────────┐
    │            │                    │
    ▼            ▼                    ▼
┌─────────┐  ┌──────────┐  ┌──────────────┐
│Research │  │Drafting  │  │Analysis      │
│Agent    │  │Agent     │  │Agent         │
│(o1 LLM) │  │(GPT-4)   │  │(Claude)      │
└────┬────┘  └────┬─────┘  └──────┬───────┘
     │            │               │
     ▼            ▼               ▼
┌─────────────────────────────────────────┐
│    VECTOR DATABASE (Custom Embeddings)   │
│    └─ Voyage-law-2 (20B+ case law)      │
│    └─ Document embeddings (1024-dim)    │
└────────┬────────────────────────────────┘
         │
    ┌────┴─────────────────┐
    ▼                      ▼
┌──────────────────┐  ┌──────────────────┐
│ BM25 Lexical     │  │ Semantic Search  │
│ (Exact terms)    │  │ (Concept match)  │
└────────┬─────────┘  └────────┬─────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌──────────────────────┐
         │ Custom Legal Reranker│
         │ (Domain-specific)    │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ Authority Weighting  │
         │ ├─ Court level       │
         │ ├─ Jurisdiction      │
         │ └─ Recency           │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ Top 8 Ranked Results │
         │ with Sources         │
         └────────┬─────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
    ▼             ▼             ▼
┌──────┐    ┌──────────┐    ┌──────────┐
│Agent1│    │Agent2    │    │Agent3    │
│Synth │    │Reason    │    │Validate  │
└──┬───┘    └────┬─────┘    └────┬─────┘
   │             │              │
   └─────────────┼──────────────┘
                 ▼
         ┌──────────────────────┐
         │  ORCHESTRATOR        │
         │  ├─ Context mgmt     │
         │  ├─ Tool routing     │
         │  └─ Fallback handling│
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │  FINAL OUTPUT        │
         │  ├─ Answer           │
         │  ├─ Citations        │
         │  ├─ Reasoning trace  │
         │  └─ Confidence score │
         └──────────────────────┘
```

**Key Features:**
- Multi-model routing by task
- Agent SDK-based orchestration
- Custom embeddings on 20B+ legal tokens
- Smart caching + predictive prefetch
- Scales to 100K+ documents

---

## ARCHITECTURE 2: COCOUNSEL DEEP RESEARCH - Multi-Agent Workflow

```
┌──────────────────────────────────────────────┐
│         USER RESEARCH QUERY                   │
│  "Research liability limits for product      │
│   defects in California under UCC § 2-316"  │
└─────────────────┬──────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │   ORCHESTRATOR      │
        │   ├─ Parse query    │
        │   ├─ Generate plan  │
        │   └─ Allocate tasks │
        └────────┬────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
    ▼            ▼            ▼              ▼
┌──────────┐ ┌──────────┐ ┌─────────┐ ┌──────────┐
│Research  │ │Discovery │ │Web      │ │Customer  │
│Agent     │ │Agent     │ │Search   │ │Document  │
│(Claude)  │ │(Claude)  │ │Agent    │ │Agent     │
└────┬─────┘ └────┬─────┘ └────┬────┘ └────┬─────┘
     │            │            │           │
     ▼            ▼            ▼           ▼
  ┌──────────────────────────────────────────┐
  │     WESTLAW + PRACTICAL LAW (RAG)        │
  │     ├─ 150 years legal knowledge         │
  │     ├─ 3,000+ subject matter experts     │
  │     ├─ Citation graph (cases, statutes)  │
  │     └─ Regulatory database               │
  └────────────┬─────────────────────────────┘
               │
   ┌───────────┴───────────┐
   ▼                       ▼
┌──────────────┐    ┌──────────────────┐
│Statute Found │    │Case Law Retrieved│
│(UCC § 2-316) │    │(Supporting cases)│
└──────┬───────┘    └────────┬─────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │  REASONING AGENT    │
        │  ├─ Cross-reference │
        │  ├─ Analyze holding │
        │  ├─ Check conflicts │
        │  └─ Synthesize      │
        └────────┬────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│Theory 1: │ │Theory 2: │ │Theory 3: │
│Strict    │ │Negligence│ │Warranty  │
│Liability │ │          │ │Breach    │
└─────┬────┘ └────┬─────┘ └────┬─────┘
      │           │            │
      └─────────┬─┴────────────┘
                │
                ▼ (IF theory 1 fails, try theory 2/3)
        ┌──────────────────────┐
        │  COMPREHENSIVE       │
        │  RESEARCH REPORT     │
        │  ├─ Multi-step plan  │
        │  ├─ Key findings     │
        │  ├─ Westlaw citations│
        │  ├─ Reasoning trace  │
        │  └─ Alternative      │
        │    theories explored │
        └──────────────────────┘
```

**Key Features:**
- 5-agent system (research, discovery, web search, customer doc, reasoning)
- Multi-step research plans with backtracking
- Westlaw citation-backed answers
- Alternative theory exploration
- Human checkpoints for cross-doc analysis

---

## ARCHITECTURE 3: LUMINANCE AI - Legal Pre-Trained Transformer + Institutional Memory

```
┌──────────────────────────────────────────┐
│       CONTRACT/LEGAL DOCUMENT             │
│       (New analysis request)              │
└─────────────┬────────────────────────────┘
              │
              ▼
     ┌─────────────────────┐
     │  LEGAL PRE-TRAINED  │
     │  TRANSFORMER (LPT)  │
     │                     │
     │  Trained on:        │
     │  └─ 150M+ verified  │
     │    legal documents  │
     │  └─ 10+ years law   │
     │    firm data        │
     └────────┬────────────┘
              │
    ┌─────────┴──────────────┐
    │                        │
    ▼                        ▼
┌──────────────────┐  ┌──────────────────┐
│MIXTURE OF EXPERTS│  │PORTFOLIO MEMORY  │
│├─ Foundational   │  │├─ Negotiation    │
││ models          │  ││  history        │
│├─ Proprietary    │  │├─ Related        │
││ fine-tuned      │  ││  contracts      │
│└─ Domain-tuned   │  │└─ Precedents     │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ RECURSIVE CONTEXTUAL │
         │ UNDERSTANDING        │
         │ ├─ Holistic analysis │
         │ ├─ Clause context    │
         │ └─ Portfolio context │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ SPECIALIST AGENTS    │
         │ FOR LIFECYCLE        │
         │ ├─ Understand context│
         │ ├─ Apply legal       │
         │ │ reasoning          │
         │ └─ Take action       │
         └────────┬─────────────┘
                  │
     ┌────────────┼───────────┐
     │            │           │
     ▼            ▼           ▼
┌──────────┐ ┌─────────┐ ┌──────────┐
│Short-term│ │Long-term│ │Audit     │
│Memory    │ │Memory   │ │Trail     │
│(Current  │ │(Deal    │ │(Source   │
│reasoning)│ │history, │ │links)    │
│          │ │prec.)   │ │          │
└─────┬────┘ └────┬────┘ └────┬─────┘
      │           │          │
      └───────────┼──────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │  AUDITABLE OUTPUT    │
         │  ├─ Answer           │
         │  ├─ Source-linked    │
         │  ├─ Reasoning        │
         │  └─ Traceability     │
         └──────────────────────┘
```

**Key Features:**
- Legal Pre-Trained Transformer (proprietary)
- Mixture-of-experts architecture
- Portfolio memory (enterprise amnesia prevention)
- Multi-agent lifecycle management
- Auditability + source linking
- Holistic recursive analysis (not clause-by-clause)

---

## ARCHITECTURE 4: LEXINTEL (CURRENT) - Advanced RAG with Verification

```
┌──────────────────────────────────────────────┐
│              USER QUERY                       │
│         (Legal matter question)               │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
      ┌──────────────────────────┐
      │  QUERY PROCESSOR         │
      │  ├─ Parse jurisdiction   │
      │  ├─ Detect task type     │
      │  └─ Identify domain      │
      └────────┬─────────────────┘
               │
    ┌──────────┴──────────────┐
    │                         │
    ▼                         ▼
┌───────────────────┐  ┌──────────────────┐
│ BM25 LEXICAL      │  │ SEMANTIC SEARCH  │
│ SEARCH            │  │                  │
│ └─ Exact terms    │  │ └─ Cohere v3.0   │
│ └─ Precise match  │  │ └─ 1024-dim      │
│ └─ Keywords       │  │ └─ Concept match │
└────────┬──────────┘  └────────┬─────────┘
         │                      │
         └──────────┬───────────┘
                    │
         ┌──────────▼──────────┐
         │ SCORE AGGREGATION   │
         │ (α-weighted hybrid) │
         └──────────┬──────────┘
                    │
                    ▼
      ┌────────────────────────┐
      │ TOP-30 DOCUMENTS       │
      │ (Score > 0.45)         │
      └────────┬───────────────┘
               │
               ▼
      ┌────────────────────────┐
      │ RERANKING              │
      │ (ms-marco cross-enc.)  │
      │ ├─ Semantic score      │
      │ ├─ Relevance signal    │
      │ └─ Document importance │
      └────────┬───────────────┘
               │
               ▼
      ┌────────────────────────┐
      │ AUTHORITY WEIGHTING    │
      │ ├─ Court level         │
      │ ├─ Jurisdiction        │
      │ ├─ Recency             │
      │ └─ Citation count      │
      └────────┬───────────────┘
               │
               ▼
      ┌────────────────────────┐
      │ TEMPORAL FILTERING     │
      │ ├─ Statute versions    │
      │ ├─ Amendment dates     │
      │ ├─ Supersession check  │
      │ └─ Effective dates     │
      └────────┬───────────────┘
               │
               ▼
      ┌────────────────────────┐
      │ TOP-8 RANKED RESULTS   │
      │ with document metadata │
      └────────┬───────────────┘
               │
         ┌─────┴──────────┐
         │                │
         ▼                ▼
    ┌─────────────┐  ┌──────────────┐
    │GENERATION   │  │VERIFICATION  │
    │(Gemini)     │  │(Parallel)    │
    │             │  │              │
    │Draft answer │  │├─ Citation   │
    │with context │  ││  lookup     │
    │             │  │├─ NLI check  │
    │             │  │├─ Authority  │
    │             │  ││  validation │
    │             │  │└─ Conflict   │
    │             │  │  detection   │
    └──────┬──────┘  └──────┬───────┘
           │                │
           └────────┬───────┘
                    │
                    ▼
         ┌──────────────────────┐
         │ VERIFIED ANSWER      │
         │ ├─ Generated text    │
         │ ├─ Citations with    │
         │ │  verification      │
         │ ├─ Confidence scores │
         │ ├─ Conflict flags    │
         │ └─ Temporal notes    │
         └──────────────────────┘
```

**Key Features (Lexintel Strengths):**
- BM25 + semantic hybrid search ✓
- Cross-encoder reranking ✓
- Authority hierarchy weighting ✓
- Temporal filtering ✓
- Citation verification (NLI + lookup) ✓
- Conflict detection ✓

**Key Gaps (vs. Competitors):**
- No multi-agent orchestration ✗
- Generic embeddings (not domain-tuned) ✗
- No knowledge graph ✗
- No long-context optimization ✗
- No institutional memory ✗
- Single-model generation ✗

---

## ARCHITECTURE 5: IDEAL LEGAL AI (2026 Parity)

```
┌──────────────────────────────────────────────────┐
│         USER COMPLEX LEGAL QUERY                  │
│  "Research precedents, verify current law,       │
│   identify conflicts across 100K documents,      │
│   and draft recommendations"                     │
└────────────┬───────────────────────────────────┘
             │
             ▼
    ┌─────────────────────┐
    │  TASK DECOMPOSER    │
    │  ├─ Research sub-   │
    │  │  question        │
    │  ├─ Verification    │
    │  │  requirements    │
    │  ├─ Conflict check  │
    │  └─ Draft plan      │
    └────────┬────────────┘
             │
    ┌────────┴────────────┬─────────────┐
    │                     │             │
    ▼                     ▼             ▼
┌────────────┐  ┌──────────────┐  ┌──────────────┐
│RESEARCH    │  │VERIFICATION  │  │SYNTHESIS     │
│AGENT       │  │AGENT         │  │AGENT         │
│            │  │              │  │              │
│1. Retrieve │  │1. Citation   │  │1. Cross-ref  │
│   docs     │  │   validation │  │   findings   │
│2. Rank by  │  │2. NLI check  │  │2. Identify   │
│   authority│  │3. Authority  │  │   conflicts  │
│3. Multi-hop│  │   hierarchy  │  │3. Reconcile  │
│   reasoning│  │4. Temporal   │  │4. Draft      │
│            │  │   accuracy   │  │              │
└────────┬───┘  └──────┬───────┘  └──────┬───────┘
         │             │                 │
         │  ┌──────────┤                 │
         │  │          │                 │
         ▼  ▼          ▼                 ▼
    ┌─────────────────────────────────────────┐
    │    HYBRID RETRIEVAL PIPELINE            │
    │    ├─ Domain-specific embeddings        │
    │    │  (Voyage-law-2 or fine-tuned)     │
    │    ├─ BM25 + semantic (hybrid)         │
    │    ├─ Legal reranker (cross-encoder)   │
    │    └─ Knowledge graph (Neo4j)          │
    │       ├─ Jurisdiction hierarchy       │
    │       ├─ Precedent relationships      │
    │       ├─ Statute amendments           │
    │       └─ Conflict detection           │
    └─────────┬───────────────────────────┘
              │
    ┌─────────┴──────────────┐
    ▼                        ▼
┌─────────────┐        ┌──────────────────┐
│Context mgmt │        │Model Routing     │
│├─ Portfolio │        │├─ Extended       │
││ reasoning  │        ││ reasoning (o1)  │
│├─ Cross-doc │        │├─ Recall search  │
││ analysis   │        ││ (GPT-4)         │
│├─ Prec. mem.│        │└─ Verification  │
││ (last 5)   │        │  (Claude)       │
│└─ Deal hist.│        └─────────────────┘
└─────────────┘
         │
         ▼
    ┌──────────────────────┐
    │ ORCHESTRATOR         │
    │ ├─ Route to agents   │
    │ ├─ Manage context    │
    │ ├─ Error recovery    │
    │ └─ Quality gates     │
    └────────┬─────────────┘
             │
         ┌───┴────┬─────────┬──────────┐
         │        │         │          │
         ▼        ▼         ▼          ▼
    ┌────────┐ ┌──────┐ ┌──────┐ ┌─────────┐
    │Research│ │Verify│ │Check │ │Synthesis│
    │Output  │ │Cit.  │ │Conf. │ │Output   │
    └────────┘ └──────┘ └──────┘ └─────────┘
         │        │        │         │
         └────────┼────────┼─────────┘
                  │        │
                  ▼        ▼
         ┌──────────────────────┐
         │  CONFIDENCE LAYER    │
         │  ├─ Citation scores  │
         │  ├─ Hallucination    │
         │  │  probability      │
         │  ├─ Conflict flags   │
         │  ├─ Authority notes  │
         │  └─ Temporal notes   │
         └────────┬─────────────┘
                  │
                  ▼
         ┌──────────────────────┐
         │ FINAL ANSWER         │
         │ ├─ Recommendation    │
         │ ├─ Verified citations│
         │ ├─ Reasoning trace   │
         │ ├─ Confidence metrics│
         │ ├─ Conflict alerts   │
         │ ├─ Temporal notes    │
         │ └─ Audit trail      │
         └──────────────────────┘
```

**Components of Ideal Architecture:**

1. **Task Decomposer** - Break complex queries into sub-tasks
2. **Multi-Agent System:**
   - Research Agent (retrieval + ranking)
   - Verification Agent (citation + NLI + authority)
   - Synthesis Agent (conflict detection + cross-ref)
3. **Hybrid Retrieval:**
   - Domain-specific embeddings
   - BM25 + semantic search
   - Legal-aware reranking
   - Knowledge graph traversal
4. **Long-Context Optimization:**
   - Portfolio-level reasoning
   - Deal history tracking
   - Cross-document analysis
5. **Knowledge Graphs:**
   - Jurisdiction hierarchy
   - Precedent relationships
   - Statute amendment chain
   - Conflict detection network
6. **Model Routing:**
   - Extended reasoning (o1) for complex analysis
   - Fast models (Haiku) for retrieval
   - Specialized models for verification
7. **Confidence & Auditability:**
   - Public hallucination metrics
   - Source linking
   - Conflict flagging
   - Temporal tracking

---

## COMPARISON MATRIX: FEATURE COMPLETENESS

```
Feature                  Lexintel  Harvey  CoCounsel  Luminance  Ideal
────────────────────────────────────────────────────────────────────────
Hybrid Search            ✓         ✓       ✓          ✓          ✓
Domain Embeddings        ✗         ✓       ✓          ✓          ✓
Cross-Encoder Reranking  ✓         ✓       ✓          ✓          ✓
Authority Weighting      ✓         ✓       ✓          ✓          ✓
Temporal Filtering       ✓         Implicit Implicit   Implicit   ✓
Multi-Agent Orchestr.    ✗         ✓       ✓          Implicit   ✓
Knowledge Graph          ✗         ✗       ✓(implicit) ✓(implicit) ✓
Citation Verification    ✓         Implicit ✓         ✓          ✓
Conflict Detection       ✓         Implicit Implicit   Implicit   ✓
Long-Context Optimize    Partial   ✓       ✓          ✓          ✓
Portfolio Memory         ✗         ✗       ✗          ✓          ✓
Model Routing            ✗         ✓       ✓          ✗          ✓
Confidence Metrics       Partial   ✗       Partial    ✓          ✓
Audit Trail/Tracing      ✓         ✓       ✓          ✓          ✓
────────────────────────────────────────────────────────────────────────
Total Features (14)      9         11      11         11          14
────────────────────────────────────────────────────────────────────────
```

---

## ROADMAP: LEXINTEL → IDEAL

```
CURRENT STATE (Mar 2026)
├─ Hybrid search ✓
├─ Cross-encoder reranking ✓
├─ Authority weighting ✓
├─ Temporal filtering ✓
├─ Citation verification ✓
├─ Conflict detection ✓
└─ Missing: Agents, domain embeddings, KG, long-context, memory

       ↓ (1-2 months)

PHASE 1: FOUNDATIONS
├─ Domain embeddings (fine-tuned or Voyage) ✓
├─ Explicit hallucination checker ✓
├─ Knowledge graph prototype ✓
└─ Multi-model routing logic ✓

       ↓ (2-3 months)

PHASE 2: AGENTIC LAYER
├─ Basic agent framework ✓
├─ Research agent (retrieval decomposition) ✓
├─ Verification agent (citation + NLI + authority) ✓
└─ Synthesis agent (conflict + cross-ref) ✓

       ↓ (3-4 months)

PHASE 3: SCALE & MEMORY
├─ Long-context optimization ✓
├─ Portfolio-level reasoning ✓
├─ Deal history tracking ✓
└─ Cross-matter precedent graph ✓

       ↓ (4-6 months)

IDEAL STATE
├─ All 14 features
├─ Agentic + verification
├─ Enterprise-ready
└─ Competitive with Harvey, CoCounsel, Luminance
```

---

**End of architecture comparison**
