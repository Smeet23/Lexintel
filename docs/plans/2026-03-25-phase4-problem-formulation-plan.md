# Phase 4: Problem Formulation Engine (Deepened)

**Created:** 2026-03-25
**Deepened:** 2026-03-25 (7 parallel research + review agents)
**Status:** Ready for implementation
**Priority:** P0 — Highest remaining engineering phase
**Estimated Duration:** 1 week (revised from 3-4 weeks after simplification review)

## Enhancement Summary

**Research agents used:** best-practices-researcher (issue spotting), best-practices-researcher (CREAC classification), framework-docs-researcher (per-issue retrieval), framework-docs-researcher (frontend UI), architecture-strategist, performance-oracle, code-simplicity-reviewer

**Key simplifications from review:**
1. Dropped `schema_organizer.py` — existing CREAC system prompt already structures output
2. Dropped `OrganizedResultsView.tsx` — LLM text carries the organization
3. Dropped `IssuePanel.tsx` interactive confirmation — show issues as read-only badges
4. Collapsed 8 Pydantic models to 2
5. Collapsed 4 feature flags to 1
6. Dropped per-issue retrieval from MVP (add later if data demands it)
7. Dropped `CompetingInterpretation` model — covered by secondary issues
8. Dropped `LegalDomain` enum — LLM generates domain labels naturally

**Estimated LOC: ~200 new (down from ~1,050)**

## Problem Statement

When a client says "My business partner won't show me financials," an expert lawyer identifies: fiduciary duty breach, shareholder oppression, access to records rights. Current Lexintel retrieves chunks based on raw query text — missing domain-specific authorities and multi-issue decomposition.

## Solution: One Service, One LLM Call

### Architecture

```
User Query
    |
    +-- [embed_query()]          ← Existing (runs in parallel)
    |
    +-- [identify_issues()]      ← NEW: 1 Groq call, ~300ms
    |       |
    |       +-- Primary issue + confidence
    |       +-- Secondary issues (max 3)
    |       +-- Missing information flags
    |
    +-- [retrieve_chunks()]      ← Existing (uses raw query embedding)
    |
    +-- [rerank_chunks()]        ← Existing
    |
    +-- [format_legal_context()] ← MODIFIED: inject issue list into prompt
    |
    +-- [generate_answer()]      ← Existing Gemini/Groq (CREAC per-issue)
    |
    +-- [verify]                 ← Existing
```

**Key design decisions (from review agents):**

1. **Formulation runs IN PARALLEL with embedding** (asyncio.gather) — zero added latency on critical path
2. **Single combined retrieval** — no per-issue retrieval in V1. Issue list injected into system prompt so LLM structures per-issue CREAC
3. **No schema organizer** — the LLM already produces CREAC structure. Issue identification gives it the structure to organize around
4. **Formulation only in rag_engine.py** — NOT duplicated in agentic_rag.py (per architecture review)
5. **Free-text domain labels** — no enum constraint (per simplicity review)

### Pydantic Schemas (2 models, not 8)

```python
class IdentifiedIssue(BaseModel):
    domain: str                   # free-text: "contract", "tort", "corporate", etc.
    issue: str                    # "duty of care standard"
    legal_question: str           # precise research question
    confidence: float             # 0.0-1.0
    key_facts: list[str]          # facts triggering this issue

class IssueAnalysis(BaseModel):
    issues: list[IdentifiedIssue] = Field(max_length=5)  # cap at 5 issues
    missing_information: list[str] = Field(default_factory=list)
```

### Research Insights

**LLM Issue Spotting Accuracy (from LegalBench):**
- GPT-4 on issue-spotting: 70.6-95%+ (varies by domain)
- Best domains: immigration (95%+), estate (95%+)
- Worst: torts (70.6%)
- Zero-shot Groq/Llama 3.3 70B: expected 70-80% (sufficient for suggestion, users confirm via follow-up)

**Why no domain enum:** The UK study (2025) achieved 87.13% F1 using free-text classification. Constrained enums reduce accuracy because the LLM forces classification into pre-defined buckets. Free-text with post-hoc bucketing for UI color is more accurate.

**Why no schema organizer:** LAMUS benchmark shows Rule vs Application slot accuracy = 46% F1 — the hardest boundary in legal text classification. The LLM does this implicitly during generation (70-76% CoT accuracy) far better than an explicit classification step.

**Prompt engineering for multi-issue detection:** Instruct the LLM: "List ALL legal issues, even low-confidence ones. It is better to over-identify than under-identify." This yields 80%+ recall on secondary issues.

**Confidence calibration:** LLM self-reported confidence is NOT well-calibrated for legal tasks. Use confidence as relative ranking (primary vs secondary) not absolute threshold.

### Performance Budget (from performance oracle)

| Step | Latency | Notes |
|------|---------|-------|
| identify_issues() | 300-500ms | Groq Llama 3.3 70B, ~500 output tokens |
| embed_query() | 100-200ms | Cohere/Voyage (runs in parallel) |
| **Net added latency** | **0ms** | Parallel execution hides formulation behind embedding |

**Gemini fallback:** 1-1.5s (acceptable since it replaces the parallel slot)

### Graceful Degradation (from architecture review)

| Failure | Behavior |
|---------|----------|
| Groq call fails | Proceed with raw query, no issue metadata |
| Groq returns garbage | Validate with Pydantic, discard if invalid |
| All issue confidence < 0.3 | Proceed with raw query, include as "suggestions" |
| `problem_formulation_enabled=False` | Skip entirely, existing pipeline unchanged |

## Implementation

### Files to Create (1)

**`backend/services/problem_formulation.py`** (~150 lines)
- `identify_issues(query: str) -> Optional[IssueAnalysis]` — main entry
- Uses `_fast_llm_json` pattern from agentic_rag.py (Groq + Gemini fallback)
- Prompt: legal issue spotter with structured JSON output
- Returns None on failure (non-blocking)

### Files to Modify (5)

**`backend/schemas.py`** — Add `IdentifiedIssue`, `IssueAnalysis`

**`backend/config.py`** — Add `problem_formulation_enabled: bool = True`

**`backend/services/rag_engine.py`** — Step 2.5:
```python
# Run formulation in parallel with embedding (zero added latency)
issue_analysis = None
if settings.problem_formulation_enabled:
    query_embedding, issue_analysis = await asyncio.gather(
        asyncio.to_thread(embed_query, query),
        identify_issues(query),
    )
else:
    query_embedding = embed_query(query)
```
Then inject issues into system prompt before generation:
```python
if issue_analysis and issue_analysis.issues:
    issue_context = "\n".join(
        f"- {i.domain.upper()}: {i.legal_question} (confidence: {i.confidence:.0%})"
        for i in issue_analysis.issues
    )
    # Prepend to context so LLM structures CREAC per-issue
    formatted_context = f"IDENTIFIED LEGAL ISSUES:\n{issue_context}\n\n{formatted_context}"
```
Return `issue_analysis` in response dict.

**`frontend/lib/types.ts`** — Add:
```typescript
interface IdentifiedIssue {
  domain: string
  issue: string
  legalQuestion: string
  confidence: number
  keyFacts: string[]
}

interface IssueAnalysis {
  issues: IdentifiedIssue[]
  missingInformation: string[]
}
```

**`frontend/components/ChatPanel.tsx`** — Add issue badges above answer:
- Small colored badges showing identified domains
- "Missing info" prompt if `missingInformation` is non-empty
- Same pattern as existing VerificationBar (30-50 lines)

### Issue Spotting Prompt (from research)

```
You are a legal issue spotter. Given facts or a legal question, identify ALL legal issues.

INSTRUCTIONS:
1. Read the facts. Identify every legally significant fact.
2. Map facts to legal triggers (e.g., "fired" + "pregnant" = employment discrimination).
3. Identify the PRIMARY issue (highest confidence, most direct).
4. Identify SECONDARY issues (plausible but lower confidence).
5. For each issue, state the precise legal question to research.
6. Flag missing information that would change classification.
7. It is BETTER to over-identify than under-identify.
8. Cap at 5 issues total.

Do NOT fabricate legal concepts. Only identify issues genuinely supported by the facts.

Return JSON: {"issues": [...], "missing_information": [...]}
```

## Success Metrics

- Issue identification accuracy >= 75% on legal domain (measured by user follow-up refinement rate)
- Zero added latency (parallel execution with embedding)
- Answer quality improvement visible in CREAC structure when issues are injected

## Future Enhancements (NOT in V1)

These are explicitly deferred based on review agent recommendations:

| Enhancement | When to Add | Trigger |
|------------|-------------|---------|
| Per-issue retrieval | When single-query retrieval precision < 60% on multi-issue queries | Measure precision first |
| Schema organizer (CREAC slot classification) | When users report answers lack structure despite issue injection | User feedback |
| Interactive issue confirmation | When > 20% of queries have wrong primary issue | Measure error rate first |
| Domain taxonomy enum | When you need filtering/aggregation by domain | Analytics requirement |
| Competing interpretation detection | When opposing counsel simulation is a product feature | Product decision |

## References (from research agents)

- LegalBench (Stanford): 162 tasks, 70-95% LLM accuracy on issue spotting
- LAMUS (March 2026): 2.9M labeled legal sentences, Rule vs Application F1=0.46
- FOLIO ontology: 18K+ concepts, Python library available (future integration)
- LIST taxonomy (Stanford): 1,100+ civil legal issues, open source
- UK legal classification study (2025): 87.13% F1 with free-text LLM classification
- BatchPrompt (ICLR 2024): Batch size 8-12 optimal for legal classification
- "Incorporating Legal Structure in RAG" (2025): Per-factor retrieval yields 8x authority score improvement
- Groq Llama 3.3 70B: 275 tok/s output, $0.59/M input + $0.79/M output
- Interface Design for Legal Reading (arxiv): "Transparency over speed" — lawyers want provenance
