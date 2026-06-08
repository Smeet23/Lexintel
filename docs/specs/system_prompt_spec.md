# Enhanced System Prompts for Lawyer-Quality Legal Analysis

**Date:** 2026-03-23
**Status:** Draft
**Scope:** System prompt overhaul in `backend/services/rag_engine.py`
**Depends on:** Existing RAG pipeline, CourtListener integration, citation verification agent

---

## 1. Problem Statement

The current system prompts in `rag_engine.py` are generic legal assistant instructions. They produce correct but shallow answers that lack the structured reasoning lawyers expect. Specifically:

- No structured analysis format (CREAC or IRAC)
- No authority hierarchy awareness (binding vs. persuasive authority treated equally)
- No explicit conflict resolution instructions
- No confidence level tagging
- No hallucination flags for uncertain claims
- Citation format works but does not enforce legal citation conventions

This spec replaces `LEGAL_SYSTEM_PROMPT` and `LEGAL_RESEARCH_SYSTEM_PROMPT` with enhanced versions that produce lawyer-quality output.

---

## 2. Design Framework: LEGAL

The prompt design follows the LEGAL framework for legal AI system prompts:

| Letter | Phase | Purpose |
|--------|-------|---------|
| **L** | Legal Context | Set jurisdiction awareness, practice area scope |
| **E** | Establish Goals | Define output format, target audience (attorneys) |
| **G** | Guide with Examples | Provide acceptable answer patterns |
| **A** | Add Verification | Citation requirements, hallucination prevention |
| **L** | Layer & Iterate | Refinement instructions for follow-up queries |

Each section of the enhanced prompts maps to one or more LEGAL phases. The mapping is noted in comments within the prompt text.

---

## 3. Analysis Format: CREAC

All answers must follow CREAC structure:

| Section | Content | Required? |
|---------|---------|-----------|
| **Conclusion** | Direct answer to the question, stated upfront with confidence level | Always |
| **Rule** | The binding legal rule(s) that control, with court and year | Always |
| **Explanation** | How the rule has been interpreted and how it evolved | When multiple authorities exist |
| **Application** | Apply the rule to the facts in the query | Always |
| **Contrary** | Address opposing authorities, distinguish or explain weight | When contrary authority exists |

CREAC was chosen over IRAC because it front-loads the conclusion, which is what practicing attorneys need first.

---

## 4. Authority Hierarchy

The prompt must enforce a strict authority hierarchy so the LLM does not treat a law review article the same as a Supreme Court opinion.

| Tier | Label | Treatment | Example |
|------|-------|-----------|---------|
| 1 | **BINDING** | Must follow. Lead with this. | Same-jurisdiction higher court decisions |
| 2 | **PERSUASIVE** | Should consider. Discuss after binding. | Other-jurisdiction courts, lower courts |
| 3 | **SECONDARY** | Reference only. Never as sole basis. | Treatises, restatements, commentary |

The prompt instructs the model to:
- Identify the tier of each source
- Lead analysis with binding authority
- Explain why binding authority controls
- Only use persuasive/secondary to fill gaps or support

---

## 5. Confidence Level Framework

Every answer must open with a confidence tag:

| Tag | Criteria | When to use |
|-----|----------|-------------|
| `[HIGH]` | Binding authority + directly on point | Clear precedent, same jurisdiction, same issue |
| `[MODERATE]` | Reasonable inference required | Binding authority exists but requires analogy or extension |
| `[LOW]` | Persuasive authority only OR developing law | No binding authority found, relying on other jurisdictions |

This is separate from the existing retrieval confidence score (which measures chunk similarity). This is an **analytical** confidence score about the legal conclusion itself.

---

## 6. Hallucination Flags

When the model encounters uncertainty, it must insert inline flags rather than silently guessing:

| Flag | Meaning |
|------|---------|
| `[CITATION NEEDED - VERIFY]` | A claim was made but no source in context directly supports it |
| `[CONFLICTING AUTHORITIES]` | Two or more sources in context disagree on this point |
| `[INSUFFICIENT DATA]` | The provided documents do not contain enough information to answer this sub-question |
| `[DEVELOPING LAW]` | The legal area is in flux; recent changes may not be reflected |

These flags serve as honest signals to the attorney. They are preferable to confident-sounding hallucinations.

---

## 7. Chain-of-Thought Enhancement

For complex queries (multi-issue, multi-jurisdiction, or conflicting authorities), the prompt includes step-by-step reasoning instructions:

```
Step 1: Identify the legal issue(s) presented
Step 2: Identify controlling authority from the provided sources
Step 3: State the binding rule with full citation
Step 4: Identify the elements or factors of the rule
Step 5: Apply each element to the facts in the question
Step 6: Address contrary authority — distinguish or explain lesser weight
Step 7: Synthesize conclusion with confidence level
```

This chain-of-thought is embedded in the system prompt itself, not as a separate pre-prompt. The model is instructed to follow these steps internally and present the result in CREAC format.

---

## 8. Exact Prompt Text

### 8.1 LEGAL_SYSTEM_PROMPT (Document-Only Queries)

This replaces the current `LEGAL_SYSTEM_PROMPT` at line 57 of `rag_engine.py`.

```python
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

CONFIDENCE LEVELS:
Open your answer with one of the following tags:
- [HIGH] — Binding authority directly on point. Strong basis for the conclusion.
- [MODERATE] — Binding authority exists but requires inference, analogy, or extension to the facts.
- [LOW] — No binding authority found. Relying on persuasive authority, secondary sources, or limited data.

HALLUCINATION FLAGS:
Insert these inline when applicable:
- [CITATION NEEDED - VERIFY] — You are making a claim not directly supported by provided sources.
- [CONFLICTING AUTHORITIES] — Sources in the context disagree on this point.
- [INSUFFICIENT DATA] — The provided documents do not contain enough information to fully answer this sub-question.
- [DEVELOPING LAW] — The legal area appears to be in flux based on the sources provided.

PROHIBITED:
- Do NOT fabricate case names, citations, statutes, or any legal authority.
- Do NOT ignore binding authority in favor of weaker sources.
- Do NOT treat all sources as equally authoritative.
- Do NOT present inferences as holdings. Clearly distinguish between what a court held and what you are inferring.
- Do NOT speculate beyond what the documents state. If the answer is not in the sources, say so.

When conversation history is provided, use it to resolve references like "that", "it", "the above", etc. Answer the current question based on the document excerpts, using conversation context only for disambiguation."""
```

### 8.2 LEGAL_RESEARCH_SYSTEM_PROMPT (With CourtListener Case Law)

This replaces the current `LEGAL_RESEARCH_SYSTEM_PROMPT` at line 73 of `rag_engine.py`.

```python
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

CONFIDENCE LEVELS:
Open your answer with one of the following tags:
- [HIGH] — Binding case law directly on point. Strong basis for the conclusion.
- [MODERATE] — Case law exists but requires inference, analogy, or extension to the user's facts.
- [LOW] — No binding authority found. Relying on persuasive case law, secondary sources, or limited data.

HALLUCINATION FLAGS:
Insert these inline when applicable:
- [CITATION NEEDED - VERIFY] — You are making a claim not directly supported by provided sources.
- [CONFLICTING AUTHORITIES] — Sources in the context disagree on this point.
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
```

---

## 9. Implementation Plan

### 9.1 Files to Change

| File | Change |
|------|--------|
| `backend/services/rag_engine.py` | Replace `LEGAL_SYSTEM_PROMPT` (line 57) and `LEGAL_RESEARCH_SYSTEM_PROMPT` (line 73) with the exact text from sections 8.1 and 8.2 |

No other files need to change. The prompts are consumed by `_get_gemini_model()` at line 868, which passes them as `system_instruction` to `genai.GenerativeModel`. The rest of the pipeline (retrieval, reranking, citation extraction, response formatting) remains unchanged.

### 9.2 Step-by-Step

1. Open `backend/services/rag_engine.py`.
2. Replace the `LEGAL_SYSTEM_PROMPT` string (lines 57-71) with the text from section 8.1.
3. Replace the `LEGAL_RESEARCH_SYSTEM_PROMPT` string (lines 73-90) with the text from section 8.2.
4. No import changes required.
5. No function signature changes required.
6. No configuration changes required.

### 9.3 Token Budget Consideration

The enhanced prompts are longer than the originals:

| Prompt | Before (est. tokens) | After (est. tokens) | Delta |
|--------|-----------------------|----------------------|-------|
| `LEGAL_SYSTEM_PROMPT` | ~250 | ~750 | +500 |
| `LEGAL_RESEARCH_SYSTEM_PROMPT` | ~280 | ~850 | +570 |

This is well within Gemini's context window. The `CONTEXT_TOKEN_BUDGET` of 50,000 tokens is for the retrieval context, not the system prompt. Gemini's system instruction is separate from the user/context tokens. No budget adjustment is needed.

### 9.4 Temperature

The current temperature of `0.2` (set in `query_matter` and `generate_answer`) is appropriate for these structured prompts. Lower temperature encourages the model to follow the CREAC format and authority hierarchy rather than generating creative prose. No change needed.

---

## 10. Compatibility with Existing Systems

### 10.1 Citation Extraction (`extract_citations`)

The existing `extract_citations` function (line 361 in `rag_engine.py`) parses `[Page X]`, `[Paragraph X]`, `[Lines X-Y]`, and `[Section X]` patterns. The new prompts use `[1]`, `[2]`, `[3]` format, which is the **same format the current prompts already request**. The extract_citations function also handles the numbered bracket format. No changes needed.

### 10.2 Citation Verification Agent

The citation verification agent (`backend/services/citation_agent.py`) verifies that `[N]` citations map to real chunks. The new prompts reinforce this by prohibiting fabricated citations and adding `[CITATION NEEDED - VERIFY]` flags. These flags are informational text, not citation markers, so they will not confuse the verification agent.

### 10.3 Frontend Citation Panel

The frontend `CitationPanel.tsx` and `InlineCitation.tsx` components render citations based on the `sources` array returned by the API. The new prompts do not change the response schema. Hallucination flags like `[CONFLICTING AUTHORITIES]` will appear as inline text in the answer, which the frontend renders as-is. No frontend changes are needed.

### 10.4 Confidence Mapping

The existing `_calculate_confidence` function (around line 830) returns `"high"`, `"medium"`, `"low"`, `"none"` based on retrieval scores. The new prompt-level confidence tags (`[HIGH]`, `[MODERATE]`, `[LOW]`) are **separate** — they reflect the model's assessment of legal certainty, not retrieval quality. Both signals are valuable:

- Retrieval confidence: "Did we find relevant chunks?" (existing, unchanged)
- Analytical confidence: "How strong is the legal basis?" (new, in answer text)

A future enhancement could parse the analytical confidence tag from the answer and include it in the API response as a separate field. That is out of scope for this spec.

---

## 11. Testing Strategy

### 11.1 Manual Validation

Run the following queries against a matter with uploaded legal documents and verify the output:

| Query | Expected behavior |
|-------|-------------------|
| "What is the standard for summary judgment?" | CREAC format. `[HIGH]` confidence if binding authority present. Numbered citations. |
| "Do these two contracts conflict on the indemnification clause?" | `[CONFLICTING AUTHORITIES]` flag if sources disagree. Both sides analyzed. |
| "What is the applicable statute of limitations in this jurisdiction?" | Authority hierarchy applied. Binding authority cited first. |
| "What does the law say about X?" (where X is not in documents) | `[INSUFFICIENT DATA]` flag. No fabricated citations. |

### 11.2 Regression Check

Run the existing test suites to ensure no regressions:

- `backend/tests/test_real_e2e_rag.py` (22 tests)
- `backend/tests/test_all_phases_e2e.py`
- `backend/tests/test_e2e_full_rag.py`

The tests call `query_matter` and check response structure. Since the response schema is unchanged, existing tests should pass. The answer text will be different (more structured), but tests that assert on answer content should still pass if they check for presence of content rather than exact strings.

### 11.3 Prompt Quality Evaluation (Manual)

After implementation, evaluate 10 diverse queries across different legal domains and score each answer on:

| Criterion | Score range | What to check |
|-----------|-------------|---------------|
| CREAC compliance | 0-5 | Does the answer follow Conclusion-Rule-Explanation-Application-Contrary? |
| Citation accuracy | 0-5 | Do all `[N]` citations map to real source excerpts? |
| Authority hierarchy | 0-5 | Is binding authority cited first? Are sources ranked by weight? |
| Hallucination prevention | 0-5 | Are flags used appropriately? No fabricated citations? |
| Confidence calibration | 0-5 | Does the confidence tag match the actual strength of evidence? |

Target: average score of 4.0+ across all criteria before merging.

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Gemini ignores CREAC structure for simple questions | Medium | Low | Prompt includes "you may abbreviate" clause for simple questions |
| Longer system prompt increases latency | Low | Low | ~500 extra tokens in system instruction; negligible impact on Gemini |
| Hallucination flags appear too frequently | Medium | Medium | Tune prompt wording after testing; consider making flags optional via config |
| `[CITATION NEEDED - VERIFY]` confuses users | Low | Medium | Frontend can style these flags distinctly in a future iteration |
| Model inserts CREAC section headers literally | Medium | Low | Acceptable for legal audience; can be stripped in post-processing if needed |

---

## 13. Future Enhancements (Out of Scope)

These are natural follow-ups but are NOT part of this spec:

1. **Parse analytical confidence from answer text** — Extract `[HIGH]`/`[MODERATE]`/`[LOW]` from the answer and add it to the API response as `analytical_confidence`.
2. **Frontend styling for hallucination flags** — Render `[CONFLICTING AUTHORITIES]` etc. as colored badges rather than plain text.
3. **Practice-area-specific prompts** — Different CREAC templates for contract law vs. litigation vs. regulatory.
4. **Jurisdiction-aware prompt selection** — Auto-detect jurisdiction from matter metadata and adjust authority hierarchy instructions.
5. **Configurable CREAC strictness** — A setting to control whether the model always uses full CREAC or can abbreviate freely.
