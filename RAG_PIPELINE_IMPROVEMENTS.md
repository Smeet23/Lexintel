# RAG Pipeline Improvements for LexIntel

## Executive Summary

This document outlines strategic improvements for the LexIntel legal RAG pipeline. Current system capabilities include multi-format document support (PDF, DOCX, TXT), citation grounding, retrieval reranking, and confidence scoring. The improvements below are organized by impact category and implementation complexity.

**Current State:**
- ✅ Multi-format support (PDF, DOCX, TXT)
- ✅ Citation grounding validation
- ✅ Retrieval reranking with cross-encoders
- ✅ Multi-factor confidence scoring (0.0-1.0)
- ✅ Hallucination detection
- ✅ Format-specific citations ([Page X], [Paragraph X], [Lines X-Y])
- ✅ Token budgeting and context management
- ✅ 87+ comprehensive tests

---

## Category 1: Answer Quality & Accuracy

### 1.1 Semantic Deduplication in Chunks

**Problem**: Retrieved chunks may contain redundant information, wasting context budget and producing verbose answers.

**Solution**: Deduplicate semantically similar chunks before generating answer using cosine similarity threshold (e.g., > 0.85).

**Implementation**:
```python
def deduplicate_chunks(chunks: List[Dict], similarity_threshold: float = 0.85) -> List[Dict]:
    """Remove semantically similar chunks to preserve context budget"""
    # Calculate embeddings for chunk content
    # Compute pairwise similarities
    # Remove duplicates, keep highest relevance scores
    # Return deduplicated list
```

**Impact**:
- Better context utilization
- Cleaner, more concise answers
- Lower API costs
- More focused citations

**Effort**: Medium (2-3 hours)

**Dependencies**: Existing embeddings infrastructure

**Integration Point**: `query_case()` in `rag_engine.py` after reranking

**Tests Needed**:
- [ ] Identical chunks deduplicated
- [ ] Similar chunks (0.85+ similarity) removed
- [ ] Low similarity chunks (< 0.85) retained
- [ ] Highest relevance chunk kept
- [ ] Empty list handled gracefully

---

### 1.2 Query Expansion & Reformulation

**Problem**: User queries may not perfectly match document terminology (e.g., "payment schedule" vs. "installment plan"). Standard vector search misses these variations.

**Solution**: Automatically expand queries with legal synonyms and related terms before vector search, boosting recall.

**Implementation**:
```python
def expand_query(query: str) -> List[str]:
    """Generate query variations for legal terminology"""
    # Define legal synonym dictionary
    # Extract key terms from query
    # Generate variations combining original + synonyms
    # Return list of expanded queries
    # Example: "payment terms" → ["payment terms", "payment obligations", "payment schedule", "installment plan"]

def expanded_vector_search(query: str, case_id: UUID, top_k: int = 10) -> List[Dict]:
    """Search using expanded queries, merge and deduplicate results"""
    # Expand query
    # Search each expanded query
    # Merge results, remove duplicates
    # Score by frequency + relevance
    # Return top_k
```

**Legal Synonym Dictionary**:
- payment terms ↔ payment obligations, payment schedule, installment plan
- liability ↔ responsibility, obligation, risk
- indemnify ↔ hold harmless, protect from loss
- terminate ↔ end, cancel, dissolve
- breach ↔ violation, default, non-performance
- contingency ↔ condition, requirement, prerequisite
- void ↔ invalid, null, unenforceable
- force majeure ↔ unforeseen circumstances, act of God

**Impact**:
- Better recall on synonym-heavy legal documents
- Finds relevant sections user queries might miss
- More natural language matching
- Improved answer completeness

**Effort**: Medium (3-4 hours)

**Dependencies**: Legal synonym dictionary (can start with basic version)

**Integration Point**: Before `search_vectors()` in `query_case()`

**Tests Needed**:
- [ ] Query expansion generates variations
- [ ] Each variation retrieved independently
- [ ] Results merged without duplicates
- [ ] Scoring prioritizes most relevant
- [ ] Dictionary lookup handles missing terms gracefully

---

### 1.3 Multi-Hop Reasoning

**Problem**: Complex legal questions often require combining information across multiple document sections. Current system returns top-K chunks without reasoning across them.

**Example Query**: "What happens if payment is late AND the document has a termination clause?"
- Requires finding: (1) late payment consequences, (2) termination conditions, (3) interaction between them

**Solution**: Detect multi-hop queries and iteratively retrieve supporting information.

**Implementation**:
```python
def detect_multi_hop_query(query: str) -> bool:
    """Detect queries requiring multi-hop reasoning"""
    # Look for logical connectors: "and", "if...then", "as a result of", "in addition to"
    # Look for comparative structures: "both...and", "between...and"
    # Look for consequence terms: "happens when", "results in", "causes"
    # Return True if multi-hop pattern detected

def multi_hop_retrieval(query: str, case_id: UUID, hops: int = 2) -> List[Dict]:
    """Iteratively retrieve supporting information"""
    # Hop 1: Retrieve chunks for main query
    # Analyze chunks, identify gaps or required connections
    # Hop 2: Formulate follow-up queries for missing information
    # Retrieve additional supporting chunks
    # Return merged, deduplicated results
    # Optional Hop 3+: Continue if needed
```

**Impact**:
- Handle complex legal reasoning questions
- More complete context for interconnected clauses
- Better answers for conditional scenarios
- Improved for documents with cross-references

**Effort**: High (6-8 hours)

**Dependencies**: LLM for reasoning about gaps

**Integration Point**: New function called from `query_case()` based on query type

**Tests Needed**:
- [ ] Multi-hop query detection accurate
- [ ] Single-hop queries don't trigger multi-hop
- [ ] First hop retrieves main information
- [ ] Gap analysis identifies missing connections
- [ ] Follow-up queries formulated correctly
- [ ] Results merged without losing information
- [ ] Token budget not exceeded

---

### 1.4 Hallucination Detection Enhancement

**Problem**: Current hallucination detection is basic (only token/claim counting). Can't catch false claims that are grammatically compatible with documents.

**Solution**: Add semantic matching check - verify answer claims are actually supported by cited chunks.

**Implementation**:
```python
def enhanced_hallucination_detection(answer: str, citations: List[Dict], chunks: List[Dict]) -> Tuple[bool, float, List[str]]:
    """Enhanced detection with semantic matching"""
    has_hallucinations = False
    hallucination_score = 0.0  # 0.0 = no hallucination, 1.0 = pure hallucination
    unsupported_claims = []

    # Extract key claims from answer
    claims = extract_claims(answer)

    for claim in claims:
        # Find which citation(s) support this claim
        supporting_citations = find_supporting_citations(claim, citations)

        if not supporting_citations:
            # Claim has no citations - clear hallucination
            has_hallucinations = True
            unsupported_claims.append(claim)
            continue

        # Check if cited chunks actually support the claim
        for citation in supporting_citations:
            chunk = find_chunk_by_id(citation['chunk_id'], chunks)

            # Semantic similarity: does chunk content match claim?
            similarity = cosine_similarity(
                embed_text(claim),
                embed_text(chunk['content'])
            )

            if similarity < 0.65:  # Low semantic match despite citation
                has_hallucinations = True
                unsupported_claims.append(f"{claim} (cited section doesn't actually support it)")
                break

    hallucination_score = min(len(unsupported_claims) * 0.15, 1.0)
    return has_hallucinations, hallucination_score, unsupported_claims

def extract_claims(text: str) -> List[str]:
    """Extract key factual claims from answer using sentence boundaries and markers"""
    # Split by sentences
    # Filter out hedging language: "may", "might", "could", "appear"
    # Keep factual assertions
    # Return list of claims
```

**Impact**:
- Catch more false claims (both uncited and poorly-cited)
- More accurate hallucination scoring
- Better confidence score calibration
- Reduce misleading answers

**Effort**: Medium (3-4 hours)

**Dependencies**: Existing embedding infrastructure, claim extraction NLP

**Integration Point**: `calculate_answer_confidence()` in `rag_engine.py`

**Tests Needed**:
- [ ] Uncited claims detected as hallucinations
- [ ] Well-cited claims not flagged
- [ ] Poorly-supported citations detected
- [ ] Hedged language filtered out
- [ ] Hallucination score calibrated 0.0-1.0
- [ ] Explanation messages clear and helpful

---

## Category 2: Retrieval Quality

### 2.1 Hybrid Search (Dense + Sparse)

**Problem**: Vector search alone may miss exact keyword matches. Legal documents are keyword-heavy (specific clause names, exact legal terms).

**Solution**: Combine vector search (semantic) with BM25 (keyword-based), intelligently merge results.

**Implementation**:
```python
# Requires adding BM25 index to Qdrant or separate BM25 system
# Option A: Use Qdrant's built-in BM25 support (recommended)
# Option B: Use Elasticsearch for BM25, combine with Qdrant vectors

def hybrid_search(query: str, case_id: UUID, top_k: int = 10) -> List[Dict]:
    """Hybrid search combining dense and sparse methods"""

    # Dense search (existing)
    dense_results = search_vectors(
        query_embedding=embed_text(query),
        case_id=case_id,
        top_k=top_k*1.5  # Get more candidates
    )

    # Sparse search (BM25)
    sparse_results = search_bm25(
        query=query,
        case_id=case_id,
        top_k=top_k*1.5
    )

    # Merge with weighted scoring
    merged = merge_results(dense_results, sparse_results)

    # Re-rank combined results
    reranked = rerank_chunks(query, merged, top_k=top_k)

    return reranked

def merge_results(dense: List[Dict], sparse: List[Dict]) -> List[Dict]:
    """Merge dense and sparse results with appropriate weighting"""
    # Normalize scores (dense 0-1, sparse 0-1)
    # Weight: 60% dense (semantic), 40% sparse (keyword)
    # Deduplicate by chunk_id
    # Sort by combined score
    # Return merged list
```

**Impact**:
- Better recall on keyword-heavy legal documents
- Catches exact terminology matches vector search misses
- More complete retrieval for multi-term queries
- Essential for contract clause matching

**Effort**: High (6-8 hours)

**Dependencies**:
- Qdrant with BM25 support enabled, OR
- Elasticsearch integration for BM25

**Integration Point**: Replace `search_vectors()` call in `query_case()` with hybrid search

**Architecture Change**: Minor (new search function, same output format)

**Tests Needed**:
- [ ] Dense search works independently
- [ ] Sparse search works independently
- [ ] Results merged without duplicates
- [ ] Scoring combines dense + sparse appropriately
- [ ] Reranking works on merged results
- [ ] Keyword matches prioritized when relevant
- [ ] Semantic relevance still prioritized overall

**Rollback**: Keep existing `search_vectors()`, add `hybrid_search()` as new function, switchable via config

---

### 2.2 Dynamic Top-K Selection

**Problem**: Always retrieving 10 chunks (fixed `RETRIEVAL_TOP_K = 10`) may not match query complexity. Simple queries don't need 10 chunks; complex queries might benefit from more.

**Solution**: Predict optimal K based on query characteristics and document context.

**Implementation**:
```python
def calculate_optimal_top_k(query: str, case: Case, base_k: int = 10) -> int:
    """Dynamically select retrieval top_k based on query complexity"""

    # Query complexity factors
    query_length = len(query.split())
    term_count = len(extract_legal_terms(query))
    connector_count = count_logical_connectors(query)  # "and", "or", "if"

    # Document factors
    doc_total_chunks = case.chunks.count()
    doc_type_factor = {"pdf": 1.0, "docx": 1.1, "txt": 0.9}[case.file_type]

    # Scoring
    complexity_score = (query_length * 0.3 + term_count * 0.4 + connector_count * 0.3)

    # Calculate K
    if complexity_score < 0.5:
        optimal_k = max(3, base_k // 2)  # Simple query
    elif complexity_score < 1.0:
        optimal_k = base_k  # Standard query
    else:
        optimal_k = min(base_k * 1.5, doc_total_chunks)  # Complex query

    return int(optimal_k)

# Usage in query_case()
optimal_k = calculate_optimal_top_k(query, case)
chunks = search_vectors(embedding, case_id, top_k=optimal_k)
```

**Impact**:
- Better resource utilization
- Faster processing for simple queries
- More context for complex queries
- Better token budget management

**Effort**: Medium (4-5 hours)

**Dependencies**: Legal term extraction, connector detection

**Integration Point**: Before `search_vectors()` in `query_case()`

**Tests Needed**:
- [ ] Simple queries get K=3-5
- [ ] Standard queries get K=10
- [ ] Complex queries get K=15+
- [ ] K doesn't exceed document chunk count
- [ ] Performance improved for simple queries
- [ ] Quality maintained for complex queries

---

### 2.3 Chunk Relevance Confidence Threshold

**Problem**: Current system uses fixed `MIN_CONFIDENCE_SCORE = 0.6`, including weakly relevant chunks. Some queries are harmed by weak context.

**Solution**: Dynamic threshold based on document length, query type, and retrieval performance.

**Implementation**:
```python
def calculate_relevance_threshold(query: str, case: Case, retrieval_stats: Dict) -> float:
    """Calculate dynamic confidence threshold"""

    # Base threshold
    base_threshold = 0.60

    # Document length factor (longer docs can afford lower threshold)
    doc_length = sum(len(chunk.content) for chunk in case.chunks)
    length_factor = 0.0
    if doc_length < 10_000:
        length_factor = 0.05  # Short docs: raise threshold
    elif doc_length > 50_000:
        length_factor = -0.05  # Long docs: lower threshold

    # Query specificity factor
    specific_terms = count_specific_legal_terms(query)
    specificity_factor = 0.05 if specific_terms > 3 else -0.02

    # Retrieval quality factor (if all top chunks are high confidence, can lower threshold)
    avg_confidence = retrieval_stats.get('avg_confidence', 0.75)
    quality_factor = 0.0
    if avg_confidence > 0.85:
        quality_factor = -0.05  # High quality results, can be more inclusive
    elif avg_confidence < 0.70:
        quality_factor = 0.10  # Low quality results, be more selective

    threshold = base_threshold + length_factor + specificity_factor + quality_factor
    return max(0.55, min(0.75, threshold))  # Clamp between 0.55-0.75

# Usage
threshold = calculate_relevance_threshold(query, case, retrieval_stats)
relevant_chunks = [c for c in chunks if c['score'] >= threshold]
```

**Impact**:
- Reduce noise from weakly relevant chunks
- Improve answer quality
- Better for specific queries
- More forgiving for broad queries

**Effort**: Low (2-3 hours)

**Dependencies**: None (uses existing scores)

**Integration Point**: After `search_vectors()` in `query_case()`

**Tests Needed**:
- [ ] Short documents get higher threshold
- [ ] Long documents get lower threshold
- [ ] Specific queries get higher threshold
- [ ] Weak retrieval results get higher threshold
- [ ] Strong retrieval results get lower threshold
- [ ] Threshold always 0.55-0.75

---

### 2.4 Query-Document Specific Chunk Weighting

**Problem**: All retrieved chunks weighted equally regardless of document structure. Some sections are legally critical (operative clauses) vs. informational (preamble).

**Solution**: Weight chunks differently based on legal significance and document structure.

**Implementation**:
```python
def identify_chunk_section_type(chunk: Dict, case: Case) -> str:
    """Identify if chunk is critical clause, definition, preamble, etc."""

    content = chunk['content'].lower()
    location = chunk['page_num']

    # Preamble: early in document, background info
    if location in ['1', 'para 1', 'line 1-50']:
        if any(word in content for word in ['hereby', 'whereas', 'background', 'recital']):
            return 'preamble'

    # Critical clauses: payment, liability, termination, indemnity
    if any(word in content for word in ['payment', 'liability', 'terminate', 'indemnif']):
        return 'critical_clause'

    # Definitions section
    if 'shall mean' in content or 'shall include' in content:
        return 'definition'

    # Standard terms
    return 'standard'

def weight_chunks_by_importance(chunks: List[Dict], case: Case, query: str) -> List[Dict]:
    """Boost score for chunks matching query context and legal importance"""

    weighted = []
    query_lower = query.lower()

    # Weights for each section type
    weights = {
        'critical_clause': 1.30,      # +30% boost
        'definition': 1.10,            # +10% boost
        'standard': 1.00,              # No change
        'preamble': 0.85               # -15% penalty
    }

    for chunk in chunks:
        section_type = identify_chunk_section_type(chunk, case)
        weight = weights.get(section_type, 1.0)

        # Additional boost if section type matches query intent
        if section_type == 'critical_clause' and any(word in query_lower for word in ['payment', 'liability', 'terminate']):
            weight *= 1.15

        chunk['original_score'] = chunk['score']
        chunk['score'] = chunk['score'] * weight
        chunk['section_type'] = section_type
        weighted.append(chunk)

    # Re-sort by weighted score
    return sorted(weighted, key=lambda x: x['score'], reverse=True)

# Usage
chunks = search_vectors(...)
chunks = weight_chunks_by_importance(chunks, case, query)
chunks = rerank_chunks(query, chunks, top_k=FINAL_CHUNK_COUNT)
```

**Impact**:
- Prioritize legally critical sections
- Reduce noise from informational preambles
- Better for queries about specific obligations
- Aligns with legal document structure

**Effort**: Medium (4-5 hours)

**Dependencies**: Section type detection (can start simple)

**Integration Point**: After `search_vectors()` in `query_case()`

**Tests Needed**:
- [ ] Critical clauses identified correctly
- [ ] Weights applied appropriately
- [ ] Preamble sections deprioritized
- [ ] Query-section matching works
- [ ] Scores re-sorted correctly
- [ ] Answer quality improved

---

## Category 3: Citation & Grounding

### 3.1 Multi-Chunk Citations

**Problem**: When answer draws from 2+ chunks, system cites each separately. Result: "[Page 5] ... [Page 7] ..." when both discuss same topic.

**Solution**: Consolidate citations when multiple chunks support same claim or are from same section.

**Implementation**:
```python
def consolidate_citations(answer: str, citations: List[Dict]) -> Tuple[str, List[Dict]]:
    """Consolidate multi-chunk citations for clarity"""

    # Group citations by location (page/paragraph/line range)
    location_groups = {}
    for citation in citations:
        location = citation['location']
        if location not in location_groups:
            location_groups[location] = []
        location_groups[location].append(citation)

    # Determine citation format
    # If all from same location: [Page 5, 7, 9]
    # If adjacent locations: [Pages 5-9]
    # If same paragraph: [Paragraph 3]

    consolidated = []
    for location, group_citations in location_groups.items():
        consolidated_citation = {
            'location': location,
            'locations_grouped': [c['location'] for c in group_citations],
            'combined_score': max(c['relevance_score'] for c in group_citations),
            'chunk_count': len(group_citations)
        }
        consolidated.append(consolidated_citation)

    # Replace citations in answer
    # Replace [Page 5] [Page 5] → [Page 5]
    # Replace [Page 5] [Page 7] → [Pages 5, 7]
    # Replace [Page 5] [Page 6] [Page 7] → [Pages 5-7]

    consolidated_answer = consolidate_answer_citations(answer, consolidated)

    return consolidated_answer, consolidated

def consolidate_answer_citations(answer: str, consolidated_citations: List[Dict]) -> str:
    """Reformat answer with consolidated citations"""
    # Find citation patterns in answer
    # Replace with consolidated format
    # Handle edge cases (citations at sentence boundaries, etc.)
    return answer
```

**Impact**:
- Cleaner, more professional citations
- Easier to read and follow
- More concise when using same source multiple times
- Better visual presentation

**Effort**: Medium (3-4 hours)

**Dependencies**: Citation parsing, string formatting

**Integration Point**: After `extract_citations()` in `query_case()`

**Tests Needed**:
- [ ] Same-location citations consolidated
- [ ] Adjacent pages grouped as ranges
- [ ] Consolidated citations don't lose information
- [ ] Answer text formatting preserved
- [ ] Citation count accuracy maintained
- [ ] Multiple consolidation levels work

---

### 3.2 Citation Span Highlighting

**Problem**: Citations show location (page/paragraph) but user must manually find exact supporting text. Makes verification difficult.

**Solution**: Return exact text spans within chunks that support each claim.

**Implementation**:
```python
def extract_citation_spans(answer: str, citations: List[Dict], chunks: List[Dict]) -> List[Dict]:
    """Extract exact text spans supporting each citation"""

    citation_spans = []

    for citation in citations:
        chunk = find_chunk_by_id(citation['chunk_id'], chunks)
        if not chunk:
            continue

        # Find which claim this citation supports
        # Parse answer for [Location] pattern
        location_pattern = rf"\[{re.escape(citation['location'])}\]"

        # Find surrounding text (sentence containing citation)
        matches = re.finditer(location_pattern, answer)
        for match in matches:
            # Extract supporting sentence
            sentence_start = answer.rfind('.', 0, match.start()) + 1
            sentence_end = answer.find('.', match.end())
            supporting_claim = answer[sentence_start:sentence_end].strip()

            # Find matching span in chunk
            span_text = find_best_span(supporting_claim, chunk['content'])

            citation_spans.append({
                'location': citation['location'],
                'claim': supporting_claim,
                'supporting_span': span_text[:200],  # First 200 chars
                'span_start_char': chunk['content'].find(span_text),
                'confidence': citation['relevance_score']
            })

    return citation_spans

def find_best_span(claim: str, chunk_text: str, window_size: int = 300) -> str:
    """Find best matching span in chunk for a claim"""
    # Tokenize claim
    # Find best matching substring in chunk
    # Return surrounding context (±150 chars)

    # Use fuzzy matching or semantic similarity
    best_score = 0
    best_span = ""

    for i in range(0, len(chunk_text) - window_size, window_size // 2):
        span = chunk_text[i:i+window_size]
        # Calculate similarity between claim and span
        similarity = calculate_similarity(claim, span)
        if similarity > best_score:
            best_score = similarity
            best_span = span

    return best_span
```

**Usage in Response**:
```json
{
  "answer": "The contract requires payment within 30 days [Page 5].",
  "confidence": 0.87,
  "citations": [
    {
      "location": "Page 5",
      "claim": "The contract requires payment within 30 days",
      "supporting_span": "...Payment Terms: The buyer shall remit payment within thirty (30) calendar days of invoice date...",
      "confidence": 0.92
    }
  ]
}
```

**Impact**:
- User can instantly verify claims
- Build trust through transparency
- Reduces need for manual document review
- Professional legal documentation

**Effort**: Medium (4-5 hours)

**Dependencies**: Sentence/claim extraction, fuzzy matching

**Integration Point**: After `extract_citations()` in `query_case()`

**Tests Needed**:
- [ ] Claim extracted from answer correctly
- [ ] Supporting span found in chunk
- [ ] Span text is accurate and relevant
- [ ] Multiple citations handled
- [ ] Edge cases (missing spans) handled gracefully
- [ ] Span formatting correct

---

### 3.3 Confidence Calibration per Citation

**Problem**: All citations treated equally regardless of support strength. A quote is stronger than a paraphrase, but both marked the same.

**Solution**: Score each citation on "support strength" (exact quote vs. relevant section vs. related info).

**Implementation**:
```python
def calibrate_citation_strength(citation: Dict, chunk: Dict, answer: str) -> Dict:
    """Calculate confidence based on how well chunk supports citation"""

    claim = citation.get('claim', '')
    chunk_content = chunk['content']
    relevance_score = citation['relevance_score']

    # Factor 1: Exact Quote Match (100% confidence)
    if any(quote in chunk_content for quote in extract_quotes(claim)):
        citation['strength'] = 'exact_quote'
        citation['strength_score'] = 0.98
        return citation

    # Factor 2: Paraphrase Match (85% confidence)
    semantic_sim = cosine_similarity(embed_text(claim), embed_text(chunk_content))
    if semantic_sim > 0.85:
        citation['strength'] = 'paraphrase'
        citation['strength_score'] = 0.85
        return citation

    # Factor 3: Related but not Direct (70% confidence)
    if semantic_sim > 0.70:
        citation['strength'] = 'related_content'
        citation['strength_score'] = 0.70
        return citation

    # Factor 4: Weak Connection (50% confidence)
    citation['strength'] = 'weak_connection'
    citation['strength_score'] = 0.50
    return citation

def recalibrate_answer_confidence(answer: str, citations: List[Dict], chunks: List[Dict]) -> float:
    """Recalculate confidence using per-citation strength"""

    if not citations:
        return 0.0

    total_strength = 0.0
    for citation in citations:
        chunk = find_chunk_by_id(citation['chunk_id'], chunks)
        if chunk:
            calibrated = calibrate_citation_strength(citation, chunk, answer)
            total_strength += calibrated['strength_score']

    avg_strength = total_strength / len(citations)

    # Combine with existing factors
    # confidence = avg_strength * 0.40 + coverage_score * 0.30 + relevance_score * 0.30

    return avg_strength

# Usage in response
citations_with_strength = [
    calibrate_citation_strength(c, find_chunk_by_id(c['chunk_id'], chunks), answer)
    for c in citations
]

refined_confidence = recalibrate_answer_confidence(answer, citations_with_strength, chunks)
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days [Page 5].",
  "confidence": {
    "overall": 0.87,
    "citations": [
      {
        "location": "Page 5",
        "strength": "exact_quote",
        "strength_score": 0.98,
        "claim": "Payment is due within 30 days"
      }
    ]
  }
}
```

**Impact**:
- Nuanced confidence scores
- Better legal precision
- User understands certainty level per claim
- Helps identify shaky citations

**Effort**: Medium (4-5 hours)

**Dependencies**: Embedding, quote extraction

**Integration Point**: In `calculate_answer_confidence()` and response formatting

**Tests Needed**:
- [ ] Exact quotes scored 0.98+
- [ ] Paraphrases scored 0.80-0.90
- [ ] Related content scored 0.65-0.75
- [ ] Weak connections scored < 0.65
- [ ] Overall confidence recalculated correctly
- [ ] Multiple citation strengths combined properly

---

### 3.4 Missing Citation Detection

**Problem**: System can't detect when answer lacks citations for important claims. Some statements should have citations but don't.

**Solution**: Check if every significant claim in answer is cited and supported.

**Implementation**:
```python
def detect_missing_citations(answer: str, citations: List[Dict]) -> Tuple[bool, List[str], float]:
    """Detect claims in answer that lack citations"""

    # Extract all claims from answer
    claims = extract_claims(answer)

    # Extract cited content
    cited_text = " ".join([f"{c.get('claim', '')} {c.get('supporting_span', '')}"
                           for c in citations])

    uncited_claims = []

    for claim in claims:
        # Check if claim is mentioned in citations
        claim_covered = False

        for citation in citations:
            # Is this claim supported by this citation?
            similarity = cosine_similarity(embed_text(claim), embed_text(cited_text))
            if similarity > 0.75:
                claim_covered = True
                break

        if not claim_covered:
            # Check if it's a hedged/uncertain claim (allowed to be uncited)
            if not is_hedged_claim(claim):
                uncited_claims.append(claim)

    has_missing_citations = len(uncited_claims) > 0

    # Calculate missing citation severity (0.0 = none, 1.0 = all uncited)
    if not claims:
        citation_coverage = 1.0
    else:
        citation_coverage = 1.0 - (len(uncited_claims) / len(claims))

    return has_missing_citations, uncited_claims, citation_coverage

def is_hedged_claim(claim: str) -> bool:
    """Check if claim uses hedging language (allowed to be uncited)"""
    hedging_words = ['may', 'might', 'could', 'appear', 'seem', 'possibly', 'likely', 'arguably']
    return any(word in claim.lower() for word in hedging_words)

# Usage in query_case()
has_missing, missing_claims, coverage = detect_missing_citations(answer, citations)

if has_missing_citations and coverage < 0.75:
    # Flag for review or regenerate answer
    logger.warning(f"Answer has unsupported claims: {missing_claims}")
    confidence *= 0.7  # Reduce confidence
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days. The amount must be in USD.",
  "confidence": 0.65,
  "citation_issues": {
    "has_missing_citations": true,
    "uncited_claims": ["The amount must be in USD"],
    "citation_coverage": 0.50,
    "recommendation": "Answer needs verification - some claims lack supporting citations"
  }
}
```

**Impact**:
- Catch unsupported claims before returning to user
- Improve answer reliability
- Flag answers needing revision
- Better quality control

**Effort**: High (5-6 hours)

**Dependencies**: Claim extraction, similarity scoring

**Integration Point**: After `extract_citations()` in `query_case()`

**Tests Needed**:
- [ ] Well-cited answers show 100% coverage
- [ ] Uncited claims detected
- [ ] Hedged claims not flagged as missing
- [ ] Coverage percentage accurate
- [ ] Multiple uncited claims handled
- [ ] Confidence reduced appropriately

---

## Category 4: User Experience & Usability

### 4.1 Answer Confidence Explanation

**Problem**: User sees confidence score (0.87) but doesn't know why or what factors contributed.

**Solution**: Return detailed explanation of confidence factors.

**Implementation**:
```python
def explain_confidence_score(answer: str, citations: List[Dict], chunks: List[Dict],
                            has_hallucinations: bool, confidence: float) -> Dict:
    """Generate human-readable explanation of confidence score"""

    # Calculate component scores
    citation_coverage = calculate_citation_coverage(answer, citations)
    avg_relevance = calculate_average_relevance(citations)
    hallucination_factor = 0.0 if not has_hallucinations else 0.15
    citation_count = len(citations)

    explanation = {
        "overall_score": round(confidence, 2),
        "rating": classify_confidence_level(confidence),
        "factors": {
            "citation_coverage": {
                "score": round(citation_coverage, 2),
                "explanation": f"{int(citation_coverage*100)}% of answer claims are cited"
            },
            "source_relevance": {
                "score": round(avg_relevance, 2),
                "explanation": f"Sources are {format_relevance(avg_relevance)}-relevant (avg {round(avg_relevance, 2)})"
            },
            "hallucination_risk": {
                "score": round(1.0 - hallucination_factor, 2),
                "explanation": "No hallucinations detected" if not has_hallucinations else "Minor hallucinations detected"
            },
            "citation_quantity": {
                "score": min(citation_count / 4, 1.0),  # Normalize to 0-1
                "explanation": f"{citation_count} source{'s' if citation_count != 1 else ''} cited"
            }
        },
        "summary": generate_confidence_summary(citation_coverage, avg_relevance, has_hallucinations, citation_count)
    }

    return explanation

def format_relevance(score: float) -> str:
    """Convert numeric score to human-readable format"""
    if score >= 0.90:
        return "highly"
    elif score >= 0.80:
        return "well"
    elif score >= 0.70:
        return "moderately"
    else:
        return "weakly"

def generate_confidence_summary(coverage: float, relevance: float, hallucinations: bool, citation_count: int) -> str:
    """Generate text summary of confidence"""
    reasons = []

    if coverage >= 0.90:
        reasons.append("well-cited")
    elif coverage >= 0.70:
        reasons.append("mostly-cited")
    else:
        reasons.append("partially-cited")

    if relevance >= 0.85:
        reasons.append("strong sources")
    elif relevance >= 0.75:
        reasons.append("good sources")
    else:
        reasons.append("moderate sources")

    if not hallucinations:
        reasons.append("no hallucinations")

    if citation_count >= 4:
        reasons.append("multiple sources")

    return f"High confidence: {', '.join(reasons)}"

# Usage in response
confidence_explanation = explain_confidence_score(answer, citations, chunks, has_hallucinations, confidence)
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days",
  "confidence": 0.87,
  "confidence_explanation": {
    "overall_score": 0.87,
    "rating": "high",
    "factors": {
      "citation_coverage": {
        "score": 0.95,
        "explanation": "95% of answer claims are cited"
      },
      "source_relevance": {
        "score": 0.89,
        "explanation": "Sources are well-relevant (avg 0.89)"
      },
      "hallucination_risk": {
        "score": 1.0,
        "explanation": "No hallucinations detected"
      },
      "citation_quantity": {
        "score": 0.75,
        "explanation": "3 sources cited"
      }
    },
    "summary": "High confidence: well-cited, strong sources, no hallucinations"
  }
}
```

**Impact**:
- User understands answer quality basis
- Build trust through transparency
- Help users identify weak answers
- Educational value

**Effort**: Low (2-3 hours)

**Dependencies**: Existing confidence calculation functions

**Integration Point**: In response formatting, new function `explain_confidence_score()`

**Tests Needed**:
- [ ] All confidence factors calculated
- [ ] Scores normalized 0.0-1.0
- [ ] Summary text clear and accurate
- [ ] Formatting consistent
- [ ] Edge cases handled (0 citations, etc.)

---

### 4.2 Follow-up Question Suggestions

**Problem**: Users must manually craft follow-ups. System knows relevant sections but doesn't suggest natural next questions.

**Solution**: Generate contextually relevant follow-up questions from cited sections.

**Implementation**:
```python
def generate_follow_up_questions(answer: str, citations: List[Dict], chunks: List[Dict]) -> List[str]:
    """Generate suggested follow-up questions based on answer context"""

    follow_ups = []

    for citation in citations[:3]:  # Use top 3 cited sources
        chunk = find_chunk_by_id(citation['chunk_id'], chunks)
        if not chunk:
            continue

        # Find related concepts in chunk
        # Use NER to extract entities: people, organizations, obligations, amounts
        entities = extract_named_entities(chunk['content'])

        # Generate questions for each entity type
        for entity_type, entity_values in entities.items():
            if entity_type == 'obligation':
                for obligation in entity_values[:1]:
                    follow_ups.append(f"What happens if {obligation} is not met?")

            elif entity_type == 'amount':
                for amount in entity_values[:1]:
                    follow_ups.append(f"Are there any penalties or interest for amounts over {amount}?")

            elif entity_type == 'date':
                for date in entity_values[:1]:
                    follow_ups.append(f"What are the consequences if deadlines after {date} are missed?")

        # Generate clarifying question for complex concept
        complex_terms = extract_complex_legal_terms(chunk['content'])
        if complex_terms:
            term = complex_terms[0]
            follow_ups.append(f"Can you clarify what '{term}' means in this context?")

    # Deduplicate and limit
    unique_follow_ups = list(dict.fromkeys(follow_ups))[:3]

    return unique_follow_ups

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """Extract legal entities, obligations, amounts, dates"""
    # Use spaCy or pattern matching
    # Return: {'obligation': ['pay within 30 days'], 'amount': ['$1000'], 'date': ['2024-01-15']}
    pass

# Usage in response
follow_ups = generate_follow_up_questions(answer, citations, chunks)
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days of invoice",
  "suggested_follow_ups": [
    "What happens if payment is not made within 30 days?",
    "Are there any penalties or interest on late payments?",
    "Can you clarify what 'invoice date' means in this context?"
  ]
}
```

**Impact**:
- Improve user engagement
- Enable deeper document analysis
- Natural conversation flow
- Discovery of related clauses

**Effort**: Medium (3-4 hours)

**Dependencies**: NER, legal term extraction

**Integration Point**: In response formatting

**Tests Needed**:
- [ ] Follow-up questions generated
- [ ] Questions are contextually relevant
- [ ] Duplicates removed
- [ ] Multiple follow-up types generated
- [ ] Edge cases (no entities) handled

---

### 4.3 Source Document Summary

**Problem**: User may not understand what document answers came from (especially multi-document queries).

**Solution**: Return brief summary of each cited document's purpose and legal significance.

**Implementation**:
```python
def generate_document_summary(case: Case) -> Dict:
    """Generate summary of document content and legal significance"""

    summary = {
        "filename": case.name,
        "file_type": case.file_type,
        "total_pages": calculate_page_count(case),
        "key_concepts": extract_key_concepts(case),
        "legal_significance": classify_legal_significance(case),
        "processing_status": case.status,
        "processed_at": case.updated_at
    }

    return summary

def extract_key_concepts(case: Case) -> List[str]:
    """Extract main legal concepts from document"""
    # Analyze all chunks
    # Identify most frequently mentioned legal terms
    # Return top 5-7 concepts

    concept_counts = {}
    legal_terms = [
        'payment', 'liability', 'warranty', 'indemnify', 'terminate',
        'breach', 'force majeure', 'arbitration', 'governing law', ...
    ]

    for chunk in case.chunks:
        for term in legal_terms:
            if term in chunk.content.lower():
                concept_counts[term] = concept_counts.get(term, 0) + 1

    # Return top concepts
    return sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:7]

def classify_legal_significance(case: Case) -> str:
    """Classify document type (contract, license, ToS, etc.)"""
    # Analyze first chunks and overall structure
    # Return classification

    first_chunk = case.chunks.first().content.lower()

    if 'terms and conditions' in first_chunk or 'terms of service' in first_chunk:
        return 'Terms of Service'
    elif 'license' in first_chunk:
        return 'License Agreement'
    elif 'privacy' in first_chunk:
        return 'Privacy Policy'
    elif 'purchase' in first_chunk or 'sale' in first_chunk:
        return 'Purchase Agreement'
    else:
        return 'Legal Document'

# Usage in response
def add_document_context_to_response(answer: str, citations: List[Dict], case: Case) -> Dict:
    """Enhance response with document context"""

    # Get unique cases from citations
    cited_cases = set(citation.get('case_id') for citation in citations)

    documents = {}
    for case_id in cited_cases:
        case = get_case(case_id)
        if case:
            documents[case.name] = generate_document_summary(case)

    return {
        "answer": answer,
        "citations": citations,
        "source_documents": documents
    }
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days",
  "citations": [...],
  "source_documents": {
    "vendor-agreement-2024.pdf": {
      "file_type": "pdf",
      "legal_significance": "Vendor Agreement",
      "key_concepts": ["payment terms", "delivery schedule", "quality standards", "termination"],
      "total_pages": 12,
      "processed_at": "2024-01-15T10:30:00Z"
    }
  }
}
```

**Impact**:
- User understands source context
- Better for multi-document queries
- Build confidence in answers
- Educational about document types

**Effort**: Low (2-3 hours)

**Dependencies**: Existing case data, term extraction

**Integration Point**: In response formatting

**Tests Needed**:
- [ ] Document summaries generated
- [ ] Key concepts extracted accurately
- [ ] Legal significance classified
- [ ] Multiple documents handled
- [ ] Formatting consistent

---

### 4.4 Query Result Caching

**Problem**: Identical questions re-processed every time, using API quota and latency.

**Solution**: Cache answer + sources for identical queries within time window.

**Implementation**:
```python
def cache_key_from_query(query: str, case_id: UUID) -> str:
    """Generate cache key from query and case"""
    # Normalize query (lowercase, remove extra spaces)
    normalized = " ".join(query.lower().split())
    # Create hash
    return f"query:{case_id}:{hashlib.md5(normalized.encode()).hexdigest()}"

async def query_case_with_cache(
    query: str,
    case_id: UUID,
    db: Session,
    redis_client: Redis,
    cache_ttl: int = 86400  # 24 hours
) -> Dict:
    """Query with caching layer"""

    cache_key = cache_key_from_query(query, case_id)

    # Check cache
    cached = redis_client.get(cache_key)
    if cached:
        logger.info(f"Cache hit for query: {cache_key}")
        return json.loads(cached)

    # Process query normally
    result = await query_case(query, case_id, db)

    # Cache result
    redis_client.setex(cache_key, cache_ttl, json.dumps(result, default=str))
    logger.info(f"Cached query result: {cache_key}")

    return result

# Configuration
CACHE_ENABLED = True
CACHE_TTL_SECONDS = 86400  # 24 hours
CACHE_EXCLUDE_PATTERNS = []  # Patterns to skip caching
```

**Impact**:
- Faster repeat queries (instant response)
- Reduced API costs (no duplicate embeddings/LLM calls)
- Better user experience for follow-ups
- Scales with repeat users

**Effort**: Low (2-3 hours)

**Dependencies**: Redis (already in stack)

**Integration Point**: Wrapper around `query_case()` function

**Configuration**:
- `CACHE_ENABLED`: Enable/disable caching
- `CACHE_TTL_SECONDS`: How long to keep cache (default 24 hours)
- `CACHE_EXCLUDE_PATTERNS`: Query patterns to never cache

**Tests Needed**:
- [ ] Cache key generated correctly
- [ ] Identical queries return same result
- [ ] Cache expires after TTL
- [ ] Different queries not cached together
- [ ] Cache performance verified

---

## Category 5: Performance & Scalability

### 5.1 Streaming Responses

**Problem**: User waits for full answer generation (3-5 seconds). No feedback during processing.

**Solution**: Stream answer generation token-by-token to client.

**Implementation**:
```python
async def query_case_streaming(
    query: str,
    case_id: UUID,
    current_user_id: UUID,
    db: Session
) -> AsyncGenerator[str, None]:
    """Stream answer generation with real-time updates"""

    # Retrieval phase (fast)
    chunks = await retrieve_and_rerank(query, case_id, db)

    # Format context
    context = format_legal_context(chunks, case.name)

    # Stream from Google AI
    client = google.generativeai.GenerativeModel("gemini-2.5-flash-lite")

    stream = await client.generate_content_async(
        contents=f"{LEGAL_SYSTEM_PROMPT}\n\n{context}\n\nQuestion: {query}",
        stream=True,
        generation_config={"temperature": 0.1}
    )

    answer_text = ""
    async for chunk in stream:
        if chunk.text:
            token = chunk.text
            answer_text += token
            yield json.dumps({"type": "token", "content": token}) + "\n"

    # Extract citations and confidence after full answer
    citations = extract_citations(answer_text, chunks)
    grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)
    confidence = calculate_answer_confidence(answer_text, grounded, chunks, has_hallucinations=False)

    # Send final metadata
    yield json.dumps({
        "type": "complete",
        "citations": grounded,
        "confidence": confidence,
        "unsupported": unsupported
    }) + "\n"

# FastAPI endpoint
@app.post("/cases/{case_id}/ask/stream")
async def ask_streaming(
    case_id: UUID,
    request: QueryRequest,
    current_user_id: UUID = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> StreamingResponse:
    """Stream answer generation"""

    return StreamingResponse(
        query_case_streaming(request.query, case_id, current_user_id, db),
        media_type="application/x-ndjson"
    )
```

**Client-side JavaScript**:
```javascript
async function askStreaming(caseId, query) {
  const response = await fetch(`/cases/${caseId}/ask/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let answer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    const line = decoder.decode(value);
    const data = JSON.parse(line);

    if (data.type === 'token') {
      answer += data.content;
      updateAnswerDisplay(answer);
    } else if (data.type === 'complete') {
      displayCitations(data.citations);
      displayConfidence(data.confidence);
    }
  }
}
```

**Impact**:
- Perceived faster responses (feedback during processing)
- Better UX for long answers
- User sees answer appearing in real-time
- Reduced perceived latency

**Effort**: Medium (4-5 hours)

**Dependencies**: Google AI async streaming, FastAPI StreamingResponse

**Integration Point**: New endpoint alongside existing `/cases/{id}/ask`

**Tests Needed**:
- [ ] Tokens streamed correctly
- [ ] Metadata sent after completion
- [ ] Multiple concurrent streams work
- [ ] Network interruption handled
- [ ] Citation extraction works on streamed answer

---

### 5.2 Embedding Cache

**Problem**: Same document section embedded multiple times (multiple queries against same case).

**Solution**: Cache embeddings for chunks, especially for repeated queries.

**Implementation**:
```python
def get_or_create_chunk_embedding(chunk: Chunk, embedding_cache: Dict[str, np.ndarray]) -> np.ndarray:
    """Get embedding from cache or create"""

    cache_key = f"chunk:{chunk.id}"

    if cache_key in embedding_cache:
        return embedding_cache[cache_key]

    # Create embedding
    embedding = embed_text(chunk.content)
    embedding_cache[cache_key] = embedding

    return embedding

class EmbeddingCacheManager:
    """Manage in-memory embedding cache"""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, chunk_id: UUID) -> Optional[np.ndarray]:
        """Get embedding from cache"""
        key = str(chunk_id)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, chunk_id: UUID, embedding: np.ndarray) -> None:
        """Store embedding in cache"""
        key = str(chunk_id)

        # LRU eviction if full
        if len(self.cache) >= self.max_size:
            # Remove oldest (simplified)
            self.cache.pop(next(iter(self.cache)))

        self.cache[key] = embedding

    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# Usage
embedding_cache = EmbeddingCacheManager(max_size=1000)

async def search_vectors_cached(
    query_embedding: np.ndarray,
    case_id: UUID,
    top_k: int = 10,
    embedding_cache: EmbeddingCacheManager = None
) -> List[Dict]:
    """Vector search with embedding cache"""

    case = db.query(Case).filter(Case.id == case_id).first()

    # Get chunk embeddings (from cache if available)
    for chunk in case.chunks:
        if embedding_cache:
            chunk_embedding = embedding_cache.get(chunk.id)
            if chunk_embedding is None:
                chunk_embedding = embed_text(chunk.content)
                embedding_cache.put(chunk.id, chunk_embedding)
        else:
            chunk_embedding = embed_text(chunk.content)

    # Continue with vector search...
```

**Impact**:
- Faster repeated queries against same case
- Reduced API calls to embedding service
- Lower costs
- Better performance on popular cases

**Effort**: Low (2-3 hours)

**Dependencies**: None (in-memory cache)

**Integration Point**: In `search_vectors()` function

**Configuration**:
- `EMBEDDING_CACHE_SIZE`: Max chunks to cache (default 1000)
- `EMBEDDING_CACHE_ENABLED`: Enable/disable caching

**Tests Needed**:
- [ ] Cache stores embeddings
- [ ] Cache returns stored embeddings
- [ ] LRU eviction works
- [ ] Hit rate calculated correctly
- [ ] Performance improvement verified

---

### 5.3 Chunk Size Optimization

**Problem**: Fixed 1500-char chunks may not align with legal sections. Some clauses span multiple chunks; some chunks contain multiple clauses.

**Solution**: Adaptive chunking based on document structure (clauses, paragraphs, sentences).

**Implementation**:
```python
def identify_logical_boundaries(content: str, file_type: str) -> List[Tuple[int, int]]:
    """Identify natural chunk boundaries based on document structure"""

    boundaries = []

    if file_type == 'pdf':
        # PDF: split on page breaks, then section headers
        sections = re.split(r'\n\s*(?=Section|Article|Clause|§)', content)
        pos = 0
        for section in sections:
            boundaries.append((pos, pos + len(section)))
            pos += len(section)

    elif file_type == 'docx':
        # DOCX: already paragraph-delimited, keep paragraphs
        paragraphs = content.split('\n\n')
        pos = 0
        for para in paragraphs:
            boundaries.append((pos, pos + len(para)))
            pos += len(para) + 2

    elif file_type == 'txt':
        # TXT: split on blank lines or sentence boundaries
        sections = re.split(r'\n\s*\n', content)
        pos = 0
        for section in sections:
            boundaries.append((pos, pos + len(section)))
            pos += len(section)

    return boundaries

def adaptive_chunk_document(
    content: str,
    file_type: str,
    min_chunk_size: int = 500,
    max_chunk_size: int = 2000,
    overlap: int = 300
) -> List[Dict]:
    """Chunk document respecting logical boundaries"""

    # Identify logical boundaries
    boundaries = identify_logical_boundaries(content, file_type)

    chunks = []
    current_pos = 0

    for start, end in boundaries:
        section = content[start:end]
        section_len = len(section)

        # If section fits in one chunk, use it as-is
        if section_len <= max_chunk_size:
            chunks.append({
                'content': section,
                'start_pos': start,
                'end_pos': end,
                'boundary_aligned': True
            })
            current_pos = end
        else:
            # Section too large, split with overlap
            chunk_start = start
            while chunk_start < end:
                chunk_end = min(chunk_start + max_chunk_size, end)

                # Find good split point (sentence boundary)
                if chunk_end < end:
                    # Look for sentence boundary near max_chunk_size
                    last_period = content.rfind('.', chunk_start, chunk_end)
                    if last_period > chunk_start + min_chunk_size:
                        chunk_end = last_period + 1

                chunk_text = content[chunk_start:chunk_end]
                chunks.append({
                    'content': chunk_text,
                    'start_pos': chunk_start,
                    'end_pos': chunk_end,
                    'boundary_aligned': chunk_end == end
                })

                # Move with overlap
                chunk_start = chunk_end - overlap

    return chunks
```

**Benefits**:
- Respect clause/paragraph boundaries
- Better semantic coherence
- Improved retrieval (chunks match logical sections)
- Fewer split clauses

**Impact**:
- Better semantic chunking
- Improved retrieval accuracy
- More natural chunk boundaries
- Better for citation extraction

**Effort**: High (7-8 hours)

**Dependencies**: Boundary detection (pattern matching, NLP)

**Integration Point**: Replace fixed chunking in `chunking.py`

**Configuration**:
- `MIN_CHUNK_SIZE`: Minimum chunk size (default 500 chars)
- `MAX_CHUNK_SIZE`: Maximum chunk size (default 2000 chars)
- `CHUNK_OVERLAP`: Overlap between chunks (default 300 chars)

**Tests Needed**:
- [ ] Logical boundaries identified
- [ ] Chunks respect boundaries
- [ ] Chunk size constraints honored
- [ ] Overlap applied correctly
- [ ] All content included
- [ ] Retrieval quality improved

---

## Category 6: Legal-Specific Features

### 6.1 Clause Extraction & Tagging

**Problem**: System treats all chunks equally, but some are critical (liability, termination, payment) while others are boilerplate.

**Solution**: Pre-identify and tag critical legal clauses during processing.

**Implementation**:
```python
CRITICAL_CLAUSE_PATTERNS = {
    'payment': r'(?:payment|invoice|remuneration|compensation|fee|charge)',
    'liability': r'(?:liability|liable|responsible|obligation|indemnifi)',
    'termination': r'(?:terminat|cancel|dissolv|end|expir)',
    'warranty': r'(?:warrant|representation|guarantee)',
    'confidentiality': r'(?:confidential|proprietary|NDA|non-disclosure)',
    'indemnity': r'(?:indemnif|hold harmless)',
    'force_majeure': r'(?:force majeure|unforeseen|act of god)',
    'dispute': r'(?:dispute|arbitrat|litigation|court)',
    'governing_law': r'(?:governing law|jurisdiction|applicable law)',
    'intellectual_property': r'(?:intellectual property|IP|patent|copyright|trademark)',
}

def classify_clause_type(content: str) -> Dict[str, float]:
    """Classify chunk by clause type with scores"""

    clause_scores = {}
    content_lower = content.lower()

    for clause_type, pattern in CRITICAL_CLAUSE_PATTERNS.items():
        matches = len(re.findall(pattern, content_lower, re.IGNORECASE))

        # Score based on match frequency and position
        if matches > 0:
            # Normalize to 0-1
            score = min(matches / 5, 1.0)

            # Boost if clause type mentioned in header/first sentence
            if matches > 0 and content[:200].lower().count(clause_type.replace('_', ' ')) > 0:
                score = min(score * 1.3, 1.0)

            clause_scores[clause_type] = score

    return clause_scores

def tag_chunks_with_clauses(chunks: List[Dict]) -> List[Dict]:
    """Add clause tags to chunks"""

    for chunk in chunks:
        clause_scores = classify_clause_type(chunk['content'])

        # Primary clause (highest score)
        if clause_scores:
            primary_clause = max(clause_scores.items(), key=lambda x: x[1])
            chunk['primary_clause_type'] = primary_clause[0]
            chunk['primary_clause_score'] = primary_clause[1]
        else:
            chunk['primary_clause_type'] = 'boilerplate'
            chunk['primary_clause_score'] = 0.0

        # All clauses above threshold
        chunk['clause_tags'] = {
            clause: score for clause, score in clause_scores.items()
            if score > 0.3
        }

        # Critical flag
        chunk['is_critical'] = chunk['primary_clause_score'] > 0.6

    return chunks

# Usage in chunking
def chunk_document_from_blob(blob_content: bytes, file_type: str = "pdf") -> List[Dict]:
    """Chunk document with clause tagging"""

    # ... existing chunking code ...
    chunks = []

    for section in extracted_sections:
        for chunk in chunked_sections:
            chunk_dict = {
                "content": chunk,
                "page_num": section["location"],
                "section_name": f"Chunk {len(chunks) + 1}"
            }
            chunks.append(chunk_dict)

    # Tag with clause types
    chunks = tag_chunks_with_clauses(chunks)

    return chunks

# Usage in query
def weight_chunks_by_clause_importance(chunks: List[Dict], query: str) -> List[Dict]:
    """Boost critical clauses in retrieval"""

    query_lower = query.lower()

    for chunk in chunks:
        # Boost if chunk is critical
        if chunk['is_critical']:
            chunk['score'] *= 1.3

        # Additional boost if query mentions clause type
        for clause in chunk.get('clause_tags', {}).keys():
            if clause.replace('_', ' ') in query_lower:
                chunk['score'] *= 1.15

    return sorted(chunks, key=lambda x: x['score'], reverse=True)
```

**Response Format**:
```json
{
  "chunks": [
    {
      "content": "...",
      "page_num": "5",
      "primary_clause_type": "payment",
      "primary_clause_score": 0.92,
      "is_critical": true,
      "clause_tags": {
        "payment": 0.92,
        "termination": 0.45
      }
    }
  ]
}
```

**Impact**:
- Prioritize legally critical sections
- Reduce noise from boilerplate
- Better for queries about obligations/risks
- More relevant retrieval

**Effort**: High (6-7 hours)

**Dependencies**: Pattern matching, legal knowledge

**Integration Point**: In `chunking.py` and `query_case()`

**Tests Needed**:
- [ ] Clause patterns match correctly
- [ ] Scoring normalized 0-1
- [ ] Critical clauses tagged
- [ ] Boilerplate deprioritized
- [ ] Scoring accuracy on legal texts
- [ ] Multiple clause types handled

---

### 6.2 Temporal Reasoning

**Problem**: Can't track "effective date", "expiration", "amendment dates". Answers may cite outdated clauses.

**Solution**: Extract and track temporal metadata, flag outdated references.

**Implementation**:
```python
from datetime import datetime
from dateutil import parser as date_parser

TEMPORAL_PATTERNS = {
    'effective_date': r'(?:effective date|effective as of|in effect as of)[:=\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
    'expiration_date': r'(?:expir(?:ation|es) date|expir(?:es|ed) on|valid until)[:=\s]+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
    'term_duration': r'(?:term of|duration|for a period of)[:=\s]+(\d+)\s+(?:years?|months?|days?)',
    'amendment_date': r'(?:amended|modified|updated)[:=\s]+(?:on\s+)?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})',
}

def extract_temporal_info(content: str) -> Dict[str, any]:
    """Extract temporal information from chunk"""

    temporal = {}

    for temporal_type, pattern in TEMPORAL_PATTERNS.items():
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            date_str = match.group(1)
            try:
                parsed_date = date_parser.parse(date_str)
                temporal[temporal_type] = parsed_date.isoformat()
            except:
                temporal[temporal_type] = date_str

    return temporal

def check_clause_currency(chunk: Dict, reference_date: datetime = None) -> Dict:
    """Check if clause is current/expired"""

    if reference_date is None:
        reference_date = datetime.now()

    temporal = chunk.get('temporal_info', {})

    status = {
        'is_current': True,
        'status': 'current',
        'reason': '',
        'urgency': 'normal'
    }

    # Check effective date
    if 'effective_date' in temporal:
        effective = datetime.fromisoformat(temporal['effective_date'])
        if effective > reference_date:
            status['is_current'] = False
            status['status'] = 'not_yet_effective'
            status['reason'] = f"Effective date is {effective.date()}"
            status['urgency'] = 'high'

    # Check expiration date
    if 'expiration_date' in temporal:
        expiration = datetime.fromisoformat(temporal['expiration_date'])
        if expiration < reference_date:
            status['is_current'] = False
            status['status'] = 'expired'
            status['reason'] = f"Expired on {expiration.date()}"
            status['urgency'] = 'critical'
        elif (expiration - reference_date).days < 30:
            status['urgency'] = 'warning'
            status['reason'] = f"Expires on {expiration.date()}"

    # Check for amendments
    if 'amendment_date' in temporal:
        amendment = datetime.fromisoformat(temporal['amendment_date'])
        status['last_amended'] = amendment.isoformat()

    return status

# Usage in chunking
def tag_chunks_with_temporal_info(chunks: List[Dict]) -> List[Dict]:
    """Add temporal metadata to chunks"""

    for chunk in chunks:
        chunk['temporal_info'] = extract_temporal_info(chunk['content'])
        chunk['currency_status'] = check_clause_currency(chunk)

    return chunks

# Usage in query
def filter_expired_clauses(chunks: List[Dict], exclude_expired: bool = False) -> List[Dict]:
    """Filter out expired clauses if needed"""

    if not exclude_expired:
        return chunks

    active_chunks = []
    for chunk in chunks:
        status = chunk.get('currency_status', {})
        if status.get('is_current', True):
            active_chunks.append(chunk)
        else:
            logger.warning(f"Skipping {status['status']} clause: {status.get('reason')}")

    return active_chunks
```

**Response Format**:
```json
{
  "answer": "Payment is due within 30 days",
  "temporal_notes": [
    {
      "location": "Page 5",
      "status": "current",
      "effective_date": "2024-01-15",
      "expiration_date": "2025-01-15",
      "urgency": "normal"
    }
  ]
}
```

**Impact**:
- Prevent relying on expired clauses
- Flag time-sensitive information
- Critical for contracts with defined terms
- Improves answer accuracy

**Effort**: High (7-8 hours)

**Dependencies**: Date parsing, temporal extraction

**Integration Point**: In `chunking.py` and `query_case()`

**Tests Needed**:
- [ ] Dates extracted correctly
- [ ] Effective dates identified
- [ ] Expiration dates flagged
- [ ] Amendments tracked
- [ ] Currency status calculated
- [ ] Expired clauses properly flagged

---

### 6.3 Entity Extraction

**Problem**: System can't distinguish between different parties, entities mentioned. Answers may conflate obligations of different parties.

**Solution**: Extract and track named entities (parties, amounts, dates) during processing.

**Implementation**:
```python
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    HAS_SPACY = True
except:
    HAS_SPACY = False

def extract_entities(content: str) -> Dict[str, List[str]]:
    """Extract legal entities, persons, amounts, dates"""

    entities = {
        'parties': [],           # Organizations, individuals
        'amounts': [],           # Monetary values
        'dates': [],             # Important dates
        'obligations': [],       # Actions parties must perform
        'penalties': []          # Consequences/penalties
    }

    if HAS_SPACY:
        doc = nlp(content)

        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['parties'].append(ent.text)
            elif ent.label_ in ['MONEY', 'QUANTITY']:
                entities['amounts'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)

    # Pattern-based extraction for legal-specific entities
    # Obligations: "must", "shall", "required to", "responsible for"
    obligation_pattern = r'(?:must|shall|required to|responsible for|agree to|obligated to)\s+([^.!?]+)'
    for match in re.finditer(obligation_pattern, content):
        obligation = match.group(1).strip()
        if len(obligation) < 100:  # Reasonable length
            entities['obligations'].append(obligation)

    # Penalties/consequences
    penalty_pattern = r'(?:penalt|fine|damages|liable for|consequence|if|fail to|breach|default).*?(?=[.!?])'
    for match in re.finditer(penalty_pattern, content, re.IGNORECASE):
        penalty = match.group(0).strip()
        if 20 < len(penalty) < 200:
            entities['penalties'].append(penalty)

    # Deduplicate
    for key in entities:
        entities[key] = list(set(entities[key]))[:5]  # Top 5 per type

    return entities

def tag_chunks_with_entities(chunks: List[Dict]) -> List[Dict]:
    """Add entity metadata to chunks"""

    for chunk in chunks:
        chunk['entities'] = extract_entities(chunk['content'])

    return chunks

# Usage in query
def filter_chunks_by_party(chunks: List[Dict], query: str) -> List[Dict]:
    """Filter chunks to specific party if mentioned in query"""

    # Extract party name from query
    # "What are acme corp's obligations?"

    # For each chunk, check if it discusses relevant party
    # This is complex - simplified version:

    party_pattern = r'(?:obligations of|(?:by|from)\s+)(\w+(?:\s+\w+)*?)(?:\s+(?:is|are|include|require)|\?|,)'
    match = re.search(party_pattern, query, re.IGNORECASE)

    if match:
        target_party = match.group(1).lower()
        relevant_chunks = []

        for chunk in chunks:
            chunk_parties = [p.lower() for p in chunk.get('entities', {}).get('parties', [])]
            if target_party in chunk_parties or any(target_party in p for p in chunk_parties):
                relevant_chunks.append(chunk)

        return relevant_chunks if relevant_chunks else chunks

    return chunks
```

**Response Format**:
```json
{
  "answer": "...",
  "entities_mentioned": {
    "parties": ["Acme Corp", "Vendor Inc"],
    "amounts": ["$1000", "$500"],
    "dates": ["2024-01-15", "2025-01-15"],
    "obligations": [
      "Acme shall pay within 30 days",
      "Vendor shall deliver by end of month"
    ]
  }
}
```

**Impact**:
- Answer specific entity questions accurately
- Distinguish between party obligations
- Reduce entity confusion
- Support entity-focused queries

**Effort**: High (6-7 hours)

**Dependencies**: spaCy for NER (optional), pattern matching

**Integration Point**: In `chunking.py` and response formatting

**Tests Needed**:
- [ ] Entities extracted correctly
- [ ] Parties identified
- [ ] Amounts extracted
- [ ] Obligations parsed
- [ ] Entity filtering works
- [ ] Deduplication accurate

---

### 6.4 Cross-Reference Resolution

**Problem**: Legal documents reference other sections ("see clause 3.2" or "per the schedule"). System can't resolve these references.

**Solution**: Automatically resolve references within same document.

**Implementation**:
```python
CROSS_REFERENCE_PATTERNS = {
    'clause_reference': r'(?:clause|section|article|paragraph)\s+(\d+\.?\d*)',
    'schedule_reference': r'(?:schedule|annex|exhibit|appendix)\s+([A-Z])',
    'page_reference': r'(?:page|p\.)\s+(\d+)',
    'internal_ref': r'(?:above|below|above-mentioned|following|preceding)',
}

def extract_cross_references(content: str) -> Dict[str, List[str]]:
    """Extract cross-references from chunk"""

    references = {}

    for ref_type, pattern in CROSS_REFERENCE_PATTERNS.items():
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            references[ref_type] = matches

    return references

def resolve_reference(reference: str, case: Case) -> Optional[Dict]:
    """Resolve a cross-reference to actual chunk"""

    # Parse reference type and target
    if reference.startswith('clause') or reference.startswith('section'):
        # "clause 3.2" -> find chunk with "3.2" or "clause 3.2" heading
        match = re.search(r'(\d+\.?\d*)', reference)
        if match:
            target = match.group(1)

            # Search chunks for this clause number
            for chunk in case.chunks:
                if target in chunk.content or f"Clause {target}" in chunk.content:
                    return {
                        'type': 'clause',
                        'reference': target,
                        'chunk_id': chunk.id,
                        'excerpt': chunk.content[:300]
                    }

    elif reference.startswith('schedule') or reference.startswith('annex'):
        # "Schedule A" -> find chunk with "Schedule A"
        match = re.search(r'([A-Z])', reference)
        if match:
            target = match.group(1)

            for chunk in case.chunks:
                if f"Schedule {target}" in chunk.content:
                    return {
                        'type': 'schedule',
                        'reference': target,
                        'chunk_id': chunk.id,
                        'excerpt': chunk.content[:300]
                    }

    elif 'page' in reference:
        # "page 5" -> find chunk with page_num = 5
        match = re.search(r'(\d+)', reference)
        if match:
            target_page = match.group(1)

            for chunk in case.chunks:
                if chunk.page_num == target_page:
                    return {
                        'type': 'page',
                        'reference': target_page,
                        'chunk_id': chunk.id,
                        'excerpt': chunk.content[:300]
                    }

    return None

def augment_chunk_with_references(chunk: Dict, case: Case) -> Dict:
    """Add resolved cross-references to chunk"""

    references = extract_cross_references(chunk['content'])
    resolved = []

    for ref_type, ref_list in references.items():
        for reference in ref_list:
            resolved_ref = resolve_reference(f"{ref_type} {reference}", case)
            if resolved_ref:
                resolved.append(resolved_ref)

    chunk['cross_references'] = references
    chunk['resolved_references'] = resolved

    return chunk

# Usage in answer generation
def enrich_context_with_references(chunks: List[Dict], case: Case) -> List[Dict]:
    """Include referenced sections in context"""

    enriched = list(chunks)
    all_resolved = set()

    # Collect all referenced chunks
    for chunk in chunks:
        for resolved_ref in chunk.get('resolved_references', []):
            all_resolved.add(resolved_ref['chunk_id'])

    # Add resolved chunks if not already included
    for chunk_id in all_resolved:
        if not any(c['chunk_id'] == chunk_id for c in enriched):
            referenced_chunk = case.chunks.filter(Chunk.id == chunk_id).first()
            if referenced_chunk:
                enriched.append({
                    **referenced_chunk.to_dict(),
                    'is_referenced': True,
                    'referenced_from': [c['chunk_id'] for c in chunks if chunk_id in
                                       [r['chunk_id'] for r in c.get('resolved_references', [])]]
                })

    return enriched
```

**Response Format**:
```json
{
  "answer": "...",
  "cross_references": {
    "clause 3.2": {
      "found": true,
      "excerpt": "...",
      "page": "7"
    },
    "Schedule A": {
      "found": true,
      "excerpt": "...",
      "page": "12"
    }
  }
}
```

**Impact**:
- Complete context by including referenced sections
- Better answer to interconnected questions
- Help users navigate document structure
- Essential for contracts with heavy internal references

**Effort**: Medium (4-5 hours)

**Dependencies**: Pattern matching, chunk lookup

**Integration Point**: In context preparation before answer generation

**Tests Needed**:
- [ ] Reference patterns identified
- [ ] References resolved correctly
- [ ] Referenced chunks included in context
- [ ] Circular references handled
- [ ] Missing references gracefully handled
- [ ] Context doesn't exceed token budget

---

## Implementation Roadmap

### Phase 1: Quick Wins (Week 1-2)
Low effort, immediate impact:
- [ ] 4.1 Answer Confidence Explanation
- [ ] 4.4 Query Result Caching
- [ ] 5.2 Embedding Cache
- [ ] 4.3 Source Document Summary

**Estimated Effort**: 6-8 hours
**Expected Impact**: Better UX, lower costs, improved transparency

---

### Phase 2: Accuracy Improvements (Week 3-4)
Medium effort, high legal impact:
- [ ] 1.4 Hallucination Detection Enhancement
- [ ] 3.2 Citation Span Highlighting
- [ ] 2.1 Semantic Deduplication
- [ ] 1.2 Query Expansion

**Estimated Effort**: 12-16 hours
**Expected Impact**: Better answer quality, improved citations, reduced false claims

---

### Phase 3: Advanced Retrieval (Week 5-6)
Higher effort, significant improvements:
- [ ] 2.1 Hybrid Search (Dense + Sparse)
- [ ] 3.1 Multi-Chunk Citations
- [ ] 2.4 Query-Document Specific Weighting
- [ ] 2.3 Chunk Relevance Confidence Threshold

**Estimated Effort**: 16-20 hours
**Expected Impact**: Better recall, cleaner citations, legal-aware retrieval

---

### Phase 4: Legal Features (Week 7-10)
Strategic legal capabilities:
- [ ] 6.1 Clause Extraction & Tagging
- [ ] 6.2 Temporal Reasoning
- [ ] 6.3 Entity Extraction
- [ ] 6.4 Cross-Reference Resolution

**Estimated Effort**: 20-28 hours
**Expected Impact**: Legal-specific reasoning, time-aware answers, entity tracking

---

### Phase 5: Complex Reasoning (Week 11+)
Advanced capabilities:
- [ ] 1.3 Multi-Hop Reasoning
- [ ] 5.1 Streaming Responses
- [ ] 5.3 Chunk Size Optimization
- [ ] 2.2 Dynamic Top-K Selection
- [ ] 3.3 Confidence Calibration per Citation
- [ ] 3.4 Missing Citation Detection
- [ ] 4.2 Follow-up Question Suggestions

**Estimated Effort**: 28-40 hours
**Expected Impact**: Complex legal questions, professional UX, advanced validation

---

## Evaluation Metrics

For each improvement, measure:

**Accuracy Metrics**:
- Citation coverage (% of claims with citations)
- Citation accuracy (% of citations supporting their claims)
- Hallucination rate (% of answers with unsupported claims)
- Entity accuracy (% of entities correctly identified)

**Retrieval Metrics**:
- Mean reciprocal rank (MRR) of correct chunks
- Recall@K for different K values
- Diversity of retrieved chunks
- Processing latency

**User Experience Metrics**:
- Query response time (p50, p95)
- Answer relevance ratings
- Citation usefulness ratings
- Feature adoption rates

**Legal-Specific Metrics**:
- Temporal accuracy (% of time-sensitive answers flagged)
- Cross-reference resolution rate
- Clause classification accuracy
- Entity mention consistency

---

## Risk Mitigation

**Implementation Risks**:
- Complex changes may break existing functionality
- **Mitigation**: Feature flags, comprehensive testing, gradual rollout

**Performance Risks**:
- Additional processing may increase latency
- **Mitigation**: Cache heavily, use async processing, profile before/after

**Cost Risks**:
- More sophisticated processing may increase API costs
- **Mitigation**: Caching, query optimization, batch processing

**Legal Risks**:
- False confident answers worse than uncertain ones
- **Mitigation**: Conservative confidence scoring, clear uncertainty indicators

---

## Success Criteria

The RAG pipeline improvements will be successful when:

1. **Answer Quality**: 90%+ of answers are well-cited and lack hallucinations
2. **User Trust**: Users report increased confidence in answers
3. **Efficiency**: Query processing remains < 5 seconds (including streaming)
4. **Legal Accuracy**: Time-sensitive information flagged appropriately
5. **User Satisfaction**: Feature adoption rate > 70% for new features

---

## Conclusion

The recommended improvements transform LexIntel's RAG system from functional to professional-grade. Phase 1-2 improvements focus on quality and transparency (critical for legal use). Phase 3-4 add specialized legal capabilities. Phase 5 enables advanced multi-document reasoning.

Start with Phase 1 quick wins to build momentum and demonstrate value, then progress based on user feedback and impact metrics.

