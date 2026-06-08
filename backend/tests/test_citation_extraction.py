"""Regression tests for rag_engine citation extraction/grounding.

Covers the fix for the bare numbered-citation format ([1], [2], [3]) that the
RAG system prompt mandates but extract_citations previously ignored (it only
matched [Page N]/[Section X]/[Lines N-M]), which left grounded `citations`
permanently empty and dropped supporting authorities from history-loaded queries.
"""


def _extract(answer, chunks):
    from backend.services.rag_engine import extract_citations
    return extract_citations(answer, chunks)


def _ground(citations, chunks, query=""):
    from backend.services.rag_engine import ground_citations_in_source
    return ground_citations_in_source(citations, chunks, query)


def _chunks():
    # Two chunks; [n] must map in SCORE-DESC order (format_legal_context's
    # numbering), NOT list order — so [1] -> highest score.
    return [
        {"chunk_id": "low", "page_num": "2", "content": "Lower scored chunk about B.", "score": 0.30},
        {"chunk_id": "high", "page_num": "1", "content": "Higher scored chunk about A.", "score": 0.90},
    ]


def test_bare_numbered_citation_maps_to_score_sorted_chunk():
    chunks = _chunks()
    _, citations, halluc = _extract("The court held A [1] and also B [2].", chunks)
    ids = sorted(c["chunk_id"] for c in citations)
    assert ids == ["high", "low"], ids          # both markers resolved
    # [1] -> highest-scored chunk, [2] -> next
    assert citations[0]["chunk_id"] == "high"
    assert citations[1]["chunk_id"] == "low"
    assert halluc is False


def test_out_of_range_marker_is_hallucinated_and_stripped():
    chunks = _chunks()
    cleaned, citations, halluc = _extract("Valid [1] but invalid [9].", chunks)
    assert halluc is True
    assert "[9]" not in cleaned
    assert "[1]" in cleaned
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "high"


def test_grounding_uses_exact_chunk_id():
    chunks = _chunks()
    _, citations, _ = _extract("Point A [1].", chunks)   # [1] -> high
    grounded, unsupported, has_unsupported = _ground(citations, chunks)
    assert len(grounded) == 1
    assert grounded[0]["chunk_id"] == "high"
    assert "Higher scored chunk" in grounded[0]["supporting_excerpt"]
    assert has_unsupported is False


def test_no_markers_yields_no_citations():
    chunks = _chunks()
    _, citations, halluc = _extract("A plain answer with no citations.", chunks)
    assert citations == []
    assert halluc is False


def test_best_excerpt_picks_query_relevant_window():
    # Large chunk (>480 chars) where the cited fact is NOT in the first 500 chars.
    from backend.services.rag_engine import _best_excerpt
    filler = "Preamble boilerplate about definitions and recitals. " * 12  # ~640 chars
    content = filler + "The governing law is the State of Delaware. " + ("Trailing clause. " * 10)
    excerpt = _best_excerpt(content, query="which state's law governs?", width=300)
    assert "Delaware" in excerpt          # picked the relevant window, not chunk[:300]
    assert "Delaware" not in content[:300]  # proves it wasn't just the leading slice


def test_best_excerpt_no_query_returns_leading_window():
    from backend.services.rag_engine import _best_excerpt
    content = "X" * 1000
    assert _best_excerpt(content, query="", width=200) == "X" * 200


def test_grounding_excerpt_contains_query_fact():
    # End-to-end through ground_citations_in_source with a realistic long chunk.
    long = ("Section 1 parties and recitals. " * 20) + "The fee is 50,000 dollars due in 30 days."
    chunks = [{"chunk_id": "c1", "page_num": "1", "content": long, "score": 0.9}]
    _, citations, _ = _extract("The fee and deadline are stated [1].", chunks)
    grounded, _, _ = _ground(citations, chunks, "what is the fee and deadline?")
    assert "50,000" in grounded[0]["supporting_excerpt"]


def test_structured_page_citation_still_works():
    # Backwards-compat: [Page N] must still resolve.
    chunks = [{"chunk_id": "c1", "page_num": "5", "content": "Page five content.", "score": 0.8}]
    _, citations, _ = _extract("As stated [Page 5].", chunks)
    assert any(c["chunk_id"] == "c1" for c in citations)
