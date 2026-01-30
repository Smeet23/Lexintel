"""Tests for citation grounding validation and confidence scoring"""
import pytest
import os

os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')

from backend.services.rag_engine import (
    ground_citations_in_source,
    calculate_answer_confidence,
    classify_confidence_level
)


class TestCitationGrounding:
    """Test citation grounding in source text"""

    def test_ground_valid_pdf_citation(self):
        """Test grounding valid PDF citation"""
        citations = [
            {
                "location": "1",
                "citation_type": "page",
                "relevance_score": 0.95,
                "chunk_id": "chunk-1"
            }
        ]
        chunks = [
            {
                "page_num": "1",
                "content": "This is page 1 content with important legal information.",
                "chunk_id": "chunk-1"
            }
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 1
        assert grounded[0]["is_grounded"] is True
        assert grounded[0]["location"] == "1"
        assert len(unsupported) == 0
        assert has_unsupported is False

    def test_ground_unsupported_citation(self):
        """Test detecting unsupported citation"""
        citations = [
            {
                "location": "5",
                "citation_type": "page",
                "relevance_score": 0.90,
                "chunk_id": "chunk-5"
            }
        ]
        chunks = [
            {
                "page_num": "1",
                "content": "This is page 1 content.",
                "chunk_id": "chunk-1"
            }
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 0
        assert len(unsupported) == 1
        assert unsupported[0]["location"] == "5"
        assert has_unsupported is True

    def test_ground_multiple_citations(self):
        """Test grounding multiple citations"""
        citations = [
            {"location": "1", "citation_type": "page", "relevance_score": 0.95, "chunk_id": "chunk-1"},
            {"location": "2", "citation_type": "page", "relevance_score": 0.92, "chunk_id": "chunk-2"},
            {"location": "99", "citation_type": "page", "relevance_score": 0.80, "chunk_id": "chunk-99"}
        ]
        chunks = [
            {"page_num": "1", "content": "Page 1 content", "chunk_id": "chunk-1"},
            {"page_num": "2", "content": "Page 2 content", "chunk_id": "chunk-2"}
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 2
        assert len(unsupported) == 1
        assert has_unsupported is True

    def test_ground_docx_citations(self):
        """Test grounding DOCX paragraph citations"""
        citations = [
            {"location": "para 3", "citation_type": "paragraph", "relevance_score": 0.91, "chunk_id": "chunk-3"}
        ]
        chunks = [
            {"page_num": "para 3", "content": "This paragraph discusses terms.", "chunk_id": "chunk-3"}
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 1
        assert grounded[0]["location"] == "para 3"
        assert grounded[0]["citation_type"] == "paragraph"
        assert has_unsupported is False

    def test_ground_txt_citations(self):
        """Test grounding TXT line range citations"""
        citations = [
            {"location": "line 1-50", "citation_type": "line_range", "relevance_score": 0.88, "chunk_id": "chunk-1"}
        ]
        chunks = [
            {"page_num": "line 1-50", "content": "First 50 lines of text...", "chunk_id": "chunk-1"}
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 1
        assert grounded[0]["citation_type"] == "line_range"
        assert has_unsupported is False

    def test_supporting_excerpt_extraction(self):
        """Test that supporting text is extracted"""
        long_content = "A" * 600  # Content longer than 500 chars
        citations = [
            {"location": "1", "citation_type": "page", "relevance_score": 0.95, "chunk_id": "chunk-1"}
        ]
        chunks = [
            {"page_num": "1", "content": long_content, "chunk_id": "chunk-1"}
        ]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        # Should extract first 500 chars
        assert len(grounded[0]["supporting_excerpt"]) == 500
        assert grounded[0]["supporting_excerpt"] == long_content[:500]


class TestConfidenceScoring:
    """Test answer confidence scoring"""

    def test_high_confidence_answer(self):
        """Test high confidence answer (all citations grounded)"""
        answer = "The contract requires payment [Page 1] within 30 days [Page 2]."
        citations = [
            {"location": "1", "relevance_score": 0.95},
            {"location": "2", "relevance_score": 0.92}
        ]
        chunks = [
            {"page_num": "1", "content": "Payment required"},
            {"page_num": "2", "content": "30 days"}
        ]

        score = calculate_answer_confidence(answer, citations, chunks, has_hallucinations=False)

        assert score >= 0.75
        assert isinstance(score, float)

    def test_low_confidence_answer_no_citations(self):
        """Test low confidence when no citations"""
        answer = "The contract is valid."
        citations = []
        chunks = []

        score = calculate_answer_confidence(answer, citations, chunks, has_hallucinations=False)

        assert score < 0.4

    def test_confidence_with_hallucinations(self):
        """Test confidence reduction with hallucinations"""
        answer = "The document states [Page 1] clearly."
        citations = [{"location": "1", "relevance_score": 0.95}]
        chunks = [{"page_num": "1", "content": "Clear statement"}]

        score_clean = calculate_answer_confidence(answer, citations, chunks, has_hallucinations=False)
        score_hallucinated = calculate_answer_confidence(answer, citations, chunks, has_hallucinations=True)

        assert score_hallucinated < score_clean
        assert score_hallucinated >= 0.0

    def test_confidence_bounded(self):
        """Test that confidence is always bounded [0.0, 1.0]"""
        answer = "Test answer" * 100
        citations = [{"location": "1", "relevance_score": 0.99}] * 10
        chunks = [{"page_num": "1", "content": "Content"}]

        score = calculate_answer_confidence(answer, citations, chunks, has_hallucinations=False)

        assert 0.0 <= score <= 1.0

    def test_confidence_increases_with_relevance(self):
        """Test that confidence increases with higher relevance scores"""
        answer = "Statement [Page 1]."
        chunks = [{"page_num": "1", "content": "Content"}]

        score_low = calculate_answer_confidence(
            answer,
            [{"location": "1", "relevance_score": 0.6}],
            chunks,
            has_hallucinations=False
        )
        score_high = calculate_answer_confidence(
            answer,
            [{"location": "1", "relevance_score": 0.95}],
            chunks,
            has_hallucinations=False
        )

        assert score_high > score_low

    def test_confidence_with_multiple_citations(self):
        """Test that more citations improve confidence"""
        chunks = [
            {"page_num": "1", "content": "Content 1"},
            {"page_num": "2", "content": "Content 2"}
        ]

        answer_one_citation = "The rule [Page 1] applies here."
        score_one = calculate_answer_confidence(
            answer_one_citation,
            [{"location": "1", "relevance_score": 0.90}],
            chunks,
            has_hallucinations=False
        )

        answer_two_citations = "The rule [Page 1] applies [Page 2] broadly."
        score_two = calculate_answer_confidence(
            answer_two_citations,
            [
                {"location": "1", "relevance_score": 0.90},
                {"location": "2", "relevance_score": 0.88}
            ],
            chunks,
            has_hallucinations=False
        )

        # More citations should increase confidence
        assert score_two > score_one


class TestConfidenceClassification:
    """Test confidence level classification"""

    def test_classify_high_confidence(self):
        """Test high confidence classification"""
        assert classify_confidence_level(0.80) == "high"
        assert classify_confidence_level(0.95) == "high"
        assert classify_confidence_level(1.0) == "high"

    def test_classify_medium_confidence(self):
        """Test medium confidence classification"""
        assert classify_confidence_level(0.75) == "high"  # Boundary
        assert classify_confidence_level(0.70) == "medium"
        assert classify_confidence_level(0.60) == "medium"

    def test_classify_low_confidence(self):
        """Test low confidence classification"""
        assert classify_confidence_level(0.59) == "low"
        assert classify_confidence_level(0.50) == "low"
        assert classify_confidence_level(0.40) == "low"

    def test_classify_none_confidence(self):
        """Test none confidence classification"""
        assert classify_confidence_level(0.39) == "none"
        assert classify_confidence_level(0.0) == "none"

    def test_classification_boundaries(self):
        """Test boundary conditions"""
        assert classify_confidence_level(0.75) == "high"
        assert classify_confidence_level(0.74999) == "medium"
        assert classify_confidence_level(0.60) == "medium"
        assert classify_confidence_level(0.5999) == "low"
        assert classify_confidence_level(0.40) == "low"
        assert classify_confidence_level(0.3999) == "none"


class TestGroundingEdgeCases:
    """Test edge cases in citation grounding"""

    def test_empty_citations(self):
        """Test with empty citations list"""
        grounded, unsupported, has_unsupported = ground_citations_in_source([], [])

        assert len(grounded) == 0
        assert len(unsupported) == 0
        assert has_unsupported is False

    def test_empty_chunks(self):
        """Test with empty chunks"""
        citations = [{"location": "1", "citation_type": "page", "relevance_score": 0.95, "chunk_id": "chunk-1"}]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, [])

        assert len(grounded) == 0
        assert len(unsupported) == 1
        assert has_unsupported is True

    def test_missing_chunk_id(self):
        """Test with missing chunk_id"""
        citations = [{"location": "1", "citation_type": "page", "relevance_score": 0.95}]
        chunks = [{"page_num": "1", "content": "Content", "chunk_id": "chunk-1"}]

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        # Should still work - chunk_id is optional
        assert len(grounded) == 1

    def test_missing_content_in_chunk(self):
        """Test with missing content in chunk"""
        citations = [{"location": "1", "citation_type": "page", "relevance_score": 0.95, "chunk_id": "chunk-1"}]
        chunks = [{"page_num": "1", "chunk_id": "chunk-1"}]  # No content key

        grounded, unsupported, has_unsupported = ground_citations_in_source(citations, chunks)

        assert len(grounded) == 1
        assert grounded[0]["supporting_excerpt"] == ""  # Empty excerpt
