"""Tests for multi-format citation extraction in RAG engine"""
import pytest
import os

# Set required env vars before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')

from backend.services.rag_engine import extract_citations


class TestPDFCitations:
    """Test PDF citation extraction [Page X] format"""

    def test_extract_single_pdf_citation(self):
        """Test extracting single [Page X] citation"""
        answer = "The contract states [Page 1] that payment is due within 30 days."
        chunks = [
            {
                "chunk_id": "test-1",
                "page_num": "1",
                "score": 0.95,
                "content": "Payment terms..."
            }
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert "[Page 1]" in answer  # Original has citation
        assert len(citations) == 1
        assert citations[0]["location"] == "1"
        assert citations[0]["citation_type"] == "page"
        assert has_hallucinations is False

    def test_extract_multiple_pdf_citations(self):
        """Test extracting multiple PDF citations"""
        answer = "As stated [Page 1], the agreement is binding. [Page 2] details the obligations."
        chunks = [
            {"chunk_id": "test-1", "page_num": "1", "score": 0.95},
            {"chunk_id": "test-2", "page_num": "2", "score": 0.92}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 2
        assert citations[0]["location"] == "1"
        assert citations[1]["location"] == "2"
        assert has_hallucinations is False

    def test_pdf_hallucinated_citation(self):
        """Test detecting hallucinated PDF citations"""
        answer = "According to [Page 1] and [Page 99], the agreement is valid."
        chunks = [
            {"chunk_id": "test-1", "page_num": "1", "score": 0.95}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 1  # Only valid citation
        assert "[Page 99]" not in cleaned_answer  # Hallucinated citation removed
        assert has_hallucinations is True

    def test_pdf_duplicate_citations(self):
        """Test that duplicate citations are not repeated"""
        answer = "As mentioned [Page 1], this is important. Also [Page 1] states that..."
        chunks = [
            {"chunk_id": "test-1", "page_num": "1", "score": 0.95}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        # Should deduplicate
        assert len(citations) == 1
        assert has_hallucinations is False


class TestDOCXCitations:
    """Test Word document citation extraction [Paragraph X] format"""

    def test_extract_single_docx_citation(self):
        """Test extracting single [Paragraph X] citation"""
        answer = "The document states [Paragraph 5] that the terms are binding."
        chunks = [
            {
                "chunk_id": "test-1",
                "page_num": "para 5",
                "score": 0.94,
                "content": "Terms and conditions..."
            }
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0]["location"] == "para 5"
        assert citations[0]["citation_type"] == "paragraph"
        assert has_hallucinations is False

    def test_extract_multiple_docx_citations(self):
        """Test extracting multiple DOCX citations"""
        answer = "First [Paragraph 3], then [Paragraph 8] explains the procedure."
        chunks = [
            {"chunk_id": "test-1", "page_num": "para 3", "score": 0.93},
            {"chunk_id": "test-2", "page_num": "para 8", "score": 0.91}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 2
        assert all(c["citation_type"] == "paragraph" for c in citations)
        assert has_hallucinations is False

    def test_docx_hallucinated_citation(self):
        """Test detecting hallucinated DOCX citations"""
        answer = "According to [Paragraph 3] and [Paragraph 100], the clause applies."
        chunks = [
            {"chunk_id": "test-1", "page_num": "para 3", "score": 0.95}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert "[Paragraph 100]" not in cleaned_answer
        assert has_hallucinations is True


class TestTXTCitations:
    """Test text file citation extraction [Lines X-Y] format"""

    def test_extract_single_txt_citation(self):
        """Test extracting single [Lines X-Y] citation"""
        answer = "As mentioned [Lines 1-50] in the document, the agreement is valid."
        chunks = [
            {
                "chunk_id": "test-1",
                "page_num": "line 1-50",
                "score": 0.92,
                "content": "First section..."
            }
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0]["location"] == "line 1-50"
        assert citations[0]["citation_type"] == "line_range"
        assert has_hallucinations is False

    def test_extract_multiple_txt_citations(self):
        """Test extracting multiple TXT citations"""
        answer = "First [Lines 1-50] shows the introduction. Then [Lines 51-100] details the terms."
        chunks = [
            {"chunk_id": "test-1", "page_num": "line 1-50", "score": 0.93},
            {"chunk_id": "test-2", "page_num": "line 51-100", "score": 0.91}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 2
        assert all(c["citation_type"] == "line_range" for c in citations)
        assert has_hallucinations is False

    def test_txt_hallucinated_citation(self):
        """Test detecting hallucinated TXT citations"""
        answer = "According to [Lines 1-50] and [Lines 501-550], the text states..."
        chunks = [
            {"chunk_id": "test-1", "page_num": "line 1-50", "score": 0.95}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 1
        assert "[Lines 501-550]" not in cleaned_answer
        assert has_hallucinations is True


class TestMixedFormatCitations:
    """Test mixed document type citations (should not happen but test robustness)"""

    def test_mixed_format_retrieval(self):
        """Test that mixed format chunks are handled correctly"""
        answer = "Document [Page 5] states, and [Paragraph 8] confirms, while [Lines 1-50] explains."
        chunks = [
            {"chunk_id": "test-1", "page_num": "5", "score": 0.95},  # PDF
            {"chunk_id": "test-2", "page_num": "para 8", "score": 0.92},  # DOCX
            {"chunk_id": "test-3", "page_num": "line 1-50", "score": 0.90}  # TXT
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 3
        assert citations[0]["citation_type"] == "page"
        assert citations[1]["citation_type"] == "paragraph"
        assert citations[2]["citation_type"] == "line_range"
        assert has_hallucinations is False

    def test_mixed_with_hallucination(self):
        """Test mixed format with some hallucinated citations"""
        answer = "Per [Page 5], [Paragraph 99], and [Lines 1-50], the terms apply."
        chunks = [
            {"chunk_id": "test-1", "page_num": "5", "score": 0.95},  # PDF - valid
            {"chunk_id": "test-3", "page_num": "line 1-50", "score": 0.90}  # TXT - valid
            # Paragraph 99 is missing
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 2  # Only valid citations
        assert "[Paragraph 99]" not in cleaned_answer
        assert has_hallucinations is True


class TestCitationEdgeCases:
    """Test edge cases in citation extraction"""

    def test_empty_answer(self):
        """Test empty answer"""
        answer = ""
        chunks = [{"chunk_id": "test-1", "page_num": "1", "score": 0.95}]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert cleaned_answer == ""
        assert len(citations) == 0
        assert has_hallucinations is False

    def test_no_citations_in_answer(self):
        """Test answer with no citations"""
        answer = "This is an answer without any citations."
        chunks = [{"chunk_id": "test-1", "page_num": "1", "score": 0.95}]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 0
        assert has_hallucinations is False

    def test_all_hallucinated_citations(self):
        """Test answer where all citations are hallucinated"""
        answer = "According to [Page 99] and [Paragraph 50], the terms are clear."
        chunks = [
            {"chunk_id": "test-1", "page_num": "1", "score": 0.95}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        assert len(citations) == 0
        assert "[Page 99]" not in cleaned_answer
        assert "[Paragraph 50]" not in cleaned_answer
        assert has_hallucinations is True

    def test_citation_with_extra_spaces(self):
        """Test citation patterns with variable spacing"""
        # PDF pattern
        answer1 = "As stated [Page  1] in the document..."
        chunks = [{"chunk_id": "test-1", "page_num": "1", "score": 0.95}]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer1, chunks)
        assert len(citations) == 1

    def test_case_sensitivity(self):
        """Test that citation patterns are case-sensitive"""
        # These should not match (wrong case)
        answer = "According to [page 1] and [paragraph 2] and [lines 1-50]..."
        chunks = [
            {"chunk_id": "test-1", "page_num": "1", "score": 0.95},
            {"chunk_id": "test-2", "page_num": "para 2", "score": 0.93},
            {"chunk_id": "test-3", "page_num": "line 1-50", "score": 0.91}
        ]

        cleaned_answer, citations, has_hallucinations = extract_citations(answer, chunks)

        # No citations should match (case matters)
        assert len(citations) == 0
        assert has_hallucinations is False  # No hallucinations, just no matches
