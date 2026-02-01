"""Test document summary generation"""
import pytest
from datetime import datetime
from unittest.mock import MagicMock
from uuid import uuid4


class TestDocumentSummary:
    """Test document summary functionality"""

    def test_extract_key_concepts(self):
        """Should extract top legal concepts from chunks"""
        from backend.services.document_summary import extract_key_concepts

        mock_case = MagicMock()
        mock_case.chunks = [
            MagicMock(content="Payment must be made. Payment terms apply."),
            MagicMock(content="Liability clause. Not liable for damages."),
            MagicMock(content="Termination notice required."),
        ]

        concepts = extract_key_concepts(mock_case)

        assert isinstance(concepts, list)
        assert len(concepts) > 0
        assert any("payment" in c.lower() for c in concepts)

    def test_classify_document_type(self):
        """Should classify document type"""
        from backend.services.document_summary import classify_legal_document_type

        test_cases = [
            ("TERMS AND CONDITIONS of Service", "Terms of Service"),
            ("SOFTWARE LICENSE AGREEMENT", "License Agreement"),
            ("PRIVACY POLICY Statement", "Privacy Policy"),
        ]

        for content, expected in test_cases:
            mock_case = MagicMock()
            mock_case.chunks = [MagicMock(content=content)]

            doc_type = classify_legal_document_type(mock_case)
            assert doc_type == expected

    def test_calculate_page_count(self):
        """Should calculate page count from chunks"""
        from backend.services.document_summary import calculate_page_count

        mock_case = MagicMock()
        mock_case.chunks = [
            MagicMock(page_num="1"),
            MagicMock(page_num="2"),
            MagicMock(page_num="5"),
        ]

        count = calculate_page_count(mock_case)

        assert isinstance(count, int)
        assert count >= 1

    def test_generate_document_summary(self):
        """Should generate complete document summary"""
        from backend.services.document_summary import generate_document_summary

        mock_case = MagicMock()
        mock_case.name = "test-agreement.pdf"
        mock_case.file_type = "pdf"
        mock_case.status = "ready"
        mock_case.updated_at = datetime.now()
        mock_case.chunks = [
            MagicMock(content="Payment terms and liability.", page_num="1"),
            MagicMock(content="Termination clause.", page_num="2"),
        ]

        summary = generate_document_summary(mock_case)

        assert summary is not None
        assert "filename" in summary
        assert summary["filename"] == "test-agreement.pdf"
        assert "file_type" in summary
        assert "key_concepts" in summary
        assert "legal_significance" in summary
        assert "total_pages" in summary
        assert "processing_status" in summary
        assert summary["processing_status"] == "ready"

    def test_empty_case_returns_empty_concepts(self):
        """Empty case should return empty concepts list"""
        from backend.services.document_summary import extract_key_concepts

        mock_case = MagicMock()
        mock_case.chunks = []

        concepts = extract_key_concepts(mock_case)

        assert isinstance(concepts, list)
        assert len(concepts) == 0
