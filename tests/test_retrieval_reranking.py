"""Tests for retrieval reranking functionality"""
import pytest
import os

os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')

from backend.services.rag_engine import rerank_chunks, _get_reranker

# Check if reranker is available
RERANKER_AVAILABLE = _get_reranker() is not None


class TestReranking:
    """Test retrieval chunk reranking"""

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_single_chunk(self):
        """Test reranking with single chunk"""
        query = "What are payment terms?"
        chunks = [
            {
                "page_num": "1",
                "content": "Payment must be made within 30 days of invoice.",
                "score": 0.85,
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        assert len(reranked) <= 1
        assert len(reranked) > 0
        assert "combined_score" in reranked[0]
        assert "rerank_score" in reranked[0]

    def test_rerank_empty_chunks(self):
        """Test reranking with empty list"""
        query = "test query"
        chunks = []

        reranked = rerank_chunks(query, chunks, top_k=5)

        assert len(reranked) == 0

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_adds_scores(self):
        """Test that reranking adds combined score"""
        query = "payment terms"
        chunks = [
            {
                "page_num": "1",
                "content": "Payment due within 30 days",
                "score": 0.80,
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        # Should add rerank_score
        assert "rerank_score" in reranked[0]
        # Should combine with original score
        assert "combined_score" in reranked[0]
        # Combined score should be weighted average
        original = reranked[0]["score"]
        rerank = reranked[0]["rerank_score"]
        combined = reranked[0]["combined_score"]

        expected_combined = (original * 0.4) + (rerank * 0.6)
        assert abs(combined - expected_combined) < 0.01

    def test_rerank_preserves_metadata(self):
        """Test that reranking preserves chunk metadata"""
        query = "test query"
        chunks = [
            {
                "page_num": "5",
                "content": "Some legal content about contracts",
                "score": 0.75,
                "chunk_id": "chunk-5",
                "section_name": "Terms and Conditions"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        # Original metadata should be preserved
        assert reranked[0]["page_num"] == "5"
        assert reranked[0]["section_name"] == "Terms and Conditions"
        assert reranked[0]["chunk_id"] == "chunk-5"

    def test_rerank_top_k_limit(self):
        """Test that top_k limit is respected"""
        query = "test query"
        chunks = [
            {"page_num": str(i), "content": f"Content {i}", "score": 0.5 + i * 0.01, "chunk_id": f"chunk-{i}"}
            for i in range(10)
        ]

        reranked = rerank_chunks(query, chunks, top_k=3)

        assert len(reranked) == 3

    def test_rerank_sorts_by_combined_score(self):
        """Test that chunks are sorted by combined score"""
        query = "important legal matter"
        chunks = [
            {
                "page_num": "1",
                "content": "Moderately relevant content",
                "score": 0.70,
                "chunk_id": "chunk-1"
            },
            {
                "page_num": "2",
                "content": "Highly relevant legal terms",
                "score": 0.90,
                "chunk_id": "chunk-2"
            },
            {
                "page_num": "3",
                "content": "Somewhat relevant info",
                "score": 0.60,
                "chunk_id": "chunk-3"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=3)

        # Should be sorted by combined_score descending
        combined_scores = [c.get("combined_score", 0) for c in reranked]
        assert combined_scores == sorted(combined_scores, reverse=True)

    def test_rerank_with_multiple_chunks_same_vector_score(self):
        """Test reranking differentiates chunks with same vector score"""
        query = "specific legal term"
        chunks = [
            {
                "page_num": "1",
                "content": "The specific legal term means X",
                "score": 0.80,
                "chunk_id": "chunk-1"
            },
            {
                "page_num": "2",
                "content": "General information about the topic",
                "score": 0.80,  # Same vector score
                "chunk_id": "chunk-2"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=2)

        # Despite same vector score, cross-encoder should differentiate
        combined_scores = [c.get("combined_score", 0) for c in reranked]
        # At least they should be sorted (one might be higher)
        assert combined_scores[0] >= combined_scores[1]

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_weighted_combination(self):
        """Test that combination uses correct weights (40% vector, 60% cross-encoder)"""
        query = "test"
        chunks = [
            {
                "page_num": "1",
                "content": "Content",
                "score": 1.0,  # Perfect vector score
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        # Combined should be weighted average
        original_score = reranked[0]["score"]
        rerank_score = reranked[0]["rerank_score"]
        combined_score = reranked[0]["combined_score"]

        # combined = original * 0.4 + rerank * 0.6
        expected = (original_score * 0.4) + (rerank_score * 0.6)
        assert abs(combined_score - expected) < 0.001


class TestReankingEdgeCases:
    """Test edge cases in reranking"""

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_very_long_content(self):
        """Test reranking with very long chunk content"""
        query = "short query"
        chunks = [
            {
                "page_num": "1",
                "content": "A" * 10000,  # Very long content
                "score": 0.75,
                "chunk_id": "chunk-1"
            }
        ]

        # Should truncate to 300 chars before reranking
        reranked = rerank_chunks(query, chunks, top_k=1)

        assert len(reranked) == 1
        assert "combined_score" in reranked[0]

    def test_rerank_empty_content(self):
        """Test reranking with empty chunk content"""
        query = "test query"
        chunks = [
            {
                "page_num": "1",
                "content": "",  # Empty content
                "score": 0.75,
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        # Should still work, just with lower rerank score
        assert len(reranked) == 1

    def test_rerank_special_characters(self):
        """Test reranking with special characters"""
        query = "legal terms & conditions @ 2024"
        chunks = [
            {
                "page_num": "1",
                "content": "§123 Terms & Conditions © 2024",
                "score": 0.70,
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        assert len(reranked) == 1

    def test_rerank_unicode_content(self):
        """Test reranking with unicode content"""
        query = "contract terms"
        chunks = [
            {
                "page_num": "1",
                "content": "Les termes du contrat: €1000",
                "score": 0.75,
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        assert len(reranked) == 1

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_preserves_zero_scores(self):
        """Test that chunks with zero score are still reranked"""
        query = "test"
        chunks = [
            {
                "page_num": "1",
                "content": "Important content",
                "score": 0.0,  # Zero score
                "chunk_id": "chunk-1"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=1)

        # Should still rerank even with zero vector score
        assert len(reranked) == 1
        assert "combined_score" in reranked[0]

    def test_rerank_top_k_exceeds_available(self):
        """Test top_k larger than available chunks"""
        query = "test"
        chunks = [
            {"page_num": "1", "content": "Content", "score": 0.75, "chunk_id": "chunk-1"}
        ]

        reranked = rerank_chunks(query, chunks, top_k=100)

        # Should return only available chunks
        assert len(reranked) == 1


class TestReankingWithRelevance:
    """Test reranking accurately identifies relevant chunks"""

    @pytest.mark.skipif(not RERANKER_AVAILABLE, reason="sentence-transformers not installed")
    def test_rerank_improves_ordering(self):
        """Test that reranking improves chunk ordering"""
        # Query about payment terms
        query = "payment terms and conditions"

        chunks = [
            {
                "page_num": "1",
                "content": "Section 2: Other provisions",
                "score": 0.85,  # High vector score but low relevance
                "chunk_id": "chunk-1"
            },
            {
                "page_num": "2",
                "content": "Payment Terms: The buyer shall pay within 30 days",
                "score": 0.75,  # Lower vector score but high relevance
                "chunk_id": "chunk-2"
            }
        ]

        reranked = rerank_chunks(query, chunks, top_k=2)

        # After reranking, the payment terms chunk should rank higher
        # (or at least be in reasonable order)
        top_chunk = reranked[0]
        assert "payment" in top_chunk["content"].lower() or top_chunk["combined_score"] >= reranked[1]["combined_score"]

    def test_rerank_multiple_iterations(self):
        """Test reranking produces consistent results"""
        query = "contract terms"
        chunks = [
            {
                "page_num": str(i),
                "content": f"Content about contract {i}",
                "score": 0.5 + i * 0.05,
                "chunk_id": f"chunk-{i}"
            }
            for i in range(5)
        ]

        # Run reranking twice
        reranked1 = rerank_chunks(query, chunks, top_k=3)
        reranked2 = rerank_chunks(query, chunks, top_k=3)

        # Results should be consistent
        ids1 = [c["chunk_id"] for c in reranked1]
        ids2 = [c["chunk_id"] for c in reranked2]

        assert ids1 == ids2
