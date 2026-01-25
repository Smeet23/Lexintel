"""Tests for OpenAI embeddings service"""
import pytest
import os
from unittest.mock import patch, MagicMock

# Set environment variables before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel_embeddings.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-key-for-testing')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')
os.environ.setdefault('DEBUG', 'True')

from backend.services.embeddings import (
    get_embeddings_client,
    embed_text,
    embed_chunks,
    estimate_embedding_cost,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL
)


class TestEmbeddingsConfiguration:
    """Test embeddings configuration"""

    def test_embedding_model_constant(self):
        """Test that embedding model is correctly configured"""
        assert EMBEDDING_MODEL == "text-embedding-3-large"

    def test_embedding_dimensions_constant(self):
        """Test that embedding dimensions match model output"""
        assert EMBEDDING_DIMENSIONS == 3072


class TestEmbeddingsClient:
    """Test embeddings client creation"""

    @patch('backend.services.embeddings.OpenAIEmbeddings')
    def test_get_embeddings_client_success(self, mock_openai_class):
        """Test successful client creation"""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        client = get_embeddings_client()

        assert client is not None
        mock_openai_class.assert_called_once()
        call_kwargs = mock_openai_class.call_args[1]
        assert call_kwargs['model'] == EMBEDDING_MODEL
        assert 'openai_api_key' in call_kwargs

    def test_get_embeddings_client_missing_api_key(self):
        """Test client creation fails without API key"""
        # Temporarily remove API key
        from backend.config import get_settings
        settings = get_settings()
        original_key = settings.openai_api_key
        settings.openai_api_key = None

        try:
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                get_embeddings_client()
        finally:
            settings.openai_api_key = original_key


class TestTextEmbedding:
    """Test single text embedding"""

    @patch('backend.services.embeddings.get_embeddings_client')
    def test_embed_text_success(self, mock_get_client):
        """Test successful text embedding"""
        # Mock embedding response
        mock_embedding = [0.1] * EMBEDDING_DIMENSIONS
        mock_client = MagicMock()
        mock_client.embed_query.return_value = mock_embedding
        mock_get_client.return_value = mock_client

        text = "What is the court judgment in this case?"
        result = embed_text(text)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == EMBEDDING_DIMENSIONS
        assert all(isinstance(x, float) for x in result)

    def test_embed_text_empty(self):
        """Test embedding empty text raises error"""
        with pytest.raises(ValueError, match="empty"):
            embed_text("")

    def test_embed_text_whitespace_only(self):
        """Test embedding whitespace-only text raises error"""
        with pytest.raises(ValueError, match="empty"):
            embed_text("   \n\t  ")

    @patch('backend.services.embeddings.get_embeddings_client')
    def test_embed_text_api_failure(self, mock_get_client):
        """Test embedding handles API failures"""
        mock_client = MagicMock()
        mock_client.embed_query.side_effect = Exception("API Error")
        mock_get_client.return_value = mock_client

        with pytest.raises(Exception, match="API Error"):
            embed_text("Sample text")


class TestChunkEmbedding:
    """Test batch chunk embedding"""

    @patch('backend.services.embeddings.get_embeddings_client')
    def test_embed_chunks_success(self, mock_get_client):
        """Test successful batch embedding"""
        # Mock embeddings response
        num_chunks = 3
        mock_embeddings = [[0.1] * EMBEDDING_DIMENSIONS for _ in range(num_chunks)]
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = mock_embeddings
        mock_get_client.return_value = mock_client

        chunks = [
            "This is chunk 1 about the plaintiff's arguments",
            "This is chunk 2 about the defendant's response",
            "This is chunk 3 about the court's judgment"
        ]
        result = embed_chunks(chunks)

        assert result is not None
        assert isinstance(result, list)
        assert len(result) == num_chunks

        for embedding in result:
            assert len(embedding) == EMBEDDING_DIMENSIONS
            assert all(isinstance(x, float) for x in embedding)

    def test_embed_chunks_empty_list(self):
        """Test embedding empty chunks list raises error"""
        with pytest.raises(ValueError, match="empty"):
            embed_chunks([])

    def test_embed_chunks_with_empty_chunk(self):
        """Test embedding chunks containing empty strings raises error"""
        chunks = ["Valid chunk", "", "Another valid chunk"]

        with pytest.raises(ValueError, match="empty"):
            embed_chunks(chunks)

    @patch('backend.services.embeddings.get_embeddings_client')
    def test_embed_chunks_dimension_mismatch(self, mock_get_client):
        """Test error handling for dimension mismatches"""
        mock_client = MagicMock()
        # Return wrong number of embeddings
        mock_client.embed_documents.return_value = [[0.1] * EMBEDDING_DIMENSIONS]
        mock_get_client.return_value = mock_client

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]

        with pytest.raises(ValueError, match="count mismatch"):
            embed_chunks(chunks)


class TestCostEstimation:
    """Test embedding cost estimation"""

    def test_estimate_cost_short_text(self):
        """Test cost estimation for short text"""
        # 100 characters ≈ 25 tokens
        cost = estimate_embedding_cost(100)
        assert cost >= 0
        assert cost < 0.01  # Should be very cheap

    def test_estimate_cost_medium_text(self):
        """Test cost estimation for medium text"""
        # 10,000 characters ≈ 2,500 tokens ≈ $0.00005
        cost = estimate_embedding_cost(10_000)
        assert cost >= 0
        assert cost < 0.0001

    def test_estimate_cost_large_text(self):
        """Test cost estimation for large text"""
        # 1M characters ≈ 250k tokens ≈ $0.005
        cost = estimate_embedding_cost(1_000_000)
        assert cost >= 0.004
        assert cost <= 0.01

    def test_estimate_cost_zero(self):
        """Test cost estimation for zero text"""
        cost = estimate_embedding_cost(0)
        assert cost == 0.0


class TestEmbeddingsIntegration:
    """Integration tests for embeddings"""

    @patch('backend.services.embeddings.get_embeddings_client')
    def test_embed_chunks_from_chunking_service(self, mock_get_client):
        """Test embeddings work with chunks from chunking service"""
        # Simulate chunks from chunking service
        chunks = [
            "Page 1, Chunk 1: The plaintiff argues that the contract was breached.",
            "Page 2, Chunk 2: The defendant claims the clause was ambiguous.",
            "Page 3, Chunk 3: The court finds in favor of the plaintiff."
        ]

        # Mock embedding response
        mock_embeddings = [[0.1] * EMBEDDING_DIMENSIONS for _ in chunks]
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = mock_embeddings
        mock_get_client.return_value = mock_client

        # Should succeed with realistic chunks
        result = embed_chunks(chunks)

        assert len(result) == len(chunks)
        assert all(len(emb) == EMBEDDING_DIMENSIONS for emb in result)
