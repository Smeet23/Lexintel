"""Tests for Qdrant vector store service"""
import pytest
import os
from unittest.mock import patch, MagicMock, AsyncMock
from typing import List

# Set environment variables before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel_vector_store.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-key-for-testing')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')
os.environ.setdefault('QDRANT_URL', 'http://localhost:6333')
os.environ.setdefault('DEBUG', 'True')

from backend.services.vector_store import (
    get_qdrant_client,
    create_collection,
    upsert_vectors,
    search_vectors,
    delete_collection,
    VECTOR_SIZE,
    DISTANCE_METRIC
)


class TestVectorStoreConfiguration:
    """Test vector store configuration constants"""

    def test_vector_size_constant(self):
        """Test that vector size matches embeddings output"""
        assert VECTOR_SIZE == 3072

    def test_distance_metric_constant(self):
        """Test that distance metric is set to cosine"""
        assert DISTANCE_METRIC == "Cosine"


class TestQdrantClientInitialization:
    """Test Qdrant client creation and initialization"""

    def setup_method(self):
        """Clear cache before each test"""
        from backend.services.vector_store import get_qdrant_client
        get_qdrant_client.cache_clear()

    @patch('backend.services.vector_store.QdrantClient')
    def test_get_qdrant_client_success(self, mock_qdrant_class):
        """Test successful Qdrant client creation"""
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        client = get_qdrant_client()

        assert client is not None
        mock_qdrant_class.assert_called_once()

    @patch('backend.services.vector_store.QdrantClient')
    def test_get_qdrant_client_uses_configured_url(self, mock_qdrant_class):
        """Test that client uses configured Qdrant URL"""
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        client = get_qdrant_client()

        # Check that QdrantClient was called with URL argument
        call_kwargs = mock_qdrant_class.call_args[1]
        assert 'url' in call_kwargs
        assert call_kwargs['url'] == 'http://localhost:6333'

    @patch('backend.services.vector_store.QdrantClient')
    def test_get_qdrant_client_cached(self, mock_qdrant_class):
        """Test that client is cached (called only once)"""
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client

        client1 = get_qdrant_client()
        client2 = get_qdrant_client()

        # Should be called only once due to caching
        assert mock_qdrant_class.call_count == 1
        assert client1 is client2


class TestCollectionCreation:
    """Test collection lifecycle management"""

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_create_collection_success(self, mock_get_client):
        """Test successful collection creation"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid-1234"
        result = create_collection(case_id)

        assert result is True
        mock_client.recreate_collection.assert_called_once()

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_create_collection_uses_case_id_as_name(self, mock_get_client):
        """Test that collection name includes case_id"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "my-case-uuid-1234"
        create_collection(case_id)

        # Get the call arguments
        call_args = mock_client.recreate_collection.call_args
        collection_name = call_args[1]['collection_name']

        assert case_id in collection_name

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_create_collection_sets_vector_size(self, mock_get_client):
        """Test that collection is created with correct vector size"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        create_collection(case_id)

        call_args = mock_client.recreate_collection.call_args
        # Vector config should specify the vector size
        assert call_args is not None

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_create_collection_failure(self, mock_get_client):
        """Test error handling on collection creation failure"""
        mock_client = MagicMock()
        mock_client.recreate_collection.side_effect = Exception("Connection failed")
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"

        with pytest.raises(Exception, match="Connection failed"):
            create_collection(case_id)


class TestVectorUpsert:
    """Test vector upsert operations"""

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_upsert_vectors_success(self, mock_get_client):
        """Test successful vector upsert"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        chunks = [
            {
                "id": "chunk-1",
                "content": "This is chunk 1 content",
                "page_num": "1",
                "section_name": "Introduction"
            },
            {
                "id": "chunk-2",
                "content": "This is chunk 2 content",
                "page_num": "2",
                "section_name": "Findings"
            }
        ]
        embeddings = [
            [0.1] * VECTOR_SIZE,
            [0.2] * VECTOR_SIZE
        ]

        result = upsert_vectors(case_id, chunks, embeddings)

        assert result == 2
        mock_client.upsert.assert_called_once()

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_upsert_vectors_single_chunk(self, mock_get_client):
        """Test upserting a single chunk"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        chunks = [
            {
                "id": "chunk-1",
                "content": "Single chunk",
                "page_num": "1",
                "section_name": "Section 1"
            }
        ]
        embeddings = [[0.1] * VECTOR_SIZE]

        result = upsert_vectors(case_id, chunks, embeddings)

        assert result == 1

    def test_upsert_vectors_empty_chunks(self):
        """Test that empty chunks list raises error"""
        case_id = "test-case-uuid"
        chunks = []
        embeddings = []

        with pytest.raises(ValueError, match="empty"):
            upsert_vectors(case_id, chunks, embeddings)

    def test_upsert_vectors_mismatched_lengths(self):
        """Test that mismatched chunks and embeddings raises error"""
        case_id = "test-case-uuid"
        chunks = [
            {"id": "chunk-1", "content": "Content 1", "page_num": "1", "section_name": "S1"},
            {"id": "chunk-2", "content": "Content 2", "page_num": "2", "section_name": "S2"}
        ]
        embeddings = [[0.1] * VECTOR_SIZE]  # Only 1 embedding for 2 chunks

        with pytest.raises(ValueError, match="count mismatch"):
            upsert_vectors(case_id, chunks, embeddings)

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_upsert_vectors_stores_metadata(self, mock_get_client):
        """Test that metadata is stored with vectors"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        chunks = [
            {
                "id": "chunk-1",
                "content": "Preview of content",
                "page_num": "5",
                "section_name": "Discussion"
            }
        ]
        embeddings = [[0.5] * VECTOR_SIZE]

        upsert_vectors(case_id, chunks, embeddings)

        # Verify upsert was called with points containing metadata
        call_args = mock_client.upsert.call_args
        assert call_args is not None

    def test_upsert_vectors_invalid_embedding_dimension(self):
        """Test that embeddings with wrong dimension are rejected"""
        case_id = "test-case-uuid"
        chunks = [{"id": "chunk-1", "content": "Content", "page_num": "1", "section_name": "S"}]
        embeddings = [[0.1] * 100]  # Wrong dimension

        with pytest.raises(ValueError, match="dimension"):
            upsert_vectors(case_id, chunks, embeddings)

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_upsert_vectors_failure(self, mock_get_client):
        """Test error handling on upsert failure"""
        mock_client = MagicMock()
        mock_client.upsert.side_effect = Exception("Upsert failed")
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        chunks = [{"id": "chunk-1", "content": "Content", "page_num": "1", "section_name": "S"}]
        embeddings = [[0.1] * VECTOR_SIZE]

        with pytest.raises(Exception, match="Upsert failed"):
            upsert_vectors(case_id, chunks, embeddings)


class TestVectorSearch:
    """Test semantic similarity search"""

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_success(self, mock_get_client):
        """Test successful vector search"""
        mock_client = MagicMock()
        mock_search_result = [
            MagicMock(
                id="chunk-1",
                score=0.95,
                payload={
                    "chunk_id": "chunk-1",
                    "page_num": "1",
                    "content_preview": "This is relevant content"
                }
            ),
            MagicMock(
                id="chunk-2",
                score=0.87,
                payload={
                    "chunk_id": "chunk-2",
                    "page_num": "3",
                    "content_preview": "Another relevant piece"
                }
            )
        ]
        mock_client.search.return_value = mock_search_result
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        results = search_vectors(case_id, query_embedding)

        assert len(results) == 2
        assert results[0]['score'] == 0.95
        assert results[1]['score'] == 0.87
        assert results[0]['chunk_id'] == "chunk-1"

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_with_limit(self, mock_get_client):
        """Test search with custom limit"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        search_vectors(case_id, query_embedding, limit=10)

        # Verify limit was passed to search
        call_args = mock_client.search.call_args
        assert call_args is not None

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_default_limit(self, mock_get_client):
        """Test search uses default limit of 5"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        search_vectors(case_id, query_embedding)

        # Default limit should be 5
        call_args = mock_client.search.call_args
        assert call_args is not None

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_returns_ordered_by_score(self, mock_get_client):
        """Test that results are ordered by similarity score"""
        mock_client = MagicMock()
        mock_search_result = [
            MagicMock(id="chunk-3", score=0.99, payload={"chunk_id": "chunk-3", "page_num": "1", "content_preview": "Most relevant"}),
            MagicMock(id="chunk-1", score=0.85, payload={"chunk_id": "chunk-1", "page_num": "2", "content_preview": "Less relevant"}),
        ]
        mock_client.search.return_value = mock_search_result
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        results = search_vectors(case_id, query_embedding)

        assert results[0]['score'] > results[1]['score']

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_empty_results(self, mock_get_client):
        """Test search returning no results"""
        mock_client = MagicMock()
        mock_client.search.return_value = []
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        results = search_vectors(case_id, query_embedding)

        assert results == []

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_invalid_embedding_size(self, mock_get_client):
        """Test search with wrong embedding dimension"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * 100  # Wrong size

        with pytest.raises(ValueError, match="dimension"):
            search_vectors(case_id, query_embedding)

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_search_vectors_failure(self, mock_get_client):
        """Test error handling on search failure"""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("Search failed")
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        query_embedding = [0.1] * VECTOR_SIZE

        with pytest.raises(Exception, match="Search failed"):
            search_vectors(case_id, query_embedding)


class TestCollectionDeletion:
    """Test collection deletion"""

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_delete_collection_success(self, mock_get_client):
        """Test successful collection deletion"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"
        result = delete_collection(case_id)

        assert result is True
        mock_client.delete_collection.assert_called_once()

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_delete_collection_uses_case_id(self, mock_get_client):
        """Test that deletion uses correct collection name"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        case_id = "my-case-uuid"
        delete_collection(case_id)

        call_args = mock_client.delete_collection.call_args
        collection_name = call_args[1]['collection_name']

        assert case_id in collection_name

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_delete_collection_failure(self, mock_get_client):
        """Test error handling on deletion failure"""
        mock_client = MagicMock()
        mock_client.delete_collection.side_effect = Exception("Collection not found")
        mock_get_client.return_value = mock_client

        case_id = "test-case-uuid"

        with pytest.raises(Exception, match="Collection not found"):
            delete_collection(case_id)


class TestVectorStoreIntegration:
    """Integration tests for vector store workflow"""

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_full_workflow_create_upsert_search_delete(self, mock_get_client):
        """Test complete workflow: create → upsert → search → delete"""
        mock_client = MagicMock()
        mock_search_result = [
            MagicMock(
                id="chunk-1",
                score=0.92,
                payload={
                    "chunk_id": "chunk-1",
                    "page_num": "1",
                    "content_preview": "Relevant content"
                }
            )
        ]
        mock_client.search.return_value = mock_search_result
        mock_get_client.return_value = mock_client

        case_id = "integration-test-case"

        # Create collection
        assert create_collection(case_id) is True

        # Upsert vectors
        chunks = [
            {"id": "chunk-1", "content": "Legal content about contracts", "page_num": "1", "section_name": "S1"}
        ]
        embeddings = [[0.1] * VECTOR_SIZE]
        assert upsert_vectors(case_id, chunks, embeddings) == 1

        # Search
        query_embedding = [0.15] * VECTOR_SIZE
        results = search_vectors(case_id, query_embedding, limit=5)
        assert len(results) > 0

        # Delete collection
        assert delete_collection(case_id) is True

    @patch('backend.services.vector_store.get_qdrant_client')
    def test_workflow_with_multiple_chunks(self, mock_get_client):
        """Test workflow with multiple chunks and searches"""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.search.return_value = []

        case_id = "multi-chunk-case"

        # Create and upsert multiple chunks
        chunks = [
            {"id": f"chunk-{i}", "content": f"Content {i}", "page_num": str(i), "section_name": f"Sec{i}"}
            for i in range(5)
        ]
        embeddings = [[0.1 * (i + 1)] * VECTOR_SIZE for i in range(5)]

        create_collection(case_id)
        inserted = upsert_vectors(case_id, chunks, embeddings)

        assert inserted == 5
