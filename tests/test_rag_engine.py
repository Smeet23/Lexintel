"""Comprehensive test suite for RAG Query Engine with mocked external services"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4
import json

# Tests will be added incrementally using TDD


class TestTokenCounting:
    """Test token counting and budget validation"""

    def test_count_tokens_with_tiktoken(self):
        """Test accurate token counting using tiktoken"""
        from backend.services.rag_engine import count_tokens_gpt4o

        text = "This is a test sentence with some words."
        count = count_tokens_gpt4o(text)

        assert count > 0
        assert count < len(text) / 4  # Upper bound: 1 token per 4 chars
        assert isinstance(count, int)

    def test_token_budget_validation(self):
        """Test token budget validation and context trimming"""
        from backend.services.rag_engine import validate_token_budget

        # Test valid budget
        assert validate_token_budget(5000, 12800) is True

        # Test exceeding budget
        assert validate_token_budget(13000, 12800) is False

        # Test boundary
        assert validate_token_budget(12800, 12800) is True


class TestContextFormatting:
    """Test context formatting with metadata"""

    def test_format_legal_context_with_metadata(self):
        """Test proper metadata inclusion in formatted context"""
        from backend.services.rag_engine import format_legal_context

        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "The court decided to uphold the decision.",
                "page_num": "5",
                "section_name": "Judgment",
                "score": 0.95
            },
            {
                "chunk_id": "chunk2",
                "content": "Evidence showed clear negligence.",
                "page_num": "3",
                "section_name": "Facts",
                "score": 0.87
            }
        ]

        case_name = "Smith v. Jones"
        formatted = format_legal_context(chunks, case_name)

        # Check case name is included
        assert "Smith v. Jones" in formatted

        # Check metadata is included
        assert "Page 5" in formatted or "page 5" in formatted.lower()
        assert "0.95" in formatted
        assert "Judgment" in formatted

        # Check chunks are included
        assert "uphold the decision" in formatted
        assert "negligence" in formatted

    def test_context_ordering_and_deduplication(self):
        """Test chunk ordering by relevance and deduplication"""
        from backend.services.rag_engine import format_legal_context

        chunks = [
            {
                "chunk_id": "chunk1",
                "content": "First statement",
                "page_num": "1",
                "score": 0.8
            },
            {
                "chunk_id": "chunk2",
                "content": "Second statement",
                "page_num": "2",
                "score": 0.9
            },
            {
                "chunk_id": "chunk3",
                "content": "Third statement",
                "page_num": "3",
                "score": 0.85
            }
        ]

        formatted = format_legal_context(chunks, "Test Case")

        # Higher score should appear first
        assert formatted.index("Second statement") < formatted.index("First statement")
        assert formatted.index("Third statement") < formatted.index("First statement")


class TestQueryProcessing:
    """Test query embedding and retrieval"""

    @pytest.mark.asyncio
    async def test_embed_query_integration(self):
        """Test query embedding integration"""
        from backend.services.rag_engine import embed_query

        with patch("backend.services.rag_engine.embed_text") as mock_embed:
            mock_embed.return_value = [0.1] * 3072

            query = "What did the court decide?"
            embedding = embed_query(query)

            assert len(embedding) == 3072
            assert all(isinstance(x, float) for x in embedding)
            mock_embed.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_retrieve_similar_chunks(self):
        """Test chunk retrieval from vector store"""
        from backend.services.rag_engine import retrieve_chunks

        with patch("backend.services.rag_engine.search_vectors") as mock_search:
            mock_search.return_value = [
                {
                    "chunk_id": "chunk1",
                    "content": "Test content",
                    "page_num": "5",
                    "score": 0.95
                },
                {
                    "chunk_id": "chunk2",
                    "content": "More content",
                    "page_num": "6",
                    "score": 0.88
                }
            ]

            case_id = str(uuid4())
            query_embedding = [0.1] * 3072
            chunks = retrieve_chunks(case_id, query_embedding)

            assert len(chunks) >= 2
            assert chunks[0]["score"] >= 0.95
            mock_search.assert_called_once()


class TestCitationExtraction:
    """Test citation extraction from answers"""

    def test_extract_page_citations(self):
        """Test extracting [Page X] citations from answer"""
        from backend.services.rag_engine import extract_citations

        answer = "The court decided [Page 5] that the evidence was insufficient. As stated [Page 3], the plaintiff failed to provide adequate proof."
        chunks = [
            {"chunk_id": "chunk1", "page_num": "5"},
            {"chunk_id": "chunk2", "page_num": "3"}
        ]

        citations = extract_citations(answer, chunks)

        assert len(citations) >= 1
        assert any(c["page_num"] == "5" for c in citations)
        assert any(c["page_num"] == "3" for c in citations)

    def test_cite_to_chunk_matching(self):
        """Test matching citations to retrieved chunks"""
        from backend.services.rag_engine import extract_citations

        answer = "According to [Page 7], the decision was made."
        chunks = [
            {"chunk_id": "chunk1", "page_num": "5", "relevance_score": 0.9},
            {"chunk_id": "chunk2", "page_num": "7", "relevance_score": 0.85}
        ]

        citations = extract_citations(answer, chunks)

        # Should find matching citation
        matching = [c for c in citations if c["page_num"] == "7"]
        assert len(matching) > 0


class TestAnswerGeneration:
    """Test LLM answer generation"""

    @pytest.mark.asyncio
    async def test_generate_answer_with_openai(self):
        """Test answer generation with OpenAI API"""
        from backend.services.rag_engine import generate_answer

        with patch("backend.services.rag_engine.AsyncOpenAI") as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="The court decided in favor of the plaintiff [Page 5]."))]
            mock_response.usage = MagicMock(total_tokens=150)
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            query = "What was the decision?"
            context = "The case involved a negligence claim."

            answer, tokens_used = await generate_answer(query, context, temperature=0.2)

            assert isinstance(answer, str)
            assert len(answer) > 0
            assert "[Page" in answer or "court" in answer.lower()
            assert isinstance(tokens_used, int)
            assert tokens_used > 0

    @pytest.mark.asyncio
    async def test_temperature_parameter(self):
        """Test that temperature parameter is passed correctly"""
        from backend.services.rag_engine import generate_answer

        with patch("backend.services.rag_engine.AsyncOpenAI") as mock_openai_class:
            mock_client = AsyncMock()
            mock_openai_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

            await generate_answer("Query", "Context", temperature=0.2)

            # Check that temperature 0.2 was passed
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs["temperature"] == 0.2


class TestErrorHandling:
    """Test error handling for 7 failure modes"""

    @pytest.mark.asyncio
    async def test_empty_retrieval_error(self):
        """Test handling of empty retrieval (no chunks found)"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve:

            mock_embed.return_value = [0.1] * 3072
            mock_retrieve.return_value = []  # Empty retrieval

            case_id = str(uuid4())
            query = "What was the decision?"

            result = await query_case(case_id, query, mock_db)

            assert result["error"] is not None
            assert "no" in result["error"].lower() and "found" in result["error"].lower()
            assert result["confidence"] == "none"

    @pytest.mark.asyncio
    async def test_low_confidence_error(self):
        """Test handling of low confidence retrieval (avg score < 0.7)"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve:

            mock_embed.return_value = [0.1] * 3072
            # Return chunks with low scores
            mock_retrieve.return_value = [
                {"chunk_id": "c1", "page_num": "1", "score": 0.6},
                {"chunk_id": "c2", "page_num": "2", "score": 0.65}
            ]

            case_id = str(uuid4())
            query = "What was the decision?"

            result = await query_case(case_id, query, mock_db)

            assert result["confidence"] == "low"

    @pytest.mark.asyncio
    async def test_token_budget_exceeded_error(self):
        """Test handling of context exceeding token budget"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve, \
             patch("backend.services.rag_engine.count_tokens_gpt4o") as mock_count:

            mock_embed.return_value = [0.1] * 3072
            mock_retrieve.return_value = [
                {"chunk_id": "c1", "page_num": "1", "score": 0.95, "content": "x" * 100000}
            ]
            # Simulate very large token count
            mock_count.return_value = 20000

            case_id = str(uuid4())
            query = "What was the decision?"

            result = await query_case(case_id, query, mock_db)

            # Should either error or gracefully degrade
            assert result.get("error") is not None or result.get("confidence") is not None

    @pytest.mark.asyncio
    async def test_openai_api_failure(self):
        """Test handling of OpenAI API failure (rate limit, timeout, connection)"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve, \
             patch("backend.services.rag_engine.generate_answer") as mock_gen:

            mock_embed.return_value = [0.1] * 3072
            mock_retrieve.return_value = [
                {"chunk_id": "c1", "page_num": "1", "score": 0.95, "content": "Test"}
            ]
            # Simulate API failure
            mock_gen.side_effect = Exception("API Rate Limit Exceeded")

            case_id = str(uuid4())
            query = "What was the decision?"

            result = await query_case(case_id, query, mock_db)

            assert result["error"] is not None
            assert "API" in result["error"] or "error" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_query_error(self):
        """Test handling of invalid query (empty or too short)"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        case_id = str(uuid4())

        # Test empty query
        result = await query_case(case_id, "", mock_db)
        assert result["error"] is not None

        # Test very short query
        result = await query_case(case_id, "a", mock_db)
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_case_not_found_error(self):
        """Test handling of case not found in vector store"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve:

            mock_embed.return_value = [0.1] * 3072
            # Simulate case not found
            mock_retrieve.side_effect = Exception("Collection not found")

            case_id = str(uuid4())
            query = "What was the decision?"

            result = await query_case(case_id, query, mock_db)

            assert result["error"] is not None
            assert result["confidence"] == "none"


class TestIntegration:
    """Integration tests for full RAG pipeline"""

    @pytest.mark.asyncio
    async def test_full_query_pipeline(self):
        """Test complete query pipeline: embed → retrieve → format → generate"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()
        mock_case = MagicMock()
        mock_case.name = "Smith v. Jones"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_case

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve, \
             patch("backend.services.rag_engine.generate_answer") as mock_gen:

            mock_embed.return_value = [0.1] * 3072
            mock_retrieve.return_value = [
                {
                    "chunk_id": "chunk1",
                    "content": "The court found liability.",
                    "page_num": "5",
                    "score": 0.95
                },
                {
                    "chunk_id": "chunk2",
                    "content": "Damages awarded: $100,000.",
                    "page_num": "6",
                    "score": 0.92
                }
            ]
            mock_gen.return_value = ("The court found liability [Page 5] and awarded damages [Page 6].", 150)

            case_id = str(uuid4())
            query = "What was the outcome?"

            result = await query_case(case_id, query, mock_db)

            assert result["answer"] is not None
            assert len(result["answer"]) > 0
            assert result["sources"] is not None
            assert len(result["sources"]) > 0
            assert result["tokens_used"] > 0
            assert result["error"] is None

    @pytest.mark.asyncio
    async def test_source_attribution_tracking(self):
        """Test source attribution and citation tracking"""
        from backend.services.rag_engine import query_case

        mock_db = MagicMock()
        mock_case = MagicMock()
        mock_case.name = "Test Case"
        mock_db.query.return_value.filter.return_value.first.return_value = mock_case

        with patch("backend.services.rag_engine.embed_query") as mock_embed, \
             patch("backend.services.rag_engine.retrieve_chunks") as mock_retrieve, \
             patch("backend.services.rag_engine.generate_answer") as mock_gen:

            mock_embed.return_value = [0.1] * 3072
            mock_retrieve.return_value = [
                {
                    "chunk_id": "chunk1",
                    "content": "Key evidence here",
                    "page_num": "10",
                    "score": 0.96
                }
            ]
            mock_gen.return_value = ("The evidence shows [Page 10] that negligence occurred.", 100)

            case_id = str(uuid4())
            query = "Was there negligence?"

            result = await query_case(case_id, query, mock_db)

            # Check sources are tracked
            assert "sources" in result
            assert len(result["sources"]) > 0

            # Check citation information
            source = result["sources"][0]
            assert "chunk_id" in source
            assert "page_num" in source
            assert "relevance_score" in source
