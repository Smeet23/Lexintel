"""Tests for OpenAI embedding generation."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from shared.embeddings import generate_embeddings_batch


def test_generate_embeddings_batch_empty_input():
    """Test that empty input raises ValueError."""
    with pytest.raises(ValueError, match="No texts provided"):
        import asyncio
        asyncio.run(generate_embeddings_batch([]))


def test_generate_embeddings_batch_empty_string():
    """Test that empty string in input raises ValueError."""
    with pytest.raises(ValueError, match="Empty text at index"):
        import asyncio
        asyncio.run(generate_embeddings_batch(["valid text", ""]))


@pytest.mark.asyncio
async def test_generate_embeddings_batch_success():
    """Test successful embedding generation."""
    mock_response = AsyncMock()
    mock_response.data = [
        AsyncMock(embedding=[0.1] * 512),
        AsyncMock(embedding=[0.2] * 512),
    ]

    with patch("shared.config.settings") as mock_settings, \
         patch("shared.embeddings.AsyncOpenAI") as mock_openai:
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        texts = ["Legal document text", "Patent description"]
        embeddings = await generate_embeddings_batch(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 512
        assert all(isinstance(e, float) for e in embeddings[0])

        # Verify API was called correctly
        mock_client.embeddings.create.assert_called_once()
        call_kwargs = mock_client.embeddings.create.call_args[1]
        assert call_kwargs["model"] == "text-embedding-3-small"
        assert call_kwargs["input"] == texts


@pytest.mark.asyncio
async def test_generate_embeddings_batch_missing_api_key():
    """Test that missing API key raises ValueError."""
    with patch("shared.config.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = None

        with pytest.raises(ValueError, match="OPENAI_API_KEY not configured"):
            await generate_embeddings_batch(["text"])


@pytest.mark.asyncio
async def test_generate_embeddings_batch_api_error():
    """Test that OpenAI API errors are propagated."""
    from openai import APIError
    from httpx import Request

    with patch("shared.config.settings") as mock_settings, \
         patch("shared.embeddings.AsyncOpenAI") as mock_openai:
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_client = AsyncMock()

        # Create a proper APIError with required request argument
        request = Request("POST", "https://api.openai.com/v1/embeddings")
        api_error = APIError("Rate limit exceeded", request=request, body={})
        mock_client.embeddings.create = AsyncMock(side_effect=api_error)
        mock_openai.return_value = mock_client

        with pytest.raises(APIError):
            await generate_embeddings_batch(["text"])


def test_generate_embeddings_batch_custom_model():
    """Test that custom embedding model is used."""
    import asyncio

    with patch("shared.config.settings") as mock_settings, \
         patch("shared.embeddings.AsyncOpenAI") as mock_openai:
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_response = AsyncMock()
        mock_response.data = [AsyncMock(embedding=[0.1] * 512)]

        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        async def run_test():
            await generate_embeddings_batch(
                ["text"],
                model="text-embedding-3-large"
            )
            call_kwargs = mock_client.embeddings.create.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-large"

        asyncio.run(run_test())
