"""Tests for embeddings error handling."""

import pytest
from unittest.mock import patch, AsyncMock
from openai import APIError, RateLimitError, AuthenticationError

from app.services.embeddings import create_chunk_embeddings
from shared import Document, DocumentChunk


@pytest.mark.asyncio
async def test_rate_limit_error_propagates(db_session):
    """Test that rate limit errors are propagated without retries."""
    doc = Document(
        id="doc-err-1",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = DocumentChunk(
        id="chunk-err-1",
        document_id="doc-err-1",
        chunk_text="Legal text about contracts",
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        mock_gen.side_effect = RateLimitError("Rate limit exceeded. Please retry after 60 seconds.")

        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await create_chunk_embeddings(
                db_session,
                "doc-err-1",
                ["Legal text about contracts"],
            )


@pytest.mark.asyncio
async def test_invalid_api_key_error(db_session):
    """Test that invalid API key error is properly handled."""
    doc = Document(
        id="doc-err-2",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = DocumentChunk(
        id="chunk-err-2",
        document_id="doc-err-2",
        chunk_text="Legal text",
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        mock_gen.side_effect = AuthenticationError("Invalid API key")

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await create_chunk_embeddings(
                db_session,
                "doc-err-2",
                ["Legal text"],
            )


@pytest.mark.asyncio
async def test_batch_partial_failure_aborts(db_session):
    """Test that failure in middle of batch aborts operation."""
    doc = Document(
        id="doc-err-3",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    # Create 4 chunks (2 batches of 2)
    for i in range(4):
        chunk = DocumentChunk(
            id=f"chunk-err-{i}",
            document_id="doc-err-3",
            chunk_text=f"Text {i}",
            chunk_index=i,
        )
        db_session.add(chunk)
    await db_session.flush()

    # Fail on second batch
    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise APIError("Transient error during batch 2")
        return [[0.1] * 512, [0.2] * 512]

    with patch("app.services.embeddings.generate_embeddings_batch", side_effect=side_effect):
        with pytest.raises(APIError, match="Transient error"):
            await create_chunk_embeddings(
                db_session,
                "doc-err-3",
                [f"Text {i}" for i in range(4)],
                batch_size=2,
            )

    # Verify only first batch was processed
    assert call_count == 2, f"Expected 2 batch calls, got {call_count}"


@pytest.mark.asyncio
async def test_missing_openai_key_error(db_session):
    """Test that missing OPENAI_API_KEY raises ValueError."""
    doc = Document(
        id="doc-err-4",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = DocumentChunk(
        id="chunk-err-4",
        document_id="doc-err-4",
        chunk_text="Text",
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.settings") as mock_settings:
        mock_settings.OPENAI_API_KEY = None

        from app.services.embeddings import generate_embeddings_batch

        with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
            mock_gen.side_effect = ValueError("OPENAI_API_KEY not configured")

            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                await create_chunk_embeddings(
                    db_session,
                    "doc-err-4",
                    ["Text"],
                )


@pytest.mark.asyncio
async def test_chunk_mismatch_error(db_session):
    """Test that chunk/embedding count mismatch raises ValueError."""
    doc = Document(
        id="doc-err-5",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    # Create only 2 chunks
    for i in range(2):
        chunk = DocumentChunk(
            id=f"chunk-mismatch-{i}",
            document_id="doc-err-5",
            chunk_text=f"Text {i}",
            chunk_index=i,
        )
        db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        # Return 3 embeddings for 2 chunks (mismatch)
        mock_gen.return_value = [[0.1] * 512, [0.2] * 512, [0.3] * 512]

        with pytest.raises(ValueError, match="Chunk count mismatch"):
            await create_chunk_embeddings(
                db_session,
                "doc-err-5",
                ["Text 0", "Text 1"],
                batch_size=10,
            )


@pytest.mark.asyncio
async def test_empty_texts_validation_error():
    """Test that empty text list raises ValueError."""
    with pytest.raises(ValueError, match="No chunk texts provided"):
        import asyncio

        async def run_test():
            await create_chunk_embeddings(
                None,  # No session needed for validation
                "doc-err-6",
                [],  # Empty texts
            )

        asyncio.run(run_test())


@pytest.mark.asyncio
async def test_network_error_during_embedding(db_session):
    """Test that network errors during embedding are propagated."""
    doc = Document(
        id="doc-err-7",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = DocumentChunk(
        id="chunk-err-7",
        document_id="doc-err-7",
        chunk_text="Text",
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        mock_gen.side_effect = APIError("Connection timeout")

        with pytest.raises(APIError, match="Connection timeout"):
            await create_chunk_embeddings(
                db_session,
                "doc-err-7",
                ["Text"],
            )


@pytest.mark.asyncio
async def test_api_error_with_status_code(db_session):
    """Test handling of API errors with specific status codes."""
    doc = Document(
        id="doc-err-8",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    db_session.add(doc)
    await db_session.flush()

    chunk = DocumentChunk(
        id="chunk-err-8",
        document_id="doc-err-8",
        chunk_text="Text",
        chunk_index=0,
    )
    db_session.add(chunk)
    await db_session.flush()

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        error = APIError("Service unavailable")
        error.status_code = 503
        mock_gen.side_effect = error

        with pytest.raises(APIError):
            await create_chunk_embeddings(
                db_session,
                "doc-err-8",
                ["Text"],
            )
