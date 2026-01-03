"""Integration tests for extraction with embedding generation."""

import pytest
from unittest.mock import patch, AsyncMock
from sqlalchemy import select

from shared import Document, DocumentChunk, ProcessingStatus
from app.services.extraction import create_text_chunks, extract_and_embed_document
from app.services.embeddings import create_chunk_embeddings


@pytest.mark.asyncio
async def test_extraction_with_embeddings_integration(async_session):
    """Test complete extraction + embedding flow."""
    # Create test document
    doc = Document(
        id="doc-embed-1",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    async_session.add(doc)
    await async_session.commit()

    # Prepare test text
    test_text = "Legal text about contracts and agreements. " * 50  # Long text to create multiple chunks

    # Mock embeddings generation
    mock_embeddings = [[float(i) / 512.0] * 512 for i in range(5)]

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        # Setup mock to return different embeddings for different calls
        mock_gen.side_effect = [
            [[float(i) / 512.0] * 512 for i in range(3)],  # First batch
            [[float(i+3) / 512.0] * 512 for i in range(2)],  # Second batch
        ]

        # Run extraction with embeddings
        chunk_ids = await extract_and_embed_document(
            async_session,
            doc.id,
            test_text,
            chunk_size=500,
            overlap=50,
            batch_size=3,
        )

    # Verify chunks were created
    assert len(chunk_ids) > 0

    # Verify chunks exist in database
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    result = await async_session.execute(stmt)
    db_chunks = result.scalars().all()

    assert len(db_chunks) > 0
    assert all(chunk.embedding is not None for chunk in db_chunks)
    assert len(db_chunks[0].embedding) == 512


@pytest.mark.asyncio
async def test_embedding_failure_preserves_extracted_text(async_session):
    """Test that embedding failure keeps document in EXTRACTED status."""
    from openai import APIError

    # Create document with extracted text
    test_text = "Extracted legal text about contracts"
    doc = Document(
        id="doc-embed-2",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
        extracted_text=test_text,
        processing_status=ProcessingStatus.EXTRACTED,
    )
    async_session.add(doc)
    await async_session.commit()

    # Create chunks
    chunk = DocumentChunk(
        id="chunk-fail-1",
        document_id="doc-embed-2",
        chunk_text=test_text,
        chunk_index=0,
    )
    async_session.add(chunk)
    await async_session.commit()

    # Mock embedding failure
    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        mock_gen.side_effect = APIError("Rate limit exceeded")

        with pytest.raises(APIError):
            await create_chunk_embeddings(
                async_session,
                "doc-embed-2",
                [test_text],
            )

    # Verify document still has extracted text
    stmt = select(Document).where(Document.id == "doc-embed-2")
    result = await async_session.execute(stmt)
    db_doc = result.scalar_one()
    assert db_doc.extracted_text == test_text
    assert db_doc.processing_status == ProcessingStatus.EXTRACTED


@pytest.mark.asyncio
async def test_chunks_have_embeddings_after_generation(async_session):
    """Test that chunks are updated with embeddings after generation."""
    # Create document and chunks
    doc = Document(
        id="doc-embed-3",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    async_session.add(doc)
    await async_session.flush()

    chunks = [
        DocumentChunk(
            id="chunk1",
            document_id="doc-embed-3",
            chunk_text="Text 1 about contracts",
            chunk_index=0,
        ),
        DocumentChunk(
            id="chunk2",
            document_id="doc-embed-3",
            chunk_text="Text 2 about patents",
            chunk_index=1,
        ),
    ]
    async_session.add_all(chunks)
    await async_session.commit()

    # Verify chunks don't have embeddings yet
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == "doc-embed-3")
    result = await async_session.execute(stmt)
    chunks_before = result.scalars().all()
    assert all(chunk.embedding is None for chunk in chunks_before)

    # Generate embeddings
    mock_embeddings = [[0.1] * 512, [0.2] * 512]

    with patch("app.services.embeddings.generate_embeddings_batch") as mock_gen:
        mock_gen.return_value = mock_embeddings

        await create_chunk_embeddings(
            async_session,
            "doc-embed-3",
            ["Text 1 about contracts", "Text 2 about patents"],
        )

    # Verify embeddings are stored
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == "doc-embed-3")
    result = await async_session.execute(stmt)
    stored_chunks = result.scalars().all()

    assert len(stored_chunks) == 2
    assert all(chunk.embedding is not None for chunk in stored_chunks)
    assert len(stored_chunks[0].embedding) == 512
    assert len(stored_chunks[1].embedding) == 512


@pytest.mark.asyncio
async def test_extract_and_embed_with_batch_processing(async_session):
    """Test that large document is processed in batches during embedding."""
    doc = Document(
        id="doc-embed-4",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    async_session.add(doc)
    await async_session.commit()

    # Create text that will result in multiple chunks
    test_text = "Legal document text. " * 200  # Will create many chunks

    batch_calls = []

    def track_batch_calls(texts, **kwargs):
        """Track how many times embeddings are called."""
        batch_calls.append(len(texts))
        return [[float(i) / 512.0] * 512 for i in range(len(texts))]

    with patch("app.services.embeddings.generate_embeddings_batch", side_effect=track_batch_calls):
        chunk_ids = await extract_and_embed_document(
            async_session,
            doc.id,
            test_text,
            chunk_size=400,
            overlap=40,
            batch_size=5,  # Small batch size to force multiple calls
        )

    # Verify batching occurred
    assert len(batch_calls) > 1, "Should have multiple batch calls"
    # First batches should be full (batch_size=5), last one might be smaller
    for batch in batch_calls[:-1]:
        assert batch == 5, f"Expected batch size 5, got {batch}"

    # Verify all chunks got embeddings
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    result = await async_session.execute(stmt)
    db_chunks = result.scalars().all()
    assert all(chunk.embedding is not None for chunk in db_chunks)


@pytest.mark.asyncio
async def test_empty_text_returns_empty_chunks(async_session):
    """Test that empty text results in no chunks created."""
    doc = Document(
        id="doc-embed-5",
        case_id="case1",
        file_path="/tmp/test.txt",
        filename="test.txt",
    )
    async_session.add(doc)
    await async_session.commit()

    chunk_ids = await extract_and_embed_document(
        async_session,
        doc.id,
        "",  # Empty text
    )

    assert chunk_ids == []

    # Verify no chunks created
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc.id)
    result = await async_session.execute(stmt)
    db_chunks = result.scalars().all()
    assert len(db_chunks) == 0
