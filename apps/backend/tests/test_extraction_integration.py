"""Integration tests for text extraction workflow"""

import pytest
import tempfile
import os
from uuid import uuid4
from unittest.mock import MagicMock, patch, AsyncMock
from app.models import Document, DocumentChunk, ProcessingStatus, DocumentType
from app.workers.tasks import _extract_and_chunk_document
from sqlalchemy import select


@pytest.mark.asyncio
async def test_end_to_end_extraction(db_session):
    """Test complete extraction workflow: upload → extract → create chunks"""

    # Setup: Create temp file with test content
    test_content = """
    Section 1: Introduction
    This is a test document for extraction.
    It contains multiple paragraphs and sections.

    Section 2: Details
    Details about the document and its content.
    More details here.

    Section 3: Conclusion
    Final thoughts and conclusions.
    """ * 100  # Repeat to have enough text for multiple chunks

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        temp_file_path = f.name

    try:
        # Create test case and document
        doc_id = str(uuid4())
        case_id = str(uuid4())

        document = Document(
            id=doc_id,
            case_id=case_id,
            title="Test Document",
            filename="test.txt",
            type=DocumentType.BRIEF,
            file_path=temp_file_path,
            file_size=len(test_content),
            processing_status=ProcessingStatus.PENDING,
        )
        db_session.add(document)
        await db_session.commit()

        # Create async context manager mock
        class AsyncSessionMock:
            async def __aenter__(self):
                return db_session
            async def __aexit__(self, *args):
                pass

        # Mock async_session callable
        mock_async_session_factory = MagicMock(return_value=AsyncSessionMock())

        with patch('app.workers.tasks.async_session', mock_async_session_factory):
            # Mock the generate_embeddings task to avoid queue issues
            with patch('app.workers.tasks.generate_embeddings') as mock_embeddings:
                # Execute extraction
                result = await _extract_and_chunk_document(doc_id)

                # Verify result
                assert result["status"] == "success"
                assert result["document_id"] == doc_id
                assert result["chunks_created"] > 0
                assert result["text_length"] > 0

                # Verify document was updated
                stmt = select(Document).where(Document.id == doc_id)
                result_doc = await db_session.execute(stmt)
                updated_doc = result_doc.scalars().first()

                assert updated_doc.processing_status == ProcessingStatus.EXTRACTED
                assert updated_doc.extracted_text is not None
                assert len(updated_doc.extracted_text) > 0

                # Verify chunks were created
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id == doc_id
                ).order_by(DocumentChunk.chunk_index)
                result = await db_session.execute(stmt)
                chunks = result.scalars().all()

                assert len(chunks) > 0

                # Verify chunk properties
                for i, chunk in enumerate(chunks):
                    assert chunk.chunk_index == i
                    assert chunk.document_id == doc_id
                    assert len(chunk.chunk_text) > 0
                    assert len(chunk.chunk_text) <= 4000  # Max chunk size
                    assert len(chunk.chunk_text) >= 200   # Min chunk size (except maybe last)

    finally:
        os.unlink(temp_file_path)


@pytest.mark.asyncio
async def test_extraction_handles_missing_file(db_session):
    """Test that extraction fails gracefully with missing file"""

    doc_id = str(uuid4())
    case_id = str(uuid4())

    # Create document with non-existent file
    document = Document(
        id=doc_id,
        case_id=case_id,
        title="Missing File",
        filename="missing.txt",
        type=DocumentType.BRIEF,
        file_path="/nonexistent/path/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(document)
    await db_session.commit()

    # Create async context manager mock
    class AsyncSessionMock:
        async def __aenter__(self):
            return db_session
        async def __aexit__(self, *args):
            pass

    # Mock async_session callable
    mock_async_session_factory = MagicMock(return_value=AsyncSessionMock())

    with patch('app.workers.tasks.async_session', mock_async_session_factory):
        # Attempt extraction - should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            await _extract_and_chunk_document(doc_id)

        # The error was correctly raised, confirming graceful handling


@pytest.mark.asyncio
async def test_extraction_empty_document(db_session):
    """Test extraction of empty document"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_file_path = f.name

    try:
        doc_id = str(uuid4())
        case_id = str(uuid4())

        document = Document(
            id=doc_id,
            case_id=case_id,
            title="Empty",
            filename="empty.txt",
            type=DocumentType.BRIEF,
            file_path=temp_file_path,
            processing_status=ProcessingStatus.PENDING,
        )
        db_session.add(document)
        await db_session.commit()

        # Create async context manager mock
        class AsyncSessionMock:
            async def __aenter__(self):
                return db_session
            async def __aexit__(self, *args):
                pass

        # Mock async_session callable
        mock_async_session_factory = MagicMock(return_value=AsyncSessionMock())

        with patch('app.workers.tasks.async_session', mock_async_session_factory):
            # Mock the generate_embeddings task to avoid queue issues
            with patch('app.workers.tasks.generate_embeddings') as mock_embeddings:
                # Execute extraction
                result = await _extract_and_chunk_document(doc_id)

                # Should succeed with 0 chunks
                assert result["status"] == "success"
                assert result["chunks_created"] == 0
                assert result["text_length"] == 0

                # Verify chunks not created
                stmt = select(DocumentChunk).where(
                    DocumentChunk.document_id == doc_id
                )
                result = await db_session.execute(stmt)
                chunks = result.scalars().all()
                assert len(chunks) == 0

    finally:
        os.unlink(temp_file_path)
