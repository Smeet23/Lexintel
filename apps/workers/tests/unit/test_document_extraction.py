"""Unit tests for document extraction worker."""

import pytest
import tempfile
from pathlib import Path
from uuid import uuid4
from unittest.mock import AsyncMock, patch, MagicMock

from shared.models import Document, DocumentChunk, ProcessingStatus, DocumentType
from shared.schemas.jobs import DocumentExtractionJob
from shared import async_session
from sqlalchemy import select


@pytest.fixture
def temp_text_file():
    """Create a temporary text file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content.\nWith multiple lines.\nFor testing extraction.")
        temp_path = f.name

    yield temp_path

    Path(temp_path).unlink()


@pytest.fixture
def sample_document(db_session, temp_text_file):
    """Create a sample document for testing."""
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test Document",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path=temp_text_file,
        processing_status=ProcessingStatus.PENDING,
        file_size=100,
    )

    db_session.add(doc)
    db_session.commit()

    return doc


@pytest.mark.asyncio
async def test_extract_text_from_document_creates_chunks(sample_document, mocker):
    """Test that extraction task creates DocumentChunks."""
    from workers.document_extraction import extract_text_from_document

    # Mock Redis publisher
    mock_redis = AsyncMock()
    mocker.patch(
        'workers.document_extraction.get_redis_client',
        return_value=mock_redis
    )

    job_payload = {
        "document_id": sample_document.id,
        "case_id": sample_document.case_id,
        "source": "upload",
    }

    # Run task
    result = extract_text_from_document(job_payload)

    # Verify result
    assert result["status"] == "success"
    assert result["document_id"] == sample_document.id
    assert result["chunks_created"] > 0
    assert result["text_length"] > 0

    # Verify chunks were created in database
    async with async_session() as db:
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == sample_document.id
        )
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        assert len(chunks) > 0
        assert all(c.document_id == sample_document.id for c in chunks)
        assert all(isinstance(c.chunk_index, int) for c in chunks)


@pytest.mark.asyncio
async def test_extract_text_updates_document_status(sample_document, mocker):
    """Test that extraction updates document status to EXTRACTED."""
    from workers.document_extraction import extract_text_from_document

    # Mock Redis publisher
    mock_redis = AsyncMock()
    mocker.patch(
        'workers.document_extraction.get_redis_client',
        return_value=mock_redis
    )

    job_payload = {
        "document_id": sample_document.id,
        "case_id": sample_document.case_id,
        "source": "upload",
    }

    # Run task
    extract_text_from_document(job_payload)

    # Verify document status was updated
    async with async_session() as db:
        stmt = select(Document).where(Document.id == sample_document.id)
        result = await db.execute(stmt)
        doc = result.scalar_one()

        assert doc.processing_status == ProcessingStatus.EXTRACTED
        assert doc.extracted_text is not None
        assert len(doc.extracted_text) > 0


@pytest.mark.asyncio
async def test_extract_text_missing_file(mocker):
    """Test extraction with non-existent file."""
    from workers.document_extraction import extract_text_from_document
    from shared import PermanentError

    # Mock Redis publisher
    mock_redis = AsyncMock()
    mocker.patch(
        'workers.document_extraction.get_redis_client',
        return_value=mock_redis
    )

    doc_id = str(uuid4())
    case_id = str(uuid4())

    # Create document with non-existent file
    async with async_session() as db:
        doc = Document(
            id=doc_id,
            case_id=case_id,
            title="Test Document",
            filename="test.txt",
            type=DocumentType.BRIEF,
            file_path="/nonexistent/file.txt",
            processing_status=ProcessingStatus.PENDING,
            file_size=100,
        )
        db.add(doc)
        await db.commit()

    job_payload = {
        "document_id": doc_id,
        "case_id": case_id,
        "source": "upload",
    }

    # Task should fail with PermanentError
    with pytest.raises(PermanentError):
        extract_text_from_document(job_payload)


@pytest.mark.asyncio
async def test_chunk_text_creates_correct_indices(temp_text_file, mocker):
    """Test that chunks have correct indices."""
    from workers.document_extraction import extract_text_from_document

    # Mock Redis publisher
    mock_redis = AsyncMock()
    mocker.patch(
        'workers.document_extraction.get_redis_client',
        return_value=mock_redis
    )

    doc_id = str(uuid4())
    case_id = str(uuid4())

    # Create document
    async with async_session() as db:
        doc = Document(
            id=doc_id,
            case_id=case_id,
            title="Test Document",
            filename="test.txt",
            type=DocumentType.BRIEF,
            file_path=temp_text_file,
            processing_status=ProcessingStatus.PENDING,
            file_size=100,
        )
        db.add(doc)
        await db.commit()

    job_payload = {
        "document_id": doc_id,
        "case_id": case_id,
        "source": "upload",
    }

    # Run task
    extract_text_from_document(job_payload)

    # Verify chunk indices
    async with async_session() as db:
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == doc_id
        ).order_by(DocumentChunk.chunk_index)
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i


@pytest.mark.asyncio
async def test_extract_text_publishes_progress(sample_document, mocker):
    """Test that extraction publishes progress updates."""
    from workers.document_extraction import extract_text_from_document

    # Mock Redis publisher
    mock_redis = AsyncMock()
    mock_publisher = AsyncMock()

    mocker.patch(
        'workers.document_extraction.get_redis_client',
        return_value=mock_redis
    )
    mocker.patch(
        'workers.document_extraction.ProgressPublisher',
        return_value=mock_publisher
    )

    job_payload = {
        "document_id": sample_document.id,
        "case_id": sample_document.case_id,
        "source": "upload",
    }

    # Run task
    extract_text_from_document(job_payload)

    # Verify progress was published
    assert mock_publisher.publish_progress.called

    # Should have at least starting and completion messages
    calls = mock_publisher.publish_progress.call_args_list
    assert len(calls) >= 2

    # First call should be starting (0%)
    assert calls[0][0][1] == 0
    assert calls[0][0][2] == "starting"

    # Last call should be completion (100%)
    assert calls[-1][0][1] == 100
    assert calls[-1][0][2] == "completed"


def test_chunk_text_function():
    """Test chunk_text function directly."""
    from workers.document_extraction import chunk_text

    text = "a" * 10000
    chunks = chunk_text(text, chunk_size=4000, overlap=400)

    assert len(chunks) > 0
    assert all(len(chunk) <= 4000 for chunk in chunks)
    assert chunks[0][:4000] == "a" * 4000


def test_clean_text_function():
    """Test clean_text function directly."""
    from workers.document_extraction import clean_text

    # Test null byte removal
    dirty = "Hello\x00World"
    clean = clean_text(dirty)
    assert '\x00' not in clean
    assert clean == "HelloWorld"

    # Test space normalization
    dirty = "Hello    World"
    clean = clean_text(dirty)
    assert clean == "Hello World"

    # Test newline normalization
    dirty = "Line1\n\n\n\nLine2"
    clean = clean_text(dirty)
    assert "\\n\\n\\n" not in clean


@pytest.mark.asyncio
async def test_extract_text_cleans_extracted_text(temp_text_file, mocker):
    """Test that extracted text is cleaned before chunking."""
    from workers.document_extraction import extract_text_from_document

    # Create a text file with dirty content
    dirty_text = "Hello\x00World\n\n\n\nMore content"
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(dirty_text)
        temp_path = f.name

    try:
        doc_id = str(uuid4())
        case_id = str(uuid4())

        # Create document
        async with async_session() as db:
            doc = Document(
                id=doc_id,
                case_id=case_id,
                title="Test Document",
                filename="test.txt",
                type=DocumentType.BRIEF,
                file_path=temp_path,
                processing_status=ProcessingStatus.PENDING,
                file_size=100,
            )
            db.add(doc)
            await db.commit()

        # Mock Redis publisher
        mock_redis = AsyncMock()
        mocker.patch(
            'workers.document_extraction.get_redis_client',
            return_value=mock_redis
        )

        job_payload = {
            "document_id": doc_id,
            "case_id": case_id,
            "source": "upload",
        }

        # Run task
        extract_text_from_document(job_payload)

        # Verify extracted text was cleaned
        async with async_session() as db:
            stmt = select(Document).where(Document.id == doc_id)
            result = await db.execute(stmt)
            doc = result.scalar_one()

            # Verify null bytes were removed
            assert '\x00' not in doc.extracted_text
            # Verify multiple newlines were normalized
            assert '\n\n\n' not in doc.extracted_text

    finally:
        Path(temp_path).unlink()
