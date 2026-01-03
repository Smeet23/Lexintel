"""Tests for extraction service"""

import pytest
import os
import tempfile
from uuid import uuid4
from app.services.extraction import extract_file, clean_text, chunk_text, create_text_chunks
from shared.models import Document, DocumentChunk, DocumentType, ProcessingStatus


# ===== EXTRACT_FILE TESTS =====
@pytest.mark.asyncio
async def test_extract_txt_file():
    """Test extracting text from TXT file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content.\nWith multiple lines.\nFor testing.")
        temp_path = f.name

    try:
        text = await extract_file(temp_path)
        assert text is not None
        assert len(text) > 0
        assert "test content" in text
        assert "multiple lines" in text
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_extract_txt_file_empty():
    """Test extracting from empty file"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_path = f.name

    try:
        text = await extract_file(temp_path)
        assert text == ""
    finally:
        os.unlink(temp_path)


@pytest.mark.asyncio
async def test_extract_unsupported_file():
    """Test that unsupported file types raise error"""
    with tempfile.NamedTemporaryFile(suffix='.xyz', delete=False) as f:
        f.write(b"content")
        temp_path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported file type"):
            await extract_file(temp_path)
    finally:
        os.unlink(temp_path)


# ===== CLEAN_TEXT TESTS =====
def test_clean_text_removes_null_bytes():
    """Test that NULL bytes are removed"""
    dirty = "Hello\x00World"
    clean = clean_text(dirty)
    assert clean == "HelloWorld"


def test_clean_text_removes_control_chars():
    """Test that control characters are removed"""
    dirty = "Hello\x01\x02World"
    clean = clean_text(dirty)
    assert clean == "HelloWorld"


def test_clean_text_normalizes_newlines():
    """Test that multiple newlines become double newlines"""
    dirty = "Line1\n\n\n\nLine2"
    clean = clean_text(dirty)
    assert clean == "Line1\n\nLine2"


def test_clean_text_normalizes_spaces():
    """Test that multiple spaces become single space"""
    dirty = "Hello    World"
    clean = clean_text(dirty)
    assert clean == "Hello World"


def test_clean_text_removes_page_numbers():
    """Test that page numbers are removed"""
    dirty = "Content\n5\nMore content"
    clean = clean_text(dirty)
    assert clean == "Content\nMore content"


def test_clean_text_trims_whitespace():
    """Test that leading/trailing whitespace is removed"""
    dirty = "   Hello World   "
    clean = clean_text(dirty)
    assert clean == "Hello World"


def test_clean_text_converts_form_feeds():
    """Test that form feeds become newlines"""
    dirty = "Section1\fSection2"
    clean = clean_text(dirty)
    assert clean == "Section1\nSection2"


def test_clean_text_normalizes_line_endings():
    """Test that CRLF becomes LF"""
    dirty = "Line1\r\nLine2"
    clean = clean_text(dirty)
    assert clean == "Line1\nLine2"


# ===== CHUNK_TEXT TESTS =====
def test_chunk_text_respects_chunk_size():
    """Test that chunks don't exceed max size"""
    text = "x" * 10000
    chunks = chunk_text(text, chunk_size=4000)
    for chunk in chunks:
        assert len(chunk) <= 4000


def test_chunk_text_minimum_size():
    """Test that chunks smaller than 200 chars are skipped"""
    text = "short" * 50
    chunks = chunk_text(text, chunk_size=200, min_size=200)
    for chunk in chunks:
        assert len(chunk) >= 200


def test_chunk_text_overlap():
    """Test that chunks have proper overlap"""
    text = "abcdefghijklmnopqrstuvwxyz" * 200
    chunks = chunk_text(text, chunk_size=1000, overlap=100)
    assert len(chunks) >= 2
    assert len(chunks[0]) <= 1000
    assert chunks[1].startswith(chunks[0][-100:])


def test_chunk_text_empty():
    """Test chunking empty text"""
    chunks = chunk_text("", chunk_size=4000)
    assert chunks == []


def test_chunk_text_smaller_than_chunk_size():
    """Test text smaller than chunk size"""
    text = "short text"
    chunks = chunk_text(text, chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == "short text"


def test_chunk_text_custom_chunk_size():
    """Test chunking with custom chunk size"""
    text = "x" * 5000
    chunks = chunk_text(text, chunk_size=2000, overlap=200)
    assert all(len(chunk) <= 2000 for chunk in chunks)
    assert len(chunks) > 1


def test_chunk_text_no_overlap():
    """Test chunking with zero overlap"""
    text = "abcdefghij" * 500
    chunks = chunk_text(text, chunk_size=1000, overlap=0)
    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk) <= 1000


# ===== CREATE_TEXT_CHUNKS TESTS =====
@pytest.mark.asyncio
async def test_create_text_chunks_creates_records(db_session):
    """Test that create_text_chunks creates DocumentChunk records"""

    # Create a test document
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path="/test/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Create text chunks
    text = "x" * 5000
    chunk_ids = await create_text_chunks(doc_id, text, db_session)

    assert len(chunk_ids) > 0
    assert all(isinstance(id, str) for id in chunk_ids)


@pytest.mark.asyncio
async def test_create_text_chunks_empty_text(db_session):
    """Test that empty text creates no chunks"""
    doc_id = str(uuid4())
    chunk_ids = await create_text_chunks(doc_id, "", db_session)
    assert chunk_ids == []


@pytest.mark.asyncio
async def test_create_text_chunks_chunk_indices(db_session):
    """Test that chunks have correct indices"""

    # Create a test document
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path="/test/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Create text chunks
    text = "x" * 5000
    chunk_ids = await create_text_chunks(doc_id, text, db_session)

    # Verify chunk indices
    from sqlalchemy import select
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index)
    result = await db_session.execute(stmt)
    chunks = result.scalars().all()

    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


@pytest.mark.asyncio
async def test_create_text_chunks_preserves_text(db_session):
    """Test that chunk text is preserved correctly"""

    # Create a test document
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path="/test/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Create text chunks
    test_text = "This is a test document with some content. " * 100
    chunk_ids = await create_text_chunks(doc_id, test_text, db_session)

    # Verify all text is in chunks
    from sqlalchemy import select
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id).order_by(DocumentChunk.chunk_index)
    result = await db_session.execute(stmt)
    chunks = result.scalars().all()

    all_text = "".join(chunk.chunk_text for chunk in chunks)
    assert "test document" in all_text


@pytest.mark.asyncio
async def test_create_text_chunks_with_custom_sizes(db_session):
    """Test creating chunks with custom chunk_size and overlap"""

    # Create a test document
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path="/test/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Create text chunks with custom sizes
    text = "x" * 10000
    chunk_ids = await create_text_chunks(
        doc_id,
        text,
        db_session,
        chunk_size=2000,
        overlap=200
    )

    from sqlalchemy import select
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    result = await db_session.execute(stmt)
    chunks = result.scalars().all()

    assert all(len(chunk.chunk_text) <= 2000 for chunk in chunks)


@pytest.mark.asyncio
async def test_create_text_chunks_multiple_documents(db_session):
    """Test creating chunks for multiple documents"""

    # Create two test documents
    doc_id_1 = str(uuid4())
    doc_id_2 = str(uuid4())
    case_id = str(uuid4())

    for doc_id in [doc_id_1, doc_id_2]:
        doc = Document(
            id=doc_id,
            case_id=case_id,
            title="Test",
            filename="test.txt",
            type=DocumentType.BRIEF,
            file_path="/test/file.txt",
            processing_status=ProcessingStatus.PENDING,
        )
        db_session.add(doc)

    await db_session.commit()

    # Create chunks for both
    text = "x" * 5000
    chunk_ids_1 = await create_text_chunks(doc_id_1, text, db_session)
    chunk_ids_2 = await create_text_chunks(doc_id_2, text, db_session)

    # Verify they have different IDs
    assert len(set(chunk_ids_1) & set(chunk_ids_2)) == 0
    assert len(chunk_ids_1) == len(chunk_ids_2)


@pytest.mark.asyncio
async def test_create_text_chunks_cleans_before_chunking(db_session):
    """Test that text is cleaned before chunking"""

    # Create a test document
    doc_id = str(uuid4())
    case_id = str(uuid4())

    doc = Document(
        id=doc_id,
        case_id=case_id,
        title="Test",
        filename="test.txt",
        type=DocumentType.BRIEF,
        file_path="/test/file.txt",
        processing_status=ProcessingStatus.PENDING,
    )
    db_session.add(doc)
    await db_session.commit()

    # Create chunks with dirty text containing null bytes and control chars
    dirty_text = "Hello\x00World\n\n\n\n" * 500
    chunk_ids = await create_text_chunks(doc_id, dirty_text, db_session)

    from sqlalchemy import select
    stmt = select(DocumentChunk).where(DocumentChunk.document_id == doc_id)
    result = await db_session.execute(stmt)
    chunks = result.scalars().all()

    for chunk in chunks:
        # Verify null bytes and control chars were removed
        assert '\x00' not in chunk.chunk_text
        assert '\n\n\n' not in chunk.chunk_text
