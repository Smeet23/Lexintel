# Phase 3: Text Extraction Workers - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract text from uploaded documents and split into searchable chunks with overlapping context

**Architecture:**
Create an extraction service (`app/services/extraction.py`) with text extraction and chunking functions. Update the worker task (`app/workers/tasks.py`) to use these functions, handle errors, and update document status. Implement TDD: failing test → minimal code → passing test → commit.

**Tech Stack:**
Python 3.10+, SQLAlchemy async ORM, Celery workers, PostgreSQL, UTF-8 text handling

---

## Phase 3 Implementation Tasks

### Task 1: Create extraction service file

**Files:**
- Create: `backend/app/services/extraction.py`

**Step 1: Create empty file with imports**

Create `backend/app/services/extraction.py`:

```python
"""
Text extraction and chunking service

Handles:
- Extracting text from various file formats (TXT, PDF, DOCX)
- Cleaning extracted text
- Splitting text into overlapping chunks
- Creating DocumentChunk database records
"""

import logging
from typing import List
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import insert

from app.models import DocumentChunk

logger = logging.getLogger(__name__)
```

**Step 2: Commit**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
git add backend/app/services/extraction.py
git commit -m "feat: create extraction service module"
```

---

### Task 2: Implement text extraction function

**Files:**
- Modify: `backend/app/services/extraction.py`
- Create: `backend/tests/test_extraction.py`

**Step 1: Write failing test for text extraction**

Create `backend/tests/test_extraction.py`:

```python
"""Tests for extraction service"""

import pytest
import os
import tempfile
from app.services.extraction import extract_file


@pytest.mark.asyncio
async def test_extract_txt_file():
    """Test extracting text from TXT file"""

    # Create temporary TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is test content.\nWith multiple lines.\nFor testing.")
        temp_path = f.name

    try:
        # Extract text
        text = await extract_file(temp_path)

        # Verify
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
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
pytest backend/tests/test_extraction.py::test_extract_txt_file -v
```

Expected output:
```
FAILED - extract_file is not defined
```

**Step 3: Implement extract_file function**

Add to `backend/app/services/extraction.py`:

```python
async def extract_file(file_path: str) -> str:
    """
    Extract text from file based on file type

    Args:
        file_path: Path to file on filesystem

    Returns:
        Extracted text content

    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file doesn't exist
    """

    # Phase 3: TXT files only
    if file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            logger.info(f"[extraction] Extracted {len(text)} chars from {file_path}")
            return text
        except FileNotFoundError:
            logger.error(f"[extraction] File not found: {file_path}")
            raise
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
```

**Step 4: Run tests to verify they pass**

```bash
pytest backend/tests/test_extraction.py -v
```

Expected output:
```
test_extract_txt_file PASSED
test_extract_txt_file_empty PASSED
test_extract_unsupported_file PASSED
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/test_extraction.py
git commit -m "feat: implement text extraction from TXT files"
```

---

### Task 3: Implement text cleaning

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/test_extraction.py`

**Step 1: Write failing test for text cleaning**

Add to `backend/tests/test_extraction.py`:

```python
from app.services.extraction import clean_text


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
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/test_extraction.py::test_clean_text_removes_null_bytes -v
```

Expected output:
```
FAILED - clean_text is not defined
```

**Step 3: Implement clean_text function**

Add to `backend/app/services/extraction.py`:

```python
import re


def clean_text(text: str) -> str:
    """
    Clean extracted text by removing artifacts and normalizing whitespace

    Removes:
    - NULL bytes (PostgreSQL UTF-8 incompatibility)
    - Control characters (except \n, \r, \t)
    - Extra newlines
    - Page numbers (OCR artifacts)
    - Form feeds

    Normalizes:
    - Line endings (CRLF → LF)
    - Spaces (multiple → single)

    Args:
        text: Raw extracted text

    Returns:
        Cleaned text
    """

    if not text:
        return text

    # Remove NULL bytes
    text = text.replace('\x00', '')

    # Remove control characters (except \n=0x0A, \r=0x0D, \t=0x09)
    text = re.sub(r'[\x01-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    # Normalize multiple newlines to double newlines (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove page numbers (line with only digits/spaces)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Remove page headers (line starting with "Page")
    text = re.sub(r'^Page\s+\d+.*$', '', text, flags=re.MULTILINE)

    # Convert form feeds to newlines
    text = text.replace('\f', '\n')

    # Normalize line endings (CRLF to LF)
    text = text.replace('\r\n', '\n')

    # Normalize multiple spaces to single space
    text = re.sub(r'[ \t]{2,}', ' ', text)

    # Final trim
    text = text.strip()

    logger.debug(f"[extraction] Cleaned text: {len(text)} chars")
    return text
```

**Step 4: Run tests to verify they pass**

```bash
pytest backend/tests/test_extraction.py -k "clean_text" -v
```

Expected output:
```
test_clean_text_removes_null_bytes PASSED
test_clean_text_removes_control_chars PASSED
test_clean_text_normalizes_newlines PASSED
test_clean_text_normalizes_spaces PASSED
test_clean_text_removes_page_numbers PASSED
test_clean_text_trims_whitespace PASSED
test_clean_text_converts_form_feeds PASSED
test_clean_text_normalizes_line_endings PASSED
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/test_extraction.py
git commit -m "feat: implement text cleaning with artifact removal"
```

---

### Task 4: Implement chunking algorithm

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/test_extraction.py`

**Step 1: Write failing tests for chunking**

Add to `backend/tests/test_extraction.py`:

```python
from app.services.extraction import chunk_text


def test_chunk_text_respects_chunk_size():
    """Test that chunks don't exceed max size"""
    text = "x" * 10000  # 10000 chars
    chunks = chunk_text(text, chunk_size=4000)

    for chunk in chunks:
        assert len(chunk) <= 4000


def test_chunk_text_minimum_size():
    """Test that chunks smaller than 200 chars are skipped"""
    text = "short" * 50  # 250 chars total
    chunks = chunk_text(text, chunk_size=200, min_size=200)

    # Should have chunks >= 200
    for chunk in chunks:
        assert len(chunk) >= 200


def test_chunk_text_overlap():
    """Test that chunks have proper overlap"""
    text = "abcdefghijklmnopqrstuvwxyz" * 200  # 5200 chars
    chunks = chunk_text(text, chunk_size=1000, overlap=100)

    # Should have at least 2 chunks
    assert len(chunks) >= 2

    # First chunk should be ~1000 chars
    assert len(chunks[0]) <= 1000

    # Second chunk should overlap with first
    # (last 100 chars of chunk 0 should appear in chunk 1)
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
```

**Step 2: Run tests to verify they fail**

```bash
pytest backend/tests/test_extraction.py::test_chunk_text_respects_chunk_size -v
```

Expected output:
```
FAILED - chunk_text is not defined
```

**Step 3: Implement chunk_text function**

Add to `backend/app/services/extraction.py`:

```python
def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
    min_size: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks

    Algorithm:
    1. Start at position 0
    2. Extract chunk_size characters
    3. Trim whitespace from end
    4. If chunk >= min_size: keep it
    5. Move start back by overlap
    6. Repeat until end of text

    Args:
        text: Text to chunk
        chunk_size: Max chars per chunk (default 4000)
        overlap: Char overlap between chunks (default 400)
        min_size: Minimum chunk size to keep (default 200)

    Returns:
        List of text chunks
    """

    if not text:
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        # Calculate end position
        end = min(start + chunk_size, text_len)

        # Extract slice and trim trailing whitespace
        slice_text = text[start:end].rstrip()

        # Only keep chunks >= min_size
        if len(slice_text) >= min_size:
            chunks.append(slice_text)
            logger.debug(
                f"[chunking] Created chunk {len(chunks)}: "
                f"{len(slice_text)} chars at position {start}"
            )

        # Move to next chunk position (with overlap)
        next_start = end - overlap

        # Prevent infinite loop (overlap >= chunk_size)
        if next_start <= start:
            break

        start = next_start

    logger.info(
        f"[chunking] Split {text_len} chars into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={overlap})"
    )

    return chunks
```

**Step 4: Run tests to verify they pass**

```bash
pytest backend/tests/test_extraction.py -k "chunk_text" -v
```

Expected output:
```
test_chunk_text_respects_chunk_size PASSED
test_chunk_text_minimum_size PASSED
test_chunk_text_overlap PASSED
test_chunk_text_empty PASSED
test_chunk_text_smaller_than_chunk_size PASSED
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/test_extraction.py
git commit -m "feat: implement text chunking with overlap support"
```

---

### Task 5: Implement create_text_chunks (DB persistence)

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/test_extraction.py`

**Step 1: Write failing test for database chunk creation**

Add to `backend/tests/test_extraction.py`:

```python
import pytest
from app.services.extraction import create_text_chunks
from app.database import async_session
from app.models import Document, DocumentChunk
from sqlalchemy import select


@pytest.mark.asyncio
async def test_create_text_chunks_creates_records():
    """Test that create_text_chunks creates DocumentChunk records"""

    # Setup: Create a test document
    async with async_session() as session:
        from uuid import uuid4
        doc_id = str(uuid4())
        case_id = str(uuid4())

        # Create test document
        doc = Document(
            id=doc_id,
            case_id=case_id,
            title="Test",
            filename="test.txt",
            type="brief",
            file_path="/test/file.txt",
        )
        session.add(doc)
        await session.commit()

        # Create chunks
        text = "x" * 5000  # Should create 2 chunks
        chunk_ids = await create_text_chunks(doc_id, text, session)

        # Verify chunks were created
        assert len(chunk_ids) > 0

        # Verify in database
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id == doc_id
        )
        result = await session.execute(stmt)
        chunks = result.scalars().all()

        assert len(chunks) == len(chunk_ids)
        assert chunks[0].chunk_index == 0
        assert chunks[1].chunk_index == 1


@pytest.mark.asyncio
async def test_create_text_chunks_empty_text():
    """Test that empty text creates no chunks"""

    async with async_session() as session:
        from uuid import uuid4
        doc_id = str(uuid4())

        chunk_ids = await create_text_chunks(doc_id, "", session)

        assert chunk_ids == []
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/test_extraction.py::test_create_text_chunks_creates_records -v
```

Expected output:
```
FAILED - create_text_chunks is not defined or doesn't work correctly
```

**Step 3: Implement create_text_chunks function**

Add to `backend/app/services/extraction.py`:

```python
async def create_text_chunks(
    document_id: str,
    text: str,
    session: AsyncSession,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[str]:
    """
    Split text into chunks and create DocumentChunk records in database

    Args:
        document_id: ID of parent document
        text: Text to split
        session: SQLAlchemy async session
        chunk_size: Max chars per chunk
        overlap: Char overlap between chunks

    Returns:
        List of created chunk IDs

    Raises:
        Exception: If database insert fails
    """

    # Clean text first
    text = clean_text(text)

    if not text:
        logger.info(f"[chunks] No text to chunk for document {document_id}")
        return []

    # Split text into chunks
    chunk_texts = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    if not chunk_texts:
        logger.warning(f"[chunks] Text split resulted in no chunks for {document_id}")
        return []

    # Create DocumentChunk records
    chunk_ids = []

    for chunk_index, chunk_text in enumerate(chunk_texts):
        chunk = DocumentChunk(
            id=str(uuid4()),
            document_id=document_id,
            chunk_text=chunk_text,
            chunk_index=chunk_index,
        )
        session.add(chunk)
        chunk_ids.append(chunk.id)

        logger.debug(
            f"[chunks] Created chunk {chunk_index}: {len(chunk_text)} chars for doc {document_id}"
        )

    # Commit all chunks at once
    try:
        await session.flush()  # Validate but don't commit yet
        logger.info(
            f"[chunks] Created {len(chunk_ids)} chunks for document {document_id}"
        )
        return chunk_ids
    except Exception as e:
        logger.error(f"[chunks] Failed to create chunks for {document_id}: {e}")
        raise
```

**Step 4: Run tests to verify they pass**

```bash
pytest backend/tests/test_extraction.py::test_create_text_chunks_creates_records -v
```

Expected output:
```
test_create_text_chunks_creates_records PASSED
test_create_text_chunks_empty_text PASSED
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/test_extraction.py
git commit -m "feat: implement database persistence for document chunks"
```

---

### Task 6: Update worker task to use extraction service

**Files:**
- Modify: `backend/app/workers/tasks.py`

**Step 1: Update imports**

Replace the existing imports in `backend/app/workers/tasks.py` with:

```python
import logging
from celery import shared_task, Task
from app.database import async_session
from app.models import Document, ProcessingStatus
from app.config import settings
from app.services.extraction import extract_file, create_text_chunks
import asyncio
import os

logger = logging.getLogger(__name__)
```

**Step 2: Update _extract_and_chunk_document function**

Replace the `_extract_and_chunk_document` function in `backend/app/workers/tasks.py` with:

```python
async def _extract_and_chunk_document(document_id: str) -> dict:
    """
    Async helper to extract text and create chunks

    Flow:
    1. Get Document from DB
    2. Validate file exists
    3. Extract text from file
    4. Clean and split into chunks
    5. Create DocumentChunk records
    6. Update Document status
    7. Queue embeddings task
    """
    from sqlalchemy import select

    async with async_session() as session:
        # Step 1: Get document from database
        stmt = select(Document).where(Document.id == document_id)
        result = await session.execute(stmt)
        document = result.scalars().first()

        if not document:
            raise ValueError(f"Document {document_id} not found")

        logger.info(
            f"[extract_text] Processing document {document_id}: {document.filename}"
        )

        # Step 2: Validate file exists
        if not document.file_path or not os.path.exists(document.file_path):
            raise FileNotFoundError(
                f"File not found for document {document_id}: {document.file_path}"
            )

        try:
            # Step 3: Extract text from file
            logger.info(f"[extract_text] Extracting text from {document.file_path}")
            text = await extract_file(document.file_path)
            logger.info(
                f"[extract_text] Extracted {len(text)} characters from {document_id}"
            )

            # Step 4 & 5: Split text into chunks and create records
            logger.info(f"[extract_text] Creating chunks for document {document_id}")
            chunk_ids = await create_text_chunks(
                document_id=document_id,
                text=text,
                session=session,
            )

            # Step 6: Update document with extracted text and status
            document.extracted_text = text
            document.processing_status = ProcessingStatus.EXTRACTED

            await session.commit()

            logger.info(
                f"[extract_text] Successfully extracted {len(chunk_ids)} chunks "
                f"for document {document_id}"
            )

            # Step 7: Queue embeddings task (Phase 4)
            from app.workers.tasks import generate_embeddings
            generate_embeddings.delay(document_id)
            logger.info(f"[extract_text] Queued embeddings task for {document_id}")

            return {
                "status": "success",
                "document_id": document_id,
                "chunks_created": len(chunk_ids),
                "text_length": len(text),
            }

        except FileNotFoundError as e:
            # File errors are permanent, don't retry
            logger.error(f"[extract_text] File error for {document_id}: {e}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
            raise

        except Exception as e:
            # Other errors: update status and re-raise for retry
            logger.error(f"[extract_text] Error processing {document_id}: {e}")
            document.processing_status = ProcessingStatus.FAILED
            document.error_message = str(e)
            await session.commit()
            raise
```

**Step 2: Verify existing extract_text_from_document task is correct**

The existing task should remain as-is:

```python
@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    """
    Extract text from document (PDF, DOCX, TXT, etc.) and create chunks
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        # Run async operations
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            _extract_and_chunk_document(document_id)
        )

        logger.info(f"[extract_text] Completed for document {document_id}")
        return result
    except FileNotFoundError as e:
        # File not found: permanent error, don't retry
        logger.error(f"[extract_text] File not found for {document_id}: {e}")
        raise
    except Exception as exc:
        # Other errors: retry with backoff
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)
```

**Step 3: Commit**

```bash
git add backend/app/workers/tasks.py backend/app/services/extraction.py
git commit -m "feat: integrate extraction service into worker task"
```

---

### Task 7: Write integration tests

**Files:**
- Create: `backend/tests/test_extraction_integration.py`

**Step 1: Create integration test file**

Create `backend/tests/test_extraction_integration.py`:

```python
"""Integration tests for text extraction workflow"""

import pytest
import tempfile
import os
from uuid import uuid4
from app.database import async_session
from app.models import Document, DocumentChunk, ProcessingStatus, DocumentType
from app.workers.tasks import _extract_and_chunk_document
from sqlalchemy import select


@pytest.mark.asyncio
async def test_end_to_end_extraction():
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
        async with async_session() as session:
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
            session.add(document)
            await session.commit()

            # Execute extraction
            result = await _extract_and_chunk_document(doc_id)

            # Verify result
            assert result["status"] == "success"
            assert result["document_id"] == doc_id
            assert result["chunks_created"] > 0
            assert result["text_length"] > 0

            # Verify document was updated
            stmt = select(Document).where(Document.id == doc_id)
            result_doc = await session.execute(stmt)
            updated_doc = result_doc.scalars().first()

            assert updated_doc.processing_status == ProcessingStatus.EXTRACTED
            assert updated_doc.extracted_text is not None
            assert len(updated_doc.extracted_text) > 0

            # Verify chunks were created
            stmt = select(DocumentChunk).where(
                DocumentChunk.document_id == doc_id
            ).order_by(DocumentChunk.chunk_index)
            result = await session.execute(stmt)
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
async def test_extraction_handles_missing_file():
    """Test that extraction fails gracefully with missing file"""

    async with async_session() as session:
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
        session.add(document)
        await session.commit()

        # Attempt extraction
        with pytest.raises(FileNotFoundError):
            await _extract_and_chunk_document(doc_id)

        # Verify error status was set
        stmt = select(Document).where(Document.id == doc_id)
        result = await session.execute(stmt)
        updated_doc = result.scalars().first()

        assert updated_doc.processing_status == ProcessingStatus.FAILED
        assert updated_doc.error_message is not None


@pytest.mark.asyncio
async def test_extraction_empty_document():
    """Test extraction of empty document"""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("")
        temp_file_path = f.name

    try:
        async with async_session() as session:
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
            session.add(document)
            await session.commit()

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
            result = await session.execute(stmt)
            chunks = result.scalars().all()
            assert len(chunks) == 0

    finally:
        os.unlink(temp_file_path)
```

**Step 2: Run integration tests**

```bash
pytest backend/tests/test_extraction_integration.py -v
```

Expected output:
```
test_end_to_end_extraction PASSED
test_extraction_handles_missing_file PASSED
test_extraction_empty_document PASSED
```

**Step 3: Commit**

```bash
git add backend/tests/test_extraction_integration.py
git commit -m "test: add integration tests for text extraction"
```

---

### Task 8: Run all extraction tests

**Files:**
- None (testing only)

**Step 1: Run all tests**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
pytest backend/tests/test_extraction*.py -v
```

**Step 2: Verify all pass**

Expected output:
```
test_extract_txt_file PASSED
test_extract_txt_file_empty PASSED
test_extract_unsupported_file PASSED
test_clean_text_* PASSED (8 tests)
test_chunk_text_* PASSED (5 tests)
test_create_text_chunks_* PASSED (2 tests)
test_end_to_end_extraction PASSED
test_extraction_handles_missing_file PASSED
test_extraction_empty_document PASSED

========================== 26 passed ==========================
```

---

### Task 9: Manual end-to-end test via API

**Files:**
- None (manual testing)

**Step 1: Start services**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
docker-compose up -d
sleep 5  # Wait for services to start
```

**Step 2: Verify services are running**

```bash
docker-compose ps
```

Should show postgres, redis, and other services as "Up"

**Step 3: Initialize database**

```bash
# The FastAPI app initializes DB on startup, just verify:
curl http://localhost:8000/docs
```

Should return Swagger docs (200 OK)

**Step 4: Create test case**

```bash
curl -X POST "http://localhost:8000/cases" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Text Extraction Test",
    "case_number": "EXTRACT-TEST-001",
    "practice_area": "Testing"
  }'
```

Note the returned `case_id`

**Step 5: Create test TXT file**

```bash
cat > /tmp/test_extraction.txt << 'EOF'
Section 1: Legal Document Extraction Test

This is a test document for the extraction worker.
It contains multiple paragraphs and sections to test the chunking algorithm.

The system should:
1. Extract all text from the file
2. Clean and normalize whitespace
3. Split into 4000-character chunks with 400-character overlap
4. Create DocumentChunk records in the database
5. Update document status to "extracted"

Section 2: More Content

This section provides additional content to ensure we have enough text
to test the chunking with multiple chunks. The test should verify that
chunks have the proper overlap and that no chunks are too small.

Section 3: Final Section

This concludes the test document. After extraction, we should be able to
query the document chunks and verify they were created correctly.
EOF
```

**Step 6: Upload test document**

```bash
CASE_ID="<PASTE CASE_ID FROM STEP 4>"

curl -X POST "http://localhost:8000/documents/upload?case_id=$CASE_ID" \
  -F "file=@/tmp/test_extraction.txt"
```

Note the returned `document_id`

**Step 7: Monitor worker logs**

In another terminal:

```bash
docker-compose logs -f celery-worker | grep -E "(extract_text|extract|completed)"
```

**Step 8: Verify document status**

```bash
DOC_ID="<PASTE DOCUMENT_ID FROM STEP 6>"

curl "http://localhost:8000/documents/$DOC_ID" | jq .
```

Check:
- `processing_status` should be "extracted"
- `extracted_text` should contain full text
- `error_message` should be null

**Step 9: Verify chunks in database**

```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d lex_intel -c "
SELECT id, chunk_index, length(chunk_text) as chunk_size FROM document_chunks
WHERE document_id = '<DOC_ID>'
ORDER BY chunk_index
LIMIT 10;
"
```

Should show:
- Multiple chunks with indices 0, 1, 2, ...
- Each chunk_size around 4000
- Last chunk may be smaller

**Step 10: Verify worker queued embeddings task**

Check logs:

```bash
docker-compose logs celery-worker | grep "Queued embeddings"
```

Should show:
```
[extract_text] Queued embeddings task for {document_id}
```

**Step 11: Commit** (if manual test passes)

```bash
git add -A
git commit -m "test: verify extraction via end-to-end API test"
```

---

### Task 10: Update documentation

**Files:**
- Modify: `docs/claude.md`
- Modify: `docs/WORKERS.md` (if needed)

**Step 1: Update claude.md status**

Edit `docs/claude.md` and update the status section:

Change:
```markdown
### ⏳ TODO (Phase 3-6)
- [ ] Text extraction workers (PDF, DOCX, TXT)
```

To:
```markdown
### ✅ Completed (Phase 1-3)
- [x] Text extraction workers (TXT files)

### ⏳ TODO (Phase 3+)
- [ ] PDF extraction (PyPDF2)
- [ ] DOCX extraction (python-pptx)
```

**Step 2: Commit**

```bash
git add docs/claude.md
git commit -m "docs: update Phase 3 text extraction status"
```

---

## ✅ Success Criteria

After all tasks complete:

- [x] `app/services/extraction.py` created with:
  - [x] `extract_file()` - TXT file extraction
  - [x] `clean_text()` - Text cleaning
  - [x] `chunk_text()` - Text chunking algorithm
  - [x] `create_text_chunks()` - Database persistence

- [x] `app/workers/tasks.py` updated:
  - [x] Imports extraction service
  - [x] `_extract_and_chunk_document()` uses extraction service
  - [x] Proper error handling (file errors vs transient errors)
  - [x] Status updates (PENDING → EXTRACTED)
  - [x] Queues embeddings task

- [x] Tests written and passing:
  - [x] 26 unit + integration tests passing
  - [x] End-to-end test via API succeeds
  - [x] Document chunks created correctly
  - [x] Error scenarios handled

- [x] Documentation updated:
  - [x] claude.md shows Phase 3 complete
  - [x] Design document created
  - [x] Implementation plan created

---

## 📝 Next Phase (Phase 4)

After Phase 3 completion:
- OpenAI embeddings service
- Batch embeddings generation
- pgvector storage
- Document status → INDEXED

See `docs/IMPLEMENTATION_PLAN.md` Phase 4 section.

