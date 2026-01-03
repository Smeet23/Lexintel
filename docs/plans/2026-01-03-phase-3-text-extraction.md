# Phase 3: Text Extraction Workers Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract text from PDF, DOCX, and TXT files, split into chunks, create DocumentChunk records, and queue embeddings generation.

**Architecture:** Create a text extraction service that handles multiple file formats, uses a configurable chunking strategy (4000 chars, 400 overlap), and integrates with Celery workers to process documents asynchronously. Tests verify each component in isolation before integration.

**Tech Stack:**
- PyPDF2 (PDF extraction)
- python-pptx (DOCX extraction)
- Native file I/O (TXT)
- SQLAlchemy async ORM
- Celery async tasks

---

## Task 1: Create Extraction Service - Chunking Logic

**Files:**
- Create: `backend/app/services/extraction.py`
- Create: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for text chunking**

Create `backend/tests/unit/test_extraction.py`:

```python
import pytest
from app.services.extraction import chunk_text


def test_chunk_text_basic():
    """Test basic text chunking"""
    text = "a" * 10000  # 10,000 characters
    chunks = chunk_text(text, chunk_size=4000, overlap=400)

    assert len(chunks) == 3  # 4000 + (4000-400) + remainder
    assert chunks[0] == "a" * 4000
    assert len(chunks[1]) == 4000
    assert chunks[1][:400] == "a" * 400  # Overlap from previous


def test_chunk_text_small():
    """Test chunking with text smaller than chunk size"""
    text = "hello world"
    chunks = chunk_text(text, chunk_size=4000, overlap=400)

    assert len(chunks) == 1
    assert chunks[0] == "hello world"


def test_chunk_text_empty():
    """Test chunking empty text"""
    text = ""
    chunks = chunk_text(text, chunk_size=4000, overlap=400)

    assert len(chunks) == 0


def test_chunk_text_exact_chunk_size():
    """Test chunking text exactly divisible by chunk size"""
    text = "a" * 8000
    chunks = chunk_text(text, chunk_size=4000, overlap=400)

    assert len(chunks) == 2
    assert chunks[0] == "a" * 4000
    assert chunks[1][:400] == "a" * 400  # Overlap
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
pytest backend/tests/unit/test_extraction.py::test_chunk_text_basic -v
```

Expected output:
```
FAILED - ModuleNotFoundError: No module named 'app.services.extraction'
```

**Step 3: Write minimal extraction service with chunking**

Create `backend/app/services/extraction.py`:

```python
from typing import List
import logging

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk (default 4000)
        overlap: Character overlap between chunks (default 400)

    Returns:
        List of text chunks
    """
    if not text or len(text) == 0:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move start position, accounting for overlap
        start = end - overlap

    return chunks
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_chunk_text_basic -v
pytest backend/tests/unit/test_extraction.py::test_chunk_text_small -v
pytest backend/tests/unit/test_extraction.py::test_chunk_text_empty -v
pytest backend/tests/unit/test_extraction.py::test_chunk_text_exact_chunk_size -v
```

Expected output:
```
test_chunk_text_basic PASSED
test_chunk_text_small PASSED
test_chunk_text_empty PASSED
test_chunk_text_exact_chunk_size PASSED

4 passed in 0.15s
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: implement text chunking logic with tests"
```

---

## Task 2: PDF Text Extraction

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for PDF extraction**

Add to `backend/tests/unit/test_extraction.py`:

```python
import pytest
from pathlib import Path
from app.services.extraction import extract_pdf


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a test PDF file"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    pdf_path = tmp_path / "test.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    c.drawString(100, 750, "Test PDF Content")
    c.drawString(100, 730, "Second line of text")
    c.save()

    return str(pdf_path)


def test_extract_pdf_text(sample_pdf):
    """Test extracting text from PDF"""
    text = extract_pdf(sample_pdf)

    assert isinstance(text, str)
    assert len(text) > 0
    assert "Test PDF Content" in text or "test" in text.lower()


def test_extract_pdf_not_found():
    """Test extraction with non-existent file"""
    with pytest.raises(FileNotFoundError):
        extract_pdf("/nonexistent/file.pdf")
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_pdf_text -v
```

Expected output:
```
FAILED - ModuleNotFoundError: No module named 'reportlab'
```

**Step 3: Update requirements.txt and install**

Update `backend/requirements.txt` - add these lines (find the document processing section):

```
pdf2image==1.17.0
PyPDF2==3.0.1
python-pptx==0.6.21
reportlab==4.0.7  # For testing PDF generation
```

Install new dependencies:

```bash
cd backend
pip install -r requirements.txt
```

**Step 4: Implement PDF extraction**

Update `backend/app/services/extraction.py` - add this function:

```python
def extract_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    from pathlib import Path
    from PyPDF2 import PdfReader

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        reader = PdfReader(file_path)
        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        text = "\n".join(text_parts)
        logger.info(f"[extraction] Extracted {len(text)} chars from PDF: {file_path}")

        return text
    except Exception as e:
        logger.error(f"[extraction] Failed to extract PDF {file_path}: {e}")
        raise
```

**Step 5: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_pdf_text -v
pytest backend/tests/unit/test_extraction.py::test_extract_pdf_not_found -v
```

Expected output:
```
test_extract_pdf_text PASSED
test_extract_pdf_not_found PASSED

2 passed in 0.25s
```

**Step 6: Commit**

```bash
git add backend/requirements.txt backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: add PDF text extraction with tests"
```

---

## Task 3: DOCX Text Extraction

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for DOCX extraction**

Add to `backend/tests/unit/test_extraction.py`:

```python
def test_extract_docx_text(tmp_path):
    """Test extracting text from DOCX"""
    from python_pptx import Document

    # Create test DOCX
    docx_path = tmp_path / "test.docx"
    doc = Document()
    doc.add_paragraph("Test DOCX Content")
    doc.add_paragraph("Second paragraph")
    doc.save(str(docx_path))

    text = extract_docx(str(docx_path))

    assert isinstance(text, str)
    assert len(text) > 0
    assert "Test DOCX Content" in text
    assert "Second paragraph" in text


def test_extract_docx_not_found():
    """Test DOCX extraction with non-existent file"""
    with pytest.raises(FileNotFoundError):
        extract_docx("/nonexistent/file.docx")
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_docx_text -v
```

Expected output:
```
FAILED - ModuleNotFoundError: No module named 'app.services.extraction.extract_docx'
```

**Step 3: Implement DOCX extraction**

Update `backend/app/services/extraction.py` - add this function:

```python
def extract_docx(file_path: str) -> str:
    """
    Extract text from DOCX file.

    Args:
        file_path: Path to DOCX file

    Returns:
        Extracted text

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    from pathlib import Path
    from python_pptx import Document

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file not found: {file_path}")

    try:
        doc = Document(file_path)
        text_parts = []

        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text_parts.append(paragraph.text)

        text = "\n".join(text_parts)
        logger.info(f"[extraction] Extracted {len(text)} chars from DOCX: {file_path}")

        return text
    except Exception as e:
        logger.error(f"[extraction] Failed to extract DOCX {file_path}: {e}")
        raise
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_docx_text -v
pytest backend/tests/unit/test_extraction.py::test_extract_docx_not_found -v
```

Expected output:
```
test_extract_docx_text PASSED
test_extract_docx_not_found PASSED

2 passed in 0.25s
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: add DOCX text extraction with tests"
```

---

## Task 4: Text File Extraction

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for TXT extraction**

Add to `backend/tests/unit/test_extraction.py`:

```python
def test_extract_txt_text(tmp_path):
    """Test extracting text from TXT"""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("Test TXT Content\nSecond line\nThird line")

    text = extract_txt(str(txt_path))

    assert isinstance(text, str)
    assert "Test TXT Content" in text
    assert "Second line" in text


def test_extract_txt_empty(tmp_path):
    """Test extracting from empty TXT"""
    txt_path = tmp_path / "empty.txt"
    txt_path.write_text("")

    text = extract_txt(str(txt_path))

    assert text == ""


def test_extract_txt_not_found():
    """Test TXT extraction with non-existent file"""
    with pytest.raises(FileNotFoundError):
        extract_txt("/nonexistent/file.txt")
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_txt_text -v
```

Expected output:
```
FAILED - NameError: name 'extract_txt' is not defined
```

**Step 3: Implement TXT extraction**

Update `backend/app/services/extraction.py` - add this function:

```python
def extract_txt(file_path: str) -> str:
    """
    Extract text from TXT file.

    Args:
        file_path: Path to TXT file

    Returns:
        Extracted text

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"TXT file not found: {file_path}")

    try:
        text = path.read_text(encoding='utf-8')
        logger.info(f"[extraction] Extracted {len(text)} chars from TXT: {file_path}")
        return text
    except UnicodeDecodeError:
        # Fallback for different encodings
        text = path.read_text(encoding='latin-1')
        logger.warning(f"[extraction] Used latin-1 encoding for: {file_path}")
        return text
    except Exception as e:
        logger.error(f"[extraction] Failed to extract TXT {file_path}: {e}")
        raise
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_txt_text -v
pytest backend/tests/unit/test_extraction.py::test_extract_txt_empty -v
pytest backend/tests/unit/test_extraction.py::test_extract_txt_not_found -v
```

Expected output:
```
test_extract_txt_text PASSED
test_extract_txt_empty PASSED
test_extract_txt_not_found PASSED

3 passed in 0.15s
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: add TXT file extraction with tests"
```

---

## Task 5: Universal Extract Function

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for universal extract**

Add to `backend/tests/unit/test_extraction.py`:

```python
def test_extract_file_by_extension_pdf(sample_pdf):
    """Test extract_file dispatches to correct extractor for PDF"""
    text = extract_file(sample_pdf)
    assert isinstance(text, str)
    assert len(text) > 0


def test_extract_file_by_extension_docx(tmp_path):
    """Test extract_file dispatches to correct extractor for DOCX"""
    from python_pptx import Document

    docx_path = tmp_path / "test.docx"
    doc = Document()
    doc.add_paragraph("DOCX Test")
    doc.save(str(docx_path))

    text = extract_file(str(docx_path))
    assert isinstance(text, str)
    assert "DOCX Test" in text


def test_extract_file_by_extension_txt(tmp_path):
    """Test extract_file dispatches to correct extractor for TXT"""
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("TXT Test Content")

    text = extract_file(str(txt_path))
    assert "TXT Test Content" in text


def test_extract_file_unsupported():
    """Test extract_file with unsupported file type"""
    from pathlib import Path

    # Create a dummy .doc file (old format)
    tmp_file = Path("/tmp/test.doc")
    tmp_file.write_text("dummy")

    with pytest.raises(ValueError, match="Unsupported file type"):
        extract_file(str(tmp_file))

    tmp_file.unlink()
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_file_by_extension_pdf -v
```

Expected output:
```
FAILED - NameError: name 'extract_file' is not defined
```

**Step 3: Implement universal extract function**

Update `backend/app/services/extraction.py` - add this function:

```python
def extract_file(file_path: str) -> str:
    """
    Extract text from any supported file format.
    Auto-detects file type by extension.

    Args:
        file_path: Path to file

    Returns:
        Extracted text

    Raises:
        ValueError: If file type not supported
        FileNotFoundError: If file doesn't exist
    """
    from pathlib import Path

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == '.pdf':
        return extract_pdf(file_path)
    elif ext == '.docx':
        return extract_docx(file_path)
    elif ext == '.txt':
        return extract_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_extract_file_by_extension_pdf -v
pytest backend/tests/unit/test_extraction.py::test_extract_file_by_extension_docx -v
pytest backend/tests/unit/test_extraction.py::test_extract_file_by_extension_txt -v
pytest backend/tests/unit/test_extraction.py::test_extract_file_unsupported -v
```

Expected output:
```
test_extract_file_by_extension_pdf PASSED
test_extract_file_by_extension_docx PASSED
test_extract_file_by_extension_txt PASSED
test_extract_file_unsupported PASSED

4 passed in 0.35s
```

**Step 5: Run all extraction tests**

```bash
pytest backend/tests/unit/test_extraction.py -v
```

Expected: All tests pass (11 total)

**Step 6: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: add universal extract_file dispatcher"
```

---

## Task 6: Document Chunking Service

**Files:**
- Modify: `backend/app/services/extraction.py`
- Modify: `backend/tests/unit/test_extraction.py`

**Step 1: Write failing test for document chunking service**

Add to `backend/tests/unit/test_extraction.py`:

```python
def test_chunk_extracted_text():
    """Test creating document chunks from extracted text"""
    from app.services.extraction import create_text_chunks

    text = "a" * 10000
    chunks = create_text_chunks(text, chunk_size=4000, overlap=400)

    assert len(chunks) == 3
    assert all(isinstance(c, dict) for c in chunks)
    assert all('text' in c and 'index' in c for c in chunks)
    assert chunks[0]['index'] == 0
    assert chunks[1]['index'] == 1


def test_create_text_chunks_empty():
    """Test creating chunks from empty text"""
    from app.services.extraction import create_text_chunks

    chunks = create_text_chunks("", chunk_size=4000, overlap=400)
    assert chunks == []
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/unit/test_extraction.py::test_chunk_extracted_text -v
```

Expected output:
```
FAILED - ImportError: cannot import name 'create_text_chunks'
```

**Step 3: Implement chunking service**

Update `backend/app/services/extraction.py` - add this function:

```python
def create_text_chunks(
    text: str,
    chunk_size: int = 4000,
    overlap: int = 400,
) -> List[dict]:
    """
    Split text into chunks with metadata.

    Args:
        text: Text to chunk
        chunk_size: Characters per chunk
        overlap: Character overlap between chunks

    Returns:
        List of dicts with 'text' and 'index'
    """
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

    return [
        {
            'text': chunk,
            'index': idx,
        }
        for idx, chunk in enumerate(chunks)
    ]
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/unit/test_extraction.py::test_chunk_extracted_text -v
pytest backend/tests/unit/test_extraction.py::test_create_text_chunks_empty -v
```

Expected output:
```
test_chunk_extracted_text PASSED
test_create_text_chunks_empty PASSED

2 passed in 0.15s
```

**Step 5: Commit**

```bash
git add backend/app/services/extraction.py backend/tests/unit/test_extraction.py
git commit -m "feat: add document chunking service with metadata"
```

---

## Task 7: Celery Task - Extract and Chunk Document

**Files:**
- Modify: `backend/app/workers/tasks.py`
- Create: `backend/tests/integration/test_text_extraction_task.py`

**Step 1: Write failing integration test**

Create `backend/tests/integration/test_text_extraction_task.py`:

```python
import pytest
from pathlib import Path
from app.database import async_session
from app.models import Document, DocumentChunk, ProcessingStatus
from app.workers.tasks import extract_text_from_document
from sqlalchemy.future import select


@pytest.fixture
async def sample_document(tmp_path):
    """Create a sample document with text file"""
    # Create test text file
    txt_path = tmp_path / "test.txt"
    txt_path.write_text("Sample document content for testing extraction")

    # Create document record
    async with async_session() as db:
        doc = Document(
            id="test-doc-123",
            case_id="test-case-123",
            title="Test Document",
            filename="test.txt",
            type="OTHER",
            file_path=str(txt_path),
            processing_status=ProcessingStatus.PENDING,
            file_size=len(txt_path.read_bytes()),
        )
        db.add(doc)
        await db.commit()
        await db.refresh(doc)

    return doc


@pytest.mark.asyncio
async def test_extract_text_from_document_creates_chunks(sample_document):
    """Test that task creates DocumentChunks"""
    # Run task synchronously for testing
    result = extract_text_from_document(sample_document.id)

    assert result['status'] == 'success'
    assert result['document_id'] == sample_document.id

    # Verify chunks were created
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
async def test_extract_text_updates_document_status(sample_document):
    """Test that document status is updated to EXTRACTED"""
    extract_text_from_document(sample_document.id)

    async with async_session() as db:
        stmt = select(Document).where(Document.id == sample_document.id)
        result = await db.execute(stmt)
        doc = result.scalar_one()

        assert doc.processing_status == ProcessingStatus.EXTRACTED
        assert doc.extracted_text is not None
        assert len(doc.extracted_text) > 0
```

**Step 2: Run test to verify it fails**

```bash
pytest backend/tests/integration/test_text_extraction_task.py::test_extract_text_from_document_creates_chunks -v
```

Expected output:
```
FAILED - AssertionError: Task returned incorrect format
```

**Step 3: Implement extract_text_from_document task**

Update `backend/app/workers/tasks.py` - replace the `extract_text_from_document` function:

```python
@shared_task(base=CallbackTask, bind=True, max_retries=3)
def extract_text_from_document(self, document_id: str):
    """
    Extract text from document and create chunks.

    Steps:
    1. Get document from database
    2. Extract text using appropriate handler
    3. Create DocumentChunk records
    4. Update document status to EXTRACTED
    5. Queue embedding generation
    """
    try:
        logger.info(f"[extract_text] Starting for document {document_id}")

        async def run():
            async with async_session() as db:
                # Get document
                from sqlalchemy.future import select
                from app.models import Document, DocumentChunk, ProcessingStatus
                from app.services.extraction import extract_file, create_text_chunks
                from uuid import uuid4

                stmt = select(Document).where(Document.id == document_id)
                result = await db.execute(stmt)
                doc = result.scalar_one_or_none()

                if not doc:
                    raise ValueError(f"Document {document_id} not found")

                if not doc.file_path:
                    raise ValueError(f"Document {document_id} has no file_path")

                # Extract text
                text = extract_file(doc.file_path)
                doc.extracted_text = text

                # Create chunks
                chunks_data = create_text_chunks(
                    text,
                    chunk_size=settings.CHUNK_SIZE,
                    overlap=settings.CHUNK_OVERLAP,
                )

                # Save chunks to database
                for chunk_data in chunks_data:
                    chunk = DocumentChunk(
                        id=str(uuid4()),
                        document_id=document_id,
                        chunk_text=chunk_data['text'],
                        chunk_index=chunk_data['index'],
                    )
                    db.add(chunk)

                # Update document status
                doc.processing_status = ProcessingStatus.EXTRACTED
                await db.commit()

                logger.info(
                    f"[extract_text] Created {len(chunks_data)} chunks "
                    f"for document {document_id}"
                )

                return {
                    "status": "success",
                    "document_id": document_id,
                    "text_length": len(text),
                    "chunks_created": len(chunks_data),
                }

        import asyncio
        result = asyncio.run(run())

        # Queue embedding generation
        generate_embeddings.delay(document_id)

        return result

    except Exception as exc:
        logger.error(f"[extract_text] Error: {exc}")
        raise self.retry(exc=exc, countdown=60)
```

**Step 4: Run test to verify it passes**

```bash
pytest backend/tests/integration/test_text_extraction_task.py::test_extract_text_from_document_creates_chunks -v
pytest backend/tests/integration/test_text_extraction_task.py::test_extract_text_updates_document_status -v
```

Expected output:
```
test_extract_text_from_document_creates_chunks PASSED
test_extract_text_updates_document_status PASSED

2 passed in 0.45s
```

**Step 5: Commit**

```bash
git add backend/app/workers/tasks.py backend/tests/integration/test_text_extraction_task.py
git commit -m "feat: implement extract_text_from_document Celery task"
```

---

## Task 8: Export Service and Add to __init__

**Files:**
- Modify: `backend/app/services/__init__.py`
- Modify: `backend/app/workers/__init__.py`

**Step 1: Write failing test for imports**

```bash
pytest -c "from app.services.extraction import extract_file, chunk_text; from app.workers.tasks import extract_text_from_document" backend/tests/unit/test_extraction.py -v
```

**Step 2: Update services __init__.py**

Update `backend/app/services/__init__.py`:

```python
from app.services.storage import storage_service
from app.services.extraction import (
    extract_file,
    extract_pdf,
    extract_docx,
    extract_txt,
    chunk_text,
    create_text_chunks,
)

__all__ = [
    "storage_service",
    "extract_file",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
    "chunk_text",
    "create_text_chunks",
]
```

**Step 3: Update workers __init__.py**

Update `backend/app/workers/__init__.py`:

```python
from app.workers.tasks import (
    extract_text_from_document,
    generate_embeddings,
    process_document_pipeline,
)

__all__ = [
    "extract_text_from_document",
    "generate_embeddings",
    "process_document_pipeline",
]
```

**Step 4: Test imports work**

```bash
python -c "from app.services.extraction import extract_file; from app.workers.tasks import extract_text_from_document; print('Imports successful')"
```

Expected output:
```
Imports successful
```

**Step 5: Commit**

```bash
git add backend/app/services/__init__.py backend/app/workers/__init__.py
git commit -m "feat: export extraction and worker functions"
```

---

## Task 9: Integration Test - Full Pipeline

**Files:**
- Modify: `backend/tests/integration/test_text_extraction_task.py`

**Step 1: Write failing test for full pipeline**

Add to `backend/tests/integration/test_text_extraction_task.py`:

```python
@pytest.mark.asyncio
async def test_full_extraction_pipeline(tmp_path):
    """Test complete extraction pipeline from file upload to chunks"""
    from app.services.storage import storage_service
    from app.workers.tasks import process_document_pipeline

    # Create test files
    txt_file = tmp_path / "contract.txt"
    txt_file.write_text("Contract terms and conditions: " + "x" * 5000)

    pdf_file = tmp_path / "agreement.pdf"
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    c = canvas.Canvas(str(pdf_file), pagesize=letter)
    c.drawString(100, 750, "Agreement content: " + "y" * 1000)
    c.save()

    # Create documents in database
    async with async_session() as db:
        case = Case(
            id="test-case-pipeline",
            name="Pipeline Test",
            case_number="2024-test",
            practice_area="contracts",
            description="Test",
        )
        db.add(case)
        await db.commit()

        doc1 = Document(
            id="test-doc-1",
            case_id="test-case-pipeline",
            title="Contract",
            filename="contract.txt",
            type="CONTRACT",
            file_path=str(txt_file),
            processing_status=ProcessingStatus.PENDING,
            file_size=txt_file.stat().st_size,
        )

        doc2 = Document(
            id="test-doc-2",
            case_id="test-case-pipeline",
            title="Agreement",
            filename="agreement.pdf",
            type="CONTRACT",
            file_path=str(pdf_file),
            processing_status=ProcessingStatus.PENDING,
            file_size=pdf_file.stat().st_size,
        )

        db.add(doc1)
        db.add(doc2)
        await db.commit()

    # Run extraction on both documents
    extract_text_from_document("test-doc-1")
    extract_text_from_document("test-doc-2")

    # Verify results
    async with async_session() as db:
        stmt = select(DocumentChunk).where(
            DocumentChunk.document_id.in_(["test-doc-1", "test-doc-2"])
        )
        result = await db.execute(stmt)
        chunks = result.scalars().all()

        assert len(chunks) >= 3  # Should have multiple chunks

        # Verify documents are marked as extracted
        stmt = select(Document).where(
            Document.id.in_(["test-doc-1", "test-doc-2"])
        )
        result = await db.execute(stmt)
        docs = result.scalars().all()

        assert all(d.processing_status == ProcessingStatus.EXTRACTED for d in docs)
        assert all(d.extracted_text for d in docs)
```

**Step 2: Run test**

```bash
pytest backend/tests/integration/test_text_extraction_task.py::test_full_extraction_pipeline -v
```

Expected output:
```
test_full_extraction_pipeline PASSED
```

**Step 3: Commit**

```bash
git add backend/tests/integration/test_text_extraction_task.py
git commit -m "test: add full pipeline integration test"
```

---

## Task 10: Manual End-to-End Test with Docker

**Files:**
- None (manual testing)

**Step 1: Start Docker services**

```bash
cd /Users/smeet/Documents/GitHub/Self-Learning/lex-intel
docker-compose up -d
docker-compose logs -f backend
```

**Step 2: Wait for services to be healthy**

```bash
# Check health
curl http://localhost:8000/health
# Expected: {"status": "ok"}
```

**Step 3: Create a case**

```bash
CASE_ID=$(curl -s -X POST http://localhost:8000/cases \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test v. Case",
    "case_number": "2024-test-001",
    "practice_area": "contracts",
    "status": "active",
    "description": "Test extraction"
  }' | python -m json.tool | grep '"id"' | head -1 | cut -d'"' -f4)

echo "Created case: $CASE_ID"
```

**Step 4: Create test document**

```bash
# Create test PDF
cat > /tmp/test_contract.txt << 'EOF'
CONTRACT AGREEMENT

This agreement is entered into on January 3, 2026.

PARTIES:
The undersigned parties agree to the following terms and conditions:

1. TERM: This agreement shall commence on the date hereof and continue for a period of one (1) year.

2. COMPENSATION: The Client agrees to pay the Contractor a fee of $5,000 per month.

3. DELIVERABLES: The Contractor shall deliver:
   - Initial consultation
   - Project planning
   - Monthly reports
   - Final documentation

4. TERMINATION: Either party may terminate this agreement with 30 days written notice.

5. CONFIDENTIALITY: Both parties agree to maintain confidentiality of proprietary information.

This contract represents the entire agreement between the parties.

[Additional legal text to ensure we have enough content for multiple chunks...]
EOF
```

**Step 5: Upload document**

```bash
RESPONSE=$(curl -s -X POST "http://localhost:8000/documents/upload?case_id=$CASE_ID" \
  -F "file=@/tmp/test_contract.txt")

echo "Upload response:"
echo $RESPONSE | python -m json.tool

DOC_ID=$(echo $RESPONSE | python -m json.tool | grep '"id"' | head -1 | cut -d'"' -f4)
echo "Document ID: $DOC_ID"
```

**Step 6: Monitor worker processing**

```bash
# In new terminal, watch celery logs
docker-compose logs -f celery-worker | grep extract_text

# Wait for processing to complete
sleep 10
```

**Step 7: Verify document was processed**

```bash
curl http://localhost:8000/documents/$DOC_ID | python -m json.tool
```

Expected output should show:
- `processing_status: "extracted"`
- `extracted_text` populated
- Document ready for next phase

**Step 8: Success**

```bash
echo "✅ Phase 3 extraction complete!"
```

---

## ✅ Phase 3 Completion Checklist

- [ ] All unit tests pass for extraction service
- [ ] All integration tests pass for Celery tasks
- [ ] Docker end-to-end test successful
- [ ] PDF extraction working
- [ ] DOCX extraction working
- [ ] TXT extraction working
- [ ] Text chunking with overlap working
- [ ] DocumentChunk records created correctly
- [ ] Document status updated to EXTRACTED
- [ ] Task logs show completion
- [ ] generate_embeddings task queued automatically

---

## Success Criteria

✅ **All Tests Pass**
```bash
pytest backend/tests/unit/test_extraction.py -v
pytest backend/tests/integration/test_text_extraction_task.py -v
```

✅ **Document Processing Works End-to-End**
- Upload → Extract → Chunk → Queue embeddings

✅ **Logs Show Progress**
```
[celery] Starting task: extract_text_from_document (ID: abc123)
[extract_text] Starting for document xyz
[extract_text] Created 5 chunks for document xyz
[celery] Completed task: extract_text_from_document (ID: abc123)
```

✅ **Database State Correct**
- Documents have `processing_status=EXTRACTED`
- `extracted_text` field populated
- DocumentChunk records exist with correct indices

---

## Notes for Implementation

1. **Async/Sync Bridge**: The Celery task must handle async code in a sync context using `asyncio.run()`

2. **File Encoding**: TXT extraction tries UTF-8 first, falls back to latin-1

3. **Error Handling**: Task retries with 60-second backoff on error

4. **Logging**: Use `[extract_text]` prefix consistently for monitoring

5. **Dependencies**: All are already in requirements.txt from Phase 1

---

**Proceed to next phase when all tests pass and docker end-to-end test succeeds.**
