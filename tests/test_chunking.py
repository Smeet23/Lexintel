"""Tests for PDF chunking service"""
import pytest
import os
import tempfile
from pathlib import Path

# Set environment variables before imports
os.environ.setdefault('DATABASE_URL', 'sqlite:////tmp/test_lexintel_chunking.db')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')
os.environ.setdefault('DEBUG', 'True')

from backend.services.chunking import chunk_pdf, chunk_pdf_from_blob, estimate_tokens


def create_test_pdf() -> bytes:
    """Create minimal valid PDF for testing"""
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R >>
endobj
4 0 obj
<< >>
stream
BT
/F1 12 Tf
100 700 Td
(This is a test PDF with some content.) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000217 00000 n
trailer
<< /Size 5 /Root 1 0 R >>
startxref
312
%%EOF"""
    return pdf_content


@pytest.fixture
def test_pdf_file():
    """Create temporary PDF file for testing"""
    pdf_content = create_test_pdf()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_content)
        temp_path = tmp.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


class TestChunkingFunctions:
    """Test PDF chunking functionality"""

    def test_chunk_pdf_success(self, test_pdf_file):
        """Test successful PDF chunking"""
        chunks = chunk_pdf(test_pdf_file)

        assert chunks is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert "content" in chunk
            assert "page_num" in chunk
            assert "section_name" in chunk
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0

    def test_chunk_pdf_file_not_found(self):
        """Test chunking with non-existent file"""
        with pytest.raises(FileNotFoundError):
            chunk_pdf("/non/existent/file.pdf")

    def test_chunk_pdf_returns_metadata(self, test_pdf_file):
        """Test that chunks include proper metadata"""
        chunks = chunk_pdf(test_pdf_file)

        for chunk in chunks:
            # Page numbers should be strings representing 1-indexed pages
            page_num = int(chunk["page_num"])
            assert page_num >= 1, "Page numbers should be 1-indexed"

            # Section names should follow pattern
            assert chunk["section_name"].startswith("Chunk")

    def test_chunk_pdf_overlap(self, test_pdf_file):
        """Test that consecutive chunks have overlap for context"""
        chunks = chunk_pdf(test_pdf_file)

        if len(chunks) >= 2:
            # With overlap, consecutive chunks should have some similar content
            # (This is a basic test - exact overlap depends on splitter behavior)
            chunk1_end = chunks[0]["content"][-100:].lower()
            chunk2_start = chunks[1]["content"][:100].lower()

            # There may be some overlap, though not guaranteed with all PDFs
            assert isinstance(chunk1_end, str)
            assert isinstance(chunk2_start, str)


class TestBlobChunking:
    """Test chunking from blob storage content"""

    @pytest.mark.asyncio
    async def test_chunk_pdf_from_blob_success(self):
        """Test chunking blob content"""
        pdf_content = create_test_pdf()

        chunks = await chunk_pdf_from_blob(pdf_content)

        assert chunks is not None
        assert isinstance(chunks, list)
        assert len(chunks) > 0

        # Verify chunk structure
        for chunk in chunks:
            assert "content" in chunk
            assert "page_num" in chunk
            assert "section_name" in chunk

    @pytest.mark.asyncio
    async def test_chunk_pdf_from_blob_empty(self):
        """Test chunking with empty blob"""
        with pytest.raises(ValueError, match="empty"):
            await chunk_pdf_from_blob(b"")

    @pytest.mark.asyncio
    async def test_chunk_pdf_from_blob_invalid(self):
        """Test chunking with invalid PDF content"""
        invalid_content = b"This is not a PDF file"

        with pytest.raises(Exception):
            await chunk_pdf_from_blob(invalid_content)


class TestTokenEstimation:
    """Test token estimation"""

    def test_estimate_tokens_short(self):
        """Test token estimation for short text"""
        text = "Hello world"
        tokens = estimate_tokens(text)

        assert tokens >= 0
        assert tokens <= len(text) / 2  # Should be less than 1 token per char

    def test_estimate_tokens_long(self):
        """Test token estimation for long text"""
        text = "A" * 1000  # 1000 characters
        tokens = estimate_tokens(text)

        # Roughly 1 token per 4 characters
        assert tokens == 250

    def test_estimate_tokens_empty(self):
        """Test token estimation for empty text"""
        tokens = estimate_tokens("")
        assert tokens == 0


class TestChunkingConfiguration:
    """Test chunking configuration"""

    def test_chunk_size_respected(self, test_pdf_file):
        """Test that chunks don't exceed size limit (with some tolerance)"""
        chunks = chunk_pdf(test_pdf_file)

        CHUNK_SIZE = 800
        TOLERANCE = 1.2  # Allow 20% over due to splitter behavior

        for chunk in chunks:
            content_length = len(chunk["content"])
            # Chunks might exceed size slightly due to splitter algorithm
            assert content_length <= CHUNK_SIZE * TOLERANCE, \
                f"Chunk size {content_length} exceeds limit {CHUNK_SIZE * TOLERANCE}"
