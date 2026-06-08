"""Shared pytest fixtures for backend tests."""
import sys
import os
import uuid

import pytest

# Ensure backend is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Mock settings before any backend imports
from unittest.mock import MagicMock
mock_settings = MagicMock()
mock_settings.google_api_key = "test-key-not-real"
mock_settings.qdrant_url = "http://localhost:6333"
mock_settings.database_url = "postgresql://test:test@localhost:5432/test"
mock_settings.azure_storage_connection_string = "UseDevelopmentStorage=true"
mock_settings.redis_url = "redis://localhost:6379/0"
mock_settings.celery_broker_url = "redis://localhost:6379/0"
mock_settings.celery_result_backend = "redis://localhost:6379/1"
mock_settings.debug = True
mock_settings.cache_enabled = False
mock_settings.cache_ttl_seconds = 0
mock_settings.allowed_origins = "http://localhost:3000"
# Numeric RRF weights — must be real numbers (not MagicMock children) so
# hybrid_search.get_rrf_weights can do arithmetic on them.
mock_settings.bm25_weight = 0.7
mock_settings.dense_weight = 0.3
mock_settings.voyage_api_key = ""
# Numeric ingest concurrency cap — must be a real int (not a MagicMock child)
# so citation_graph._gather_bounded can build an asyncio.Semaphore.
mock_settings.citation_graph_max_concurrency = 8

import backend.config
backend.config.get_settings = lambda: mock_settings


@pytest.fixture
def pdf_bytes():
    """Create a sample legal PDF for testing."""
    import fitz  # PyMuPDF
    import tempfile

    doc = fitz.open()

    # Page 1: Contract header + definitions
    page1 = doc.new_page()
    page1.insert_text(fitz.Point(72, 72), """MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into as of January 15, 2026,
by and between TechCorp Inc., a Delaware corporation ("Company"), and Legal Solutions LLC.

ARTICLE I - DEFINITIONS

Section 1.1 "Confidential Information" means any and all non-public information.
Section 1.2 "Deliverables" means any work product created by Provider.
Section 1.3 "Services" means the professional consulting and advisory services.""", fontsize=10)

    # Page 2: Representations and liability
    page2 = doc.new_page()
    page2.insert_text(fitz.Point(72, 72), """ARTICLE II - REPRESENTATIONS AND WARRANTIES

Section 2.1 Provider represents and warrants that it has the legal right and authority.
Section 2.2 Company represents and warrants that it will provide timely access.

ARTICLE III - LIMITATION OF LIABILITY

Section 3.1 NEITHER PARTY SHALL BE LIABLE FOR INDIRECT DAMAGES.
Section 3.2 Total aggregate liability shall not exceed the total fees paid.""", fontsize=10)

    # Page 3: Termination and governing law
    page3 = doc.new_page()
    page3.insert_text(fitz.Point(72, 72), """ARTICLE IV - TERMINATION

Section 4.1 Either party may terminate upon thirty (30) days written notice.

GOVERNING LAW

This Agreement shall be governed by the laws of the State of New York.

EXHIBIT A - STATEMENT OF WORK

The Provider shall deliver legal document analysis and classification services.""", fontsize=10)

    pdf_path = tempfile.mktemp(suffix=".pdf")
    doc.save(pdf_path)
    doc.close()

    with open(pdf_path, "rb") as f:
        data = f.read()

    os.unlink(pdf_path)
    return data


@pytest.fixture
def chunks(pdf_bytes):
    """Create chunks from sample PDF with pre-assigned UUIDs."""
    from unittest.mock import patch
    from services.chunking import chunk_document_from_blob

    with patch('backend.services.chunking._get_semantic_chunker', return_value=None):
        raw_chunks = chunk_document_from_blob(pdf_bytes, file_type="pdf")

    # Pre-assign UUIDs like tasks.py does
    for idx, chunk in enumerate(raw_chunks):
        chunk["id"] = str(uuid.uuid4())
        chunk["chunk_sequence"] = idx

    return raw_chunks
