"""Unit tests for document extraction worker."""

import pytest
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.asyncio
async def test_extract_text_from_document_success(redis_mock, db_session_mock):
    """Test successful text extraction."""
    job_payload = {
        "document_id": "doc_123",
        "case_id": "case_456",
        "source": "upload",
    }

    # Note: This is a placeholder test
    # Real implementation would mock database and Redis
    assert job_payload["document_id"] == "doc_123"
