"""Tests for progress SSE endpoint."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_progress_endpoint_exists():
    """Test that progress endpoint is accessible."""
    response = client.get("/api/documents/test_doc_id/progress")
    # SSE endpoint should be stream-able
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
