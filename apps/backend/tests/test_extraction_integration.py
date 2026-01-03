"""Integration tests for text extraction workflow

NOTE: This test file previously tested internal worker task logic by directly importing
_extract_and_chunk_document from app.workers.tasks. Since the monorepo refactoring moved
worker tasks to the separate /apps/workers/ service, these tests have been refactored to:

1. Focus on API-level behavior (e.g., document upload triggering extraction)
2. Mock Celery tasks rather than importing them directly
3. Leave worker-specific tests to /apps/workers/tests/

The worker extraction logic is now tested in:
- /apps/workers/tests/unit/test_document_extraction.py
- /apps/workers/tests/integration/test_extraction_workflow.py
"""

import pytest
from uuid import uuid4
from unittest.mock import patch, AsyncMock
from app.models import Document, ProcessingStatus, DocumentType
from sqlalchemy import select


# REMOVED: test_end_to_end_extraction
# This test was testing internal worker task logic (_extract_and_chunk_document).
# The extraction workflow is now tested in /apps/workers/tests/ using the Celery task.
# For backend API testing, use mock Celery tasks instead of importing them directly.


# REMOVED: test_extraction_handles_missing_file
# This test was testing internal worker error handling.
# Error handling for extraction is now tested in /apps/workers/tests/unit/test_document_extraction.py


# REMOVED: test_extraction_empty_document
# This test was testing internal worker logic for empty documents.
# Empty document handling is now tested in /apps/workers/tests/
