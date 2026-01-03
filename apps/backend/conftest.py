"""Pytest configuration and fixtures"""

import os
import sys
import tempfile
from pathlib import Path

# Set test environment variables BEFORE any imports
# This must happen at module load time, not in a fixture
os.environ["DATABASE_URL"] = "postgresql+asyncpg://lex_user:lex_password@postgres:5432/lex_intel_test"
os.environ["REDIS_URL"] = "redis://redis:6379"
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["AZURE_STORAGE_CONNECTION_STRING"] = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXOU+FH+fNGuNGVGyWjnRZGuyBC0wWgyWkVDclxwGXQ15j0Dhn4XbJXg==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
os.environ["UPLOAD_DIR"] = tempfile.gettempdir()

import pytest
