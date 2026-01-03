"""Test configuration and fixtures."""

import pytest
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_mock():
    """Mock Redis client for testing."""
    from unittest.mock import AsyncMock
    return AsyncMock()


@pytest.fixture
async def db_session_mock():
    """Mock database session for testing."""
    from unittest.mock import AsyncMock
    return AsyncMock()
