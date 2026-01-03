"""Test configuration and fixtures."""

import pytest
import asyncio
import os
from pathlib import Path


# Set up test database URL before importing shared
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")

from shared import async_session, engine, Base


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def setup_db():
    """Set up test database."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def db_session(event_loop, setup_db):
    """Provide a database session for testing."""
    # For sync fixtures with async setup
    event_loop.run_until_complete(setup_db)

    # Create a sync wrapper for the async session
    class SyncSessionWrapper:
        def __init__(self):
            self._session = None

        def add(self, obj):
            if self._session is None:
                self._session = event_loop.run_until_complete(async_session())
            self._session.add(obj)

        def commit(self):
            if self._session:
                event_loop.run_until_complete(self._session.commit())

        def rollback(self):
            if self._session:
                event_loop.run_until_complete(self._session.rollback())

    session = SyncSessionWrapper()
    yield session
    if session._session:
        event_loop.run_until_complete(session._session.close())


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
