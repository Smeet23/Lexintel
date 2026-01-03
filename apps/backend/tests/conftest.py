import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import MetaData, Table
from shared.models import Base, Document, DocumentChunk


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def db_session():
    """Provide a database session for async tests."""
    # Create in-memory SQLite database for testing
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    # Create only the tables we need (Document and DocumentChunk)
    async with engine.begin() as conn:
        # Create Document table
        await conn.run_sync(Document.__table__.create, checkfirst=True)
        # Create DocumentChunk table
        await conn.run_sync(DocumentChunk.__table__.create, checkfirst=True)

    # Create session
    async_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_factory() as session:
        yield session

    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(DocumentChunk.__table__.drop, checkfirst=True)
        await conn.run_sync(Document.__table__.drop, checkfirst=True)

    await engine.dispose()
