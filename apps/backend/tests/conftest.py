import pytest
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import MetaData, Table, Column, String, Text, Integer, ForeignKey
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

    # Create tables manually for SQLite (doesn't support pgvector Vector type)
    async with engine.begin() as conn:
        # Create Document table using raw SQL
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                case_id TEXT NOT NULL,
                title TEXT,
                filename TEXT NOT NULL,
                type TEXT,
                extracted_text TEXT,
                page_count INTEGER,
                file_size INTEGER,
                file_path TEXT,
                processing_status TEXT DEFAULT 'pending',
                error_message TEXT,
                indexed_at TIMESTAMP,
                created_at TIMESTAMP,
                updated_at TIMESTAMP
            )
        """)

        # Create DocumentChunk table using raw SQL (embedding as TEXT for SQLite)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding TEXT,
                search_vector TEXT,
                FOREIGN KEY(document_id) REFERENCES documents(id)
            )
        """)

    # Create session
    async_session_factory = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session_factory() as session:
        yield session

    # Cleanup
    async with engine.begin() as conn:
        await conn.execute("DROP TABLE IF EXISTS document_chunks")
        await conn.execute("DROP TABLE IF EXISTS documents")

    await engine.dispose()
