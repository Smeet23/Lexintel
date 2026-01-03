from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from typing import AsyncGenerator
from os import getenv

# Import Base from models
from shared.models import Base

# Get database URL from environment
DATABASE_URL = getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/lex_intel"
)
DATABASE_ECHO = getenv("DATABASE_ECHO", "false").lower() == "true"

# Create async engine with asyncpg driver
engine = create_async_engine(
    DATABASE_URL,
    echo=DATABASE_ECHO,
    future=True,
    poolclass=NullPool,
    connect_args={"server_settings": {"application_name": "lex-intel"}},
)

# Create async session factory
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Initialize database tables"""
    from sqlalchemy import text

    # Try to enable extensions (non-critical for MVP)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    except Exception:
        pass  # pgvector not available

    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm"))
    except Exception:
        pass  # pg_trgm not available

    # Create all tables (critical)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
