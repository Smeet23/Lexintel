"""Database configuration and session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base


def init_db(database_url: str):
    """Initialize database with given URL"""
    engine = create_engine(
        database_url,
        echo=False,  # Set to True for SQL debugging
        pool_pre_ping=True,  # Test connections before using
    )
    return engine, sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Runtime initialization (called once at startup)
_engine = None
_SessionLocal = None


def get_engine():
    """Get database engine (initialized at runtime)"""
    global _engine
    if _engine is None:
        from backend.config import get_settings
        settings = get_settings()
        _engine, _ = init_db(settings.database_url)
    return _engine


def get_session_factory():
    """Get session factory (initialized at runtime)"""
    global _SessionLocal
    if _SessionLocal is None:
        from backend.config import get_settings
        settings = get_settings()
        _, _SessionLocal = init_db(settings.database_url)
    return _SessionLocal


def get_db():
    """Dependency for getting database session"""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
