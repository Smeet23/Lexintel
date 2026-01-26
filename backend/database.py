"""Database configuration and session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Handle both module import styles
try:
    from backend.models import Base
except ImportError:
    from models import Base

# Export SessionLocal for backward compatibility
SessionLocal = None


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
        try:
            from backend.config import get_settings
        except ImportError:
            from config import get_settings
        settings = get_settings()
        _engine, _ = init_db(settings.database_url)
    return _engine


def get_session_factory():
    """Get session factory (initialized at runtime)"""
    global _SessionLocal, SessionLocal
    if _SessionLocal is None:
        try:
            from backend.config import get_settings
        except ImportError:
            from config import get_settings
        settings = get_settings()
        _, _SessionLocal = init_db(settings.database_url)
        SessionLocal = _SessionLocal  # Export it
    return _SessionLocal


def get_db():
    """Dependency for getting database session"""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
