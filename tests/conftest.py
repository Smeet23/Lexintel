"""Pytest configuration and fixtures"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.models import Base
from backend.config import get_settings


@pytest.fixture
def db():
    """Test database session"""
    # Use in-memory SQLite for speed
    engine = create_engine("sqlite:///:memory:")

    # Create tables
    Base.metadata.create_all(bind=engine)

    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()

    yield db

    db.rollback()
    db.close()

    # Drop all tables
    Base.metadata.drop_all(bind=engine)
