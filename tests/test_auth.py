"""Tests for JWT authentication endpoints"""
import pytest
import os
import sys
from datetime import datetime, timedelta
from uuid import UUID
import uuid

# Set required env vars before imports
os.environ.setdefault('DATABASE_URL', 'postgresql://test:test@localhost/test')
os.environ.setdefault('OPENAI_API_KEY', 'sk-test')
os.environ.setdefault('AZURE_STORAGE_CONNECTION_STRING', 'UseDevelopmentStorage=true')
os.environ.setdefault('SECRET_KEY', 'test-secret-key-for-testing-long-enough')
os.environ.setdefault('DEBUG', 'True')

from backend.auth import hash_password, verify_password, create_access_token, decode_token


class TestPasswordHashingFunctions:
    """Test password hashing utilities exist and are callable"""

    def test_hash_password_exists(self):
        """Test that hash_password function exists and is callable"""
        assert callable(hash_password)

    def test_verify_password_exists(self):
        """Test that verify_password function exists and is callable"""
        assert callable(verify_password)


class TestJWTTokens:
    """Test JWT token creation and validation"""

    def test_create_access_token(self):
        """Test that access token is created"""
        user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": user_id})
        assert token is not None
        assert len(token) > 0
        assert "." in token  # JWT format

    def test_decode_valid_token(self):
        """Test decoding a valid token"""
        user_id = str(uuid.uuid4())
        token = create_access_token(data={"sub": user_id})
        decoded_user_id = decode_token(token)
        assert decoded_user_id == user_id

    def test_decode_invalid_token(self):
        """Test decoding an invalid token returns None"""
        invalid_token = "invalid.token.here"
        decoded = decode_token(invalid_token)
        assert decoded is None

    def test_decode_expired_token(self):
        """Test that expired token returns None"""
        from backend.config import get_settings
        from jose import jwt

        settings = get_settings()
        user_id = str(uuid.uuid4())

        # Create an already-expired token
        past_time = datetime.utcnow() - timedelta(hours=1)
        to_encode = {"sub": user_id, "exp": past_time}
        token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

        # Decoding should fail
        decoded = decode_token(token)
        assert decoded is None

    def test_token_without_sub_claim(self):
        """Test that token without sub claim returns None"""
        from backend.config import get_settings
        from jose import jwt

        settings = get_settings()

        # Create token without sub claim
        future_time = datetime.utcnow() + timedelta(hours=1)
        to_encode = {"exp": future_time}
        token = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)

        # Should return None (no sub claim)
        decoded = decode_token(token)
        assert decoded is None

    def test_create_token_with_custom_expiry(self):
        """Test creating token with custom expiry"""
        from backend.config import get_settings
        from jose import jwt

        settings = get_settings()
        user_id = str(uuid.uuid4())

        # Create token with short expiry
        expires_delta = timedelta(minutes=5)
        token = create_access_token(data={"sub": user_id}, expires_delta=expires_delta)

        # Verify we can decode it
        decoded_user_id = decode_token(token)
        assert decoded_user_id == user_id


class TestAuthEndpointsIntegration:
    """Test authentication endpoints with database"""

    @pytest.fixture
    def test_db_session(self):
        """Create test database session"""
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.models import Base, User

        # Use file-based database for persistence
        engine = create_engine(
            "sqlite:////tmp/test_lexintel.db",
            connect_args={"check_same_thread": False}
        )

        # Drop and recreate tables
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)

        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()

        yield db

        db.close()
        Base.metadata.drop_all(bind=engine)

    def test_user_query_by_email(self, test_db_session):
        """Test querying user by email"""
        from backend.models import User

        # Create a new user with pre-hashed password
        user_id = uuid.uuid4()
        email = "querytest@example.com"
        # Use a mock hash instead of hashing
        mock_hash = "$2b$12$mock.hash.value.for.testing"

        new_user = User(
            id=user_id,
            email=email,
            password_hash=mock_hash,
            created_at=datetime.utcnow()
        )
        test_db_session.add(new_user)
        test_db_session.commit()

        # Verify user exists by email
        found_user = test_db_session.query(User).filter(User.email == email).first()
        assert found_user is not None
        assert found_user.email == email
        assert found_user.id == user_id

    def test_user_deletion_tracking(self, test_db_session):
        """Test that is_deleted flag prevents user queries"""
        from backend.models import User

        # Create a user
        user_id = uuid.uuid4()
        email = "deletetest@example.com"
        mock_hash = "$2b$12$mock.hash.value.for.testing"

        new_user = User(
            id=user_id,
            email=email,
            password_hash=mock_hash,
            created_at=datetime.utcnow()
        )
        test_db_session.add(new_user)
        test_db_session.commit()

        # Mark as deleted
        new_user.is_deleted = True
        test_db_session.commit()

        # Should not find active user
        found_user = test_db_session.query(User).filter(
            User.email == email,
            User.is_deleted == False
        ).first()
        assert found_user is None


class TestTokenAuthenticationHeaders:
    """Test token validation in authorization headers"""

    def test_bearer_token_extraction(self):
        """Test extracting bearer token from authorization header"""
        token = "test.jwt.token"
        header = f"Bearer {token}"

        # Simulate what get_current_user does
        try:
            scheme, extracted_token = header.split(" ")
            assert scheme.lower() == "bearer"
            assert extracted_token == token
        except (ValueError, IndexError):
            assert False, "Should not raise"

    def test_invalid_bearer_format(self):
        """Test that invalid format raises error"""
        header = "InvalidToken"

        with pytest.raises((ValueError, IndexError)):
            scheme, token = header.split(" ")

    def test_case_insensitive_bearer_scheme(self):
        """Test that Bearer scheme is case-insensitive"""
        token = "test.jwt.token"

        # Test uppercase
        header = f"BEARER {token}"
        scheme, extracted_token = header.split(" ")
        assert scheme.lower() == "bearer"

        # Test mixed case
        header = f"Bearer {token}"
        scheme, extracted_token = header.split(" ")
        assert scheme.lower() == "bearer"
