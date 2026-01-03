"""Unified configuration for all services in the monorepo."""

from pydantic_settings import BaseSettings
from typing import Optional
import logging


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All services (backend, workers) should use this unified configuration
    to ensure consistency and avoid duplication.
    """

    # ===== DATABASE =====
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/lex_intel"
    """PostgreSQL connection string with asyncpg driver."""

    # ===== REDIS =====
    REDIS_URL: str = "redis://localhost:6379/0"
    """Redis connection string."""

    # ===== CELERY =====
    CELERY_BROKER_URL: Optional[str] = None
    """Celery broker URL (defaults to REDIS_URL if not set)."""

    CELERY_RESULT_BACKEND: Optional[str] = None
    """Celery result backend (defaults to REDIS_URL if not set)."""

    CELERY_TASK_SERIALIZER: str = "json"
    """Celery task serializer format."""

    CELERY_RESULT_SERIALIZER: str = "json"
    """Celery result serializer format."""

    CELERY_ACCEPT_CONTENT: list = ["json"]
    """Celery accepted content types."""

    CELERY_TIMEZONE: str = "UTC"
    """Celery timezone."""

    # ===== WORKER SETTINGS =====
    WORKER_PREFETCH_MULTIPLIER: int = 1
    """Celery worker prefetch multiplier."""

    TASK_SOFT_TIME_LIMIT: int = 1500  # 25 minutes
    """Celery task soft time limit in seconds."""

    TASK_TIME_LIMIT: int = 1800  # 30 minutes
    """Celery task hard time limit in seconds."""

    # ===== EXTRACTION SETTINGS =====
    CHUNK_SIZE: int = 4000
    """Default chunk size for text extraction in characters."""

    CHUNK_OVERLAP: int = 400
    """Default overlap between chunks in characters."""

    # ===== OPENAI SETTINGS =====
    OPENAI_API_KEY: Optional[str] = None
    """OpenAI API key for embeddings generation."""

    OPENAI_MODEL: str = "text-embedding-3-small"
    """OpenAI model for embeddings."""

    # ===== AZURE SETTINGS =====
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = None
    """Azure Storage account name."""

    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = None
    """Azure Storage account key."""

    AZURE_STORAGE_CONTAINER_NAME: str = "documents"
    """Azure Storage container name for documents."""

    # ===== LOGGING =====
    LOG_LEVEL: str = logging.getLevelName(logging.INFO)
    """Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

    # ===== APPLICATION =====
    ENV: str = "development"
    """Environment: development, staging, production."""

    DEBUG: bool = False
    """Enable debug mode."""

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **data):
        """Initialize settings with defaults for Celery URLs."""
        super().__init__(**data)

        # Default Celery URLs to Redis URL if not explicitly set
        if self.CELERY_BROKER_URL is None:
            self.CELERY_BROKER_URL = self.REDIS_URL
        if self.CELERY_RESULT_BACKEND is None:
            self.CELERY_RESULT_BACKEND = self.REDIS_URL


# Singleton instance
settings = Settings()

__all__ = ["Settings", "settings"]
