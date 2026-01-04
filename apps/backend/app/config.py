"""Backend-specific configuration extending shared settings."""

from shared import Settings as SharedSettings
from typing import Optional


class Settings(SharedSettings):
    """Backend application settings extending shared configuration.

    This class extends the shared Settings to add backend-specific
    configuration while maintaining consistency for shared settings.
    """

    # ===== API SETTINGS =====
    API_HOST: str = "localhost"
    """API host address."""

    API_PORT: int = 8000
    """API port number."""

    # ===== DATABASE =====
    DATABASE_ECHO: bool = False
    """Enable SQLAlchemy echo for SQL debugging."""

    # ===== FILE UPLOAD =====
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    """Maximum upload size in bytes."""

    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.doc,.pptx"
    """Comma-separated list of allowed file extensions."""

    # ===== OPENAI CHAT =====
    OPENAI_CHAT_MODEL: str = "gpt-4o"
    """OpenAI model for chat/RAG operations."""

    # ===== AZURE STORAGE =====
    AZURE_STORAGE_CONNECTION_STRING: Optional[str] = None
    """Azure Storage connection string (used instead of account name/key for backend)."""

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = True


# Singleton instance
settings = Settings()
