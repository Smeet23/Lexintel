from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # App
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    API_HOST: str = "localhost"
    API_PORT: int = 8000

    # Database
    DATABASE_URL: str
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_URL: str = "redis://localhost:6379"

    # OpenAI
    OPENAI_API_KEY: str
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-4o"

    # Azure Storage (Azurite locally)
    AZURE_STORAGE_CONNECTION_STRING: str

    # Upload settings
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE: int = 104857600  # 100MB
    ALLOWED_EXTENSIONS: str = ".pdf,.docx,.txt,.doc,.pptx"

    # Processing
    CHUNK_SIZE: int = 4000
    CHUNK_OVERLAP: int = 400

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
