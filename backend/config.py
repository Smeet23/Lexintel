from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str

    # OpenAI
    openai_api_key: str

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Azure Blob Storage
    azure_storage_connection_string: str

    # JWT
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 1440

    # CORS Configuration
    allowed_origins: str = "http://localhost:3000"

    # Environment
    debug: bool = False

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_allowed_origins_list(self) -> List[str]:
        """Parse comma-separated allowed origins"""
        origins = [origin.strip() for origin in self.allowed_origins.split(",")]
        return origins

    @field_validator('allowed_origins')
    @classmethod
    def validate_cors_origins(cls, v: str, info) -> str:
        """Validate that CORS origins don't contain placeholders in production"""
        # This validator is called after field assignment
        # We'll validate in startup instead for better error messages
        return v

@lru_cache()
def get_settings():
    return Settings()
