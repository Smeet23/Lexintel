from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # Database
    database_url: str

    # Google AI (Gemini) - used for RAG LLM
    google_api_key: str

    # Cohere - used for embeddings (required — no default so app fails fast if missing)
    cohere_api_key: str

    # Gemini LLM model
    gemini_model: str = "gemini-2.5-flash-lite"

    # Qdrant
    qdrant_url: str = "http://localhost:6333"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Azure Blob Storage
    azure_storage_connection_string: str

    # CORS Configuration
    allowed_origins: str = "http://localhost:3000"

    # Environment
    debug: bool = False

    # Query Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 86400  # 24 hours

    # CourtListener API (for on-demand legal research)
    courtlistener_api_token: str = ""

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_allowed_origins_list(self) -> List[str]:
        """Parse comma-separated allowed origins"""
        origins = [origin.strip() for origin in self.allowed_origins.split(",")]
        return origins

@lru_cache()
def get_settings():
    return Settings()
