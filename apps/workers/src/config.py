"""Worker-specific configuration."""

import os
from dotenv import load_dotenv

load_dotenv()


class WorkerConfig:
    """Worker configuration settings."""

    # Redis configuration
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Database configuration
    DATABASE_URL = os.getenv(
        "DATABASE_URL",
        "postgresql+asyncpg://postgres:postgres@localhost/lex_intel"
    )

    # Celery configuration
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SERIALIZER = "json"
    CELERY_RESULT_SERIALIZER = "json"
    CELERY_ACCEPT_CONTENT = ["json"]
    CELERY_TIMEZONE = "UTC"

    # Worker settings
    WORKER_PREFETCH_MULTIPLIER = int(os.getenv("WORKER_PREFETCH_MULTIPLIER", "1"))
    TASK_SOFT_TIME_LIMIT = int(os.getenv("TASK_SOFT_TIME_LIMIT", "1500"))  # 25 min
    TASK_TIME_LIMIT = int(os.getenv("TASK_TIME_LIMIT", "1800"))  # 30 min

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
