"""Celery application for async task processing"""
from celery import Celery

# Handle both import styles for config
try:
    from backend.config import get_settings
except ImportError:
    from config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "lexintel",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    # Auto-discover tasks in these modules
    include=["backend.tasks"]
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes hard limit
    task_soft_time_limit=25 * 60,  # 25 minutes soft limit
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=1000,  # Worker respawn after 1000 tasks
    broker_connection_retry_on_startup=True,  # Retry broker connection on startup (Celery 6.0 compatibility)
)

__all__ = ["celery_app"]
