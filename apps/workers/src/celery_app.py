"""Celery application configuration using shared settings."""

from celery import Celery
from shared import settings

# Create Celery app
celery_app = Celery(__name__)

# Configure Celery with shared settings
celery_app.conf.update(
    broker_url=settings.CELERY_BROKER_URL,
    result_backend=settings.CELERY_RESULT_BACKEND,
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    timezone=settings.CELERY_TIMEZONE,
    # Worker-specific settings
    task_acks_late=True,
    worker_prefetch_multiplier=settings.WORKER_PREFETCH_MULTIPLIER,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
    task_time_limit=settings.TASK_TIME_LIMIT,
    task_track_started=True,
)

# Auto-discover tasks from workers module
celery_app.autodiscover_tasks(["workers"])
