"""Celery application configuration."""

from celery import Celery
from config import WorkerConfig

# Create Celery app
celery_app = Celery(__name__)

# Configure Celery
celery_app.conf.update(
    broker_url=WorkerConfig.CELERY_BROKER_URL,
    result_backend=WorkerConfig.CELERY_RESULT_BACKEND,
    task_serializer=WorkerConfig.CELERY_TASK_SERIALIZER,
    result_serializer=WorkerConfig.CELERY_RESULT_SERIALIZER,
    accept_content=WorkerConfig.CELERY_ACCEPT_CONTENT,
    timezone=WorkerConfig.CELERY_TIMEZONE,
    task_acks_late=True,
    worker_prefetch_multiplier=WorkerConfig.WORKER_PREFETCH_MULTIPLIER,
    task_reject_on_worker_lost=True,
    task_soft_time_limit=WorkerConfig.TASK_SOFT_TIME_LIMIT,
    task_time_limit=WorkerConfig.TASK_TIME_LIMIT,
    task_track_started=True,
)

# Auto-discover tasks from workers module
celery_app.autodiscover_tasks(["workers"])
