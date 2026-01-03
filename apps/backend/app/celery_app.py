from celery import Celery
from celery.signals import task_prerun, task_postrun
from app.config import settings
import logging

logger = logging.getLogger(__name__)

# Initialize Celery with settings
celery_app = Celery(
    "lex-intel",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
)

# Configure Celery with shared settings
celery_app.conf.update(
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=True,
    task_track_started=True,
    task_time_limit=settings.TASK_TIME_LIMIT,
    task_soft_time_limit=settings.TASK_SOFT_TIME_LIMIT,
    worker_prefetch_multiplier=settings.WORKER_PREFETCH_MULTIPLIER,
    worker_max_tasks_per_child=1000,
)

@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, **kwargs):
    logger.info(f"[celery] Starting task: {task.name} (ID: {task_id})")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, state=None, **kwargs):
    logger.info(f"[celery] Completed task: {task.name} (ID: {task_id})")

# Note: Task discovery is handled by the separate /apps/workers/ service
# Backend sends tasks by name via celery_app.send_task() without needing autodiscover
