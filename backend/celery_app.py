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
    # Queue routing — the producer (API send_task) and the worker MUST agree on
    # the queue name. Celery's built-in default is "celery", but run_worker.sh
    # consumes "-Q default"; without this, every task lands in "celery" and is
    # never consumed, leaving documents stuck in "processing" forever.
    task_default_queue="default",
    # Reliability: ack a task only AFTER it completes (not on receipt), and
    # re-queue it if the worker is lost mid-task (crash/deploy/restart). With
    # early ack (the default) a worker restart silently drops the in-flight task
    # and the document hangs in "processing" with no error.
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)

__all__ = ["celery_app"]
