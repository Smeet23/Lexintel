"""Celery application configuration"""
import os
from celery import Celery

# Create Celery app
celery_app = Celery("lexintel")

# Configure from environment variables
celery_app.conf.broker_url = os.getenv(
    "CELERY_BROKER_URL",
    "redis://localhost:6379/0"
)
celery_app.conf.result_backend = os.getenv(
    "CELERY_RESULT_BACKEND",
    "redis://localhost:6379/1"
)

# Configure Celery settings
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Auto-discover tasks from all apps
celery_app.autodiscover_tasks(["backend"])

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery"""
    print(f"Request: {self.request!r}")
