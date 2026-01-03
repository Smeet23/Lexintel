"""Worker service entry point."""

import signal
import sys
import asyncio
import logging
from celery_app import celery_app
from lib import close_redis

logger = logging.getLogger(__name__)


def shutdown_handler(signum, frame):  # noqa: ARG001
    """Handle graceful shutdown."""
    logger.info("Received shutdown signal, gracefully stopping...")
    celery_app.control.shutdown()
    try:
        asyncio.run(close_redis())
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")
    logger.info("Worker shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    logger.info("Starting worker service...")

    # Start Celery worker
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
    ])
