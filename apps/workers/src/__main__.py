"""Worker service entry point."""

import signal
import sys
import asyncio
from celery_app import celery_app
from lib import close_redis


def shutdown_handler(signum, frame):
    """Handle graceful shutdown."""
    print("Received shutdown signal, gracefully stopping...")
    celery_app.control.shutdown()
    try:
        asyncio.run(close_redis())
    except:
        pass
    print("Worker shutdown complete")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    print("Starting worker service...")

    # Start Celery worker
    celery_app.worker_main([
        "worker",
        "--loglevel=info",
        "--concurrency=4",
    ])
