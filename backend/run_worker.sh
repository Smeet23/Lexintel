#!/bin/bash
# Start Celery worker for document processing

# Navigate to backend directory
cd "$(dirname "$0")"

# Run Celery worker
# --loglevel=info: Set logging level
# -c 4: Number of concurrent worker processes
# -Q default: Process tasks from "default" queue
# --max-tasks-per-child=1000: Respawn worker after 1000 tasks
python -m celery -A celery_app worker \
    --loglevel=info \
    -c 4 \
    -Q default \
    --max-tasks-per-child=1000
