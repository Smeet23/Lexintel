#!/bin/bash
# Start Celery worker for document processing

# Run from the PROJECT ROOT (parent of backend/) and ensure it is on PYTHONPATH.
# celery_app.py uses include=["backend.tasks"] and the codebase imports
# `backend.*`, so the `backend` package must be importable — which it is NOT when
# the worker is started from inside backend/ with `-A celery_app` (that was the
# old behaviour and failed with `ModuleNotFoundError: No module named 'backend'`).
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH}"

# Run Celery worker
# --loglevel=info: Set logging level
# -c 4: Number of concurrent worker processes
# -Q default: Process tasks from "default" queue
# --max-tasks-per-child=1000: Respawn worker after 1000 tasks
python -m celery -A backend.celery_app worker \
    --loglevel=info \
    -c 4 \
    -Q default \
    --max-tasks-per-child=1000
