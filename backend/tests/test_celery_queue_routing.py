"""Regression guard: the producer (API send_task) and the worker MUST agree on
the Celery queue name.

Two real outages came from this mismatch:
  1. task_default_queue defaulted to "celery" while run_worker.sh consumes -Q default.
  2. main.py send_task(..., queue="celery") hardcoded the wrong queue, overriding (1).
Either one silently strands every uploaded document in a never-consumed queue
("processing" forever, no error). These tests fail fast if it regresses.
"""
import pathlib
import re

import pytest

MAIN = pathlib.Path(__file__).resolve().parents[1] / "main.py"
TASKS = pathlib.Path(__file__).resolve().parents[1] / "tasks.py"
RUN_WORKER = pathlib.Path(__file__).resolve().parents[1] / "run_worker.sh"


def test_task_default_queue_is_default():
    from backend.celery_app import celery_app
    assert celery_app.conf.task_default_queue == "default"


def test_acks_late_enabled_for_restart_safety():
    from backend.celery_app import celery_app
    assert celery_app.conf.task_acks_late is True


def test_no_hardcoded_celery_queue_in_dispatch():
    # Cover BOTH the API producer (main.py send_task) and the worker-side
    # self-dispatch (tasks.py reindex_all_matters_task.apply_async).
    for f in (MAIN, TASKS):
        src = f.read_text()
        assert "queue='celery'" not in src and 'queue="celery"' not in src, (
            f"{f.name} hardcodes the 'celery' queue but the worker consumes "
            "'default' — tasks will be stranded. Use queue='default'."
        )


def test_send_task_queue_matches_worker_queue():
    """Every dispatch(queue=...) in main.py + tasks.py must target the worker queue."""
    src = MAIN.read_text() + "\n" + TASKS.read_text()
    dispatch_queues = set(re.findall(r"queue=['\"]([a-z_]+)['\"]", src))
    worker_q = re.search(r"-Q\s+([a-z_]+)", RUN_WORKER.read_text())
    assert worker_q, "run_worker.sh must pin a queue with -Q"
    worker_queue = worker_q.group(1)
    assert dispatch_queues <= {worker_queue}, (
        f"send_task queues {dispatch_queues} must all match the worker queue "
        f"'{worker_queue}'"
    )
