"""Async/event loop utilities."""

import asyncio
from typing import Any


def run_async(coro) -> Any:
    """Run async code from sync context (e.g., Celery tasks).

    Handles event loop creation/reuse gracefully for different environments.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    return loop.run_until_complete(coro)


__all__ = ["run_async"]
