"""Redis pub/sub service for real-time progress updates."""
import json
import logging
import redis
from typing import Optional

try:
    from backend.config import get_settings
except ImportError:
    try:
        from config import get_settings
    except ImportError:
        from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Initialize Redis client from Celery broker URL
redis_client = redis.from_url(settings.celery_broker_url, decode_responses=True)

# Processing stages definition
STAGES = [
    ("uploaded", "Uploaded"),
    ("downloading", "Downloading"),
    ("chunking", "Chunking document"),
    ("embedding", "Generating embeddings"),
    ("indexing", "Indexing vectors"),
    ("storing", "Storing metadata"),
    ("ready", "Ready"),
]

STAGE_KEYS = [s[0] for s in STAGES]


def get_step_number(stage: str) -> int:
    """Get the step number (1-indexed) for a given stage key."""
    try:
        return STAGE_KEYS.index(stage) + 1
    except ValueError:
        # For special stages like 'error' or 'retrying'
        return 0


def get_channel_name(case_id: str) -> str:
    """Get the Redis channel name for a case."""
    return f"lexintel:case:{case_id}:progress"


def publish_progress(
    case_id: str,
    stage: str,
    progress: int = 0,
    message: Optional[str] = None,
    detail: str = "",
    retry_attempt: Optional[int] = None,
    error: Optional[str] = None
) -> bool:
    """
    Publish a progress event to Redis channel.

    Args:
        case_id: UUID of the case being processed
        stage: Current processing stage key
        progress: Progress percentage (0-100) for current stage
        message: Human-readable status message (defaults to stage label)
        detail: Additional detail text (e.g., "54 of 120 chunks")
        retry_attempt: Current retry attempt number if retrying
        error: Error message if stage is 'error'

    Returns:
        True if published successfully, False otherwise
    """
    channel = get_channel_name(case_id)

    # Get default message from stage label
    if message is None:
        stage_labels = dict(STAGES)
        message = stage_labels.get(stage, stage.capitalize())

    payload = {
        "stage": stage,
        "step": get_step_number(stage),
        "total_steps": len(STAGES),
        "progress": min(100, max(0, progress)),  # Clamp to 0-100
        "message": message,
        "detail": detail,
        "retry_attempt": retry_attempt,
        "error": error,
    }

    try:
        redis_client.publish(channel, json.dumps(payload))
        logger.debug(f"Published progress for case {case_id}: {stage} ({progress}%)")
        return True
    except redis.RedisError as e:
        logger.error(f"Failed to publish progress for case {case_id}: {e}")
        return False


def publish_uploaded(case_id: str, filename: str) -> bool:
    """Publish 'uploaded' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="uploaded",
        progress=100,
        message="Document uploaded",
        detail=filename
    )


def publish_downloading(case_id: str) -> bool:
    """Publish 'downloading' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="downloading",
        progress=0,
        message="Fetching document from storage..."
    )


def publish_chunking(case_id: str, progress: int = 0, current: int = 0, total: int = 0) -> bool:
    """Publish 'chunking' stage event."""
    detail = f"{current} of {total} pages" if total > 0 else ""
    return publish_progress(
        case_id=case_id,
        stage="chunking",
        progress=progress,
        message="Splitting document into chunks...",
        detail=detail
    )


def publish_embedding(case_id: str, progress: int = 0, current: int = 0, total: int = 0) -> bool:
    """Publish 'embedding' stage event."""
    detail = f"{current} of {total} chunks" if total > 0 else ""
    return publish_progress(
        case_id=case_id,
        stage="embedding",
        progress=progress,
        message="Generating embeddings...",
        detail=detail
    )


def publish_indexing(case_id: str, progress: int = 0, detail: str = "") -> bool:
    """Publish 'indexing' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="indexing",
        progress=progress,
        message="Indexing vectors...",
        detail=detail
    )


def publish_storing(case_id: str) -> bool:
    """Publish 'storing' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="storing",
        progress=0,
        message="Storing metadata..."
    )


def publish_ready(case_id: str, chunk_count: int) -> bool:
    """Publish 'ready' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="ready",
        progress=100,
        message="Processing complete!",
        detail=f"{chunk_count} chunks ready for analysis"
    )


def publish_error(case_id: str, error_message: str, retry_attempt: Optional[int] = None) -> bool:
    """Publish 'error' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="error",
        progress=0,
        message="Processing failed",
        error=error_message,
        retry_attempt=retry_attempt
    )


def publish_retrying(case_id: str, attempt: int, max_attempts: int, error_message: str) -> bool:
    """Publish 'retrying' stage event."""
    return publish_progress(
        case_id=case_id,
        stage="retrying",
        progress=0,
        message=f"Retrying... (attempt {attempt}/{max_attempts})",
        retry_attempt=attempt,
        error=error_message
    )
