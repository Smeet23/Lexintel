"""Progress tracking via Redis Pub/Sub."""

import json
from redis.asyncio import Redis


class ProgressPublisher:
    """Publish document processing progress to Redis."""

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    async def publish_progress(
        self,
        document_id: str,
        progress: int,
        step: str,
        message: str,
    ):
        """Publish progress update for document."""
        payload = {
            "document_id": document_id,
            "progress": progress,
            "step": step,
            "message": message,
        }

        channel = f"progress:{document_id}"
        await self.redis_client.publish(
            channel,
            json.dumps(payload)
        )
