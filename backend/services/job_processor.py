"""Background job processor for case document analysis"""
from sqlalchemy.orm import Session
from backend.models import ProcessingJob
from typing import List


def get_pending_jobs(db: Session, limit: int = 5) -> List[ProcessingJob]:
    """Get pending jobs in FIFO order"""
    return db.query(ProcessingJob).filter(
        ProcessingJob.status == "pending"
    ).order_by(ProcessingJob.created_at).limit(limit).all()


def calculate_retry_delay(attempt: int) -> int:
    """Calculate retry delay in seconds based on attempt number"""
    delays = {1: 0, 2: 5, 3: 10}
    return delays.get(attempt, 10)
