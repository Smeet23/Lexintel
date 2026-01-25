"""Test background job processor for case document analysis"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.models import User, Case, ProcessingJob, CaseStatus


class TestProcessingJob:
    """Test ProcessingJob model"""

    def test_processing_job_creation(self, db: Session):
        """ProcessingJob can be created and stored in database"""
        # Create a user and case first
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Smith v. Jones",
            blob_storage_path="cases/abc123.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create processing job
        job = ProcessingJob(
            id=str(uuid4()),
            case_id=case.id,
            status="pending"
        )
        db.add(job)
        db.commit()

        # Verify it was stored
        retrieved = db.query(ProcessingJob).filter(ProcessingJob.id == job.id).first()
        assert retrieved is not None
        assert retrieved.case_id == case.id
        assert retrieved.status == "pending"
        assert retrieved.attempts == 0
        assert retrieved.max_attempts == 3
        assert retrieved.error_message is None
        assert retrieved.next_retry_at is None

    def test_processing_job_status_transition(self, db: Session):
        """ProcessingJob status can transition between states"""
        # Create a user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Smith v. Jones",
            blob_storage_path="cases/abc123.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(
            id=str(uuid4()),
            case_id=case.id,
            status="pending"
        )
        db.add(job)
        db.commit()

        # Transition: pending -> processing
        job.status = "processing"
        job.started_at = datetime.now(timezone.utc)
        job.attempts = 1
        db.commit()

        retrieved = db.query(ProcessingJob).filter(ProcessingJob.id == job.id).first()
        assert retrieved.status == "processing"
        assert retrieved.attempts == 1
        assert retrieved.started_at is not None

        # Transition: processing -> completed
        job.status = "completed"
        job.completed_at = datetime.now(timezone.utc)
        db.commit()

        retrieved = db.query(ProcessingJob).filter(ProcessingJob.id == job.id).first()
        assert retrieved.status == "completed"
        assert retrieved.completed_at is not None
