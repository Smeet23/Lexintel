"""Test background job processor for case document analysis"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from backend.models import User, Case, ProcessingJob, CaseStatus
from backend.services.job_processor import get_pending_jobs, calculate_retry_delay


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


class TestJobHelper:
    """Test helper functions"""

    def test_get_pending_jobs(self, db: Session):
        """get_pending_jobs returns pending jobs in FIFO order"""
        # Create user and cases
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case1 = Case(user_id=user.id, name="Case 1", blob_storage_path="p1.pdf", status="processing")
        case2 = Case(user_id=user.id, name="Case 2", blob_storage_path="p2.pdf", status="processing")
        case3 = Case(user_id=user.id, name="Case 3", blob_storage_path="p3.pdf", status="processing")
        db.add_all([case1, case2, case3])
        db.commit()

        # Create jobs with different statuses
        job1 = ProcessingJob(id=str(uuid4()), case_id=case1.id, status="pending")
        job2 = ProcessingJob(id=str(uuid4()), case_id=case2.id, status="pending")
        job3 = ProcessingJob(id=str(uuid4()), case_id=case3.id, status="completed")
        db.add_all([job1, job2, job3])
        db.commit()

        # Get pending jobs
        pending = get_pending_jobs(db, limit=2)
        assert len(pending) == 2
        assert pending[0].case_id == case1.id
        assert pending[1].case_id == case2.id

    def test_calculate_retry_delay(self):
        """calculate_retry_delay returns correct backoff times"""
        assert calculate_retry_delay(1) == 0
        assert calculate_retry_delay(2) == 5
        assert calculate_retry_delay(3) == 10
