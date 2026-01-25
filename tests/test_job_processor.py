"""Test background job processor for case document analysis"""
import pytest
from uuid import uuid4
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock
from sqlalchemy.orm import Session
from backend.models import User, Case, ProcessingJob, CaseStatus, Chunk
from backend.services.job_processor import (
    get_pending_jobs,
    calculate_retry_delay,
    process_case,
)


@pytest.fixture
def mock_storage(monkeypatch):
    """Mock Azure Blob Storage"""
    mock = AsyncMock()
    mock.return_value = b"%PDF-1.4..."  # valid PDF bytes
    monkeypatch.setattr("backend.services.job_processor.download_pdf_from_blob", mock)
    return mock


@pytest.fixture
def mock_chunking(monkeypatch):
    """Mock chunking service"""
    mock = Mock()
    mock.return_value = [
        {"id": "chunk-1", "content": "Legal text 1", "page_num": "1", "section_name": "Intro"},
        {"id": "chunk-2", "content": "Legal text 2", "page_num": "2", "section_name": "Body"},
    ]
    monkeypatch.setattr("backend.services.job_processor.chunk_pdf_from_blob", mock)
    return mock


@pytest.fixture
def mock_embeddings(monkeypatch):
    """Mock embeddings service"""
    mock = Mock()
    mock.return_value = [[0.1] * 3072, [0.2] * 3072]  # 2 embeddings, 3072 dims
    monkeypatch.setattr("backend.services.job_processor.embed_chunks", mock)
    return mock


@pytest.fixture
def mock_vector_store(monkeypatch):
    """Mock vector store"""
    create_mock = Mock()
    upsert_mock = Mock(return_value=2)  # 2 vectors upserted
    monkeypatch.setattr("backend.services.job_processor.create_collection", create_mock)
    monkeypatch.setattr("backend.services.job_processor.upsert_vectors", upsert_mock)
    return {"create": create_mock, "upsert": upsert_mock}


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


class TestCaseProcessing:
    """Test case processing pipeline"""

    @pytest.mark.asyncio
    async def test_process_case_success(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """process_case successfully processes a case"""
        # Create user and case
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

        # Process case
        result = await process_case(case.id, db)

        assert result["success"] is True
        assert result["chunks_created"] == 2

        # Verify case status changed
        updated_case = db.query(Case).filter(Case.id == case.id).first()
        assert updated_case.status == "ready"

        # Verify job status changed
        updated_job = db.query(ProcessingJob).filter(ProcessingJob.id == job.id).first()
        assert updated_job.status == "completed"

    @pytest.mark.asyncio
    async def test_process_case_chunks_database_storage(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """Chunks are properly stored in database with metadata"""
        # Create user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Case with metadata",
            blob_storage_path="cases/test.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(id=str(uuid4()), case_id=case.id, status="pending")
        db.add(job)
        db.commit()

        # Process case
        await process_case(case.id, db)

        # Verify chunks in database
        chunks = db.query(Chunk).filter(Chunk.case_id == case.id).all()
        assert len(chunks) == 2
        assert chunks[0].page_num == "1"
        assert chunks[0].section_name == "Intro"
        assert chunks[0].content == "Legal text 1"
        assert chunks[0].chunk_sequence == 1
        assert chunks[1].page_num == "2"
        assert chunks[1].section_name == "Body"
        assert chunks[1].content == "Legal text 2"
        assert chunks[1].chunk_sequence == 2

    @pytest.mark.asyncio
    async def test_process_case_vector_storage(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """Vectors are properly stored in vector store"""
        # Create user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Case for vectors",
            blob_storage_path="cases/test.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(id=str(uuid4()), case_id=case.id, status="pending")
        db.add(job)
        db.commit()

        # Process case
        await process_case(case.id, db)

        # Verify vector store was called
        mock_vector_store["create"].assert_called_once()
        mock_vector_store["upsert"].assert_called_once()


class TestErrorHandling:
    """Test error scenarios"""

    @pytest.mark.asyncio
    async def test_process_case_chunking_failure(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """process_case handles chunking failure gracefully"""
        # Mock chunking to raise error
        mock_chunking.side_effect = ValueError("Invalid PDF")

        # Create user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Case with error",
            blob_storage_path="cases/bad.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(id=str(uuid4()), case_id=case.id, status="pending")
        db.add(job)
        db.commit()

        # Process case
        result = await process_case(case.id, db)

        assert result["success"] is False
        assert "Invalid PDF" in result["error"]

    @pytest.mark.asyncio
    async def test_process_case_embedding_failure(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """process_case handles embedding failure gracefully"""
        # Mock embeddings to raise error
        mock_embeddings.side_effect = Exception("Embedding API error")

        # Create user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Case with embedding error",
            blob_storage_path="cases/test.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(id=str(uuid4()), case_id=case.id, status="pending")
        db.add(job)
        db.commit()

        # Process case
        result = await process_case(case.id, db)

        assert result["success"] is False
        assert "Embedding API error" in result["error"]

    @pytest.mark.asyncio
    async def test_process_case_vector_store_failure(
        self, db: Session, mock_storage, mock_chunking, mock_embeddings, mock_vector_store
    ):
        """process_case handles vector store failure gracefully"""
        # Mock vector store upsert to raise error
        mock_vector_store["upsert"].side_effect = Exception("Vector store error")

        # Create user and case
        user = User(email="lawyer@example.com", password_hash="hash")
        db.add(user)
        db.commit()

        case = Case(
            user_id=user.id,
            name="Case with vector store error",
            blob_storage_path="cases/test.pdf",
            status="processing"
        )
        db.add(case)
        db.commit()

        # Create job
        job = ProcessingJob(id=str(uuid4()), case_id=case.id, status="pending")
        db.add(job)
        db.commit()

        # Process case
        result = await process_case(case.id, db)

        assert result["success"] is False
        assert "Vector store error" in result["error"]
