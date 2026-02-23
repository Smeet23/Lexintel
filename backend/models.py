"""SQLAlchemy ORM models for legal RAG app"""
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, Integer, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone
import uuid

# Demo user UUID (seeded in migration 2)
DEMO_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")

# Create Base for model definitions
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), nullable=False, unique=True, index=True)
    password_hash = Column(String(255), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)


class Case(Base):
    __tablename__ = "cases"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, default=DEMO_USER_ID, index=True)
    name = Column(String(255), nullable=False)
    blob_storage_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False, default="pdf")
    status = Column(String(50), default="processing", nullable=False, index=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    chunks = relationship("Chunk", back_populates="case")
    queries = relationship("Query", back_populates="case")

    def __repr__(self):
        return f"<Case(id={self.id}, name={self.name}, status={self.status})>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False, index=True)
    page_num = Column(String(50), nullable=True)
    section_name = Column(String(255), nullable=True)
    section_type = Column(String(100), nullable=True)  # Legal section type (article, exhibit, etc.)
    content = Column(Text, nullable=False)
    embedding_hash = Column(String(255), nullable=True)  # SHA256 hash for deduplication
    chunk_sequence = Column(Integer, nullable=True)  # Order within case
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    case = relationship("Case", back_populates="chunks")

    __table_args__ = (
        Index('idx_case_id_sequence', 'case_id', 'chunk_sequence'),
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, case_id={self.case_id}, page={self.page_num})>"


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False, index=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, default=DEMO_USER_ID, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True, default=list)  # List of citation dicts
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    case = relationship("Case", back_populates="queries")

    __table_args__ = (
        Index('idx_case_id_created', 'case_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Query(id={self.id}, case_id={self.case_id})>"


class ProcessingJob(Base):
    """Background processing job for case document analysis"""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    case_id = Column(UUID(as_uuid=True), ForeignKey("cases.id"), nullable=False, index=True)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(String(500), nullable=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    next_retry_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, case_id={self.case_id}, status={self.status})>"
