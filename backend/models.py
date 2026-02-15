"""SQLAlchemy ORM models for legal RAG app"""
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, Integer, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone
import uuid
import enum

# Create Base for model definitions
Base = declarative_base()


class MatterStatus(str, enum.Enum):
    """Matter processing status"""
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


class FileType(str, enum.Enum):
    """Supported document file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class Matter(Base):
    __tablename__ = "matters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    blob_storage_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False, default="pdf")
    status = Column(String(50), default="processing", nullable=False, index=True)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    chunks = relationship("Chunk", back_populates="matter")
    queries = relationship("Query", back_populates="matter")

    def __repr__(self):
        return f"<Matter(id={self.id}, name={self.name}, status={self.status})>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    page_num = Column(String(50), nullable=True)
    section_name = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    embedding_hash = Column(String(255), nullable=True)  # SHA256 hash for deduplication
    chunk_sequence = Column(Integer, nullable=True)  # Order within matter
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="chunks")

    __table_args__ = (
        Index('idx_matter_id_sequence', 'matter_id', 'chunk_sequence'),
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, matter_id={self.matter_id}, page={self.page_num})>"


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True, default=list)  # List of citation dicts
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    matter = relationship("Matter", back_populates="queries")

    __table_args__ = (
        Index('idx_matter_id_created', 'matter_id', 'created_at'),
    )

    def __repr__(self):
        return f"<Query(id={self.id}, matter_id={self.matter_id})>"


class ProcessingJob(Base):
    """Background processing job for matter document analysis"""
    __tablename__ = "processing_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    status = Column(String(50), default="pending")  # pending, processing, completed, failed
    error_message = Column(String(500), nullable=True)
    attempts = Column(Integer, default=0)
    max_attempts = Column(Integer, default=3)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    next_retry_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<ProcessingJob(id={self.id}, matter_id={self.matter_id}, status={self.status})>"
