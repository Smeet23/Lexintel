"""SQLAlchemy ORM models for legal RAG app"""
import enum
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, Integer, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime, timezone
import uuid

# Create Base for model definitions
Base = declarative_base()


class MatterStatus(str, enum.Enum):
    """Matter processing status"""
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"
    CANCELLED = "cancelled"


class Matter(Base):
    __tablename__ = "matters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    blob_storage_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False, default="pdf")
    status = Column(String(50), default="processing", nullable=False, index=True)
    celery_task_id = Column(String(255), nullable=True)  # For cancelling processing
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="matter")
    chunks = relationship("Chunk", back_populates="matter")
    queries = relationship("Query", back_populates="matter")
    conversations = relationship("Conversation", back_populates="matter")

    def __repr__(self):
        return f"<Matter(id={self.id}, name={self.name}, status={self.status})>"


class Document(Base):
    """A single uploaded file belonging to a matter"""
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    blob_storage_path = Column(String(500), nullable=False)
    file_type = Column(String(10), nullable=False, default="pdf")
    status = Column(String(50), default="processing", nullable=False, index=True)
    celery_task_id = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)  # Gemini-generated 1-2 sentence summary
    document_type = Column(String(100), nullable=True)  # statute/contract/judgment/regulation/other
    jurisdiction = Column(String(100), nullable=True)  # US/UK/EU/AU/CA/SG/IN/UN/other
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")

    def __repr__(self):
        return f"<Document(id={self.id}, name={self.name}, status={self.status})>"


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    page_num = Column(String(50), nullable=True)
    section_name = Column(String(255), nullable=True)
    section_type = Column(String(100), nullable=True)  # Legal section type (article, exhibit, etc.)
    content = Column(Text, nullable=False)
    embedding_hash = Column(String(255), nullable=True)  # SHA256 hash for deduplication
    concepts = Column(JSON, nullable=True, default=list)  # YAKE-extracted keywords
    chunk_sequence = Column(Integer, nullable=True)  # Order within document
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index('idx_matter_id_sequence', 'matter_id', 'chunk_sequence'),
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, matter_id={self.matter_id}, page={self.page_num})>"


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    title = Column(String(255), nullable=True)  # Auto-generated from first question
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="conversations")
    queries = relationship("Query", back_populates="conversation", order_by="Query.created_at")

    __table_args__ = (
        Index('idx_conv_matter_active', 'matter_id', 'is_deleted', 'updated_at'),
    )

    def __repr__(self):
        return f"<Conversation(id={self.id}, matter_id={self.matter_id}, title={self.title})>"


class Query(Base):
    __tablename__ = "queries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.id"), nullable=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True, default=list)  # List of citation dicts
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    # Relationships
    matter = relationship("Matter", back_populates="queries")
    conversation = relationship("Conversation", back_populates="queries")

    __table_args__ = (
        Index('idx_matter_id_created', 'matter_id', 'created_at'),
        Index('idx_query_conversation_created', 'conversation_id', 'created_at'),
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


class ContractReview(Base):
    """Contract risk analysis results for a document"""
    __tablename__ = "contract_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False, index=True)
    risks = Column(JSON, nullable=True, default=list)  # [{clause, risk_level, explanation, remedy}]
    summary = Column(JSON, nullable=True, default=dict)  # {total_clauses, high_risk, medium_risk, low_risk}
    missing_clauses = Column(JSON, nullable=True, default=list)
    overall_score = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<ContractReview(id={self.id}, matter_id={self.matter_id}, document_id={self.document_id})>"


class Draft(Base):
    """AI-generated legal document drafts"""
    __tablename__ = "drafts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    document_type = Column(String(100), nullable=False)
    instructions = Column(Text, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(JSON, nullable=True, default=list)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<Draft(id={self.id}, matter_id={self.matter_id}, document_type={self.document_type})>"


class AuditLog(Base):
    """Activity audit trail for matters"""
    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=False, index=True)
    action = Column(String(100), nullable=False)
    user = Column(String(255), default="System", nullable=False)
    details = Column(Text, nullable=True)
    sources = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)

    __table_args__ = (
        Index('idx_audit_matter_created', 'matter_id', 'created_at'),
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, matter_id={self.matter_id}, action={self.action})>"


class SavedPrecedent(Base):
    """User-saved precedent research results"""
    __tablename__ = "saved_precedents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    query = Column(Text, nullable=False)
    document_name = Column(String(255), nullable=True)
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=True)
    chunk_content = Column(Text, nullable=True)
    page_num = Column(Integer, nullable=True)
    section_name = Column(String(255), nullable=True)
    relevance_score = Column(String(10), nullable=True)
    tags = Column(JSON, nullable=True, default=list)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    def __repr__(self):
        return f"<SavedPrecedent(id={self.id}, title={self.title})>"
