"""SQLAlchemy ORM models for legal RAG app"""
import enum
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, Text, JSON, Integer, Index, Float
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
    # Authoritative per-matter embedding model name (e.g. "voyage-law-2",
    # "embed-english-v3.0"). Drives admin status + all-matters reindex filter.
    # NULL = needs reindex (vector space unknown).
    embedding_model = Column(String(64), nullable=True, index=True)
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


class AmendmentChain(Base):
    """Groups versioned documents together (statute v1.0 → v1.1 → v2.0)"""
    __tablename__ = "amendment_chains"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # nullable=True: migration 17 drops NOT NULL so ON DELETE SET NULL can fire.
    # The named index idx_chain_matter_canonical in __table_args__ covers this column.
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=True, index=True)
    canonical_document_id = Column(String(255), nullable=False)
    # Text matches migration 12 (sa.Text); String(500) would diverge.
    canonical_name = Column(Text, nullable=False)
    jurisdiction = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    documents = relationship("Document", back_populates="amendment_chain")

    __table_args__ = (
        Index('idx_chain_matter_canonical', 'matter_id', 'canonical_document_id'),
    )

    def __repr__(self):
        return f"<AmendmentChain(id={self.id}, name={self.canonical_name})>"


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
    # Temporal awareness (populated by temporal_extractor during ingestion)
    effective_date = Column(DateTime(timezone=True), nullable=True)
    superseded_date = Column(DateTime(timezone=True), nullable=True)
    version_number = Column(String(20), nullable=True)
    # nullable=False and server_default="unknown" match migration 12 which added
    # this column NOT NULL with server_default="current" then backfilled to "unknown".
    document_status = Column(String(50), nullable=False, server_default="unknown")
    amendment_chain_id = Column(UUID(as_uuid=True), ForeignKey("amendment_chains.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="documents")
    chunks = relationship("Chunk", back_populates="document")
    amendment_chain = relationship("AmendmentChain", back_populates="documents")

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
    authority_metadata = Column(JSON, nullable=True)  # Authority hierarchy data from authority_detector
    # Per-chunk embedding provenance (model name) — enables partial-failure
    # resume during a re-index (only re-embed chunks not yet on the target).
    embedding_model = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    matter = relationship("Matter", back_populates="chunks")
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index('idx_matter_id_sequence', 'matter_id', 'chunk_sequence'),
    )

    def __repr__(self):
        return f"<Chunk(id={self.id}, matter_id={self.matter_id}, page={self.page_num})>"


class CitationNode(Base):
    """A unique legal citation (case, statute, or regulation) in the knowledge graph."""
    __tablename__ = "citation_nodes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    node_type = Column(String(20), nullable=False, index=True)  # case/statute/regulation
    citation_text = Column(String(500), nullable=False)
    name = Column(String(500), nullable=True)
    court = Column(String(200), nullable=True)
    # index=True removed: migration 13 already creates idx_citation_node_year explicitly.
    year = Column(Integer, nullable=True)
    jurisdiction = Column(String(20), nullable=True, index=True)  # US/UK/EU/IN/AU/SG
    authority_score = Column(Float, nullable=True, default=0.0)  # PageRank-based authority score
    # Named index idx_citation_node_document_id created in migration 17.
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=True)
    # Named index idx_citation_node_matter_id created in migration 13 and in __table_args__.
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=True)
    courtlistener_id = Column(String(100), nullable=True)
    source_url = Column(String(500), nullable=True)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_good_law = Column(Boolean, nullable=True)  # None = unknown
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    outgoing_edges = relationship(
        "CitationEdge",
        foreign_keys="CitationEdge.source_id",
        back_populates="source_node",
        cascade="all, delete-orphan",
    )
    incoming_edges = relationship(
        "CitationEdge",
        foreign_keys="CitationEdge.target_id",
        back_populates="target_node",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_citation_node_text", "citation_text"),
        Index("idx_citation_node_type_jurisdiction", "node_type", "jurisdiction"),
        Index("idx_citation_node_matter_id", "matter_id"),
    )

    def __repr__(self):
        return f"<CitationNode(id={self.id}, type={self.node_type}, text={self.citation_text[:60]!r})>"


class CitationEdge(Base):
    """A directed relationship between two citation nodes in the knowledge graph."""
    __tablename__ = "citation_edges"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Named indexes idx_citation_edge_source and idx_citation_edge_source_target in migration 13.
    source_id = Column(
        UUID(as_uuid=True),
        ForeignKey("citation_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Named index idx_citation_edge_target in migration 13.
    target_id = Column(
        UUID(as_uuid=True),
        ForeignKey("citation_nodes.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Named index idx_citation_edge_treatment in migration 13 and __table_args__.
    treatment = Column(String(20), nullable=False)  # FOLLOWS/OVERRULES/CITES/etc.
    # Named index idx_citation_edge_kind in migration 16.
    edge_kind = Column(String(20), nullable=False, default="document_cites", server_default="document_cites")  # values: 'document_cites' (doc→case provenance) | 'case_cites' (real case→case)
    confidence = Column(Float, nullable=True)
    context_excerpt = Column(Text, nullable=True)
    # Named index idx_citation_edge_matter_id in migration 13 and __table_args__.
    matter_id = Column(UUID(as_uuid=True), ForeignKey("matters.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    # Relationships
    source_node = relationship(
        "CitationNode",
        foreign_keys=[source_id],
        back_populates="outgoing_edges",
    )
    target_node = relationship(
        "CitationNode",
        foreign_keys=[target_id],
        back_populates="incoming_edges",
    )

    __table_args__ = (
        Index("idx_citation_edge_source_target", "source_id", "target_id"),
        Index("idx_citation_edge_treatment", "treatment"),
        Index("idx_citation_edge_matter_id", "matter_id"),
    )

    def __repr__(self):
        return f"<CitationEdge(id={self.id}, source={self.source_id}, target={self.target_id}, treatment={self.treatment})>"


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
    citation_verification = Column(JSON, nullable=True)  # Citation verification results
    claim_verification = Column(JSON, nullable=True)  # Claim verification results
    conflict_analysis = Column(JSON, nullable=True)  # Conflict detection results
    # Agentic RAG observability (nullable; non-agentic queries leave these NULL)
    routing_decision = Column(String(50), nullable=True)  # "simple" | "complex" | fast-path label
    agent_iterations = Column(Integer, nullable=True)  # rewrite_count + 1
    agent_latency_ms = Column(Integer, nullable=True)  # total agentic wall-clock
    agentic_metadata = Column(JSON, nullable=True)  # full agentic_metadata blob
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
