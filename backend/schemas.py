"""Pydantic v2 schemas for request/response validation"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from uuid import UUID


# ============================================
# MATTER SCHEMAS
# ============================================

class MatterCreate(BaseModel):
    """Create matter request"""
    name: str = Field(..., min_length=1, max_length=255)


class MatterResponse(BaseModel):
    """Matter response (output)"""
    id: UUID
    name: str
    status: str
    file_type: str
    blob_storage_path: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================
# CHUNK SCHEMAS
# ============================================

class ChunkResponse(BaseModel):
    """Chunk response (output)"""
    id: UUID
    matter_id: UUID
    page_num: Optional[str] = None
    section_name: Optional[str] = None
    content: str
    embedding_hash: Optional[str] = None
    chunk_sequence: Optional[int] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ============================================
# QUERY SCHEMAS
# ============================================

class QueryCreate(BaseModel):
    """Query request (ask question)"""
    question: str = Field(..., min_length=1, max_length=1000)
    include_legal_research: bool = Field(False, description="Include CourtListener case law in results")
    conversation_id: Optional[UUID] = Field(None, description="Conversation thread ID to attach this query to")
    as_of_date: Optional[datetime] = Field(
        None,
        description=(
            "Retrieve the law as it stood on this date. When set, includes documents "
            "in force on this date (even if since superseded) and excludes future versions. "
            "Overrides any natural-language temporal intent in the question."
        ),
    )


class CitationData(BaseModel):
    """Citation metadata"""
    page: str
    section: Optional[str] = None
    content_snippet: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_type: Optional[str] = Field(None, description="'document' or 'case_law'")
    url: Optional[str] = Field(None, description="External URL for case law sources")
    # Authority hierarchy fields
    court_level: Optional[str] = Field(
        None,
        description="Court tier: supreme, appellate, trial, administrative, unknown",
    )
    jurisdiction_code: Optional[str] = Field(
        None,
        description="Hierarchical jurisdiction code (e.g. US.federal, US.state.CA, UK)",
    )
    authority_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Combined authority score (court level + jurisdiction + binding)",
    )
    binding_authority: Optional[bool] = Field(
        None,
        description="Whether this source is binding authority in its jurisdiction",
    )
    # Temporal awareness fields
    document_status: Optional[str] = Field(
        None,
        description="Temporal status: current, superseded, repealed, draft, unknown",
    )
    effective_date: Optional[str] = Field(
        None,
        description="ISO date the document/version took effect, if known",
    )


class AgenticMetadata(BaseModel):
    """Observability metadata emitted by the agentic RAG pipeline.

    All fields optional — only populated when settings.agentic_rag_enabled
    handled the query. Non-agentic queries omit this entirely.
    """
    complexity: Optional[str] = None
    query_type: Optional[str] = None
    jurisdiction: Optional[str] = None
    entities: list[str] = []
    rewrite_count: int = 0
    retrieval_score: float = 0.0
    sub_questions: list[str] = []
    requires_external_research: Optional[bool] = None
    temporal_scope: Optional[str] = None
    research_plan: Optional[dict] = None
    strategy_log: list[str] = []
    sufficiency_signals: Optional[dict] = None
    routing_decision: Optional[str] = None
    agent_iterations: Optional[int] = None
    agent_latency_ms: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class QueryResponse(BaseModel):
    """Query response (output)"""
    id: UUID
    question: str
    answer: str
    citations: list[CitationData] = []
    created_at: datetime
    routing_decision: Optional[str] = None
    agent_iterations: Optional[int] = None
    agent_latency_ms: Optional[int] = None
    agentic_metadata: Optional[AgenticMetadata] = None

    model_config = ConfigDict(from_attributes=True)


# ============================================
# CITATION VERIFICATION SCHEMAS
# ============================================

class CitationVerificationItem(BaseModel):
    """Single citation verification result"""
    index: int
    raw_text: str
    case_name: Optional[str] = None
    court: Optional[str] = None
    date: Optional[str] = None
    jurisdiction: Optional[str] = None
    status: str = "unverifiable"  # verified, partial, unverified, not_found, unverifiable
    case_status: Optional[str] = None  # good_law, caution, bad_law, unknown
    confidence: float = 0.0
    source_url: Optional[str] = None
    citation_count: Optional[int] = None
    quote_match: Optional[float] = None
    matched_source_idx: Optional[int] = None
    verification_method: Optional[str] = None


class CitationVerificationSummary(BaseModel):
    """Summary of verification results"""
    total: int = 0
    verified: int = 0
    partial: int = 0
    unverified: int = 0
    not_found: int = 0
    unverifiable: int = 0


# ============================================
# CLAIM VERIFICATION SCHEMAS (Grounding)
# ============================================

class ClaimVerificationItem(BaseModel):
    """Single claim verification result"""
    claim_text: str
    cited_sources: list[int] = []
    status: str = "uncited"  # supported, partially_supported, unsupported, uncited
    confidence: float = 0.0
    verification_tier: int = 0  # 0=uncited, 1=rule, 2=cosine, 3=LLM
    issues: list[str] = []
    source_excerpt: Optional[str] = None
    requires_review: bool = False


class ClaimVerificationSummary(BaseModel):
    """Summary of claim verification results"""
    total: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    uncited: int = 0


# ============================================
# CONFLICT DETECTION SCHEMAS
# ============================================

class ConflictChunkInfo(BaseModel):
    """One side of a detected conflict"""
    document_id: str
    document_name: str
    page_num: str
    section_name: Optional[str] = None
    content_snippet: str
    credibility: dict  # {score, authority_type, authority_weight, recency_weight, ...}


class ConflictItem(BaseModel):
    """Single detected conflict between two sources"""
    id: str
    severity: str  # "high" or "medium"
    contradiction_score: float = Field(ge=0.0, le=1.0)
    chunk_a: ConflictChunkInfo
    chunk_b: ConflictChunkInfo
    recommended_source: str  # "a" or "b"
    explanation: str


class ConflictSummary(BaseModel):
    """Summary of all conflicts detected"""
    total_conflicts: int = 0
    high_severity: int = 0
    medium_severity: int = 0
    recommended_action: str = ""


# ============================================
# Problem Formulation (Issue Classification)
# ============================================

class IdentifiedIssue(BaseModel):
    domain: str = ""  # free-text: "contract", "tort", "corporate"
    issue: str = ""  # "duty of care standard"
    legal_question: str = ""  # precise research question
    confidence: float = 0.0  # 0.0-1.0
    key_facts: list = Field(default_factory=list)  # facts triggering this issue


class IssueAnalysis(BaseModel):
    issues: list = Field(default_factory=list)  # list of IdentifiedIssue dicts
    missing_information: list = Field(default_factory=list)  # what info would help



# ============================================
# /ask FULL RESPONSE (typed contract for OpenAPI)
# ============================================

class AskResponse(BaseModel):
    """Full response for POST /matters/{id}/ask.

    Permissive (``extra='allow'``) and all-optional so the standard and agentic
    pipelines can add/omit fields without anything being dropped from the payload
    — this publishes the contract in OpenAPI without changing the response.
    """
    answer: Optional[str] = None
    sources: list = Field(default_factory=list)
    citations: list = Field(default_factory=list)
    matter_id: Optional[str] = None
    query: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    confidence: Optional[dict] = None
    citation_verification: Optional[dict] = None
    claim_verification: Optional[dict] = None
    conflict_analysis: Optional[dict] = None
    issue_analysis: Optional[dict] = None
    temporal_filter_relaxed: Optional[bool] = None
    as_of_date: Optional[str] = None
    error: Optional[str] = None
    query_id: Optional[str] = None
    # agentic-path extras
    routing_decision: Optional[str] = None
    agentic_metadata: Optional[dict] = None

    model_config = ConfigDict(extra="allow")
