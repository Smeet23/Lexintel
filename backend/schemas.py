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


class CitationData(BaseModel):
    """Citation metadata"""
    page: str
    section: Optional[str] = None
    content_snippet: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)
    source_type: Optional[str] = Field(None, description="'document' or 'case_law'")
    url: Optional[str] = Field(None, description="External URL for case law sources")


class QueryResponse(BaseModel):
    """Query response (output)"""
    id: UUID
    question: str
    answer: str
    citations: list[CitationData] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
