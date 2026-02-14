"""Pydantic v2 schemas for request/response validation"""
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import Optional
from uuid import UUID


# ============================================
# CASE SCHEMAS
# ============================================

class CaseCreate(BaseModel):
    """Create case request"""
    name: str = Field(..., min_length=1, max_length=255)


class CaseResponse(BaseModel):
    """Case response (output)"""
    id: UUID
    name: str
    status: str
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
    case_id: UUID
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


class CitationData(BaseModel):
    """Citation metadata"""
    page: str
    section: Optional[str] = None
    content_snippet: str
    score: Optional[float] = Field(None, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    """Query response (output)"""
    id: UUID
    question: str
    answer: str
    citations: list[CitationData] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
