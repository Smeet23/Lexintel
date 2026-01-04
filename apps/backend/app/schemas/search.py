"""Search API schemas"""

from pydantic import BaseModel, Field
from typing import List, Optional


class SearchChunkResult(BaseModel):
    """Single search result"""

    chunk_id: str
    document_id: str
    document_title: str
    chunk_text: str
    chunk_index: int
    relevance_score: float
    search_type: str  # "full_text", "semantic", "hybrid"


class FullTextSearchRequest(BaseModel):
    """Full-text search request"""

    case_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "case-123",
                "query": "contract terms",
                "limit": 10,
            }
        }


class SemanticSearchRequest(BaseModel):
    """Semantic search request"""

    case_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "case-123",
                "query": "What are the main obligations in the contract?",
                "limit": 10,
                "threshold": 0.5,
            }
        }


class HybridSearchRequest(BaseModel):
    """Hybrid search request"""

    case_id: str
    query: str = Field(..., min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=100)
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    ft_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    class Config:
        json_schema_extra = {
            "example": {
                "case_id": "case-123",
                "query": "contract terms and conditions",
                "limit": 10,
                "threshold": 0.5,
                "ft_weight": 0.5,
                "semantic_weight": 0.5,
            }
        }


class SearchResponse(BaseModel):
    """Search response"""

    results: List[SearchChunkResult]
    total: int
    search_type: str
    execution_time_ms: float = 0.0

    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "chunk_id": "chunk-1",
                        "document_id": "doc-123",
                        "document_title": "Contract Agreement",
                        "chunk_text": "The parties agree to the following terms...",
                        "chunk_index": 0,
                        "relevance_score": 0.95,
                        "search_type": "hybrid",
                    }
                ],
                "total": 1,
                "search_type": "hybrid",
                "execution_time_ms": 45.5,
            }
        }
