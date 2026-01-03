"""Type-safe job payload definitions."""

from pydantic import BaseModel
from typing import Optional, List


class DocumentExtractionJob(BaseModel):
    """Job payload for document text extraction."""
    document_id: str
    case_id: str
    source: str = "upload"

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "case_id": "case_456",
                "source": "upload",
            }
        }


class EmbeddingGenerationJob(BaseModel):
    """Job payload for embedding generation."""
    document_id: str
    chunk_ids: Optional[List[str]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "doc_123",
                "chunk_ids": ["chunk_1", "chunk_2"],
            }
        }
