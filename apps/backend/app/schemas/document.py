from pydantic import BaseModel
from datetime import datetime
from typing import Optional
from app.models import DocumentType, ProcessingStatus

class DocumentBase(BaseModel):
    title: str
    filename: str
    type: DocumentType

class DocumentResponse(DocumentBase):
    id: str
    case_id: str
    extracted_text: Optional[str] = None
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    file_path: Optional[str] = None
    processing_status: ProcessingStatus
    error_message: Optional[str] = None
    indexed_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

class DocumentChunkResponse(BaseModel):
    id: str
    document_id: str
    chunk_text: str
    chunk_index: int

    class Config:
        from_attributes = True
