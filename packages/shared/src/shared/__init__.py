"""LexIntel shared package - models, schemas, and utilities."""

from .database import async_session, engine
from .models import Base, Case, Document, DocumentChunk, ProcessingStatus
from .utils import (
    setup_logging,
    PermanentError,
    RetryableError,
    DocumentNotFound,
    FileNotFound,
    ExtractionFailed,
)
from .schemas.jobs import DocumentExtractionJob, EmbeddingGenerationJob
from .extraction import (
    chunk_text,
    clean_text,
    extract_file,
    extract_pdf,
    extract_docx,
    extract_txt,
)

__version__ = "0.1.0"
__all__ = [
    "async_session",
    "engine",
    "Base",
    "Case",
    "Document",
    "DocumentChunk",
    "ProcessingStatus",
    "setup_logging",
    "PermanentError",
    "RetryableError",
    "DocumentNotFound",
    "FileNotFound",
    "ExtractionFailed",
    "DocumentExtractionJob",
    "EmbeddingGenerationJob",
    "chunk_text",
    "clean_text",
    "extract_file",
    "extract_pdf",
    "extract_docx",
    "extract_txt",
]
