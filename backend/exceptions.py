"""Custom exception hierarchy for LexIntel"""


class LexIntelException(Exception):
    """Base exception for all LexIntel errors"""

    def __init__(self, message: str, detail: str = None):
        """
        Initialize LexIntelException

        Args:
            message: Human-readable error message
            detail: Additional context or details about the error
        """
        self.message = message
        self.detail = detail
        super().__init__(message)


# ============================================
# Storage Exceptions
# ============================================

class StorageException(LexIntelException):
    """Base exception for storage-related errors"""

    pass


class BlobUploadException(StorageException):
    """Raised when uploading to blob storage fails"""

    pass


class BlobDownloadException(StorageException):
    """Raised when downloading from blob storage fails"""

    pass


class BlobNotFoundException(BlobDownloadException):
    """Raised when the requested blob does not exist (distinct from a transient
    storage failure). Subclasses BlobDownloadException so existing
    ``except BlobDownloadException`` handlers still catch it as a fallback, while
    callers that care can map it to HTTP 404 instead of 500."""

    pass


class BlobDeleteException(StorageException):
    """Raised when deleting from blob storage fails"""

    pass


# ============================================
# RAG Exceptions
# ============================================

class RAGException(LexIntelException):
    """Base exception for RAG pipeline errors"""

    pass


class EmbeddingException(RAGException):
    """Raised when embedding generation fails"""

    pass


class VectorStoreException(RAGException):
    """Raised when vector store operations fail"""

    pass


class QueryProcessingException(RAGException):
    """Raised when query processing or LLM answer generation fails"""

    pass


# ============================================
# Document Processing Exceptions
# ============================================

class DocumentProcessingException(LexIntelException):
    """Base exception for document processing errors"""

    pass


class ChunkingException(DocumentProcessingException):
    """Raised when document chunking fails"""

    pass


class InvalidPDFException(DocumentProcessingException):
    """Raised when PDF validation fails"""

    pass


# ============================================
# Validation Exceptions
# ============================================

class ValidationException(LexIntelException):
    """Raised when input validation fails"""

    pass
